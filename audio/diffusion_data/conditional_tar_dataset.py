"""Train-only conditional legal-YTB streaming dataset.

Only signed per-TAR metadata sidecars are read.  Audio archives are opened in
forward-only ``r|*`` mode and members are decoded from memory; no member is ever
extracted.  Rows with no valid conditional target were removed before this loader
performs rank/worker assignment.
"""

from __future__ import annotations

import io
import math
import os
import random
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .conditional_metadata import (
    CONDITIONAL_METADATA_SCHEMA_VERSION,
    TARGET_COLUMNS,
    VIEW_COUNT,
    VIEW_NAMES,
    VIEW_REQUESTS,
    ConditionalTarPartition,
    conditional_metadata_schema,
    load_conditional_metadata_index,
    select_view_id,
    sha256_file,
    stable_view_permutation,
)
from .local_tar_dataset import (
    _bounded_shuffle,
    _normalise_member_name,
    _resolve_distributed,
    _safe_relative_path,
    _stable_seed,
    video_key_from_sample_key,
)


@dataclass(slots=True)
class ConditionalDatasetCounters:
    epoch: int
    rank: int
    world_size: int
    worker_id: int
    num_workers: int
    assigned_tar_files: int = 0
    metadata_rows_read: int = 0
    all_invalid_rows_skipped: int = 0
    duplicate_member_rows: int = 0
    tar_files_opened: int = 0
    tar_members_scanned: int = 0
    audio_payload_bytes_read: int = 0
    audio_decode_errors: int = 0
    missing_audio_members: int = 0
    samples_trimmed_for_rank_balance: int = 0
    samples_yielded: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _ConditionalRow:
    sample_key: str
    audio_tar_path: str
    audio_member: str
    duration_sec: float
    country_code: str
    content_has_code_switching: bool
    targets: tuple[str | None, str | None, str | None, str | None]
    valid_view_mask: tuple[bool, bool, bool, bool]
    view_permutation: tuple[int, int, int, int]
    applied_code_switching_mask: tuple[bool, bool, bool, bool]
    target_token_counts: tuple[int, int, int, int]
    canvas_sizes: tuple[int, int, int, int]


class ConditionalLocalTarDataset(IterableDataset[dict[str, Any]]):
    """Stream one deterministic valid target view per train sample and epoch."""

    def __init__(
        self,
        *,
        local_root: str | Path,
        metadata_index_path: str | Path,
        view_seed: int,
        seed: int = 0,
        shuffle_buffer_size: int = 32,
        parquet_batch_size: int = 65_536,
        target_sample_rate: int | None = 16_000,
        rank: int | None = None,
        world_size: int | None = None,
        validate_sidecar_hashes: bool = True,
        validate_tar_files: bool = True,
        strict_missing_members: bool = True,
        error_policy: Literal["raise", "skip"] = "raise",
        max_audio_bytes: int = 64 << 20,
        balance_distributed_shards: bool = True,
    ) -> None:
        super().__init__()
        self.local_root = Path(local_root).expanduser().resolve()
        self.metadata_index_path = Path(metadata_index_path).expanduser().resolve()
        self.view_seed = int(view_seed)
        self.seed = int(seed)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.parquet_batch_size = int(parquet_batch_size)
        self.target_sample_rate = target_sample_rate
        self.rank, self.world_size = _resolve_distributed(rank, world_size)
        self.strict_missing_members = bool(strict_missing_members)
        self.error_policy = error_policy
        self.max_audio_bytes = int(max_audio_bytes)
        self.balance_distributed_shards = bool(balance_distributed_shards)
        self._epoch = 0
        self._last_counters: ConditionalDatasetCounters | None = None

        if self.shuffle_buffer_size < 1 or self.parquet_batch_size < 1:
            raise ValueError("shuffle/parquet batch sizes must be positive")
        if target_sample_rate is not None and target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be positive or None")
        if error_policy not in ("raise", "skip"):
            raise ValueError("error_policy must be 'raise' or 'skip'")
        if self.max_audio_bytes <= 0:
            raise ValueError("max_audio_bytes must be positive")

        self._index = load_conditional_metadata_index(
            self.metadata_index_path,
            # Authenticate the small index first. Sidecar bytes are distributed
            # across ranks below so DDP128 performs one corpus-wide hash pass,
            # not 128 identical passes.
            validate_sidecar_hashes=False,
        )
        index_view_seed = self._index.policy.get("view_seed")
        if index_view_seed is None or int(index_view_seed) != self.view_seed:
            raise ValueError(
                f"view_seed {self.view_seed} does not match signed metadata "
                f"view_seed {index_view_seed!r}"
            )
        self._partitions = {item.tar_path: item for item in self._index.partitions}
        if validate_tar_files:
            missing = [
                tar_path
                for tar_path in self._partitions
                if not (self.local_root / _safe_relative_path(tar_path)).is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"conditional metadata references missing TARs: {missing[:8]}"
                )
        self._tar_sample_counts = {
            item.tar_path: item.any_valid_count for item in self._index.partitions
        }
        if validate_sidecar_hashes:
            for tar_path in self.assigned_tar_paths():
                partition = self._partitions[tar_path]
                sidecar = self.metadata_index_path.parent / partition.sidecar_path
                if sha256_file(sidecar) != partition.sidecar_sha256:
                    raise ValueError(f"conditional sidecar hash mismatch: {sidecar}")
        if self.balance_distributed_shards and self.world_size > 1:
            loads = self.samples_per_rank()
            assert loads is not None
            if min(loads) == 0:
                raise ValueError(
                    "distributed TAR assignment leaves at least one rank empty; "
                    "reduce world_size"
                )

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def last_counters(self) -> ConditionalDatasetCounters | None:
        return self._last_counters

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("epoch must be non-negative")
        self._epoch = int(epoch)

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "seed": self.seed,
            "view_seed": self.view_seed,
            "repo_id": self._index.repo_id,
            "revision": self._index.revision,
            "metadata_index_sha256": self._index.content_sha256,
            "split_manifest_sha256": self._index.split_manifest_sha256,
            "tokenizer_sha256": self._index.tokenizer_sha256,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for key, expected in (
            ("seed", self.seed),
            ("view_seed", self.view_seed),
            ("repo_id", self._index.repo_id),
            ("revision", self._index.revision),
            ("metadata_index_sha256", self._index.content_sha256),
            ("split_manifest_sha256", self._index.split_manifest_sha256),
            ("tokenizer_sha256", self._index.tokenizer_sha256),
        ):
            if key in state and state[key] != expected:
                raise ValueError(
                    f"conditional dataset state mismatch for {key}: "
                    f"{state[key]!r} != {expected!r}"
                )
        self.set_epoch(int(state["epoch"]))

    def _balanced_assignments(
        self, paths: Sequence[str], bucket_count: int
    ) -> tuple[tuple[tuple[str, ...], ...], tuple[int, ...]]:
        if bucket_count <= 0:
            raise ValueError("bucket_count must be positive")
        ordered = sorted(
            paths, key=lambda path: self._tar_sample_counts[path], reverse=True
        )
        assignments: list[list[str]] = [[] for _ in range(bucket_count)]
        loads = [0] * bucket_count
        for path in ordered:
            bucket = min(range(bucket_count), key=lambda index: (loads[index], index))
            assignments[bucket].append(path)
            loads[bucket] += self._tar_sample_counts[path]
        return tuple(tuple(items) for items in assignments), tuple(loads)

    def _epoch_tar_order(self) -> list[str]:
        ordered = sorted(self._partitions)
        random.Random(_stable_seed(self.seed, self.epoch, "conditional-tar-order")).shuffle(
            ordered
        )
        return ordered

    def samples_per_rank(self) -> tuple[int, ...] | None:
        if not self.balance_distributed_shards or self.world_size <= 1:
            return None
        _, loads = self._balanced_assignments(self._epoch_tar_order(), self.world_size)
        return loads

    def assigned_tar_paths(
        self, *, worker_id: int = 0, num_workers: int = 1
    ) -> tuple[str, ...]:
        if num_workers <= 0 or worker_id < 0 or worker_id >= num_workers:
            raise ValueError("invalid worker assignment")
        ordered = self._epoch_tar_order()
        if not self.balance_distributed_shards or self.world_size <= 1:
            rank_paths = ordered[self.rank :: self.world_size]
            return tuple(rank_paths[worker_id::num_workers])
        rank_assignments, _ = self._balanced_assignments(ordered, self.world_size)
        rank_paths = list(rank_assignments[self.rank])
        random.Random(
            _stable_seed(self.seed, self.epoch, "conditional-rank-order", self.rank)
        ).shuffle(rank_paths)
        worker_assignments, _ = self._balanced_assignments(rank_paths, num_workers)
        worker_paths = list(worker_assignments[worker_id])
        random.Random(
            _stable_seed(
                self.seed,
                self.epoch,
                "conditional-worker-order",
                self.rank,
                worker_id,
            )
        ).shuffle(worker_paths)
        return tuple(worker_paths)

    def assigned_sidecar_paths(
        self, *, worker_id: int = 0, num_workers: int = 1
    ) -> tuple[str, ...]:
        """Return absolute signed sidecars corresponding to assigned TARs."""

        return tuple(
            str(self.metadata_index_path.parent / self._partitions[tar].sidecar_path)
            for tar in self.assigned_tar_paths(
                worker_id=worker_id, num_workers=num_workers
            )
        )

    @staticmethod
    def _fair_capacity_quotas(capacities: Sequence[int], total: int) -> tuple[int, ...]:
        if total < 0 or total > sum(capacities):
            raise ValueError("quota total is outside worker capacity")
        quotas = [0] * len(capacities)
        remaining = total
        while remaining:
            active = [i for i, capacity in enumerate(capacities) if quotas[i] < capacity]
            if not active:
                raise RuntimeError("could not distribute worker quota")
            share = max(1, remaining // len(active))
            for index in active:
                added = min(share, capacities[index] - quotas[index], remaining)
                quotas[index] += added
                remaining -= added
                if not remaining:
                    break
        return tuple(quotas)

    def assigned_sample_quota(
        self, *, worker_id: int = 0, num_workers: int = 1
    ) -> int | None:
        loads = self.samples_per_rank()
        if loads is None:
            return None
        if num_workers <= 0 or worker_id < 0 or worker_id >= num_workers:
            raise ValueError("invalid worker assignment")
        ordered = self._epoch_tar_order()
        rank_assignments, _ = self._balanced_assignments(ordered, self.world_size)
        rank_paths = list(rank_assignments[self.rank])
        random.Random(
            _stable_seed(self.seed, self.epoch, "conditional-rank-order", self.rank)
        ).shuffle(rank_paths)
        _, worker_loads = self._balanced_assignments(rank_paths, num_workers)
        return self._fair_capacity_quotas(worker_loads, min(loads))[worker_id]

    def configuration_summary(self) -> dict[str, Any]:
        rank_loads = self.samples_per_rank()
        return {
            "loader": type(self).__name__,
            "local_root": str(self.local_root),
            "metadata_index": str(self.metadata_index_path),
            "schema_version": self._index.raw["schema_version"],
            "repo_id": self._index.repo_id,
            "revision": self._index.revision,
            "metadata_index_sha256": self._index.content_sha256,
            "source_metadata_sha256": self._index.source_metadata_sha256,
            "split_manifest_sha256": self._index.split_manifest_sha256,
            "tokenizer_sha256": self._index.tokenizer_sha256,
            "preprocessing_code_sha256": self._index.preprocessing_code_sha256,
            "sidecar_count": len(self._index.partitions),
            "total_train_rows": self._index.total_train_rows,
            "per_view_valid_counts": self._index.per_view_valid_counts,
            "any_valid_rows": self._index.any_valid_rows,
            "rank": self.rank,
            "world_size": self.world_size,
            "seed": self.seed,
            "view_seed": self.view_seed,
            "epoch": self.epoch,
            "shuffle_buffer_size": self.shuffle_buffer_size,
            "target_sample_rate": self.target_sample_rate,
            "balance_distributed_shards": self.balance_distributed_shards,
            "samples_per_rank_before_trim": rank_loads,
            "samples_per_rank_after_trim": None if rank_loads is None else min(rank_loads),
        }

    def _metadata_for_partition(
        self,
        partition: ConditionalTarPartition,
        counters: ConditionalDatasetCounters,
    ) -> dict[str, _ConditionalRow]:
        sidecar = self.metadata_index_path.parent / partition.sidecar_path
        parquet = pq.ParquetFile(sidecar)
        if not parquet.schema_arrow.equals(conditional_metadata_schema(), check_metadata=False):
            raise ValueError(f"conditional sidecar schema mismatch: {sidecar}")
        rows: dict[str, _ConditionalRow] = {}
        observed_rows = 0
        observed_any_valid = 0
        columns = list(conditional_metadata_schema().names)
        for batch in parquet.iter_batches(
            batch_size=self.parquet_batch_size,
            columns=columns,
            use_threads=False,
        ):
            counters.metadata_rows_read += batch.num_rows
            observed_rows += batch.num_rows
            for raw in batch.to_pylist():
                if raw["metadata_schema_version"] != CONDITIONAL_METADATA_SCHEMA_VERSION:
                    raise ValueError("conditional row schema version mismatch")
                if raw["audio_tar_path"] != partition.tar_path:
                    raise ValueError("conditional row TAR path differs from its partition")
                valid = tuple(bool(value) for value in raw["valid_view_mask"])
                permutation = tuple(int(value) for value in raw["view_permutation"])
                if len(valid) != VIEW_COUNT or sorted(permutation) != list(range(VIEW_COUNT)):
                    raise ValueError("invalid view mask/permutation in conditional sidecar")
                if not any(valid):
                    counters.all_invalid_rows_skipped += 1
                    continue
                observed_any_valid += 1
                sample_key = str(raw["sample_key"])
                if permutation != stable_view_permutation(sample_key, self.view_seed):
                    raise ValueError("stored conditional view permutation mismatch")
                expected_view = select_view_id(
                    sample_key, self.view_seed, self.epoch, valid
                )
                # The sidecar is signed, but verify that its stored permutation and
                # the live selection agree.  This catches an incompatible algorithm
                # change instead of silently altering resumed training.
                filtered = tuple(view for view in permutation if valid[view])
                if expected_view != filtered[self.epoch % len(filtered)]:
                    raise ValueError("conditional view selection mismatch")
                member = _normalise_member_name(str(raw["audio_member"]))
                if member in rows:
                    counters.duplicate_member_rows += 1
                    if self.error_policy == "raise":
                        raise ValueError(
                            f"duplicate conditional metadata member {partition.tar_path}:{member}"
                        )
                    continue
                targets = tuple(raw[name] for name in TARGET_COLUMNS)
                row = _ConditionalRow(
                    sample_key=str(raw["sample_key"]),
                    audio_tar_path=partition.tar_path,
                    audio_member=member,
                    duration_sec=float(raw["duration_sec"]),
                    country_code=str(raw["country_code"]),
                    content_has_code_switching=bool(raw["content_has_code_switching"]),
                    targets=targets,  # type: ignore[arg-type]
                    valid_view_mask=valid,  # type: ignore[arg-type]
                    view_permutation=permutation,  # type: ignore[arg-type]
                    applied_code_switching_mask=tuple(
                        bool(value) for value in raw["applied_code_switching_mask"]
                    ),  # type: ignore[arg-type]
                    target_token_counts=tuple(
                        int(value) for value in raw["target_token_counts"]
                    ),  # type: ignore[arg-type]
                    canvas_sizes=tuple(int(value) for value in raw["canvas_sizes"]),  # type: ignore[arg-type]
                )
                rows[member] = row
        if observed_rows != partition.row_count:
            raise ValueError(f"conditional sidecar row count mismatch: {sidecar}")
        if observed_any_valid != partition.any_valid_count:
            raise ValueError(f"conditional sidecar valid row count mismatch: {sidecar}")
        return rows

    def _decode_audio(self, payload: bytes) -> tuple[np.ndarray, int]:
        try:
            import soundfile as sf
        except ImportError:
            import wave

            with wave.open(io.BytesIO(payload), "rb") as wav:
                if wav.getcomptype() != "NONE":
                    raise ValueError(f"unsupported compressed WAV: {wav.getcomptype()}")
                channels = wav.getnchannels()
                width = wav.getsampwidth()
                sample_rate = wav.getframerate()
                raw = wav.readframes(wav.getnframes())
            if channels <= 0:
                raise ValueError("WAV has no channels")
            if width == 1:
                decoded = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
            elif width == 2:
                decoded = np.frombuffer(raw, np.dtype("<i2")).astype(np.float32) / 32768.0
            elif width == 3:
                octets = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
                integer = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
                decoded = ((integer ^ 0x800000) - 0x800000).astype(np.float32) / 8388608.0
            elif width == 4:
                decoded = np.frombuffer(raw, np.dtype("<i4")).astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"unsupported PCM WAV sample width: {width}")
            if decoded.size % channels:
                raise ValueError("WAV payload is not divisible by channel count")
            samples = decoded.reshape(-1, channels)
        else:
            samples, sample_rate = sf.read(
                io.BytesIO(payload), dtype="float32", always_2d=True
            )
        if samples.shape[0] == 0:
            raise ValueError("decoded audio is empty")
        waveform = samples.mean(axis=1, dtype=np.float32)
        sample_rate = int(sample_rate)
        if self.target_sample_rate is not None and sample_rate != self.target_sample_rate:
            from scipy.signal import resample_poly

            divisor = math.gcd(sample_rate, self.target_sample_rate)
            waveform = resample_poly(
                waveform,
                self.target_sample_rate // divisor,
                sample_rate // divisor,
            ).astype(np.float32, copy=False)
            sample_rate = self.target_sample_rate
        return np.ascontiguousarray(waveform, dtype=np.float32), sample_rate

    def _sample_from_member(
        self,
        row: _ConditionalRow,
        payload: bytes,
        counters: ConditionalDatasetCounters,
    ) -> dict[str, Any] | None:
        view_id = select_view_id(
            row.sample_key, self.view_seed, self.epoch, row.valid_view_mask
        )
        transcription = row.targets[view_id]
        if not isinstance(transcription, str) or not transcription.strip():
            raise ValueError("selected valid conditional target is empty")
        try:
            audio, sampling_rate = self._decode_audio(payload)
        except Exception:
            if self.error_policy == "raise":
                raise
            counters.audio_decode_errors += 1
            return None
        request_code_switching, request_diacritics = VIEW_REQUESTS[view_id]
        applied_code_switching = row.applied_code_switching_mask[view_id]
        targets = dict(zip(TARGET_COLUMNS, row.targets, strict=True))
        target = {
            "target_transcription": transcription,
            "transcription": transcription,
            "country_code": row.country_code,
            "content_has_code_switching": row.content_has_code_switching,
            "has_code_switching": row.content_has_code_switching,
            "request_code_switching": request_code_switching,
            "request_diacritics": request_diacritics,
            "applied_code_switching": applied_code_switching,
            "view_id": view_id,
            "view_name": VIEW_NAMES[view_id],
        }
        return {
            "sample_key": row.sample_key,
            "video_key": video_key_from_sample_key(row.sample_key),
            "audio_tar_path": row.audio_tar_path,
            "audio_member": row.audio_member,
            "duration_sec": row.duration_sec,
            "audio": audio,
            "sampling_rate": sampling_rate,
            "target_transcription": transcription,
            "transcription": transcription,
            "country_code": row.country_code,
            "content_has_code_switching": row.content_has_code_switching,
            "request_code_switching": request_code_switching,
            "request_diacritics": request_diacritics,
            "applied_code_switching": applied_code_switching,
            "view_id": view_id,
            "view_name": VIEW_NAMES[view_id],
            "valid_view_mask": row.valid_view_mask,
            "view_permutation": row.view_permutation,
            "applied_code_switching_mask": row.applied_code_switching_mask,
            "target_token_counts": row.target_token_counts,
            "canvas_sizes": row.canvas_sizes,
            "target_token_count": row.target_token_counts[view_id],
            "canvas_size": row.canvas_sizes[view_id],
            **targets,
            "target": target,
        }

    def _stream_partition(
        self,
        partition: ConditionalTarPartition,
        metadata: dict[str, _ConditionalRow],
        counters: ConditionalDatasetCounters,
    ) -> Iterator[dict[str, Any]]:
        if not metadata:
            return
        tar_file = self.local_root / _safe_relative_path(partition.tar_path)
        counters.tar_files_opened += 1
        with tarfile.open(tar_file, mode="r|*") as archive:
            for member in archive:
                counters.tar_members_scanned += 1
                if not member.isfile():
                    continue
                name = _normalise_member_name(member.name)
                row = metadata.pop(name, None)
                if row is None:
                    continue
                if member.size < 0 or member.size > self.max_audio_bytes:
                    if self.error_policy == "raise":
                        raise ValueError(
                            f"audio member size out of bounds: {partition.tar_path}:{name}"
                        )
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    if self.error_policy == "raise":
                        raise ValueError(f"could not read TAR member: {name}")
                    continue
                payload = stream.read(self.max_audio_bytes + 1)
                if len(payload) != member.size:
                    if self.error_policy == "raise":
                        raise ValueError(f"short TAR member read: {name}")
                    continue
                counters.audio_payload_bytes_read += len(payload)
                sample = self._sample_from_member(row, payload, counters)
                if sample is not None:
                    yield sample
        if metadata:
            counters.missing_audio_members += len(metadata)
            if self.strict_missing_members:
                raise FileNotFoundError(
                    f"{partition.tar_path} is missing metadata members: "
                    f"{list(metadata)[:8]}"
                )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        tar_paths = self.assigned_tar_paths(worker_id=worker_id, num_workers=num_workers)
        quota = self.assigned_sample_quota(worker_id=worker_id, num_workers=num_workers)
        counters = ConditionalDatasetCounters(
            epoch=self.epoch,
            rank=self.rank,
            world_size=self.world_size,
            worker_id=worker_id,
            num_workers=num_workers,
            assigned_tar_files=len(tar_paths),
        )
        self._last_counters = counters

        def source() -> Iterator[dict[str, Any]]:
            for tar_path in tar_paths:
                partition = self._partitions[tar_path]
                metadata = self._metadata_for_partition(partition, counters)
                yield from self._stream_partition(partition, metadata, counters)

        shuffled = _bounded_shuffle(
            source(),
            buffer_size=self.shuffle_buffer_size,
            rng=random.Random(
                _stable_seed(
                    self.seed,
                    self.epoch,
                    "conditional-sample-shuffle",
                    self.rank,
                    worker_id,
                )
            ),
        )
        for sample_index, sample in enumerate(shuffled):
            if quota is not None and sample_index >= quota:
                counters.samples_trimmed_for_rank_balance += 1
                continue
            counters.samples_yielded += 1
            yield sample


__all__ = ["ConditionalDatasetCounters", "ConditionalLocalTarDataset"]
