"""Local, bounded-storage streaming loader for the legal YouTube ASR subset.

The loader intentionally accepts only an explicit subset manifest and local files. It
does not import ``datasets`` or ``huggingface_hub``, cannot discover additional remote
shards, and opens every TAR with :mod:`tarfile`'s forward-only ``r|*`` mode. Audio is
decoded from memory; archive members are never extracted to disk.

Work is assigned at TAR granularity. In distributed mode, usable metadata rows are
counted once and archives are deterministically bin-packed across ranks and DataLoader
workers. Every rank is capped to the same sample quota and the dropped tail is counted
explicitly. This equalizes samples, not DataLoader batches produced independently by
multiple workers; distributed trainers must batch after worker merging or validate an
equal batch count. Metadata is read in bounded PyArrow record batches and only rows for
that worker's TARs are retained while its archives are streamed.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .exclusions import SampleExclusionManifest, load_sample_exclusion_manifest


_PARQUET_COLUMNS = (
    "sample_key",
    "audio_tar_path",
    "audio_member",
    "duration_sec",
    "target_json",
)


def video_key_from_sample_key(sample_key: str) -> str:
    """Return the video-group key used to keep train and evaluation disjoint."""

    return sample_key.rsplit("__", 1)[0] if "__" in sample_key else sample_key


def _stable_seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Manifest path must be a safe relative POSIX path: {value!r}")
    return Path(*path.parts)


def _normalise_member_name(value: str) -> str:
    while value.startswith("./"):
        value = value[2:]
    return PurePosixPath(value).as_posix()


@dataclass(slots=True)
class StreamingDatasetCounters:
    """Per-iterator counters (per rank and DataLoader worker)."""

    epoch: int
    rank: int
    world_size: int
    worker_id: int
    num_workers: int
    assigned_tar_files: int = 0
    parquet_rows_scanned: int = 0
    metadata_rows_assigned: int = 0
    eval_rows_excluded: int = 0
    invalid_rows_skipped: int = 0
    duplicate_member_rows: int = 0
    tar_files_opened: int = 0
    tar_members_scanned: int = 0
    audio_payload_bytes_read: int = 0
    audio_decode_errors: int = 0
    missing_audio_members: int = 0
    samples_trimmed_for_rank_balance: int = 0
    samples_excluded_by_manifest: int = 0
    samples_yielded: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _ManifestFile:
    path: str
    size: int


@dataclass(frozen=True, slots=True)
class _SubsetManifest:
    repo_id: str
    revision: str
    max_bytes: int
    selected_bytes: int
    files: tuple[_ManifestFile, ...]
    parquet_path: str
    tar_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _MetadataRow:
    sample_key: str
    audio_tar_path: str
    audio_member: str
    duration_sec: float
    target_json: str


def _read_subset_manifest(path: Path) -> _SubsetManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
        raise ValueError(f"Invalid subset manifest: {path}")

    files: list[_ManifestFile] = []
    seen_paths: set[str] = set()
    for raw_file in payload["files"]:
        if not isinstance(raw_file, dict):
            raise ValueError("Every manifest file entry must be an object")
        relative_path = str(raw_file["path"])
        _safe_relative_path(relative_path)
        if relative_path in seen_paths:
            raise ValueError(f"Duplicate manifest path: {relative_path}")
        seen_paths.add(relative_path)
        size = int(raw_file["size"])
        if size < 0:
            raise ValueError(f"Negative manifest size for {relative_path}")
        files.append(_ManifestFile(relative_path, size))

    parquet_paths = [item.path for item in files if item.path.endswith(".parquet")]
    tar_paths = tuple(item.path for item in files if item.path.endswith(".tar"))
    if len(parquet_paths) != 1:
        raise ValueError(
            f"Expected exactly one Parquet metadata file, got {parquet_paths}"
        )
    if not tar_paths:
        raise ValueError("The subset manifest contains no TAR archive")

    selected_bytes = int(payload.get("selected_bytes", sum(item.size for item in files)))
    max_bytes = int(payload.get("max_bytes", selected_bytes))
    file_bytes = sum(item.size for item in files)
    if selected_bytes != file_bytes:
        raise ValueError(
            f"Manifest selected_bytes={selected_bytes} but file entries total {file_bytes}"
        )
    if selected_bytes > max_bytes:
        raise ValueError("Subset manifest exceeds its declared byte cap")
    declared_tar_count = payload.get("selected_tar_files")
    if declared_tar_count is not None and int(declared_tar_count) != len(tar_paths):
        raise ValueError("selected_tar_files does not match the manifest file list")

    return _SubsetManifest(
        repo_id=str(payload.get("repo_id", "")),
        revision=str(payload.get("revision", "")),
        max_bytes=max_bytes,
        selected_bytes=selected_bytes,
        files=tuple(files),
        parquet_path=parquet_paths[0],
        tar_paths=tar_paths,
    )


def _load_eval_video_keys(path: Path) -> frozenset[str]:
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in eval manifest {path}:{line_number}"
                ) from error
            key = row.get("video_key")
            if not key and row.get("sample_key"):
                key = video_key_from_sample_key(str(row["sample_key"]))
            if not isinstance(key, str) or not key:
                raise ValueError(f"Missing video_key in {path}:{line_number}")
            keys.add(key)
    if not keys:
        raise ValueError(f"Evaluation manifest contains no video keys: {path}")
    return frozenset(keys)


def _resolve_distributed(rank: int | None, world_size: int | None) -> tuple[int, int]:
    if (rank is None) != (world_size is None):
        raise ValueError("rank and world_size must be provided together")
    if rank is None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = int(os.environ.get("RANK", "0"))
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
    assert world_size is not None
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid distributed assignment rank={rank}, world_size={world_size}")
    return rank, world_size


def _bounded_shuffle(
    source: Iterable[dict[str, Any]],
    *,
    buffer_size: int,
    rng: random.Random,
) -> Iterator[dict[str, Any]]:
    if buffer_size == 1:
        yield from source
        return
    buffer: list[dict[str, Any]] = []
    for item in source:
        if len(buffer) < buffer_size:
            buffer.append(item)
            continue
        selected = rng.randrange(len(buffer))
        yield buffer[selected]
        buffer[selected] = item
    rng.shuffle(buffer)
    yield from buffer


class LocalTarParquetASRDataset(IterableDataset[dict[str, Any]]):
    """Stream an explicit local ASR subset without extracting its TAR archives.

    Args:
        local_root: Root containing paths listed in ``subset_manifest_path``.
        subset_manifest_path: Size-capped JSON manifest produced by the project data
            preparation script. Only TARs in its ``files`` array are ever opened.
        eval_manifest_path: JSONL evaluation manifest. Every row sharing one of its
            video keys is excluded from training.
        seed: Dataset-order seed shared by all comparable experiments.
        shuffle_buffer_size: Maximum number of decoded samples kept for bounded
            sample-level shuffle. Set to one to disable sample shuffling. Audio can be
            large, so the conservative default is intentionally small.
        target_sample_rate: Resample decoded audio in memory to this rate. ``None``
            preserves the archive rate.
        error_policy: ``"raise"`` fails on invalid target/audio. ``"skip"`` records a
            counter and moves on. Missing archive members are controlled separately.
    """

    def __init__(
        self,
        *,
        local_root: str | Path,
        subset_manifest_path: str | Path,
        eval_manifest_path: str | Path,
        seed: int = 0,
        shuffle_buffer_size: int = 32,
        parquet_batch_size: int = 65_536,
        target_sample_rate: int | None = 16_000,
        rank: int | None = None,
        world_size: int | None = None,
        validate_file_sizes: bool = True,
        strict_missing_members: bool = True,
        error_policy: Literal["raise", "skip"] = "raise",
        max_audio_bytes: int = 64 << 20,
        balance_distributed_shards: bool = True,
        sample_exclusion_manifest_path: str | Path | None = None,
        expected_excluded_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.local_root = Path(local_root).expanduser().resolve()
        self.subset_manifest_path = Path(subset_manifest_path).expanduser().resolve()
        self.eval_manifest_path = Path(eval_manifest_path).expanduser().resolve()
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
        self._last_counters: StreamingDatasetCounters | None = None

        if self.shuffle_buffer_size < 1:
            raise ValueError("shuffle_buffer_size must be at least one")
        if self.parquet_batch_size < 1:
            raise ValueError("parquet_batch_size must be positive")
        if target_sample_rate is not None and target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be positive or None")
        if error_policy not in ("raise", "skip"):
            raise ValueError(f"Unsupported error_policy: {error_policy}")
        if self.max_audio_bytes <= 0:
            raise ValueError("max_audio_bytes must be positive")
        if not self.subset_manifest_path.is_file():
            raise FileNotFoundError(self.subset_manifest_path)
        if not self.eval_manifest_path.is_file():
            raise FileNotFoundError(self.eval_manifest_path)

        self._manifest = _read_subset_manifest(self.subset_manifest_path)
        self._eval_video_keys = _load_eval_video_keys(self.eval_manifest_path)
        self._subset_manifest_sha256 = _sha256_file(self.subset_manifest_path)
        self._eval_manifest_sha256 = _sha256_file(self.eval_manifest_path)
        self._sample_exclusion_manifest: SampleExclusionManifest | None = None
        if sample_exclusion_manifest_path is not None:
            self._sample_exclusion_manifest = load_sample_exclusion_manifest(
                sample_exclusion_manifest_path,
                expected_dataset_revision=self._manifest.revision,
                expected_subset_manifest_sha256=self._subset_manifest_sha256,
                expected_count=expected_excluded_samples,
            )
        elif expected_excluded_samples not in (None, 0):
            raise ValueError(
                "expected_excluded_samples requires sample_exclusion_manifest_path"
            )
        self._excluded_sample_keys = (
            frozenset()
            if self._sample_exclusion_manifest is None
            else self._sample_exclusion_manifest.sample_keys
        )
        self._parquet_path = self._local_path(self._manifest.parquet_path)
        self._validate_local_files(validate_file_sizes=validate_file_sizes)
        # TAR-count striding is badly imbalanced on the real subset. Only DDP
        # needs equal rank lengths, so avoid this extra metadata scan for the
        # common world_size=1 inference/debug path.
        self._tar_sample_counts = (
            self._count_training_samples_by_tar()
            if self.balance_distributed_shards and self.world_size > 1
            else None
        )
        if self._tar_sample_counts is not None:
            rank_loads = self.samples_per_rank()
            assert rank_loads is not None
            if min(rank_loads) == 0:
                raise ValueError(
                    "distributed TAR assignment leaves at least one rank empty; "
                    "reduce world_size or select more non-empty archives"
                )

    def _local_path(self, relative_path: str) -> Path:
        return self.local_root / _safe_relative_path(relative_path)

    def _validate_local_files(self, *, validate_file_sizes: bool) -> None:
        missing: list[str] = []
        wrong_sizes: list[tuple[str, int, int]] = []
        for item in self._manifest.files:
            path = self._local_path(item.path)
            if not path.is_file():
                missing.append(item.path)
                continue
            if validate_file_sizes:
                actual_size = path.stat().st_size
                if actual_size != item.size:
                    wrong_sizes.append((item.path, item.size, actual_size))
        if missing or wrong_sizes:
            raise FileNotFoundError(
                "Local subset does not match its manifest: "
                f"missing={missing[:8]}, wrong_sizes={wrong_sizes[:8]}"
            )

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def last_counters(self) -> StreamingDatasetCounters | None:
        """Counters from the last iterator in this process.

        With ``num_workers > 0``, each worker owns its own dataset copy; aggregate
        counters in the worker/training process if cross-worker reporting is needed.
        """

        return self._last_counters

    def set_epoch(self, epoch: int) -> None:
        """Select a deterministic epoch ordering.

        Recreate the DataLoader iterator after calling this method. When persistent
        workers are enabled, their private copies do not observe this plain Python
        attribute; use non-persistent workers or propagate the epoch explicitly.
        """

        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._epoch = int(epoch)

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "seed": self.seed,
            "repo_id": self._manifest.repo_id,
            "revision": self._manifest.revision,
            "selected_tar_files": len(self._manifest.tar_paths),
            "subset_manifest_sha256": self._subset_manifest_sha256,
            "eval_manifest_sha256": self._eval_manifest_sha256,
            "sample_exclusion_manifest_sha256": (
                None
                if self._sample_exclusion_manifest is None
                else self._sample_exclusion_manifest.content_sha256
            ),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for key, expected in (
            ("seed", self.seed),
            ("repo_id", self._manifest.repo_id),
            ("revision", self._manifest.revision),
            ("selected_tar_files", len(self._manifest.tar_paths)),
            ("subset_manifest_sha256", self._subset_manifest_sha256),
            ("eval_manifest_sha256", self._eval_manifest_sha256),
            (
                "sample_exclusion_manifest_sha256",
                None
                if self._sample_exclusion_manifest is None
                else self._sample_exclusion_manifest.content_sha256,
            ),
        ):
            if key in state and state[key] != expected:
                raise ValueError(
                    f"Dataset state mismatch for {key}: {state[key]!r} != {expected!r}"
                )
        self.set_epoch(int(state["epoch"]))

    def assigned_tar_paths(
        self, *, worker_id: int = 0, num_workers: int = 1
    ) -> tuple[str, ...]:
        """Return this rank/worker's deterministic, rank-first TAR assignment."""

        if num_workers <= 0 or worker_id < 0 or worker_id >= num_workers:
            raise ValueError(
                f"Invalid worker assignment worker_id={worker_id}, num_workers={num_workers}"
            )
        ordered = sorted(self._manifest.tar_paths)
        random.Random(_stable_seed(self.seed, self.epoch, "tar-order")).shuffle(ordered)
        if self._tar_sample_counts is None:
            rank_paths = ordered[self.rank :: self.world_size]
            return tuple(rank_paths[worker_id::num_workers])

        rank_assignments, _ = self._balanced_assignments(ordered, self.world_size)
        rank_paths = list(rank_assignments[self.rank])
        random.Random(
            _stable_seed(self.seed, self.epoch, "rank-tar-stream-order", self.rank)
        ).shuffle(rank_paths)
        worker_assignments, _ = self._balanced_assignments(
            rank_paths, num_workers
        )
        worker_paths = list(worker_assignments[worker_id])
        random.Random(
            _stable_seed(
                self.seed,
                self.epoch,
                "worker-tar-stream-order",
                self.rank,
                worker_id,
            )
        ).shuffle(worker_paths)
        return tuple(worker_paths)

    def _count_training_samples_by_tar(self) -> dict[str, int]:
        """Count selected, non-eval metadata rows for balanced DDP assignment."""

        selected = set(self._manifest.tar_paths)
        counts = {path: 0 for path in self._manifest.tar_paths}
        parquet = pq.ParquetFile(self._parquet_path)
        required = {"sample_key", "audio_tar_path"}
        missing = required.difference(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"Parquet is missing columns: {sorted(missing)}")
        tar_type = parquet.schema_arrow.field("audio_tar_path").type
        selected_values = pa.array(sorted(selected), type=tar_type)
        for batch in parquet.iter_batches(
            batch_size=self.parquet_batch_size,
            columns=["sample_key", "audio_tar_path"],
            use_threads=False,
        ):
            batch = batch.filter(pc.is_in(batch.column(1), value_set=selected_values))
            if batch.num_rows == 0:
                continue
            sample_keys = batch.column(0).to_pylist()
            tar_paths = batch.column(1).to_pylist()
            for sample_key, tar_path in zip(sample_keys, tar_paths, strict=True):
                sample_key = str(sample_key)
                if (
                    sample_key not in self._excluded_sample_keys
                    and video_key_from_sample_key(sample_key) not in self._eval_video_keys
                ):
                    counts[str(tar_path)] += 1
        return counts

    def _balanced_assignments(
        self,
        paths: Sequence[str],
        bucket_count: int,
    ) -> tuple[tuple[tuple[str, ...], ...], tuple[int, ...]]:
        """LPT-bin-pack TARs by usable row count with deterministic tie order."""

        if bucket_count <= 0:
            raise ValueError("bucket_count must be positive")
        if self._tar_sample_counts is None:
            raise RuntimeError("sample-count balancing was not initialized")
        # ``paths`` already has an epoch-seeded order. Stable sorting preserves
        # that order among equal-size TARs, rotating ties across epochs.
        ordered = sorted(paths, key=lambda path: self._tar_sample_counts[path], reverse=True)
        assignments: list[list[str]] = [[] for _ in range(bucket_count)]
        loads = [0] * bucket_count
        for path in ordered:
            bucket = min(range(bucket_count), key=lambda index: (loads[index], index))
            assignments[bucket].append(path)
            loads[bucket] += self._tar_sample_counts[path]
        return (
            tuple(tuple(bucket) for bucket in assignments),
            tuple(loads),
        )

    def samples_per_rank(self) -> tuple[int, ...] | None:
        """Expected usable samples per DDP rank before equal-length trimming."""

        if self._tar_sample_counts is None:
            return None
        ordered = sorted(self._manifest.tar_paths)
        random.Random(_stable_seed(self.seed, self.epoch, "tar-order")).shuffle(ordered)
        _, loads = self._balanced_assignments(ordered, self.world_size)
        return loads

    @staticmethod
    def _fair_capacity_quotas(capacities: Sequence[int], total: int) -> tuple[int, ...]:
        if total < 0 or total > sum(capacities):
            raise ValueError("quota total is outside worker capacity")
        quotas = [0] * len(capacities)
        remaining = total
        while remaining:
            active = [index for index, capacity in enumerate(capacities) if quotas[index] < capacity]
            if not active:
                raise RuntimeError("could not distribute sample quota")
            share = max(1, remaining // len(active))
            progressed = 0
            for index in active:
                added = min(share, capacities[index] - quotas[index], remaining)
                quotas[index] += added
                remaining -= added
                progressed += added
                if remaining == 0:
                    break
            if progressed == 0:
                raise RuntimeError("sample quota distribution made no progress")
        return tuple(quotas)

    def assigned_sample_quota(self, *, worker_id: int = 0, num_workers: int = 1) -> int | None:
        """Equal-length DDP quota for this worker, or ``None`` outside balanced DDP."""

        if self._tar_sample_counts is None:
            return None
        if num_workers <= 0 or worker_id < 0 or worker_id >= num_workers:
            raise ValueError("invalid worker assignment")
        ordered = sorted(self._manifest.tar_paths)
        random.Random(_stable_seed(self.seed, self.epoch, "tar-order")).shuffle(ordered)
        rank_assignments, rank_loads = self._balanced_assignments(ordered, self.world_size)
        rank_paths = list(rank_assignments[self.rank])
        random.Random(
            _stable_seed(self.seed, self.epoch, "rank-tar-stream-order", self.rank)
        ).shuffle(rank_paths)
        worker_assignments, worker_loads = self._balanced_assignments(
            rank_paths, num_workers
        )
        del worker_assignments
        rank_quota = min(rank_loads)
        return self._fair_capacity_quotas(worker_loads, rank_quota)[worker_id]

    def configuration_summary(self) -> dict[str, Any]:
        """Return immutable provenance suitable for the resolved run config."""

        rank_loads = self.samples_per_rank()
        return {
            "loader": type(self).__name__,
            "local_root": str(self.local_root),
            "subset_manifest": str(self.subset_manifest_path),
            "eval_manifest": str(self.eval_manifest_path),
            "repo_id": self._manifest.repo_id,
            "revision": self._manifest.revision,
            "subset_manifest_sha256": self._subset_manifest_sha256,
            "eval_manifest_sha256": self._eval_manifest_sha256,
            "sample_exclusion_manifest": (
                None
                if self._sample_exclusion_manifest is None
                else str(self._sample_exclusion_manifest.path)
            ),
            "sample_exclusion_manifest_sha256": (
                None
                if self._sample_exclusion_manifest is None
                else self._sample_exclusion_manifest.content_sha256
            ),
            "excluded_sample_keys_sha256": (
                None
                if self._sample_exclusion_manifest is None
                else self._sample_exclusion_manifest.sample_keys_sha256
            ),
            "excluded_sample_count": len(self._excluded_sample_keys),
            "selected_bytes": self._manifest.selected_bytes,
            "selected_tar_files": len(self._manifest.tar_paths),
            "excluded_eval_video_groups": len(self._eval_video_keys),
            "rank": self.rank,
            "world_size": self.world_size,
            "seed": self.seed,
            "epoch": self.epoch,
            "shuffle_buffer_size": self.shuffle_buffer_size,
            "target_sample_rate": self.target_sample_rate,
            "balance_distributed_shards": self.balance_distributed_shards,
            "samples_per_rank_before_trim": rank_loads,
            "samples_per_rank_after_trim": (
                None if rank_loads is None else min(rank_loads)
            ),
        }

    def _metadata_for_tar_paths(
        self,
        tar_paths: Sequence[str],
        counters: StreamingDatasetCounters,
    ) -> dict[str, dict[str, _MetadataRow]]:
        selected = set(tar_paths)
        rows_by_tar: dict[str, dict[str, _MetadataRow]] = {
            path: {} for path in tar_paths
        }
        parquet = pq.ParquetFile(self._parquet_path)
        missing_columns = set(_PARQUET_COLUMNS).difference(parquet.schema_arrow.names)
        if missing_columns:
            raise ValueError(f"Parquet is missing columns: {sorted(missing_columns)}")

        tar_path_type = parquet.schema_arrow.field("audio_tar_path").type
        selected_values = pa.array(sorted(selected), type=tar_path_type)
        for batch in parquet.iter_batches(
            batch_size=self.parquet_batch_size,
            columns=list(_PARQUET_COLUMNS),
            use_threads=False,
        ):
            counters.parquet_rows_scanned += batch.num_rows
            # Filter in Arrow before converting strings/JSON to Python. The real
            # metadata has 4.2M rows but only ~418k belong to the local subset.
            batch = batch.filter(pc.is_in(batch.column(1), value_set=selected_values))
            counters.metadata_rows_assigned += batch.num_rows
            if batch.num_rows == 0:
                continue
            columns = [column.to_pylist() for column in batch.columns]
            for sample_key, tar_path, member, duration, target_json in zip(
                *columns, strict=True
            ):
                sample_key = str(sample_key)
                if video_key_from_sample_key(sample_key) in self._eval_video_keys:
                    counters.eval_rows_excluded += 1
                    continue
                if sample_key in self._excluded_sample_keys:
                    counters.samples_excluded_by_manifest += 1
                    continue
                duration = float(duration)
                if not math.isfinite(duration) or duration <= 0.0:
                    if self.error_policy == "raise":
                        raise ValueError(f"Invalid duration for {sample_key}: {duration}")
                    counters.invalid_rows_skipped += 1
                    continue
                member = _normalise_member_name(str(member))
                tar_rows = rows_by_tar[str(tar_path)]
                if member in tar_rows:
                    counters.duplicate_member_rows += 1
                    if self.error_policy == "raise":
                        raise ValueError(f"Duplicate metadata member {tar_path}:{member}")
                    continue
                tar_rows[member] = _MetadataRow(
                    sample_key=sample_key,
                    audio_tar_path=str(tar_path),
                    audio_member=member,
                    duration_sec=duration,
                    target_json=str(target_json),
                )
        return rows_by_tar

    def _decode_audio(self, payload: bytes) -> tuple[np.ndarray, int]:
        try:
            import soundfile as sf
        except ImportError:
            # The immutable MI210 image has libsndfile but not its Python binding.
            # The selected legal-ytb subset stores PCM WAV members, so provide a
            # dependency-free decoder and fail explicitly for compressed WAV.
            import wave

            with wave.open(io.BytesIO(payload), "rb") as wav:
                if wav.getcomptype() != "NONE":
                    raise ValueError(
                        f"unsupported compressed WAV type: {wav.getcomptype()}"
                    )
                channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                sample_rate = wav.getframerate()
                raw = wav.readframes(wav.getnframes())
            if channels <= 0:
                raise ValueError("WAV has no channels")
            if sample_width == 1:
                decoded = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
                decoded = (decoded - 128.0) / 128.0
            elif sample_width == 2:
                decoded = np.frombuffer(raw, dtype=np.dtype("<i2")).astype(np.float32)
                decoded /= 32768.0
            elif sample_width == 3:
                octets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
                integer = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
                integer = (integer ^ 0x800000) - 0x800000
                decoded = integer.astype(np.float32) / 8388608.0
            elif sample_width == 4:
                decoded = np.frombuffer(raw, dtype=np.dtype("<i4")).astype(np.float32)
                decoded /= 2147483648.0
            else:
                raise ValueError(f"unsupported PCM WAV sample width: {sample_width}")
            if decoded.size % channels:
                raise ValueError("WAV payload is not divisible by its channel count")
            samples = decoded.reshape(-1, channels)
        else:
            samples, sample_rate = sf.read(
                io.BytesIO(payload), dtype="float32", always_2d=True
            )
        if samples.shape[0] == 0:
            raise ValueError("Decoded audio is empty")
        waveform = samples.mean(axis=1, dtype=np.float32)
        sample_rate = int(sample_rate)
        if self.target_sample_rate is not None and sample_rate != self.target_sample_rate:
            try:
                from scipy.signal import resample_poly
            except ImportError as error:  # pragma: no cover - environment-specific
                raise RuntimeError("scipy is required for in-memory resampling") from error
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
        row: _MetadataRow,
        payload: bytes,
        counters: StreamingDatasetCounters,
    ) -> dict[str, Any] | None:
        try:
            target = json.loads(row.target_json)
            if not isinstance(target, dict):
                raise ValueError("target_json must decode to an object")
            transcription = target.get("transcription")
            if not isinstance(transcription, str) or not transcription.strip():
                raise ValueError("target_json has no non-empty transcription")
        except Exception:
            if self.error_policy == "raise":
                raise
            counters.invalid_rows_skipped += 1
            return None
        try:
            audio, sample_rate = self._decode_audio(payload)
        except Exception:
            if self.error_policy == "raise":
                raise
            counters.invalid_rows_skipped += 1
            counters.audio_decode_errors += 1
            return None

        return {
            "sample_key": row.sample_key,
            "video_key": video_key_from_sample_key(row.sample_key),
            "audio_tar_path": row.audio_tar_path,
            "audio_member": row.audio_member,
            "duration_sec": row.duration_sec,
            "audio": audio,
            "sampling_rate": sample_rate,
            "transcription": transcription,
            "country_code": target.get("country_code"),
            "target": target,
        }

    def _stream_tar_samples(
        self,
        tar_path: str,
        metadata: dict[str, _MetadataRow],
        counters: StreamingDatasetCounters,
    ) -> Iterator[dict[str, Any]]:
        if not metadata:
            return
        # This mapping is owned by the current iterator, so consuming it in place
        # releases metadata steadily as the sequential archive scan advances.
        pending = metadata
        local_tar_path = self._local_path(tar_path)
        counters.tar_files_opened += 1
        with tarfile.open(local_tar_path, mode="r|*") as archive:
            for member in archive:
                counters.tar_members_scanned += 1
                if not member.isfile():
                    continue
                member_name = _normalise_member_name(member.name)
                row = pending.pop(member_name, None)
                if row is None:
                    continue
                if member.size < 0 or member.size > self.max_audio_bytes:
                    message = (
                        f"Audio member size out of bounds ({member.size} bytes): "
                        f"{tar_path}:{member_name}"
                    )
                    if self.error_policy == "raise":
                        raise ValueError(message)
                    counters.invalid_rows_skipped += 1
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    if self.error_policy == "raise":
                        raise OSError(f"Could not read {tar_path}:{member_name}")
                    counters.invalid_rows_skipped += 1
                    counters.audio_decode_errors += 1
                    continue
                payload = extracted.read()
                counters.audio_payload_bytes_read += len(payload)
                if len(payload) != member.size:
                    if self.error_policy == "raise":
                        raise OSError(
                            f"Short read for {tar_path}:{member_name}: "
                            f"expected {member.size}, got {len(payload)} bytes"
                        )
                    counters.invalid_rows_skipped += 1
                    counters.audio_decode_errors += 1
                    continue
                sample = self._sample_from_member(row, payload, counters)
                if sample is not None:
                    yield sample

        if pending:
            counters.missing_audio_members += len(pending)
            if self.strict_missing_members:
                examples = sorted(pending)[:8]
                raise FileNotFoundError(
                    f"{len(pending)} metadata members missing from {tar_path}: {examples}"
                )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        tar_paths = self.assigned_tar_paths(
            worker_id=worker_id, num_workers=num_workers
        )
        counters = StreamingDatasetCounters(
            epoch=self.epoch,
            rank=self.rank,
            world_size=self.world_size,
            worker_id=worker_id,
            num_workers=num_workers,
            assigned_tar_files=len(tar_paths),
        )
        self._last_counters = counters
        if not tar_paths:
            return

        sample_quota = self.assigned_sample_quota(
            worker_id=worker_id, num_workers=num_workers
        )
        if sample_quota is not None:
            assigned_capacity = sum(self._tar_sample_counts[path] for path in tar_paths)
            counters.samples_trimmed_for_rank_balance = assigned_capacity - sample_quota
            if sample_quota == 0:
                return

        rows_by_tar = self._metadata_for_tar_paths(tar_paths, counters)

        def source() -> Iterator[dict[str, Any]]:
            for tar_path in tar_paths:
                metadata = rows_by_tar.pop(tar_path)
                yield from self._stream_tar_samples(
                    tar_path, metadata, counters
                )

        rng = random.Random(
            _stable_seed(self.seed, self.epoch, self.rank, worker_id, "sample-shuffle")
        )
        for sample in _bounded_shuffle(
            source(), buffer_size=self.shuffle_buffer_size, rng=rng
        ):
            counters.samples_yielded += 1
            yield sample
            if sample_quota is not None and counters.samples_yielded >= sample_quota:
                return
        if sample_quota is not None and counters.samples_yielded != sample_quota:
            raise RuntimeError(
                "balanced distributed iterator ended before its declared sample quota: "
                f"rank={self.rank}, worker={worker_id}, "
                f"yielded={counters.samples_yielded}, quota={sample_quota}"
            )
