#!/usr/bin/env python3
"""Prepare signed train-only four-view legal-YTB metadata sidecars.

The command supports an idempotent Slurm array without copying or extracting
audio.  Array task ``i`` owns every TAR for which
``sha256(tar_path) % task_count == i``; it scans the bounded source Parquet,
filters held-out video groups and foreign TARs before expensive text work, and
writes one atomic Parquet sidecar per owned TAR.  ``--finalize`` authenticates
all task manifests and creates ``metadata-index.json``.

Typical production use::

    python examples/prepare_legal_ytb_conditional_metadata.py \
      --task-count 64 --task-index "$SLURM_ARRAY_TASK_ID" ...
    python examples/prepare_legal_ytb_conditional_metadata.py \
      --task-count 64 --finalize ...

With the default ``--task-count 1`` and no task index, preparation and
finalization run in one process, which is convenient for a dry-run fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from audio.diffusion_data.conditional_metadata import (  # noqa: E402
    CONDITIONAL_INDEX_SCHEMA_VERSION,
    TARGET_COLUMNS,
    VIEW_COUNT,
    VIEW_NAMES,
    canonical_json_sha256,
    conditional_metadata_schema,
    prepare_conditional_row,
    sha256_file,
)
from audio.diffusion_data.local_tar_dataset import (  # noqa: E402
    video_key_from_sample_key,
)


DEFAULT_REPO_ID = "local/user-provided-audio"
DEFAULT_REVISION = "user-provided-dataset-revision"
TOKENIZER_FILENAMES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)
SOURCE_COLUMNS = (
    "sample_key",
    "audio_tar_path",
    "audio_member",
    "duration_sec",
    "target_json",
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _combined_file_fingerprint(files: Iterable[Path]) -> tuple[str, list[dict[str, str]]]:
    records = []
    for path in sorted((item.resolve(strict=True) for item in files), key=str):
        records.append({"path": path.name, "sha256": sha256_file(path)})
    if not records:
        raise ValueError("fingerprint input contains no regular file")
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest, records


def tokenizer_fingerprint(tokenizer_path: Path) -> tuple[str, list[dict[str, str]]]:
    files = [
        tokenizer_path / name
        for name in TOKENIZER_FILENAMES
        if (tokenizer_path / name).is_file()
    ]
    return _combined_file_fingerprint(files)


def preprocessing_code_fingerprint() -> tuple[str, list[dict[str, str]]]:
    files = [
        Path(__file__),
        REPOSITORY_ROOT / "audio/diffusion_data/conditional_metadata.py",
        REPOSITORY_ROOT / "audio/diffusion_data/clean_17k_preprocessing_allowed_shadda.py",
    ]
    return _combined_file_fingerprint(files)


def load_excluded_video_keys(path: Path) -> frozenset[str]:
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid split JSON at {path}:{line_number}") from error
            value = row.get("video_key")
            if not isinstance(value, str) or not value:
                raise ValueError(f"missing video_key at {path}:{line_number}")
            keys.add(value)
    if not keys:
        raise ValueError(f"split exclusion manifest is empty: {path}")
    return frozenset(keys)


def task_for_tar(tar_path: str, task_count: int) -> int:
    if task_count <= 0:
        raise ValueError("task_count must be positive")
    digest = hashlib.sha256(tar_path.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % task_count


def sidecar_relative_path(tar_path: str) -> str:
    stem = Path(tar_path).name
    if stem.endswith(".tar"):
        stem = stem[:-4]
    prefix = hashlib.sha256(tar_path.encode("utf-8")).hexdigest()[:16]
    return f"sidecars/{prefix}-{stem}.parquet"


def _signed_task_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload["content_sha256"] = canonical_json_sha256(payload)
    return payload


def _policy(
    *,
    canvas_sizes: Sequence[int],
    tokens_per_second: float,
    margin_tokens: float,
    view_seed: int,
) -> dict[str, Any]:
    return {
        "canvas_sizes": [int(value) for value in canvas_sizes],
        "duration_tokens_per_second": float(tokens_per_second),
        "duration_margin_tokens": float(margin_tokens),
        "turn_end_positions": 1,
        "view_seed": int(view_seed),
        "view_selection": "sha256_permutation_then_filter_valid_v1",
        "country_source": "target_json.country_code",
    }


def _provenance(
    *,
    source_metadata: Path,
    split_manifest: Path,
    tokenizer_path: Path,
) -> dict[str, Any]:
    tokenizer_sha, tokenizer_files = tokenizer_fingerprint(tokenizer_path)
    code_sha, code_files = preprocessing_code_fingerprint()
    return {
        "source_metadata_path": str(source_metadata),
        "source_metadata_sha256": sha256_file(source_metadata),
        "split_manifest_path": str(split_manifest),
        "split_manifest_sha256": sha256_file(split_manifest),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": tokenizer_sha,
        "tokenizer_files": tokenizer_files,
        "preprocessing_code_sha256": code_sha,
        "preprocessing_code_files": code_files,
    }


def _load_tokenizer(tokenizer_path: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("transformers is required to load the MawJais tokenizer") from error
    return AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=True,
    )


class _TarWriter:
    def __init__(self, final_path: Path) -> None:
        self.final_path = final_path
        self.temporary_path = final_path.with_name(
            f".{final_path.name}.incomplete-{os.getpid()}"
        )
        self.temporary_path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = pq.ParquetWriter(
            self.temporary_path,
            conditional_metadata_schema(),
            compression="zstd",
            compression_level=6,
            use_dictionary=True,
            write_statistics=True,
        )
        self.row_count = 0
        self.any_valid_count = 0
        self.per_view_valid_counts = [0] * VIEW_COUNT

    def write(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=conditional_metadata_schema())
        self.writer.write_table(table)
        self.row_count += len(rows)
        for row in rows:
            mask = row["valid_view_mask"]
            self.any_valid_count += int(any(mask))
            for view_id, value in enumerate(mask):
                self.per_view_valid_counts[view_id] += int(bool(value))

    def close(self) -> dict[str, Any]:
        self.writer.close()
        temporary_sha = sha256_file(self.temporary_path)
        if self.final_path.exists():
            if sha256_file(self.final_path) != temporary_sha:
                raise FileExistsError(
                    f"existing sidecar differs from deterministic retry: {self.final_path}"
                )
            self.temporary_path.unlink()
        else:
            self.temporary_path.replace(self.final_path)
        return {
            "sidecar_sha256": temporary_sha,
            "row_count": self.row_count,
            "any_valid_count": self.any_valid_count,
            "per_view_valid_counts": self.per_view_valid_counts,
        }


def _validate_existing_task_manifest(
    path: Path,
    *,
    expected_task_count: int,
    expected_task_index: int,
    expected_provenance: Mapping[str, Any],
    expected_policy: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if canonical_json_sha256(payload) != payload.get("content_sha256"):
        raise ValueError(f"existing task manifest is not authentic: {path}")
    expected = {
        "task_count": expected_task_count,
        "task_index": expected_task_index,
        "provenance": dict(expected_provenance),
        "policy": dict(expected_policy),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"existing task manifest differs for {key}: {path}")
    for partition in payload.get("partitions", []):
        sidecar = path.parents[1] / partition["sidecar_path"]
        if sha256_file(sidecar) != partition["sidecar_sha256"]:
            raise ValueError(f"existing task sidecar hash mismatch: {sidecar}")
    return payload


def run_task(
    *,
    source_metadata: Path,
    split_manifest: Path,
    output_dir: Path,
    tokenizer_path: Path,
    tokenizer: Any | None,
    repo_id: str,
    revision: str,
    canvas_sizes: Sequence[int],
    tokens_per_second: float,
    margin_tokens: float,
    view_seed: int,
    task_count: int,
    task_index: int,
    parquet_batch_size: int,
    sidecar_row_group_size: int,
    max_source_rows: int | None = None,
) -> dict[str, Any]:
    """Run one deterministic, restartable preprocessing shard."""

    if not 0 <= task_index < task_count:
        raise ValueError("task_index must be in [0, task_count)")
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = _provenance(
        source_metadata=source_metadata,
        split_manifest=split_manifest,
        tokenizer_path=tokenizer_path,
    )
    policy = _policy(
        canvas_sizes=canvas_sizes,
        tokens_per_second=tokens_per_second,
        margin_tokens=margin_tokens,
        view_seed=view_seed,
    )
    task_manifest = output_dir / "tasks" / f"task-{task_index:05d}-of-{task_count:05d}.json"
    existing = _validate_existing_task_manifest(
        task_manifest,
        expected_task_count=task_count,
        expected_task_index=task_index,
        expected_provenance=provenance,
        expected_policy=policy,
    )
    if existing is not None:
        print(json.dumps({"status": "already-complete", "task": task_index}))
        return existing

    excluded_video_keys = load_excluded_video_keys(split_manifest)
    tokenizer = _load_tokenizer(tokenizer_path) if tokenizer is None else tokenizer
    parquet = pq.ParquetFile(source_metadata)
    missing = set(SOURCE_COLUMNS).difference(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"source metadata is missing columns: {sorted(missing)}")

    writers: dict[str, _TarWriter] = {}
    buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    row_exclusions: list[dict[str, Any]] = []
    source_rows_scanned = 0
    heldout_rows_excluded = 0
    foreign_task_rows_skipped = 0
    assigned_train_rows = 0
    stop = False
    for batch in parquet.iter_batches(
        batch_size=parquet_batch_size,
        columns=list(SOURCE_COLUMNS),
        use_threads=False,
    ):
        if max_source_rows is not None:
            remaining = max_source_rows - source_rows_scanned
            if remaining <= 0:
                break
            if batch.num_rows > remaining:
                batch = batch.slice(0, remaining)
                stop = True
        source_rows_scanned += batch.num_rows
        # A source batch contains only a small number of TAR paths on the real
        # corpus. Filter in Arrow before materializing Python strings/JSON so a
        # 64-way array does not execute 64 Python loops over all 4.2M rows.
        tar_column = batch.column(SOURCE_COLUMNS.index("audio_tar_path"))
        owned_tars = [
            str(value)
            for value in pc.unique(tar_column).to_pylist()
            if task_for_tar(str(value), task_count) == task_index
        ]
        if not owned_tars:
            foreign_task_rows_skipped += batch.num_rows
            if stop:
                break
            continue
        owned_values = pa.array(owned_tars, type=tar_column.type)
        owned_batch = batch.filter(pc.is_in(tar_column, value_set=owned_values))
        foreign_task_rows_skipped += batch.num_rows - owned_batch.num_rows
        for row in owned_batch.to_pylist():
            sample_key = str(row["sample_key"])
            if video_key_from_sample_key(sample_key) in excluded_video_keys:
                heldout_rows_excluded += 1
                continue
            tar_path = str(row["audio_tar_path"])
            assigned_train_rows += 1
            try:
                prepared = prepare_conditional_row(
                    row,
                    tokenizer=tokenizer,
                    canvas_sizes=canvas_sizes,
                    tokens_per_second=tokens_per_second,
                    margin_tokens=margin_tokens,
                    view_seed=view_seed,
                )
            except Exception as error:
                row_exclusions.append(
                    {
                        "sample_key": sample_key,
                        "audio_tar_path": tar_path,
                        "reason": f"{type(error).__name__}:{error}",
                    }
                )
                continue
            if not any(prepared["valid_view_mask"]):
                row_exclusions.append(
                    {
                        "sample_key": sample_key,
                        "audio_tar_path": tar_path,
                        "reason": "all_views_invalid",
                        "view_errors": prepared["view_errors"],
                    }
                )
                continue
            buffer = buffers[tar_path]
            buffer.append(prepared)
            if len(buffer) >= sidecar_row_group_size:
                writer = writers.get(tar_path)
                if writer is None:
                    writer = _TarWriter(output_dir / sidecar_relative_path(tar_path))
                    writers[tar_path] = writer
                writer.write(buffer)
                buffer.clear()
        if stop:
            break

    for tar_path, buffer in buffers.items():
        if buffer:
            writer = writers.get(tar_path)
            if writer is None:
                writer = _TarWriter(output_dir / sidecar_relative_path(tar_path))
                writers[tar_path] = writer
            writer.write(buffer)
    partitions = []
    for tar_path in sorted(writers):
        stats = writers[tar_path].close()
        partitions.append(
            {
                "tar_path": tar_path,
                "sidecar_path": sidecar_relative_path(tar_path),
                **stats,
            }
        )

    exclusion_path = output_dir / "exclusions" / f"task-{task_index:05d}-of-{task_count:05d}.jsonl"
    exclusion_path.parent.mkdir(parents=True, exist_ok=True)
    exclusion_tmp = exclusion_path.with_name(f".{exclusion_path.name}.tmp-{os.getpid()}")
    with exclusion_tmp.open("w", encoding="utf-8") as stream:
        for row in sorted(row_exclusions, key=lambda item: item["sample_key"]):
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    exclusion_tmp.replace(exclusion_path)

    payload = _signed_task_payload(
        {
            "schema_version": 1,
            "repo_id": repo_id,
            "revision": revision,
            "task_count": task_count,
            "task_index": task_index,
            "dry_run": max_source_rows is not None,
            "max_source_rows": max_source_rows,
            "provenance": provenance,
            "policy": policy,
            "statistics": {
                "source_rows_scanned": source_rows_scanned,
                "heldout_rows_excluded": heldout_rows_excluded,
                "foreign_task_rows_skipped": foreign_task_rows_skipped,
                "assigned_train_rows": assigned_train_rows,
                "all_invalid_rows_excluded": len(row_exclusions),
                "indexed_train_rows": sum(item["row_count"] for item in partitions),
            },
            "exclusions": {
                "path": exclusion_path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(exclusion_path),
                "count": len(row_exclusions),
            },
            "partitions": partitions,
        }
    )
    _atomic_json(task_manifest, payload)
    return payload


def _load_task_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if canonical_json_sha256(payload) != payload.get("content_sha256"):
        raise ValueError(f"task manifest hash mismatch: {path}")
    return payload


def finalize_tasks(
    *,
    output_dir: Path,
    task_count: int,
    expected_train_rows: int | None,
) -> dict[str, Any]:
    manifests = [
        output_dir / "tasks" / f"task-{index:05d}-of-{task_count:05d}.json"
        for index in range(task_count)
    ]
    missing = [str(path) for path in manifests if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing preprocessing task manifests: {missing[:8]}")
    tasks = [_load_task_manifest(path) for path in manifests]
    reference = tasks[0]
    for task_index, task in enumerate(tasks):
        if task.get("task_count") != task_count or task.get("task_index") != task_index:
            raise ValueError("preprocessing task identity mismatch")
        for key in ("repo_id", "revision", "provenance", "policy", "dry_run", "max_source_rows"):
            if task.get(key) != reference.get(key):
                raise ValueError(f"preprocessing tasks disagree on {key}")

    partitions: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    seen_tars: set[str] = set()
    for task in tasks:
        exclusion = task["exclusions"]
        exclusion_path = output_dir / exclusion["path"]
        if sha256_file(exclusion_path) != exclusion["sha256"]:
            raise ValueError(f"task exclusion hash mismatch: {exclusion_path}")
        exclusions.append(exclusion)
        for partition in task["partitions"]:
            if partition["tar_path"] in seen_tars:
                raise ValueError("duplicate TAR ownership across preprocessing tasks")
            seen_tars.add(partition["tar_path"])
            sidecar = output_dir / partition["sidecar_path"]
            if sha256_file(sidecar) != partition["sidecar_sha256"]:
                raise ValueError(f"sidecar hash mismatch during finalize: {sidecar}")
            partitions.append(partition)
    partitions.sort(key=lambda item: item["tar_path"])
    total_rows = sum(item["row_count"] for item in partitions)
    any_valid_rows = sum(item["any_valid_count"] for item in partitions)
    per_view = [
        sum(item["per_view_valid_counts"][view_id] for item in partitions)
        for view_id in range(VIEW_COUNT)
    ]
    all_invalid = sum(item["count"] for item in exclusions)
    assigned_rows = sum(task["statistics"]["assigned_train_rows"] for task in tasks)
    if assigned_rows != total_rows + all_invalid:
        raise ValueError("indexed plus excluded rows do not equal assigned train rows")
    if expected_train_rows is not None and assigned_rows != expected_train_rows:
        raise ValueError(
            f"prepared train row count {assigned_rows} != expected {expected_train_rows}"
        )
    index = {
        "schema_version": CONDITIONAL_INDEX_SCHEMA_VERSION,
        "repo_id": reference["repo_id"],
        "revision": reference["revision"],
        "view_names": list(VIEW_NAMES),
        "dry_run": reference["dry_run"],
        "provenance": reference["provenance"],
        "policy": reference["policy"],
        "totals": {
            "train_rows": total_rows,
            "assigned_train_rows_including_exclusions": assigned_rows,
            "all_invalid_rows_excluded": all_invalid,
            "any_valid_rows": any_valid_rows,
            "per_view_valid_counts": per_view,
            "tar_partitions": len(partitions),
        },
        "task_manifests": [
            {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": task["content_sha256"],
            }
            for path, task in zip(manifests, tasks, strict=True)
        ],
        "exclusion_manifests": exclusions,
        "partitions": partitions,
    }
    index["content_sha256"] = canonical_json_sha256(index)
    _atomic_json(output_dir / "metadata-index.json", index)
    return index


def prepare_validation_metadata(
    *,
    validation_manifest: Path,
    output_dir: Path,
    tokenizer: Any,
    canvas_sizes: Sequence[int],
    tokens_per_second: float,
    margin_tokens: float,
    view_seed: int,
) -> dict[str, Any]:
    """Prepare the small paired validation manifest with the train schema."""

    rows = []
    with validation_manifest.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            source = {
                "sample_key": raw["sample_key"],
                "audio_tar_path": raw["audio_tar_path"],
                "audio_member": raw["audio_member"],
                "duration_sec": raw["duration_sec"],
                "target_json": json.dumps(
                    {
                        "transcription": raw["reference"],
                        "country_code": raw["country_code"],
                        "has_code_switching": bool(raw.get("has_code_switching", False)),
                    },
                    ensure_ascii=False,
                ),
            }
            rows.append(
                prepare_conditional_row(
                    source,
                    tokenizer=tokenizer,
                    canvas_sizes=canvas_sizes,
                    tokens_per_second=tokens_per_second,
                    margin_tokens=margin_tokens,
                    view_seed=view_seed,
                )
            )
    output = output_dir / "validation-metadata.parquet"
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    pq.write_table(
        pa.Table.from_pylist(rows, schema=conditional_metadata_schema()),
        temporary,
        compression="zstd",
        compression_level=6,
    )
    temporary.replace(output)
    payload = {
        "schema_version": 1,
        "source_manifest": str(validation_manifest),
        "source_manifest_sha256": sha256_file(validation_manifest),
        "metadata_path": output.name,
        "metadata_sha256": sha256_file(output),
        "rows": len(rows),
        "view_names": list(VIEW_NAMES),
    }
    payload["content_sha256"] = canonical_json_sha256(payload)
    _atomic_json(output_dir / "validation-metadata.manifest.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--canvas-sizes", type=int, nargs="+", default=[64, 128, 256, 384, 512])
    parser.add_argument("--tokens-per-second", type=float, default=9.0)
    parser.add_argument("--margin-tokens", type=float, default=100.0)
    parser.add_argument("--view-seed", type=int, default=3463)
    parser.add_argument("--task-count", type=int, default=1)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--expected-train-rows", type=int, default=None,
                        help="Optional exact corpus-size assertion when finalizing")
    parser.add_argument("--parquet-batch-size", type=int, default=65_536)
    parser.add_argument("--sidecar-row-group-size", type=int, default=4096)
    parser.add_argument("--max-source-rows", type=int)
    parser.add_argument("--validation-manifest", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_metadata = args.source_metadata.expanduser().resolve(strict=True)
    split_manifest = args.split_manifest.expanduser().resolve(strict=True)
    tokenizer_path = args.tokenizer_path.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    if args.max_source_rows is not None and args.max_source_rows <= 0:
        raise ValueError("max_source_rows must be positive")
    if args.finalize:
        index = finalize_tasks(
            output_dir=output_dir,
            task_count=args.task_count,
            expected_train_rows=(
                None if args.max_source_rows is not None else args.expected_train_rows
            ),
        )
        print(json.dumps(index["totals"], indent=2))
        if args.validation_manifest is not None:
            tokenizer = _load_tokenizer(tokenizer_path)
            validation = prepare_validation_metadata(
                validation_manifest=args.validation_manifest.expanduser().resolve(strict=True),
                output_dir=output_dir,
                tokenizer=tokenizer,
                canvas_sizes=args.canvas_sizes,
                tokens_per_second=args.tokens_per_second,
                margin_tokens=args.margin_tokens,
                view_seed=args.view_seed,
            )
            print(json.dumps(validation, indent=2))
        return 0

    task_index = 0 if args.task_index is None else args.task_index
    if args.task_index is None and args.task_count != 1:
        raise ValueError("--task-index is required when --task-count is greater than one")
    # Fail early with a clear message on unusually restrictive environments.
    soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft_limit < 256:
        raise RuntimeError("preprocessing requires an open-file limit of at least 256")
    tokenizer = _load_tokenizer(tokenizer_path)
    task = run_task(
        source_metadata=source_metadata,
        split_manifest=split_manifest,
        output_dir=output_dir,
        tokenizer_path=tokenizer_path,
        tokenizer=tokenizer,
        repo_id=args.repo_id,
        revision=args.revision,
        canvas_sizes=args.canvas_sizes,
        tokens_per_second=args.tokens_per_second,
        margin_tokens=args.margin_tokens,
        view_seed=args.view_seed,
        task_count=args.task_count,
        task_index=task_index,
        parquet_batch_size=args.parquet_batch_size,
        sidecar_row_group_size=args.sidecar_row_group_size,
        max_source_rows=args.max_source_rows,
    )
    print(json.dumps(task["statistics"], indent=2))
    if args.task_count == 1:
        index = finalize_tasks(
            output_dir=output_dir,
            task_count=1,
            expected_train_rows=(
                None if args.max_source_rows is not None else args.expected_train_rows
            ),
        )
        print(json.dumps(index["totals"], indent=2))
        if args.validation_manifest is not None:
            validation = prepare_validation_metadata(
                validation_manifest=args.validation_manifest.expanduser().resolve(strict=True),
                output_dir=output_dir,
                tokenizer=tokenizer,
                canvas_sizes=args.canvas_sizes,
                tokens_per_second=args.tokens_per_second,
                margin_tokens=args.margin_tokens,
                view_seed=args.view_seed,
            )
            print(json.dumps(validation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
