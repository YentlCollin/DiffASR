#!/usr/bin/env python3
"""Create a deterministic, video-disjoint legal-YTB train/test split.

The script never copies or extracts audio.  It writes lightweight metadata and
video-group exclusion manifests; the streaming loader continues to read audio
from the single local TAR snapshot.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_REVISION = "user-provided-dataset-revision"
DEFAULT_SEED = "legal-ytb-diffasr-test-v1"
ARABIC_DIACRITICS = re.compile(
    "[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]"
)
TARGET_FLAGS = (
    "has_code_switching",
    "has_disfluency",
    "has_vocalizations_or_emotions",
    "has_overlapping_speech",
    "has_numbers",
    "has_uncertainty",
    "has_elongation",
    "audio_quality_issues",
)
SOURCE_COLUMNS = (
    "sample_key",
    "audio_tar_path",
    "audio_member",
    "duration_sec",
    "target_json",
)
TEST_PARQUET_SCHEMA = pa.schema(
    [
        ("sample_key", pa.large_string()),
        ("video_key", pa.large_string()),
        ("audio_tar_path", pa.large_string()),
        ("audio_member", pa.large_string()),
        ("duration_sec", pa.float64()),
        ("target_json", pa.large_string()),
        ("reference", pa.large_string()),
        ("country_code", pa.string()),
        ("has_code_switching", pa.bool_()),
        ("has_arabic_diacritics", pa.bool_()),
    ]
)
GROUP_PARQUET_SCHEMA = pa.schema(
    [("video_key", pa.large_string()), ("split", pa.string())]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument(
        "--validation-manifest", type=Path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-revision", default=DEFAULT_REVISION)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=65_536)
    return parser.parse_args()


def video_key_from_sample_key(sample_key: str) -> str:
    if "__" not in sample_key:
        raise ValueError(f"Cannot derive video key from {sample_key!r}")
    return sample_key.rsplit("__", 1)[0]


def source_country_from_sample_key(sample_key: str) -> str:
    parts = sample_key.split("__", 3)
    return parts[1] if len(parts) >= 3 and parts[1] else "UNK"


def stable_fraction(seed: str, video_key: str) -> float:
    payload = f"{seed}\0{video_key}".encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(payload).digest(), "big")
    return integer / (1 << 256)


def load_validation_video_keys(path: Path) -> frozenset[str]:
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            key = row.get("video_key")
            if not key and row.get("sample_key"):
                key = video_key_from_sample_key(str(row["sample_key"]))
            if not isinstance(key, str) or not key:
                raise ValueError(f"Missing video key at {path}:{line_number}")
            keys.add(key)
    if not keys:
        raise ValueError(f"Validation manifest contains no video groups: {path}")
    return frozenset(keys)


def split_for_video(
    video_key: str,
    *,
    validation_keys: frozenset[str],
    seed: str,
    test_fraction: float,
) -> str:
    if video_key in validation_keys:
        return "validation"
    return "test" if stable_fraction(seed, video_key) < test_fraction else "train"


def duration_bin(seconds: float) -> str:
    if seconds < 10:
        return "lt_10s"
    if seconds < 20:
        return "10_20s"
    if seconds < 30:
        return "20_30s"
    if seconds <= 35:
        return "30_35s"
    return "gt_35s"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def empty_stats() -> dict[str, Any]:
    return {
        "rows": 0,
        "duration_seconds": 0.0,
        "code_switch_rows": 0,
        "code_switch_seconds": 0.0,
        "diacritized_rows": 0,
    }


def update_stats(
    stats: dict[str, Any],
    *,
    duration: float,
    code_switch: bool,
    has_diacritics: bool,
) -> None:
    stats["rows"] += 1
    stats["duration_seconds"] += duration
    stats["code_switch_rows"] += int(code_switch)
    stats["code_switch_seconds"] += duration if code_switch else 0.0
    stats["diacritized_rows"] += int(has_diacritics)


def json_ready_stats(stats: dict[str, Any]) -> dict[str, Any]:
    result = dict(stats)
    result["duration_hours"] = stats["duration_seconds"] / 3600.0
    result["code_switch_hours"] = stats["code_switch_seconds"] / 3600.0
    return result


def ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def write_parquet_rows(
    writer: pq.ParquetWriter,
    rows: list[dict[str, Any]],
    schema: pa.Schema,
) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        rows.clear()


def write_group_files(
    *,
    seen_groups: set[str],
    validation_keys: frozenset[str],
    seed: str,
    test_fraction: float,
    output_dir: Path,
) -> dict[str, int]:
    test_groups_path = output_dir / "test-video-groups.jsonl"
    heldout_groups_path = output_dir / "train-excluded-video-groups.jsonl"
    group_parquet_path = output_dir / "video-group-splits.parquet"
    counts: Counter[str] = Counter()
    parquet_rows: list[dict[str, str]] = []
    with (
        test_groups_path.open("w", encoding="utf-8") as test_output,
        heldout_groups_path.open("w", encoding="utf-8") as heldout_output,
        pq.ParquetWriter(
            group_parquet_path,
            GROUP_PARQUET_SCHEMA,
            compression="zstd",
            use_dictionary=["split"],
        ) as parquet_writer,
    ):
        for video_key in sorted(seen_groups):
            split = split_for_video(
                video_key,
                validation_keys=validation_keys,
                seed=seed,
                test_fraction=test_fraction,
            )
            counts[split] += 1
            row = {"video_key": video_key, "split": split}
            parquet_rows.append(row)
            if split == "test":
                test_output.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
            if split != "train":
                heldout_output.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
            if len(parquet_rows) >= 65_536:
                write_parquet_rows(parquet_writer, parquet_rows, GROUP_PARQUET_SCHEMA)
        write_parquet_rows(parquet_writer, parquet_rows, GROUP_PARQUET_SCHEMA)
    return dict(counts)


def distribution_report(
    values: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for label, split_values in sorted(values.items()):
        train = split_values["train"]
        test = split_values["test"]
        available_rows = train["rows"] + test["rows"]
        available_seconds = train["duration_seconds"] + test["duration_seconds"]
        report[label] = {
            "train": json_ready_stats(train),
            "test": json_ready_stats(test),
            "validation": json_ready_stats(split_values["validation"]),
            "test_fraction_excluding_validation": {
                "rows": ratio(test["rows"], available_rows),
                "duration": ratio(test["duration_seconds"], available_seconds),
            },
        }
    return report


def ensure_output_is_new(output_dir: Path, names: Iterable[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in names if (output_dir / name).exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing split artifacts: {joined}")


def main() -> None:
    args = parse_args()
    if not 0.0 < args.test_fraction < 1.0:
        raise SystemExit("--test-fraction must be strictly between zero and one")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if not args.parquet.is_file():
        raise FileNotFoundError(args.parquet)
    if not args.validation_manifest.is_file():
        raise FileNotFoundError(args.validation_manifest)

    output_names = (
        "test-manifest.jsonl",
        "test-metadata.parquet",
        "test-video-groups.jsonl",
        "train-excluded-video-groups.jsonl",
        "video-group-splits.parquet",
        "split-summary.json",
        "SHA256SUMS",
    )
    ensure_output_is_new(args.output_dir, output_names)
    validation_keys = load_validation_video_keys(args.validation_manifest)
    parquet = pq.ParquetFile(args.parquet)
    missing = set(SOURCE_COLUMNS).difference(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Training Parquet is missing columns: {sorted(missing)}")

    split_stats = {split: empty_stats() for split in ("train", "test", "validation")}
    country_stats: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: {split: empty_stats() for split in ("train", "test", "validation")}
    )
    source_country_stats: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: {split: empty_stats() for split in ("train", "test", "validation")}
    )
    duration_stats: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: {split: empty_stats() for split in ("train", "test", "validation")}
    )
    seen_groups: set[str] = set()
    seen_samples: set[str] = set()
    matched_validation_groups: set[str] = set()
    test_rows_buffer: list[dict[str, Any]] = []
    test_manifest_path = args.output_dir / "test-manifest.jsonl"
    test_parquet_path = args.output_dir / "test-metadata.parquet"

    with (
        test_manifest_path.open("w", encoding="utf-8") as test_manifest,
        pq.ParquetWriter(
            test_parquet_path,
            TEST_PARQUET_SCHEMA,
            compression="zstd",
            use_dictionary=["audio_tar_path", "country_code"],
        ) as test_parquet,
    ):
        for batch in parquet.iter_batches(
            batch_size=args.batch_size,
            columns=list(SOURCE_COLUMNS),
            use_threads=True,
        ):
            columns = [column.to_pylist() for column in batch.columns]
            for sample_key, tar_path, member, duration, target_json in zip(
                *columns, strict=True
            ):
                sample_key = str(sample_key)
                if sample_key in seen_samples:
                    raise ValueError(f"Duplicate sample key: {sample_key}")
                seen_samples.add(sample_key)
                video_key = video_key_from_sample_key(sample_key)
                seen_groups.add(video_key)
                split = split_for_video(
                    video_key,
                    validation_keys=validation_keys,
                    seed=args.seed,
                    test_fraction=args.test_fraction,
                )
                if split == "validation":
                    matched_validation_groups.add(video_key)

                duration = float(duration)
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError(f"Invalid duration for {sample_key}: {duration}")
                try:
                    target = json.loads(str(target_json))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid target_json for {sample_key}") from error
                if not isinstance(target, dict):
                    raise ValueError(f"target_json is not an object for {sample_key}")
                reference = target.get("transcription")
                if not isinstance(reference, str) or not reference.strip():
                    raise ValueError(f"Missing transcription for {sample_key}")
                country = str(target.get("country_code") or "UNK").upper()
                source_country = source_country_from_sample_key(sample_key).upper()
                code_switch = bool(target.get("has_code_switching", False))
                has_diacritics = ARABIC_DIACRITICS.search(reference) is not None
                for stats in (
                    split_stats[split],
                    country_stats[country][split],
                    source_country_stats[source_country][split],
                    duration_stats[duration_bin(duration)][split],
                ):
                    update_stats(
                        stats,
                        duration=duration,
                        code_switch=code_switch,
                        has_diacritics=has_diacritics,
                    )

                if split != "test":
                    continue
                eval_row = {
                    "sample_key": sample_key,
                    "video_key": video_key,
                    "audio_tar_path": str(tar_path),
                    "audio_member": str(member),
                    "duration_sec": duration,
                    "reference": reference.strip(),
                    "country_code": country,
                    **{flag: bool(target.get(flag, False)) for flag in TARGET_FLAGS},
                    "detected_cs_lang": target.get("detected_cs_lang", []),
                }
                test_manifest.write(
                    json.dumps(eval_row, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                test_rows_buffer.append(
                    {
                        "sample_key": sample_key,
                        "video_key": video_key,
                        "audio_tar_path": str(tar_path),
                        "audio_member": str(member),
                        "duration_sec": duration,
                        "target_json": str(target_json),
                        "reference": reference.strip(),
                        "country_code": country,
                        "has_code_switching": code_switch,
                        "has_arabic_diacritics": has_diacritics,
                    }
                )
                if len(test_rows_buffer) >= args.batch_size:
                    write_parquet_rows(
                        test_parquet, test_rows_buffer, TEST_PARQUET_SCHEMA
                    )
        write_parquet_rows(test_parquet, test_rows_buffer, TEST_PARQUET_SCHEMA)

    missing_validation = validation_keys.difference(matched_validation_groups)
    if missing_validation:
        raise ValueError(
            f"{len(missing_validation)} validation video groups are absent from the Parquet"
        )
    group_counts = write_group_files(
        seen_groups=seen_groups,
        validation_keys=validation_keys,
        seed=args.seed,
        test_fraction=args.test_fraction,
        output_dir=args.output_dir,
    )
    for split in split_stats:
        split_stats[split]["video_groups"] = group_counts.get(split, 0)

    train_test_rows = split_stats["train"]["rows"] + split_stats["test"]["rows"]
    train_test_seconds = (
        split_stats["train"]["duration_seconds"]
        + split_stats["test"]["duration_seconds"]
    )
    train_test_cs_rows = (
        split_stats["train"]["code_switch_rows"]
        + split_stats["test"]["code_switch_rows"]
    )
    train_test_cs_seconds = (
        split_stats["train"]["code_switch_seconds"]
        + split_stats["test"]["code_switch_seconds"]
    )
    overall_test_ratios = {
        "rows": ratio(split_stats["test"]["rows"], train_test_rows),
        "duration": ratio(split_stats["test"]["duration_seconds"], train_test_seconds),
        "code_switch_rows": ratio(
            split_stats["test"]["code_switch_rows"], train_test_cs_rows
        ),
        "code_switch_duration": ratio(
            split_stats["test"]["code_switch_seconds"], train_test_cs_seconds
        ),
    }
    tolerance = 0.01
    ratio_checks = {
        name: value is not None and abs(value - args.test_fraction) <= tolerance
        for name, value in overall_test_ratios.items()
    }
    if not all(ratio_checks.values()):
        raise RuntimeError(
            f"Test split falls outside ±{tolerance:.2%} tolerance: {overall_test_ratios}"
        )

    artifact_names = output_names[:5]
    artifacts = {
        name: {
            "path": str((args.output_dir / name).resolve()),
            "bytes": (args.output_dir / name).stat().st_size,
            "sha256": sha256_file(args.output_dir / name),
        }
        for name in artifact_names
    }
    summary = {
        "schema_version": 1,
        "method": "SHA-256 threshold by complete video group; validation override",
        "seed": args.seed,
        "requested_test_fraction": args.test_fraction,
        "ratio_tolerance": tolerance,
        "dataset": {
            "parquet": str(args.parquet.resolve()),
            "parquet_sha256": sha256_file(args.parquet),
            "revision": args.dataset_revision,
            "rows": parquet.metadata.num_rows,
        },
        "validation": {
            "manifest": str(args.validation_manifest.resolve()),
            "manifest_sha256": sha256_file(args.validation_manifest),
            "requested_video_groups": len(validation_keys),
            "matched_video_groups": len(matched_validation_groups),
        },
        "splits": {
            split: json_ready_stats(stats) for split, stats in split_stats.items()
        },
        "test_fraction_excluding_validation": overall_test_ratios,
        "distributions": {
            "target_country": distribution_report(country_stats),
            "source_country": distribution_report(source_country_stats),
            "segment_duration": distribution_report(duration_stats),
        },
        "checks": {
            "assignment_unit": "video_key",
            "video_overlap_possible": False,
            "all_validation_groups_matched": not missing_validation,
            "unique_sample_keys": len(seen_samples) == parquet.metadata.num_rows,
            "test_ratio_within_one_percentage_point": ratio_checks,
            "audio_copied_or_extracted": False,
        },
        "artifacts": artifacts,
    }
    summary_path = args.output_dir / "split-summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    checksums_path = args.output_dir / "SHA256SUMS"
    checksum_names = (*artifact_names, "split-summary.json")
    checksums_path.write_text(
        "".join(
            f"{sha256_file(args.output_dir / name)}  {name}\n"
            for name in checksum_names
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
