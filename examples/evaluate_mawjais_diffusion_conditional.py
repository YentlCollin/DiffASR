#!/usr/bin/env python3
"""Generate or merge a paired four-prompt conditional DiffASR evaluation.

Examples
--------
Quick 64-audio gate on one GPU::

    python examples/evaluate_mawjais_diffusion_conditional.py run \
      --checkpoint CHECKPOINT --manifest validation-metadata.parquet \
      --dataset-root DATASET --output-jsonl gate.jsonl --limit 64 --nfe 4

Full validation can be split across eight GPUs by launching ``run`` eight
times with ``--num-shards 8 --shard-index 0..7`` and distinct output paths,
then merged with the ``score`` subcommand.  Sharding is by ``sample_key`` and
never separates the four views of one audio.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import html
import importlib.util
import io
import json
import math
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time
from types import ModuleType, SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from audio.diffusion.canvas import (  # noqa: E402
    DurationBucketSelector,
    truncate_at_first_turn_end,
)
from audio.diffusion.conditional import CONDITIONAL_ASR_VIEWS  # noqa: E402
from audio.diffusion.conditional_evaluation import (  # noqa: E402
    score_conditional_predictions,
)
from audio.diffusion.config import ConfidenceSamplerConfig  # noqa: E402
from audio.diffusion.collator import build_mawjais_asr_prefix  # noqa: E402
from audio.diffusion.sampler import sample_diffusion  # noqa: E402


TARGET_FIELDS = tuple(view.target_field for view in CONDITIONAL_ASR_VIEWS)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest_signature(path: Path) -> dict[str, Any] | None:
    """Verify the signed companion emitted for production validation Parquet."""

    if path.suffix.lower() != ".parquet":
        return None
    companion = path.with_suffix(".manifest.json")
    if not companion.is_file():
        raise FileNotFoundError(
            f"signed validation manifest companion is missing: {companion}"
        )
    try:
        payload = json.loads(companion.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid validation manifest signature: {companion}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported validation manifest signature schema")
    if payload.get("metadata_path") != path.name:
        raise ValueError("validation signature metadata_path does not name the Parquet")
    actual_hash = _sha256_file(path)
    if payload.get("metadata_sha256") != actual_hash:
        raise ValueError("validation Parquet SHA-256 does not match its signature")
    claimed_content_hash = payload.get("content_sha256")
    unsigned = dict(payload)
    unsigned.pop("content_sha256", None)
    actual_content_hash = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if claimed_content_hash != actual_content_hash:
        raise ValueError("validation manifest content_sha256 mismatch")
    return {
        "path": str(companion.resolve()),
        "sha256": _sha256_file(companion),
        "content_sha256": actual_content_hash,
    }


def _stable_shard(sample_key: str, num_shards: int) -> int:
    digest = hashlib.sha256(sample_key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def _sampling_seed(evaluation_seed: int, sample_key: str) -> int:
    payload = f"{evaluation_seed}\0{sample_key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _load_inference_module() -> ModuleType:
    path = REPOSITORY_ROOT / "examples" / "infer_mawjais_diffusion.py"
    name = "_mawjais_conditional_inference_runtime"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - installation guard
        raise RuntimeError(f"could not import inference runtime from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(row)
    return rows


def read_evaluation_manifest(path: Path) -> list[dict[str, Any]]:
    """Read the compact validation metadata generated by conditional prep."""

    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".jsonl", ".json"}:
        rows = read_jsonl(path)
    elif path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as error:  # pragma: no cover - runtime dependency
            raise RuntimeError("pyarrow is required to read a Parquet manifest") from error
        rows = pq.read_table(path).to_pylist()
    else:
        raise ValueError("evaluation manifest must be .parquet or .jsonl")

    seen: set[str] = set()
    for index, row in enumerate(rows):
        sample_key = row.get("sample_key")
        if not isinstance(sample_key, str) or not sample_key:
            raise ValueError(f"manifest row {index} has no sample_key")
        if sample_key in seen:
            raise ValueError(f"duplicate manifest sample_key: {sample_key}")
        seen.add(sample_key)
    return rows


def _valid_view_mask(row: Mapping[str, Any]) -> tuple[bool, bool, bool, bool]:
    raw = row.get("valid_view_mask")
    if isinstance(raw, int) and not isinstance(raw, bool):
        if raw < 0 or raw > 15:
            raise ValueError("integer valid_view_mask must be in [0, 15]")
        return tuple(  # type: ignore[return-value]
            bool(raw & (1 << view.view_id)) for view in CONDITIONAL_ASR_VIEWS
        )
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError("valid_view_mask must contain exactly four booleans")
    if any(not isinstance(value, (bool, np.bool_)) for value in raw):
        raise TypeError("valid_view_mask entries must be bool")
    return tuple(bool(value) for value in raw)  # type: ignore[return-value]


def select_compatible_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    limit: int | None,
    num_shards: int,
    shard_index: int,
    shard_mode: str = "sample_hash",
) -> tuple[list[dict[str, Any]], dict[str, int | float | str]]:
    """Keep all-four-view rows, apply a global limit, then deterministic sharding."""

    compatible: list[dict[str, Any]] = []
    incompatible = 0
    for source in rows:
        row = dict(source)
        mask = _valid_view_mask(row)
        targets_valid = all(
            isinstance(row.get(field), str) and bool(str(row[field]).strip())
            for field in TARGET_FIELDS
        )
        if not all(mask) or not targets_valid:
            incompatible += 1
            continue
        for field in ("audio_tar_path", "audio_member", "country_code"):
            if not isinstance(row.get(field), str) or not str(row[field]).strip():
                raise ValueError(
                    f"compatible sample {row.get('sample_key')!r} has no {field}"
                )
        duration = row.get("duration_sec")
        if (
            not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or duration <= 0
        ):
            raise ValueError(
                f"compatible sample {row.get('sample_key')!r} has invalid duration_sec"
            )
        content = row.get("content_has_code_switching")
        if not isinstance(content, (bool, np.bool_)):
            raise TypeError("content_has_code_switching must be bool")
        row["content_has_code_switching"] = bool(content)
        compatible.append(row)

    compatible.sort(key=lambda row: str(row["sample_key"]))
    global_selected = compatible if limit is None else compatible[:limit]
    if shard_mode == "sample_hash":
        selected = [
            row
            for row in global_selected
            if _stable_shard(str(row["sample_key"]), num_shards) == shard_index
        ]
    elif shard_mode == "tar_balanced":
        # Assign complete TAR archives to workers with deterministic
        # longest-processing-time balancing.  This prevents every worker from
        # reopening most archives during large distributed evaluations.
        tar_durations: dict[str, float] = defaultdict(float)
        for row in global_selected:
            tar_durations[str(row["audio_tar_path"])] += float(row["duration_sec"])
        shard_loads = [0.0] * num_shards
        tar_to_shard: dict[str, int] = {}
        for tar_name, duration in sorted(
            tar_durations.items(), key=lambda item: (-item[1], item[0])
        ):
            destination = min(range(num_shards), key=lambda idx: (shard_loads[idx], idx))
            tar_to_shard[tar_name] = destination
            shard_loads[destination] += duration
        selected = [
            row
            for row in global_selected
            if tar_to_shard[str(row["audio_tar_path"])] == shard_index
        ]
    else:
        raise ValueError(f"unsupported shard_mode: {shard_mode!r}")
    return selected, {
        "manifest_rows": len(compatible) + incompatible,
        "compatible_rows": len(compatible),
        "incompatible_rows_skipped": incompatible,
        "globally_selected_rows": len(global_selected),
        "shard_selected_rows": len(selected),
        "shard_selected_duration_seconds": sum(
            float(row["duration_sec"]) for row in selected
        ),
        "shard_mode": shard_mode,
    }


def merge_prediction_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Merge shards, collapsing exact duplicates and rejecting conflicts."""

    merged: dict[tuple[str, int], dict[str, Any]] = {}
    provenance: dict[str, Any] | None = None
    provenance_fields = (
        "checkpoint",
        "checkpoint_update",
        "manifest_sha256",
        "sampler",
        "nfe",
        "evaluation_seed",
    )
    for path in paths:
        for row in read_jsonl(path):
            key = (str(row.get("sample_key", "")), int(row.get("view_id", -1)))
            if not key[0] or not 0 <= key[1] <= 3:
                raise ValueError(f"invalid prediction identity in {path}: {key}")
            row_provenance = {field: row.get(field) for field in provenance_fields}
            if provenance is None:
                provenance = row_provenance
            elif row_provenance != provenance:
                raise ValueError(
                    f"prediction provenance mismatch in {path}: {row_provenance} != {provenance}"
                )
            previous = merged.get(key)
            if previous is not None and previous != row:
                raise ValueError(f"conflicting duplicate prediction for {key}")
            merged[key] = row
    return [merged[key] for key in sorted(merged)]


def _safe_relative_path(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"unsafe relative TAR path: {value!r}")
    return Path(*pure.parts)


def _resolve_tar_path(dataset_root: Path, value: str) -> Path:
    candidate = Path(value)
    path = candidate if candidate.is_absolute() else dataset_root / _safe_relative_path(value)
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def decode_audio(payload: bytes, target_sample_rate: int = 16_000) -> np.ndarray:
    """Decode corpus audio in memory without extracting TAR members."""

    try:
        import soundfile as sf
    except ImportError:
        import wave

        with wave.open(io.BytesIO(payload), "rb") as stream:
            if stream.getcomptype() != "NONE":
                raise ValueError(f"unsupported compressed WAV type: {stream.getcomptype()}")
            channels = stream.getnchannels()
            width = stream.getsampwidth()
            sample_rate = stream.getframerate()
            raw = stream.readframes(stream.getnframes())
        if channels <= 0:
            raise ValueError("WAV has no channels")
        if width == 1:
            decoded = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif width == 2:
            decoded = np.frombuffer(raw, dtype=np.dtype("<i2")).astype(np.float32) / 32768.0
        elif width == 3:
            octets = np.frombuffer(raw, dtype=np.uint8)
            if octets.size % 3:
                raise ValueError("invalid 24-bit PCM WAV payload")
            octets = octets.reshape(-1, 3).astype(np.int32)
            integer = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
            integer = (integer ^ 0x800000) - 0x800000
            decoded = integer.astype(np.float32) / 8388608.0
        elif width == 4:
            decoded = np.frombuffer(raw, dtype=np.dtype("<i4")).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"unsupported PCM WAV width: {width}")
        if decoded.size % channels:
            raise ValueError("WAV payload is not divisible by channel count")
        samples = decoded.reshape(-1, channels)
    else:
        samples, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
    if samples.shape[0] == 0:
        raise ValueError("decoded audio is empty")
    waveform = samples.mean(axis=1, dtype=np.float32)
    sample_rate = int(sample_rate)
    if sample_rate != target_sample_rate:
        from scipy.signal import resample_poly

        divisor = math.gcd(sample_rate, target_sample_rate)
        waveform = resample_poly(
            waveform, target_sample_rate // divisor, sample_rate // divisor
        ).astype(np.float32, copy=False)
    return np.ascontiguousarray(waveform, dtype=np.float32)


def _processor_prefix(
    processor: Any, waveform: np.ndarray, prompt_tag: str
) -> dict[str, torch.Tensor]:
    output = dict(
        processor(
            text=[build_mawjais_asr_prefix(prompt_tag)],
            audio=[waveform],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
    )
    required = {
        "input_ids",
        "attention_mask",
        "audio_position_mask",
        "input_features",
        "feature_attention_mask",
    }
    missing = required.difference(output)
    if missing:
        raise ValueError(f"processor output is missing fields: {sorted(missing)}")
    if output["audio_position_mask"].dtype != torch.bool:
        raise TypeError("processor audio_position_mask must be torch.bool")
    return output


def _canvas_sizes(row: Mapping[str, Any], diffusion: Any) -> tuple[int, int, int, int]:
    raw = row.get("canvas_sizes")
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        sizes = tuple(int(value) for value in raw)
    else:
        selector = DurationBucketSelector(
            diffusion.canvas.sizes,
            tokens_per_second=diffusion.canvas.duration_tokens_per_second,
            margin_tokens=diffusion.canvas.duration_margin_tokens,
        )
        selected = selector.select_or_raise(float(row["duration_sec"]))
        sizes = (selected, selected, selected, selected)
    if any(size not in diffusion.canvas.sizes for size in sizes):
        raise ValueError(
            f"manifest canvas sizes {sizes} are incompatible with trained buckets "
            f"{diffusion.canvas.sizes}"
        )
    return sizes  # type: ignore[return-value]


def _decode_prediction(processor: Any, ids: Sequence[int], turn_end_token_id: int) -> str:
    truncated = truncate_at_first_turn_end(
        ids, turn_end_token_id=turn_end_token_id, include_turn_end=False
    )
    return processor.tokenizer.decode(
        truncated,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ).strip()


@torch.no_grad()
def generate_four_views(
    *,
    row: Mapping[str, Any],
    waveform: np.ndarray,
    model: torch.nn.Module,
    processor: Any,
    diffusion: Any,
    noise_process: Any,
    sampler_config: ConfidenceSamplerConfig,
    forbidden_output_token_ids: Sequence[int],
    layout: Any,
    manifest_sha256: str,
    evaluation_seed: int,
    only_view_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Sample all requested views with one shared audio-tower projection."""

    sizes = _canvas_sizes(row, diffusion)
    device = next(model.parameters()).device
    sample_seed = _sampling_seed(evaluation_seed, str(row["sample_key"]))
    projected_audio: torch.Tensor | None = None
    reference_feature_mask: torch.Tensor | None = None
    outputs: list[dict[str, Any]] = []
    for view in CONDITIONAL_ASR_VIEWS:
        if only_view_ids is not None and view.view_id not in only_view_ids:
            continue
        started = time.perf_counter()
        prefix = _processor_prefix(processor, waveform, view.prompt_tag)
        prefix_ids = prefix["input_ids"].to(device)
        prefix_valid = prefix["attention_mask"].to(device=device, dtype=torch.bool)
        audio_positions = prefix["audio_position_mask"].to(device)
        feature_mask = prefix["feature_attention_mask"].to(device)
        if projected_audio is None:
            projected_audio = model.encode_audio(prefix["input_features"].to(device), feature_mask)
            reference_feature_mask = feature_mask.detach().cpu()
        elif not torch.equal(feature_mask.detach().cpu(), reference_feature_mask):
            raise ValueError("audio feature mask changed between conditional prompts")

        canvas_size = sizes[view.view_id]
        canvas_valid = torch.ones((1, canvas_size), device=device, dtype=torch.bool)

        def denoiser(canvas: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
            return model.forward_diffusion(
                prefix_input_ids=prefix_ids,
                prefix_attention_mask=prefix_valid,
                audio_position_mask=audio_positions,
                feature_attention_mask=feature_mask,
                projected_audio_embeddings=projected_audio,
                noisy_canvas_ids=canvas,
                canvas_valid_mask=canvas_valid,
                noise_level=noise_level,
                denoiser_output_alignment=diffusion.denoiser_output_alignment,
            ).logits

        # The same seed is reset for each view.  Thus paired Uniform runs start
        # from the same x_T rather than confounding prompt effects with prior noise.
        generator = torch.Generator(device=device).manual_seed(sample_seed)
        sampled = sample_diffusion(
            denoiser,
            noise_process,
            (1, canvas_size),
            device=device,
            config=ConfidenceSamplerConfig(
                sampler=sampler_config.sampler,
                max_nfe=sampler_config.max_nfe,
                prediction_mode=sampler_config.prediction_mode,
                temperature_start=sampler_config.temperature_start,
                temperature_end=sampler_config.temperature_end,
                entropy_budget_gamma=sampler_config.entropy_budget_gamma,
                position_bias_lambda=sampler_config.position_bias_lambda,
                forbidden_output_token_ids=tuple(forbidden_output_token_ids),
            ),
            generator=generator,
            canvas_valid_mask=canvas_valid,
            padding_token_id=diffusion.canvas.pad_token_id,
        )
        full_ids = [int(value) for value in sampled.canvas[0].tolist()]
        elapsed = time.perf_counter() - started
        content = bool(row["content_has_code_switching"])
        outputs.append(
            {
                "sample_key": str(row["sample_key"]),
                "view_id": view.view_id,
                "view_name": view.name,
                "request_code_switching": view.request_code_switching,
                "request_diacritics": view.request_diacritics,
                "content_has_code_switching": content,
                "applied_code_switching": bool(content and view.request_code_switching),
                "country_code": str(row["country_code"]),
                "reference": str(row[view.target_field]),
                "hypothesis": _decode_prediction(
                    processor,
                    full_ids,
                    turn_end_token_id=diffusion.canvas.turn_end_token_id,
                ),
                "duration_sec": float(row["duration_sec"]),
                "generation_seconds": elapsed,
                "inference_prompt_tag": view.prompt_tag,
                "canvas_size": canvas_size,
                "terminated": diffusion.canvas.turn_end_token_id in full_ids,
                "nfe": int(sampled.nfe),
                "sampler": sampler_config.sampler,
                "sampling_seed": sample_seed,
                "evaluation_seed": int(evaluation_seed),
                "checkpoint": str(layout.root),
                "checkpoint_update": layout.metadata.get("update"),
                "manifest_sha256": manifest_sha256,
            }
        )
    return outputs


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _dashboard_metrics(report: Mapping[str, Any]) -> list[tuple[str, float | None]]:
    views = report["views"]
    slices = report["instruction_slices"]
    return [
        (
            f"{view.name}: CER lexical",
            views[view.name]["cer_without_diacritics"],
        )
        for view in CONDITIONAL_ASR_VIEWS
    ] + [
        ("Diacritics: DER", slices["diacritics_enabled"]["der"]),
        (
            "Sans Diacritics: fuite",
            slices["diacritics_disabled"]["diacritic_leakage_mark_rate"],
        ),
        ("Code-switch: rappel latin", slices["code_switch_enabled"]["latin_word_recall"]),
        (
            "Code-switch sans contenu: hallucination",
            slices["code_switch_enabled_no_content"][
                "latin_hallucination_utterance_rate"
            ],
        ),
    ]


def _render_metrics_svg(report: Mapping[str, Any]) -> str:
    """Render a dependency-free checkpoint dashboard as deterministic SVG."""

    metrics = _dashboard_metrics(report)
    finite = [value for _, value in metrics if value is not None and math.isfinite(value)]
    scale = max([1.0, *finite])
    width = 960
    label_width = 390
    bar_width = 430
    row_height = 42
    height = 95 + row_height * len(metrics)
    bars = []
    for index, (label, value) in enumerate(metrics):
        y = 75 + index * row_height
        amount = 0.0 if value is None else max(0.0, float(value))
        pixels = bar_width * min(amount / scale, 1.0)
        display = "n/a" if value is None else f"{100.0 * value:.3f}%"
        bars.append(
            f'<text x="20" y="{y + 17}" font-size="14">{html.escape(label)}</text>'
            f'<rect x="{label_width}" y="{y}" width="{bar_width}" height="24" '
            'fill="#e8edf3" rx="3"/>'
            f'<rect x="{label_width}" y="{y}" width="{pixels:.2f}" height="24" '
            'fill="#3769b0" rx="3"/>'
            f'<text x="{label_width + bar_width + 12}" y="{y + 17}" '
            f'font-size="14">{html.escape(display)}</text>'
        )
    checkpoint = html.escape(str(report.get("checkpoint", "conditional checkpoint")))
    update = html.escape(str(report.get("checkpoint_update", "unknown")))
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        '<text x="20" y="30" font-size="21" font-weight="bold">'
        'Évaluation DiffASR conditionnelle</text>\n'
        f'<text x="20" y="53" font-size="12">checkpoint={checkpoint} · update={update}</text>\n'
        + "\n".join(bars)
        + "\n</svg>\n"
    )


def write_scoring_artifacts(
    report: dict[str, Any],
    *,
    artifacts_dir: Path,
    summary_path: Path | None,
    merged_path: Path | None,
) -> dict[str, str]:
    """Write deterministic CSV/SVG plus hashes for one checkpoint report."""

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    csv_path = artifacts_dir / "conditional_metrics.csv"
    svg_path = artifacts_dir / "conditional_metrics.svg"
    manifest_path = artifacts_dir / "artifact_manifest.json"
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer, lineterminator="\n")
    writer.writerow(("metric", "value", "percent"))
    for label, value in _dashboard_metrics(report):
        writer.writerow(
            (
                label,
                "" if value is None else f"{value:.12g}",
                "" if value is None else f"{100.0 * value:.9f}",
            )
        )
    _write_text_atomic(csv_path, csv_buffer.getvalue())
    _write_text_atomic(svg_path, _render_metrics_svg(report))
    paths = {
        "metrics_csv": str(csv_path.resolve()),
        "metrics_svg": str(svg_path.resolve()),
        "artifact_manifest": str(manifest_path.resolve()),
    }
    report["artifacts"] = {"directory": str(artifacts_dir.resolve()), **paths}
    if summary_path is not None:
        _write_json_atomic(summary_path, report)
    files = {
        "metrics_csv": {"path": csv_path.name, "sha256": _sha256_file(csv_path)},
        "metrics_svg": {"path": svg_path.name, "sha256": _sha256_file(svg_path)},
    }
    if summary_path is not None:
        files["summary_json"] = {
            "path": str(summary_path.resolve()),
            "sha256": _sha256_file(summary_path),
        }
    if merged_path is not None:
        files["merged_jsonl"] = {
            "path": str(merged_path.resolve()),
            "sha256": _sha256_file(merged_path),
        }
    artifact_manifest = {
        "schema_version": 1,
        "evaluation": report["evaluation"],
        "checkpoint": report.get("checkpoint"),
        "checkpoint_update": report.get("checkpoint_update"),
        "files": files,
    }
    _write_json_atomic(manifest_path, artifact_manifest)
    return paths


def _sampling_config(
    inference: ModuleType,
    resolved: Mapping[str, Any],
    diffusion: Any,
    args: argparse.Namespace,
) -> ConfidenceSamplerConfig:
    if args.sampler in ("argmax_udm", "udm_marginalization"):
        if diffusion.noise_type != "uniform":
            raise ValueError("UDM samplers require a checkpoint trained with uniform noise")
        return ConfidenceSamplerConfig(
            sampler=args.sampler,
            max_nfe=int(args.nfe if args.nfe is not None else 32),
            prediction_mode="argmax" if args.sampler == "argmax_udm" else "categorical",
            temperature_start=1.0,
            temperature_end=1.0,
        )
    controls = inference._resolve_sampling_controls(
        resolved,
        SimpleNamespace(
            sampler="confidence_schedule",
            nfe=args.nfe,
            prediction_mode="argmax",
            temperature_start=None,
            temperature_end=None,
            entropy_budget_gamma=None,
            position_bias_lambda=None,
            adaptive_stopping=False,
        ),
    )
    inference._validate_sampling_semantics(diffusion, controls)
    return ConfidenceSamplerConfig(
        sampler="confidence_schedule",
        max_nfe=int(controls["max_nfe"]),
        prediction_mode="argmax",
        temperature_start=float(controls["temperature_start"]),
        temperature_end=float(controls["temperature_end"]),
        entropy_budget_gamma=float(controls["entropy_budget_gamma"]),
        position_bias_lambda=float(controls["position_bias_lambda"]),
    )


def run_generation(args: argparse.Namespace) -> dict[str, Any]:
    manifest = args.manifest.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    manifest_signature = validate_manifest_signature(manifest)
    manifest_rows = read_evaluation_manifest(manifest)
    selected, selection = select_compatible_rows(
        manifest_rows,
        limit=args.limit,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        shard_mode=args.shard_mode,
    )
    manifest_sha256 = _sha256_file(manifest)

    inference = _load_inference_module()
    layout = inference.read_diffusion_checkpoint_layout(args.checkpoint)
    resolved = inference._load_resolved_config(layout, args.config)
    diffusion = inference._resolve_semantics(
        resolved,
        SimpleNamespace(noise_type=None, time_conditioning=None, alignment=None),
    )
    sampler_config = _sampling_config(inference, resolved, diffusion, args)
    dry_payload = {
        "schema_version": 1,
        "evaluation": "mawjais_diffusion_conditional_four_view_paired",
        "status": "VALIDATED" if args.dry_run else "RUNNING",
        "checkpoint": str(layout.root),
        "checkpoint_update": layout.metadata.get("update"),
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256,
        "manifest_signature": manifest_signature,
        "sampler": sampler_config.sampler,
        "prediction_mode": sampler_config.prediction_mode,
        "nfe": sampler_config.max_nfe,
        "evaluation_seed": args.seed,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        **selection,
    }
    if args.dry_run:
        return dry_payload
    if not selected:
        empty_report = {
            **dry_payload,
            "status": "EMPTY_SHARD",
            "prediction_rows": 0,
        }
        output_path = args.output_jsonl.expanduser().resolve()
        _write_jsonl_atomic(output_path, [])
        summary = args.summary_json or output_path.with_suffix(".metrics.json")
        _write_json_atomic(summary.expanduser().resolve(), empty_report)
        return empty_report

    model, processor = inference._load_trained_model(layout, diffusion, args)
    token_policy = inference.resolve_runtime_token_policy(
        model, processor, resolved, diffusion, layout
    )
    noise_process = inference._noise_process(
        diffusion, uniform_noise_ids=token_policy.uniform_noise_ids
    )

    output_path = args.output_jsonl.expanduser().resolve()
    existing = read_jsonl(output_path) if output_path.is_file() else []
    selected_keys = {str(row["sample_key"]) for row in selected}
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in existing:
        key = (str(row.get("sample_key", "")), int(row.get("view_id", -1)))
        if key[0] not in selected_keys:
            raise ValueError(f"resumed output contains out-of-shard sample {key[0]!r}")
        expected_provenance = {
            "checkpoint": str(layout.root),
            "manifest_sha256": manifest_sha256,
            "sampler": sampler_config.sampler,
            "nfe": sampler_config.max_nfe,
            "evaluation_seed": args.seed,
        }
        actual = {field: row.get(field) for field in expected_provenance}
        if actual != expected_provenance:
            raise ValueError(
                f"resumed prediction provenance mismatch for {key}: {actual}"
            )
        if key in completed and completed[key] != row:
            raise ValueError(f"conflicting resumed prediction for {key}")
        completed[key] = row

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        if any(
            (str(row["sample_key"]), view.view_id) not in completed
            for view in CONDITIONAL_ASR_VIEWS
        ):
            grouped[str(row["audio_tar_path"])].append(row)

    with output_path.open("a", encoding="utf-8") as output:
        done_samples = sum(
            all((key, view.view_id) in completed for view in CONDITIONAL_ASR_VIEWS)
            for key in selected_keys
        )
        for tar_name in sorted(grouped):
            tar_path = _resolve_tar_path(dataset_root, tar_name)
            with tarfile.open(tar_path, mode="r:") as archive:
                for row in grouped[tar_name]:
                    sample_key = str(row["sample_key"])
                    missing = {
                        view.view_id
                        for view in CONDITIONAL_ASR_VIEWS
                        if (sample_key, view.view_id) not in completed
                    }
                    member = archive.extractfile(str(row["audio_member"]))
                    if member is None:
                        raise FileNotFoundError(f"{tar_path}:{row['audio_member']}")
                    waveform = decode_audio(member.read())
                    predictions = generate_four_views(
                        row=row,
                        waveform=waveform,
                        model=model,
                        processor=processor,
                        diffusion=diffusion,
                        noise_process=noise_process,
                        sampler_config=sampler_config,
                        forbidden_output_token_ids=token_policy.sampling_forbidden_ids,
                        layout=layout,
                        manifest_sha256=manifest_sha256,
                        evaluation_seed=args.seed,
                        only_view_ids=missing,
                    )
                    for prediction in predictions:
                        key = (sample_key, int(prediction["view_id"]))
                        output.write(
                            json.dumps(prediction, ensure_ascii=False, sort_keys=True)
                            + "\n"
                        )
                        output.flush()
                        completed[key] = prediction
                    done_samples += 1
                    print(
                        f"shard={args.shard_index}/{args.num_shards} "
                        f"evaluated_samples={done_samples}/{len(selected)}",
                        flush=True,
                    )

    ordered = [completed[key] for key in sorted(completed)]
    report = score_conditional_predictions(ordered)
    report.update(
        {
            "status": "COMPLETE",
            "checkpoint": str(layout.root),
            "checkpoint_update": layout.metadata.get("update"),
            "manifest": str(manifest),
            "manifest_sha256": manifest_sha256,
            "manifest_signature": manifest_signature,
            "sampler": sampler_config.sampler,
            "prediction_mode": sampler_config.prediction_mode,
            "nfe": sampler_config.max_nfe,
            "evaluation_seed": args.seed,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            **selection,
        }
    )
    summary = args.summary_json or output_path.with_suffix(".metrics.json")
    _write_json_atomic(summary.expanduser().resolve(), report)
    return report


def run_scoring(args: argparse.Namespace) -> dict[str, Any]:
    paths = [path.expanduser().resolve() for path in args.predictions]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = merge_prediction_rows(paths)
    manifest = getattr(args, "manifest", None)
    if manifest is not None:
        manifest = manifest.expanduser().resolve()
        validate_manifest_signature(manifest)
        expected_rows, _ = select_compatible_rows(
            read_evaluation_manifest(manifest),
            limit=None,
            num_shards=1,
            shard_index=0,
        )
        expected = {
            (str(row["sample_key"]), view.view_id)
            for row in expected_rows
            for view in CONDITIONAL_ASR_VIEWS
        }
        actual = {
            (str(row["sample_key"]), int(row["view_id"])) for row in rows
        }
        missing = expected - actual
        unexpected = actual - expected
        if missing or unexpected:
            raise ValueError(
                "prediction coverage does not match manifest: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
    report = score_conditional_predictions(
        rows, num_workers=int(getattr(args, "score_workers", 1))
    )
    if rows:
        report.update(
            {
                "checkpoint": rows[0].get("checkpoint"),
                "checkpoint_update": rows[0].get("checkpoint_update"),
                "manifest_sha256": rows[0].get("manifest_sha256"),
                "sampler": rows[0].get("sampler"),
                "nfe": rows[0].get("nfe"),
                "evaluation_seed": rows[0].get("evaluation_seed"),
            }
        )
    merged_path = (
        None if args.merged_jsonl is None else args.merged_jsonl.expanduser().resolve()
    )
    summary_path = (
        None if args.summary_json is None else args.summary_json.expanduser().resolve()
    )
    if args.merged_jsonl is not None:
        assert merged_path is not None
        _write_jsonl_atomic(merged_path, rows)
    artifact_root = (
        args.artifacts_dir.expanduser().resolve()
        if args.artifacts_dir is not None
        else (summary_path or merged_path or paths[0]).parent / "artifacts"
    )
    write_scoring_artifacts(
        report,
        artifacts_dir=artifact_root,
        summary_path=summary_path,
        merged_path=merged_path,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Generate one resumable evaluation shard.")
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--config")
    run.add_argument("--base-checkpoint")
    run.add_argument(
        "--fsdp-merged-weights",
        help=(
            "Temporary model.safetensors (or its directory) exported from an "
            "Accelerate FSDP sharded checkpoint."
        ),
    )
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--dataset-root", type=Path, required=True)
    run.add_argument("--output-jsonl", type=Path, required=True)
    run.add_argument("--summary-json", type=Path)
    run.add_argument("--limit", type=int)
    run.add_argument("--num-shards", type=int, default=1)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument(
        "--shard-mode",
        choices=("sample_hash", "tar_balanced"),
        default="sample_hash",
    )
    run.add_argument("--nfe", type=int)
    run.add_argument(
        "--sampler",
        choices=("confidence_schedule", "argmax_udm", "udm_marginalization"),
        default="confidence_schedule",
    )
    run.add_argument("--seed", type=int, default=3421)
    run.add_argument("--device", default="cuda")
    run.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    run.add_argument("--attention-backend", choices=("eager", "sdpa"), default="sdpa")
    run.add_argument("--dry-run", action="store_true")

    score = commands.add_parser(
        "score", help="Merge completed shards and compute exact corpus metrics."
    )
    score.add_argument("--predictions", type=Path, nargs="+", required=True)
    score.add_argument("--merged-jsonl", type=Path)
    score.add_argument("--summary-json", type=Path)
    score.add_argument("--score-workers", type=int, default=32)
    score.add_argument(
        "--manifest",
        type=Path,
        help="When provided, require exact four-view coverage of this manifest.",
    )
    score.add_argument(
        "--artifacts-dir",
        type=Path,
        help="Default: an artifacts/ directory beside metrics.json/all.jsonl.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.command == "score":
        if args.score_workers <= 0:
            raise ValueError("--score-workers must be positive")
        return
    if args.command != "run":
        return
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards)")
    if args.nfe is not None and args.nfe <= 0:
        raise ValueError("--nfe must be positive")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
        report = run_generation(args) if args.command == "run" else run_scoring(args)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (FileNotFoundError, ImportError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
        return 2  # pragma: no cover - argparse exits


if __name__ == "__main__":
    raise SystemExit(main())
