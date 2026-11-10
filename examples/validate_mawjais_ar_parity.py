#!/usr/bin/env python3
"""Validate the repository PyTorch AR path before enabling diffusion training.

The check uses a transcript already validated by the independent vLLM path. It
tests one no-cache teacher-forced forward (the path relevant to diffusion) and,
by default, cached greedy generation (the legacy AR regression). A trainer gate
artifact is written only when both checks reproduce the reference exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Sequence

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from audio.diffusion.collator import build_mawjais_asr_prefix  # noqa: E402
from audio.diffusion.loading import load_mawjais_r2_for_diffusion  # noqa: E402


DEFAULT_MODEL = "/path/to/authorized-base-model"
DEFAULT_REVISION = "user-provided-revision"
TURN_END_TOKEN_ID = 250019


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_audio(path: Path, target_sample_rate: int = 16_000) -> np.ndarray:
    try:
        import soundfile as sf

        waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        waveform = waveform.mean(axis=1, dtype=np.float32)
    except ImportError:
        # The validated MI210 container deliberately has a small Python image and
        # may not ship the optional soundfile binding.  The parity fixture is PCM
        # WAV, so retain a dependency-free path instead of asking a diagnostic
        # job to mutate its environment at runtime.
        import wave

        if path.suffix.lower() != ".wav":
            raise RuntimeError(
                "soundfile is required for non-WAV parity audio; use a PCM WAV "
                "fixture or install soundfile in the immutable ROCm image"
            )
        with wave.open(str(path), "rb") as stream:
            channels = stream.getnchannels()
            sample_width = stream.getsampwidth()
            sample_rate = stream.getframerate()
            frame_count = stream.getnframes()
            payload = stream.readframes(frame_count)
        dtype_and_scale = {
            1: (np.uint8, 128.0),
            2: (np.dtype("<i2"), 32768.0),
            4: (np.dtype("<i4"), 2147483648.0),
        }
        if sample_width not in dtype_and_scale:
            raise ValueError(f"unsupported PCM WAV sample width: {sample_width} bytes")
        sample_dtype, scale = dtype_and_scale[sample_width]
        decoded = np.frombuffer(payload, dtype=sample_dtype)
        if sample_width == 1:
            decoded = decoded.astype(np.float32) - 128.0
        waveform = decoded.astype(np.float32).reshape(-1, channels).mean(axis=1) / scale
    if waveform.size == 0:
        raise ValueError(f"audio is empty: {path}")
    sample_rate = int(sample_rate)
    if sample_rate != target_sample_rate:
        from scipy.signal import resample_poly
        import math

        divisor = math.gcd(sample_rate, target_sample_rate)
        waveform = resample_poly(
            waveform,
            target_sample_rate // divisor,
            sample_rate // divisor,
        ).astype(np.float32, copy=False)
    return np.ascontiguousarray(waveform, dtype=np.float32)


def _resolve_checkpoint(model: str, revision: str) -> str:
    local = Path(model).expanduser()
    if local.exists():
        return str(local.resolve())
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model, revision=revision)


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _move_processor_batch(
    batch: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for name, value in batch.items():
        if not isinstance(value, torch.Tensor):
            moved[name] = value
            continue
        value = value.to(device)
        if name == "input_features":
            value = value.to(dtype=dtype)
        moved[name] = value
    return moved


def _target_ids(tokenizer: Any, transcript: str) -> torch.Tensor:
    encoded = tokenizer(transcript, add_special_tokens=False)["input_ids"]
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("reference tokenizer unexpectedly returned a batch")
        encoded = encoded[0]
    ids = [int(token_id) for token_id in encoded]
    if not ids:
        raise ValueError("reference transcript tokenized to an empty sequence")
    if TURN_END_TOKEN_ID in ids:
        raise ValueError("reference transcript must not contain the assistant turn-end token")
    return torch.tensor(ids + [TURN_END_TOKEN_ID], dtype=torch.long)


@torch.no_grad()
def run_parity_checks(
    model: torch.nn.Module,
    processor: Any,
    *,
    waveform: np.ndarray,
    reference_transcript: str,
    prompt_tag: str,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int,
    check_generation: bool,
) -> dict[str, Any]:
    prompt = build_mawjais_asr_prefix(prompt_tag)
    prefix = processor(
        text=[prompt],
        audio=[waveform],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    prefix = _move_processor_batch(dict(prefix), device=device, dtype=dtype)
    if prefix["audio_position_mask"].dtype != torch.bool:
        raise TypeError("processor audio_position_mask must be torch.bool")
    expected = _target_ids(processor.tokenizer, reference_transcript).to(device)
    prefix_length = prefix["input_ids"].shape[1]
    full_ids = torch.cat((prefix["input_ids"], expected[None, :]), dim=1)
    full_attention = torch.cat(
        (
            prefix["attention_mask"],
            torch.ones((1, expected.numel()), device=device, dtype=prefix["attention_mask"].dtype),
        ),
        dim=1,
    )
    full_audio_positions = torch.cat(
        (
            prefix["audio_position_mask"],
            torch.zeros((1, expected.numel()), device=device, dtype=torch.bool),
        ),
        dim=1,
    )
    output = model(
        input_ids=full_ids,
        attention_mask=full_attention,
        audio_position_mask=full_audio_positions,
        input_features=prefix["input_features"],
        feature_attention_mask=prefix["feature_attention_mask"],
        use_cache=False,
    )
    prediction_logits = output.logits[:, prefix_length - 1 : prefix_length - 1 + expected.numel()]
    prediction = prediction_logits.argmax(dim=-1)[0]
    token_matches = prediction.eq(expected)
    mismatches = torch.nonzero(~token_matches, as_tuple=False).flatten()
    first_mismatch = None
    if mismatches.numel():
        index = int(mismatches[0].item())
        first_mismatch = {
            "index": index,
            "expected_id": int(expected[index].item()),
            "predicted_id": int(prediction[index].item()),
            "expected_token": processor.tokenizer.decode([int(expected[index].item())]),
            "predicted_token": processor.tokenizer.decode([int(prediction[index].item())]),
        }

    generated_exact = None
    generated_transcript = None
    generated_ids: list[int] | None = None
    if check_generation:
        generated = model.generate(
            **prefix,
            max_new_tokens=max(max_new_tokens, expected.numel()),
            do_sample=False,
            use_cache=True,
        )[0, prefix_length:]
        generated_list = [int(token_id) for token_id in generated.tolist()]
        if TURN_END_TOKEN_ID in generated_list:
            generated_list = generated_list[: generated_list.index(TURN_END_TOKEN_ID)]
        generated_ids = generated_list
        expected_without_end = expected[:-1].tolist()
        generated_exact = generated_list == expected_without_end
        generated_transcript = processor.tokenizer.decode(
            generated_list,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ).strip()

    return {
        "audio_placeholder_count": int(prefix["audio_position_mask"].sum().item()),
        "prefix_token_count": int(prefix_length),
        "reference_token_count_with_turn_end": int(expected.numel()),
        "teacher_forced_exact": bool(token_matches.all().item()),
        # Compute the report in Python scalar arithmetic. An all-True vector
        # must serialize as exactly 1.0 rather than a float32 reduction rounded
        # just below one; the boolean exactness remains the primary gate field.
        "teacher_forced_token_accuracy": int(token_matches.sum().item())
        / int(token_matches.numel()),
        "teacher_forced_first_mismatch": first_mismatch,
        "generation_checked": check_generation,
        "generation_exact": generated_exact,
        "generated_transcript": generated_transcript,
        "generated_token_ids": generated_ids,
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_passed_artifact(
    path: Path,
    *,
    checkpoint: str,
    revision: str,
    audio: Path,
    reference_transcript: str,
    checks: dict[str, Any],
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"parity artifact already exists: {path}; pass --overwrite")
    if not checks["teacher_forced_exact"] or checks["generation_exact"] is not True:
        raise ValueError("refusing to write a passed artifact because AR parity did not pass")
    payload = {
        "schema_version": 1,
        "status": "passed",
        "pytorch_ar_parity": True,
        "checkpoint": checkpoint,
        "revision": revision,
        "checks": checks,
        "provenance": {
            "audio_path": str(audio),
            "audio_sha256": _sha256_file(audio),
            "reference_transcript_sha256": hashlib.sha256(
                reference_transcript.encode("utf-8")
            ).hexdigest(),
            "repository_commit": _git_commit(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, type=Path)
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument("--reference-text")
    reference.add_argument("--reference-file", type=Path)
    parser.add_argument("--prompt-tag", default="[Diacritics]")
    parser.add_argument("--checkpoint", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attention-backend", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--teacher-force-only",
        action="store_true",
        help="Diagnostic only; a passed trainer artifact cannot be written in this mode.",
    )
    parser.add_argument("--write-artifact", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate local arguments without resolving/loading the 4B checkpoint.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        audio = args.audio.expanduser().resolve()
        if not audio.is_file():
            raise FileNotFoundError(audio)
        if args.reference_file is not None:
            reference_file = args.reference_file.expanduser().resolve()
            reference_transcript = reference_file.read_text(encoding="utf-8").strip()
        else:
            reference_transcript = args.reference_text.strip()
        if not reference_transcript:
            raise ValueError("reference transcript is empty")
        if args.max_new_tokens <= 0:
            raise ValueError("--max-new-tokens must be positive")
        if args.write_artifact is not None and args.teacher_force_only:
            raise ValueError("a trainer artifact requires the cached generation regression")
        if args.dry_run:
            print(json.dumps({"status": "validated", "model_loaded": False}, indent=2))
            return 0

        checkpoint_path = _resolve_checkpoint(args.checkpoint, args.revision)
        dtype = _dtype(args.dtype)
        device = torch.device(args.device)
        model, processor = load_mawjais_r2_for_diffusion(
            checkpoint_path,
            repository_root=REPOSITORY_ROOT,
            dtype=dtype,
            attn_implementation=args.attention_backend,
            device_map={"": str(device)} if device.type != "cpu" else None,
        )
        if device.type == "cpu":
            model.to(device=device, dtype=dtype)
        model.eval()
        waveform = _load_audio(audio, processor.feature_extractor.sampling_rate)
        checks = run_parity_checks(
            model,
            processor,
            waveform=waveform,
            reference_transcript=reference_transcript,
            prompt_tag=args.prompt_tag,
            device=device,
            dtype=dtype,
            max_new_tokens=args.max_new_tokens,
            check_generation=not args.teacher_force_only,
        )
        result = {
            "checkpoint": args.checkpoint,
            "revision": args.revision,
            "checks": checks,
            "passed": checks["teacher_forced_exact"]
            and (args.teacher_force_only or checks["generation_exact"] is True),
        }
        if args.write_artifact is not None and result["passed"]:
            _write_passed_artifact(
                args.write_artifact.expanduser().resolve(),
                checkpoint=args.checkpoint,
                revision=args.revision,
                audio=audio,
                reference_transcript=reference_transcript,
                checks=checks,
                overwrite=args.overwrite,
            )
            result["artifact"] = str(args.write_artifact.expanduser().resolve())
        elif args.write_artifact is not None:
            result["artifact"] = None
            result["artifact_reason"] = "parity checks failed; no gate artifact was written"
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["passed"] else 1
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
        return 2  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
