#!/usr/bin/env python3
"""Merge a sharded DiffASR FSDP checkpoint for single-process inference.

The training checkpoint remains immutable.  The merged safetensors file is
written to an explicit, normally node-local, output directory.  A sidecar is
published only after the weights have been validated so inference never treats
a partial export as complete.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable, Iterator, Mapping, Sequence


CHECKPOINT_MANIFEST_NAME = "diffusion_checkpoint.json"
EXPORT_MANIFEST_NAME = "fsdp_export_manifest.json"
WEIGHTS_NAME = "model.safetensors"
EXPECTED_CHECKPOINT_FORMAT = "accelerate_fsdp_sharded_state_dict"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

MergeFunction = Callable[..., Any]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Required JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _sha256_file(path: Path, *, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _checkpoint_source(checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve(strict=True)
    if not checkpoint.is_dir():
        raise ValueError(f"Checkpoint is not a directory: {checkpoint}")

    manifest_path = checkpoint / CHECKPOINT_MANIFEST_NAME
    manifest = _read_json(manifest_path)
    if manifest.get("strategy") != "full":
        raise ValueError(
            "FSDP inference export requires checkpoint strategy='full'; "
            f"got {manifest.get('strategy')!r}"
        )
    if manifest.get("checkpoint_format") != EXPECTED_CHECKPOINT_FORMAT:
        raise ValueError(
            "FSDP inference export requires checkpoint_format="
            f"{EXPECTED_CHECKPOINT_FORMAT!r}; got "
            f"{manifest.get('checkpoint_format')!r}"
        )

    update = manifest.get("update")
    if isinstance(update, bool) or not isinstance(update, int) or update < 0:
        raise ValueError(f"Checkpoint manifest has an invalid update: {update!r}")

    artifacts = manifest.get("artifact_sha256")
    trained_model_sha256 = (
        artifacts.get("trained_model") if isinstance(artifacts, Mapping) else None
    )
    if not isinstance(trained_model_sha256, str) or not SHA256_RE.fullmatch(
        trained_model_sha256
    ):
        raise ValueError(
            "Checkpoint manifest is missing a valid "
            "artifact_sha256.trained_model SHA-256"
        )

    full_model_dir = manifest.get("full_model_dir")
    if not isinstance(full_model_dir, str) or not full_model_dir:
        raise ValueError("Checkpoint manifest is missing full_model_dir")
    full_model_path = (checkpoint / full_model_dir).resolve(strict=True)
    if not _is_relative_to(full_model_path, checkpoint):
        raise ValueError("full_model_dir must resolve inside the checkpoint")
    source_dcp_dir = (full_model_path / "pytorch_model_fsdp_0").resolve(strict=True)
    if not _is_relative_to(source_dcp_dir, checkpoint):
        raise ValueError("FSDP model state directory must be inside the checkpoint")
    if not source_dcp_dir.is_dir():
        raise ValueError(f"FSDP model state is not a directory: {source_dcp_dir}")
    if not (source_dcp_dir / ".metadata").is_file():
        raise ValueError(f"FSDP model state is missing .metadata: {source_dcp_dir}")
    if not any(source_dcp_dir.glob("*.distcp")):
        raise ValueError(f"FSDP model state contains no .distcp shards: {source_dcp_dir}")

    return {
        "checkpoint": str(checkpoint),
        "checkpoint_update": update,
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": _sha256_file(manifest_path),
        "source_trained_model_sha256": trained_model_sha256,
        "source_dcp_dir": str(source_dcp_dir),
    }


def _normalized_wrapper_keys(keys: Sequence[str]) -> list[str]:
    if keys and all(key.startswith("model.") for key in keys):
        return [key.removeprefix("model.") for key in keys]
    return list(keys)


def _read_and_validate_weights_keys(weights_path: Path) -> list[str]:
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - exercised in the runtime image
        raise RuntimeError("safetensors is required to validate the merged export") from exc

    if not weights_path.is_file():
        raise ValueError(f"Accelerate did not create {weights_path}")
    try:
        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            keys = sorted(handle.keys())
    except Exception as exc:
        raise ValueError(f"Invalid merged safetensors file {weights_path}: {exc}") from exc
    if not keys:
        raise ValueError(f"Merged safetensors file has no tensors: {weights_path}")

    normalized = _normalized_wrapper_keys(keys)
    required_exact = {
        "diffusion_mask_embedding",
        "autoregressive_model.lm_head.weight",
        "autoregressive_model.model.language_model.embed_tokens.weight",
    }
    missing = sorted(required_exact.difference(normalized))
    required_prefixes = (
        "autoregressive_model.model.language_model.",
        "autoregressive_model.model.audio_tower.",
        "autoregressive_model.model.embed_audio.",
    )
    missing_prefixes = [
        prefix for prefix in required_prefixes if not any(key.startswith(prefix) for key in normalized)
    ]
    if missing or missing_prefixes:
        details = []
        if missing:
            details.append(f"missing exact keys: {missing}")
        if missing_prefixes:
            details.append(f"missing key prefixes: {missing_prefixes}")
        raise ValueError(
            "Merged weights do not contain the expected DiffASR wrapper state ("
            + "; ".join(details)
            + ")"
        )
    return keys


def _same_source(export_manifest: Mapping[str, Any], source: Mapping[str, Any]) -> bool:
    fields = (
        "checkpoint",
        "checkpoint_update",
        "source_trained_model_sha256",
        "source_dcp_dir",
    )
    return all(export_manifest.get(field) == source.get(field) for field in fields)


def _validate_existing_export(
    output_dir: Path, source: Mapping[str, Any]
) -> dict[str, Any] | None:
    manifest_path = output_dir / EXPORT_MANIFEST_NAME
    weights_path = output_dir / WEIGHTS_NAME
    if not manifest_path.exists():
        return None

    manifest = _read_json(manifest_path)
    if not _same_source(manifest, source):
        raise ValueError(
            f"Output directory already contains an export for another checkpoint: {output_dir}"
        )
    if manifest.get("weights_file") != WEIGHTS_NAME:
        return None
    expected_sha256 = manifest.get("weights_sha256")
    expected_key_count = manifest.get("key_count")
    if not isinstance(expected_sha256, str) or not SHA256_RE.fullmatch(expected_sha256):
        return None
    if isinstance(expected_key_count, bool) or not isinstance(expected_key_count, int):
        return None
    if not weights_path.is_file() or _sha256_file(weights_path) != expected_sha256:
        return None
    keys = _read_and_validate_weights_keys(weights_path)
    if len(keys) != expected_key_count:
        return None
    return manifest


def _write_json_fsync(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with path.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _exclusive_export_lock(output_dir: Path) -> Iterator[None]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / f".{output_dir.name}.fsdp-export.lock"
    with lock_path.open("a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


def export_fsdp_diffusion_checkpoint(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    merge_function: MergeFunction | None = None,
) -> dict[str, Any]:
    """Create or reuse a verified node-local inference export."""

    source = _checkpoint_source(Path(checkpoint))
    destination = Path(output_dir).expanduser().resolve(strict=False)
    source_dcp_dir = Path(source["source_dcp_dir"])
    if destination == source_dcp_dir or _is_relative_to(destination, source_dcp_dir):
        raise ValueError("The export output directory cannot be inside the source DCP state")

    with _exclusive_export_lock(destination):
        destination.mkdir(parents=True, exist_ok=True)
        existing = _validate_existing_export(destination, source)
        if existing is not None:
            return existing

        if merge_function is None:
            try:
                from accelerate.utils import merge_fsdp_weights
            except ImportError as exc:  # pragma: no cover - runtime dependency
                raise RuntimeError("accelerate is required to merge FSDP weights") from exc
            merge_function = merge_fsdp_weights

        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
        )
        try:
            merge_function(
                str(source_dcp_dir),
                str(temporary),
                safe_serialization=True,
                remove_checkpoint_dir=False,
            )
            temporary_weights = temporary / WEIGHTS_NAME
            keys = _read_and_validate_weights_keys(temporary_weights)
            weights_sha256 = _sha256_file(temporary_weights)
            export_manifest = {
                "schema_version": 1,
                **source,
                "weights_file": WEIGHTS_NAME,
                "weights_sha256": weights_sha256,
                "key_count": len(keys),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            temporary_manifest = temporary / EXPORT_MANIFEST_NAME
            _write_json_fsync(temporary_manifest, export_manifest)

            # The sidecar is the completion marker and is deliberately replaced
            # last.  A reader can therefore reject every interrupted publication.
            os.replace(temporary_weights, destination / WEIGHTS_NAME)
            os.replace(temporary_manifest, destination / EXPORT_MANIFEST_NAME)
            _fsync_directory(destination)
            return export_manifest
        finally:
            shutil.rmtree(temporary, ignore_errors=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge an Accelerate FSDP DiffASR checkpoint for inference."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = export_fsdp_diffusion_checkpoint(args.checkpoint, args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
