"""Strict, checksum-verified discovery for resumable training checkpoints."""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .checkpointing import read_diffusion_checkpoint_manifest


RESUME_STATE_SCHEMA_VERSION = 2
SUPPORTED_RESUME_STATE_SCHEMA_VERSIONS = (1, 2)
CHECKPOINT_DIRECTORY_PATTERN = re.compile(r"checkpoint-([0-9]{6})\Z")
_ALLOWED_RUN_FILES = {
    "resolved_config.yaml",
    "training_summary.json",
    "fork_provenance.json",
}
_REQUIRED_HASHES = {
    "resolved_config",
    "diffusion_state",
    "training_state",
    "trained_model",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: str | Path) -> str:
    """Hash a directory tree by relative path and file content.

    Symlinks and non-regular entries are rejected: a training checkpoint must
    remain self-contained and must not change meaning when moved or mounted in
    another container.
    """

    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"checkpoint model artifact is not a directory: {root}")
    digest = hashlib.sha256()
    files = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    for item in files:
        if item.is_symlink():
            raise ValueError(f"checkpoint artifact contains a symlink: {item}")
        if item.is_dir():
            continue
        if not item.is_file():
            raise ValueError(f"checkpoint artifact is not a regular file: {item}")
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def mapping_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def dataset_identity(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Remove mutable/rank-local selectors from shared dataset provenance."""

    identity = dict(summary)
    identity.pop("epoch", None)
    # Every rank authenticates the same distributed dataset; the rank-local
    # selector is saved separately in that rank's training-loop cursor.
    identity.pop("rank", None)
    return identity


def _direct_child(
    checkpoint_dir: Path,
    relative: Any,
    *,
    expect_directory: bool,
) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).name != relative:
        raise ValueError(f"checkpoint artifact must be one direct relative name: {relative!r}")
    path = checkpoint_dir / relative
    if path.is_symlink():
        raise ValueError(f"checkpoint artifact must not be a symlink: {path}")
    if expect_directory:
        if not path.is_dir():
            raise ValueError(f"checkpoint directory artifact is missing: {path}")
    elif not path.is_file():
        raise ValueError(f"checkpoint file artifact is missing: {path}")
    return path


@dataclass(frozen=True)
class ResumeCheckpoint:
    path: Path
    manifest: dict[str, Any]
    update: int
    resolved_config: Path
    diffusion_state: Path
    training_state: Path
    trained_model: Path


def inspect_resume_checkpoint(
    checkpoint_dir: str | Path,
    *,
    verify_trained_model_hash: bool = True,
) -> ResumeCheckpoint:
    """Validate one complete resume checkpoint without unpickling state.

    ``verify_trained_model_hash=False`` is reserved for non-primary ranks after
    rank zero has authenticated the complete distributed artifact and
    broadcast its selected path. Small control files are still hashed on every
    rank; this avoids every worker rereading the full sharded model+optimizer
    directory from shared storage.
    """

    checkpoint = Path(checkpoint_dir).expanduser().resolve(strict=True)
    match = CHECKPOINT_DIRECTORY_PATTERN.fullmatch(checkpoint.name)
    if match is None or checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ValueError(f"invalid checkpoint directory: {checkpoint}")
    manifest_path = checkpoint / "diffusion_checkpoint.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"checkpoint manifest is missing: {manifest_path}")
    manifest = read_diffusion_checkpoint_manifest(manifest_path)
    update = manifest.get("update")
    if not isinstance(update, int) or isinstance(update, bool) or update < 0:
        raise ValueError(f"checkpoint update is invalid: {update!r}")
    if update != int(match.group(1)):
        raise ValueError(
            f"checkpoint directory/update mismatch: {checkpoint.name} vs {update}"
        )
    if manifest.get("resume_schema_version") not in SUPPORTED_RESUME_STATE_SCHEMA_VERSIONS:
        raise ValueError("checkpoint has no supported resumable-training schema")

    resolved_config = _direct_child(
        checkpoint, manifest.get("resolved_config"), expect_directory=False
    )
    diffusion_state = _direct_child(
        checkpoint, manifest.get("diffusion_state"), expect_directory=False
    )
    training_state = _direct_child(
        checkpoint, manifest.get("training_state"), expect_directory=False
    )
    strategy_field = (
        "adapter_dir" if manifest.get("strategy") == "lora" else "full_model_dir"
    )
    trained_model = _direct_child(
        checkpoint, manifest.get(strategy_field), expect_directory=True
    )

    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != _REQUIRED_HASHES:
        raise ValueError(
            "checkpoint artifact hashes are incomplete: "
            f"expected={sorted(_REQUIRED_HASHES)}, "
            f"actual={sorted(hashes) if isinstance(hashes, Mapping) else hashes!r}"
        )
    actual_hashes = {
        "resolved_config": sha256_file(resolved_config),
        "diffusion_state": sha256_file(diffusion_state),
        "training_state": sha256_file(training_state),
    }
    if verify_trained_model_hash:
        actual_hashes["trained_model"] = sha256_directory(trained_model)
    mismatches = {
        name: {"saved": hashes.get(name), "actual": actual}
        for name, actual in actual_hashes.items()
        if hashes.get(name) != actual
    }
    if mismatches:
        raise ValueError(f"checkpoint artifact checksum mismatch: {mismatches}")

    for identity_name in (
        "training_config_sha256",
        "dataset_identity_sha256",
        "token_policy_sha256",
        "trainable_parameters_sha256",
        "base_snapshot_sha256",
        "source_code_sha256",
    ):
        value = manifest.get(identity_name)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"checkpoint identity hash is invalid: {identity_name}")
    return ResumeCheckpoint(
        path=checkpoint,
        manifest=manifest,
        update=update,
        resolved_config=resolved_config,
        diffusion_state=diffusion_state,
        training_state=training_state,
        trained_model=trained_model,
    )


def _validate_run_directory_entries(output_dir: Path) -> tuple[Path, ...]:
    checkpoints: list[Path] = []
    unexpected: list[str] = []
    for entry in output_dir.iterdir():
        match = CHECKPOINT_DIRECTORY_PATTERN.fullmatch(entry.name)
        if match is not None:
            if entry.is_symlink() or not entry.is_dir():
                unexpected.append(entry.name)
            else:
                checkpoints.append(entry)
        elif entry.name not in _ALLOWED_RUN_FILES or not entry.is_file() or entry.is_symlink():
            unexpected.append(entry.name)
    if unexpected:
        raise ValueError(
            "resume output contains unexpected or unsafe entries: "
            f"{sorted(unexpected)}"
        )
    return tuple(checkpoints)


def resolve_resume_checkpoint(
    request: str | Path | None,
    *,
    output_dir: str | Path,
) -> ResumeCheckpoint | None:
    """Resolve explicit/latest resume while preserving fresh-run behavior.

    ``latest`` is intentionally special: a missing or empty output starts a new
    run, making one command safe for its first Slurm allocation and later
    requeues/resubmissions. A non-empty output without any complete checkpoint
    is an error and never silently restarts from R2.
    """

    output = Path(output_dir).expanduser().resolve()
    if request is None or str(request).strip() in ("", "-"):
        if output.exists() and any(output.iterdir()):
            raise ValueError(
                "training output is non-empty; pass --resume-from-checkpoint "
                "explicitly or choose a new output directory"
            )
        return None

    value = str(request).strip()
    if value == "latest":
        if not output.exists() or not any(output.iterdir()):
            return None
        checkpoint_dirs = _validate_run_directory_entries(output)
        ignored: list[tuple[str, str]] = []
        for checkpoint_dir in sorted(checkpoint_dirs, reverse=True):
            try:
                checkpoint = inspect_resume_checkpoint(checkpoint_dir)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                # A manifest is the final atomic marker. Interrupted higher
                # checkpoints are deliberately ignored in favor of the last
                # complete one.
                ignored.append((checkpoint_dir.name, str(error)))
                continue
            for name, reason in ignored:
                warnings.warn(
                    f"ignoring incomplete or invalid resume candidate {name}: {reason}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return checkpoint
        if not checkpoint_dirs:
            raise ValueError(
                "resume requested from latest, but the non-empty output contains "
                "no complete checksum-verified checkpoint; "
                f"ignored={ignored}"
            )
        raise ValueError(
            "resume requested from latest, but the non-empty output contains "
            "no complete checksum-verified checkpoint; "
            f"ignored={ignored}"
        )

    if not output.exists():
        raise ValueError("explicit resume requires the existing checkpoint output directory")
    _validate_run_directory_entries(output)
    checkpoint = inspect_resume_checkpoint(value)
    if checkpoint.path.parent != output:
        raise ValueError(
            "resume checkpoint must be a direct child of --output-dir: "
            f"{checkpoint.path.parent} != {output}"
        )
    return checkpoint


def validate_resume_identities(
    checkpoint: ResumeCheckpoint,
    *,
    training_config_sha256: str,
    dataset_identity_sha256: str,
    token_policy_sha256: str,
    trainable_parameters_sha256: str,
    base_snapshot_sha256: str,
    source_code_sha256: str,
    strategy: str,
    base_checkpoint: str,
    revision: str,
) -> None:
    expected = {
        "training_config_sha256": training_config_sha256,
        "dataset_identity_sha256": dataset_identity_sha256,
        "token_policy_sha256": token_policy_sha256,
        "trainable_parameters_sha256": trainable_parameters_sha256,
        "base_snapshot_sha256": base_snapshot_sha256,
        "source_code_sha256": source_code_sha256,
        "strategy": strategy,
        "base_checkpoint": base_checkpoint,
        "revision": revision,
    }
    mismatches = {
        name: {"checkpoint": checkpoint.manifest.get(name), "current": value}
        for name, value in expected.items()
        if checkpoint.manifest.get(name) != value
    }
    if mismatches:
        raise ValueError(f"resume checkpoint identity mismatch: {mismatches}")


__all__ = [
    "CHECKPOINT_DIRECTORY_PATTERN",
    "RESUME_STATE_SCHEMA_VERSION",
    "SUPPORTED_RESUME_STATE_SCHEMA_VERSIONS",
    "ResumeCheckpoint",
    "dataset_identity",
    "inspect_resume_checkpoint",
    "mapping_sha256",
    "resolve_resume_checkpoint",
    "sha256_directory",
    "sha256_file",
    "validate_resume_identities",
]
