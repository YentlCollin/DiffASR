import json
from pathlib import Path

import pytest

from audio.diffusion.checkpointing import write_diffusion_checkpoint_manifest
from audio.diffusion.training_resume import (
    dataset_identity,
    inspect_resume_checkpoint,
    mapping_sha256,
    resolve_resume_checkpoint,
    sha256_directory,
    sha256_file,
    validate_resume_identities,
)


IDENTITIES = {
    "training_config_sha256": "1" * 64,
    "dataset_identity_sha256": "2" * 64,
    "token_policy_sha256": "3" * 64,
    "trainable_parameters_sha256": "4" * 64,
    "base_snapshot_sha256": "5" * 64,
    "source_code_sha256": "6" * 64,
}


def _complete_checkpoint(output: Path, update: int) -> Path:
    checkpoint = output / f"checkpoint-{update:06d}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "resolved_config.yaml").write_text(
        "experiment_name: resume-test\n", encoding="utf-8"
    )
    (checkpoint / "diffusion_state.pt").write_bytes(b"diffusion")
    (checkpoint / "training_state.pt").write_bytes(b"training")
    adapter = checkpoint / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(f"adapter-{update}".encode())
    manifest = {
        "strategy": "lora",
        "base_checkpoint": "org/model",
        "revision": "revision",
        "update": update,
        "diffusion_state": "diffusion_state.pt",
        "resolved_config": "resolved_config.yaml",
        "adapter_dir": "adapter",
        "resume_schema_version": 1,
        "training_state": "training_state.pt",
        **IDENTITIES,
        "artifact_sha256": {
            "resolved_config": sha256_file(checkpoint / "resolved_config.yaml"),
            "diffusion_state": sha256_file(checkpoint / "diffusion_state.pt"),
            "training_state": sha256_file(checkpoint / "training_state.pt"),
            "trained_model": sha256_directory(adapter),
        },
    }
    write_diffusion_checkpoint_manifest(checkpoint, manifest)
    return checkpoint


def test_latest_starts_only_from_absent_or_empty_output(tmp_path):
    output = tmp_path / "run"
    assert resolve_resume_checkpoint("latest", output_dir=output) is None
    output.mkdir()
    assert resolve_resume_checkpoint("latest", output_dir=output) is None

    (output / "resolved_config.yaml").write_text("partial\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no complete checksum-verified"):
        resolve_resume_checkpoint("latest", output_dir=output)

    with pytest.raises(ValueError, match="non-empty"):
        resolve_resume_checkpoint(None, output_dir=output)


def test_latest_ignores_incomplete_higher_checkpoint(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "resolved_config.yaml").write_text("root\n", encoding="utf-8")
    first = _complete_checkpoint(output, 1000)
    incomplete = output / "checkpoint-002000"
    incomplete.mkdir()
    (incomplete / "training_state.pt.tmp").write_bytes(b"interrupted")

    with pytest.warns(RuntimeWarning, match="checkpoint-002000"):
        selected = resolve_resume_checkpoint("latest", output_dir=output)
    assert selected is not None
    assert selected.path == first.resolve()
    assert selected.update == 1000


def test_latest_rejects_unexpected_run_entries_even_with_checkpoint(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    _complete_checkpoint(output, 1)
    (output / "notes.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected or unsafe"):
        resolve_resume_checkpoint("latest", output_dir=output)


def test_checkpoint_hash_and_identity_validation_precede_state_load(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    checkpoint = _complete_checkpoint(output, 12)
    layout = inspect_resume_checkpoint(checkpoint)
    validate_resume_identities(
        layout,
        **IDENTITIES,
        strategy="lora",
        base_checkpoint="org/model",
        revision="revision",
    )

    with pytest.raises(ValueError, match="identity mismatch"):
        validate_resume_identities(
            layout,
            **{**IDENTITIES, "token_policy_sha256": "f" * 64},
            strategy="lora",
            base_checkpoint="org/model",
            revision="revision",
        )

    (checkpoint / "training_state.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        inspect_resume_checkpoint(checkpoint)


def test_non_primary_rank_mode_skips_only_the_large_model_hash(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    checkpoint = _complete_checkpoint(output, 13)
    (checkpoint / "adapter" / "adapter_model.safetensors").write_bytes(b"tampered")

    # Rank zero's default audit rejects the modified model artifact.
    with pytest.raises(ValueError, match="checksum mismatch"):
        inspect_resume_checkpoint(checkpoint)
    # A non-primary rank may skip that one large hash only after rank zero has
    # already authenticated and broadcast the selected directory.
    layout = inspect_resume_checkpoint(
        checkpoint,
        verify_trained_model_hash=False,
    )
    assert layout.update == 13

    (checkpoint / "training_state.pt").write_bytes(b"tampered-control")
    with pytest.raises(ValueError, match="checksum mismatch"):
        inspect_resume_checkpoint(
            checkpoint,
            verify_trained_model_hash=False,
        )


def test_explicit_checkpoint_must_belong_to_output(tmp_path):
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    first_output.mkdir()
    second_output.mkdir()
    checkpoint = _complete_checkpoint(first_output, 7)
    with pytest.raises(ValueError, match="direct child"):
        resolve_resume_checkpoint(checkpoint, output_dir=second_output)


def test_dataset_and_mapping_identities_are_stable():
    first = {"revision": "r", "seed": 4, "epoch": 0, "paths": ["a", "b"]}
    second = {"paths": ["a", "b"], "epoch": 9, "seed": 4, "revision": "r"}
    assert dataset_identity(first) == dataset_identity(second)
    assert mapping_sha256(dataset_identity(first)) == mapping_sha256(
        dataset_identity(second)
    )
