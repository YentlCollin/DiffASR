import json
from pathlib import Path

import pytest
import torch

from audio.diffusion.checkpointing import write_diffusion_checkpoint_manifest
from audio.diffusion.training_fork import (
    apply_learning_rate_fork,
    load_fork_training_state,
    read_fork_provenance,
    resolve_explicit_training_fork,
    training_source_identity,
    validate_fork_configuration,
    validate_fork_immutable_identities,
    validate_replay_snapshot,
    validate_weight_continuation_configuration,
    weight_continuation_dataset_identity,
    write_fork_provenance,
)
from audio.diffusion.training_resume import sha256_directory, sha256_file


IDENTITIES = {
    "training_config_sha256": "1" * 64,
    "dataset_identity_sha256": "2" * 64,
    "token_policy_sha256": "3" * 64,
    "trainable_parameters_sha256": "4" * 64,
    "base_snapshot_sha256": "5" * 64,
    "source_code_sha256": "6" * 64,
}


def _checkpoint(root: Path, update: int = 2530) -> Path:
    checkpoint = root / f"checkpoint-{update:06d}"
    checkpoint.mkdir(parents=True)
    resolved = checkpoint / "resolved_config.yaml"
    resolved.write_text(
        "experiment_name: uniform_no_t_shifted\n"
        "training:\n"
        "  learning_rate: 0.0001\n"
        "runtime:\n"
        "  elapsed: 1.0\n",
        encoding="utf-8",
    )
    diffusion = checkpoint / "diffusion_state.pt"
    diffusion.write_bytes(b"diffusion")
    state = checkpoint / "training_state.pt"
    torch.save(
        {
            "update": update,
            "identities": dict(IDENTITIES),
            "optimizer": {"moments": "preserved"},
        },
        state,
    )
    adapter = checkpoint / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    manifest = {
        "strategy": "lora",
        "base_checkpoint": "org/model",
        "revision": "user-provided-revision",
        "update": update,
        "diffusion_state": diffusion.name,
        "resolved_config": resolved.name,
        "adapter_dir": adapter.name,
        "resume_schema_version": 1,
        "training_state": state.name,
        **IDENTITIES,
        "artifact_sha256": {
            "resolved_config": sha256_file(resolved),
            "diffusion_state": sha256_file(diffusion),
            "training_state": sha256_file(state),
            "trained_model": sha256_directory(adapter),
        },
    }
    write_diffusion_checkpoint_manifest(checkpoint, manifest)
    return checkpoint


def _parent_config():
    return {
        "experiment_name": "uniform_no_t_shifted",
        "training": {"learning_rate": 1.0e-4},
        "runtime": {"elapsed": 99},
    }


def test_fork_config_delta_is_exact_and_safety_allowlisted():
    child = _parent_config()
    child["training"]["max_grad_norm"] = 1.0
    child["runtime"] = {"different_observation": True}
    changes = validate_fork_configuration(
        _parent_config(), child, allowed_changes=["training.max_grad_norm"]
    )
    assert [change.path for change in changes] == ["training.max_grad_norm"]
    assert changes[0].parent_present is False
    assert changes[0].child_value == 1.0

    child["data"] = {"seed": 9}
    with pytest.raises(ValueError, match="undeclared=.*data"):
        validate_fork_configuration(
            _parent_config(), child, allowed_changes=["training.max_grad_norm"]
        )
    with pytest.raises(ValueError, match="outside the safety allowlist"):
        validate_fork_configuration(
            _parent_config(), _parent_config(), allowed_changes=["data.seed"]
        )
    with pytest.raises(ValueError, match="declared_but_unchanged"):
        validate_fork_configuration(
            _parent_config(), _parent_config(), allowed_changes=["training.learning_rate"]
        )


def test_weight_continuation_delta_allows_only_the_audited_topology_and_schedule():
    parent = {
        "experiment_name": "mask_no_t_same_sdpa_no_gc",
        "training": {
            "max_updates": 50000,
            "gradient_accumulation_steps": 2,
            "global_batch_size": 16,
            "supported_world_sizes": [1, 2, 4, 8],
            "learning_rate": 1.0e-4,
        },
        "runtime": {"world_size": 8},
    }
    child = {
        "experiment_name": "mask_no_t_same_sdpa_no_gc_continue_50k_to_100k_16gpu",
        "training": {
            "max_updates": 100000,
            "gradient_accumulation_steps": 1,
            "global_batch_size": 16,
            "supported_world_sizes": [1, 2, 4, 8, 16],
            "learning_rate": 1.0e-5,
            "scheduler_type": "continuation_cosine",
            "scheduler_warmup_updates": 1000,
            "scheduler_min_lr_ratio": 0.1,
            "continuation_start_update": 50000,
            "continuation_start_epoch": 2,
        },
    }
    changes = validate_weight_continuation_configuration(parent, child)
    assert {change.path for change in changes} == {
        "experiment_name",
        "training.max_updates",
        "training.gradient_accumulation_steps",
        "training.supported_world_sizes",
        "training.learning_rate",
        "training.scheduler_type",
        "training.scheduler_warmup_updates",
        "training.scheduler_min_lr_ratio",
        "training.continuation_start_update",
        "training.continuation_start_epoch",
    }

    child["data"] = {"seed": 999}
    with pytest.raises(ValueError, match="unsafe=.*data"):
        validate_weight_continuation_configuration(parent, child)


def test_weight_continuation_delta_can_preserve_ddp16_and_scheduler_at_200k():
    parent = {
        "experiment_name": "mask_no_t_same_sdpa_no_gc_continue_50k_to_100k_16gpu",
        "training": {
            "max_updates": 100000,
            "gradient_accumulation_steps": 1,
            "global_batch_size": 16,
            "supported_world_sizes": [1, 2, 4, 8, 16],
            "learning_rate": 1.0e-5,
            "scheduler_type": "continuation_cosine",
            "scheduler_warmup_updates": 1000,
            "scheduler_min_lr_ratio": 0.1,
            "continuation_start_update": 50000,
            "continuation_start_epoch": 2,
        },
        "runtime": {"world_size": 16},
    }
    child = {
        "experiment_name": "mask_no_t_same_sdpa_no_gc_continue_100k_to_200k_16gpu",
        "training": {
            **parent["training"],
            "max_updates": 200000,
            "continuation_start_update": 100000,
            "continuation_start_epoch": 4,
        },
    }
    changes = validate_weight_continuation_configuration(parent, child)
    assert {change.path for change in changes} == {
        "experiment_name",
        "training.max_updates",
        "training.continuation_start_update",
        "training.continuation_start_epoch",
    }

    child["training"]["global_batch_size"] = 32
    with pytest.raises(ValueError, match="unsafe=.*global_batch_size"):
        validate_weight_continuation_configuration(parent, child)

    child["training"]["global_batch_size"] = 16
    child["training"]["learning_rate"] = 2.0e-5
    with pytest.raises(ValueError, match="initial LR"):
        validate_weight_continuation_configuration(parent, child)


def test_same_topology_repeated_continuation_accepts_bounded_lr_warm_restart():
    parent = {
        "experiment_name": "lora_u058838_to_u117676",
        "training": {
            "expected_epoch_updates": [29419] * 4,
            "max_updates": 117676,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "global_batch_size": 128,
            "supported_world_sizes": [32, 64, 128],
            "learning_rate": 1.0e-5,
            "scheduler_type": "continuation_cosine",
            "scheduler_warmup_updates": 625,
            "scheduler_min_lr_ratio": 0.3,
            "save_at_updates": [88257, 117676],
            "continuation_start_update": 58838,
            "continuation_start_epoch": 2,
        },
    }
    child = {
        "experiment_name": "lora_u117676_to_u205933_warmrestart",
        "training": {
            **parent["training"],
            "expected_epoch_updates": [29419] * 7,
            "max_updates": 205933,
            "learning_rate": 3.0e-5,
            "scheduler_warmup_updates": 1000,
            "scheduler_min_lr_ratio": 0.1,
            "save_at_updates": [147095, 176514, 205933],
            "continuation_start_update": 117676,
            "continuation_start_epoch": 4,
        },
    }

    changes = validate_weight_continuation_configuration(parent, child)
    assert {change.path for change in changes} == {
        "experiment_name",
        "training.expected_epoch_updates",
        "training.learning_rate",
        "training.max_updates",
        "training.save_at_updates",
        "training.scheduler_min_lr_ratio",
        "training.scheduler_warmup_updates",
        "training.continuation_start_update",
        "training.continuation_start_epoch",
    }


def test_same_topology_continuation_accepts_optimizer_preserving_policy():
    parent = {
        "experiment_name": "parent",
        "training": {
            "expected_epoch_updates": [100, 100],
            "max_updates": 200,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "global_batch_size": 128,
            "supported_world_sizes": [32, 64, 128],
            "learning_rate": 1.0e-5,
            "scheduler_type": "continuation_cosine",
            "scheduler_warmup_updates": 625,
            "scheduler_min_lr_ratio": 0.3,
            "save_at_updates": [200],
            "continuation_start_update": 100,
            "continuation_start_epoch": 1,
        },
    }
    child = {
        "experiment_name": "child",
        "training": {
            **parent["training"],
            "expected_epoch_updates": [100, 100, 100],
            "max_updates": 300,
            "learning_rate": 5.0e-6,
            "scheduler_warmup_updates": 2000,
            "scheduler_min_lr_ratio": 0.6,
            "save_at_updates": [300],
            "continuation_start_update": 200,
            "continuation_start_epoch": 2,
            "continuation_state_policy": (
                "preserve_optimizer_scaler_rng_reset_scheduler"
            ),
        },
    }

    changes = validate_weight_continuation_configuration(parent, child)
    assert "training.continuation_state_policy" in {
        change.path for change in changes
    }


def test_same_topology_warm_restart_rejects_peak_above_three_times_parent():
    parent = {
        "experiment_name": "parent",
        "training": {
            "expected_epoch_updates": [100, 100],
            "max_updates": 200,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "global_batch_size": 128,
            "supported_world_sizes": [32, 64, 128],
            "learning_rate": 1.0e-5,
            "scheduler_type": "continuation_cosine",
            "scheduler_warmup_updates": 1000,
            "scheduler_min_lr_ratio": 0.3,
            "save_at_updates": [200],
            "continuation_start_update": 100,
            "continuation_start_epoch": 1,
        },
    }
    child = {
        "experiment_name": "child",
        "training": {
            **parent["training"],
            "expected_epoch_updates": [100, 100, 100],
            "max_updates": 300,
            "learning_rate": 4.0e-5,
            "scheduler_warmup_updates": 2000,
            "scheduler_min_lr_ratio": 0.075,
            "save_at_updates": [300],
            "continuation_start_update": 200,
            "continuation_start_epoch": 2,
        },
    }

    with pytest.raises(ValueError, match="3x warm-restart bound"):
        validate_weight_continuation_configuration(parent, child)


def test_repeated_weight_continuation_allows_declared_full_noise_intervention():
    parent = {
        "experiment_name": "uniform_no_t_same_continue_100k_to_200k_16gpu",
        "diffusion": {"full_noise_probability": 0.0},
        "training": {
            "max_updates": 200000,
            "gradient_accumulation_steps": 1,
            "global_batch_size": 16,
            "supported_world_sizes": [1, 2, 4, 8, 16],
            "learning_rate": 1.0e-5,
            "scheduler_type": "continuation_cosine",
            "scheduler_warmup_updates": 1000,
            "scheduler_min_lr_ratio": 0.1,
            "continuation_start_update": 100000,
            "continuation_start_epoch": 4,
        },
    }
    child = {
        **parent,
        "experiment_name": "uniform_no_t_same_full_noise_p010_200k_to_225k",
        "diffusion": {"full_noise_probability": 0.10},
        "training": {
            **parent["training"],
            "max_updates": 225000,
            "continuation_start_update": 200000,
            "continuation_start_epoch": 8,
        },
    }

    changes = validate_weight_continuation_configuration(parent, child)
    assert {change.path for change in changes} == {
        "experiment_name",
        "diffusion.full_noise_probability",
        "training.max_updates",
        "training.continuation_start_update",
        "training.continuation_start_epoch",
    }

    child["diffusion"]["eos_tail_loss_weight"] = 0.25
    with pytest.raises(ValueError, match="unsafe=.*eos_tail_loss_weight"):
        validate_weight_continuation_configuration(parent, child)

    child["diffusion"].pop("eos_tail_loss_weight")
    child["diffusion"]["full_noise_probability"] = 0.20
    with pytest.raises(ValueError, match="requires exactly"):
        validate_weight_continuation_configuration(parent, child)


def test_repeated_weight_continuation_requires_parent_boundary_and_new_epoch():
    parent = {
        "experiment_name": "uniform_with_t_same_continue_50k_to_100k_16gpu",
        "training": {
            "max_updates": 100000,
            "gradient_accumulation_steps": 1,
            "global_batch_size": 16,
            "supported_world_sizes": [1, 2, 4, 8, 16],
            "learning_rate": 1.0e-5,
            "scheduler_type": "continuation_cosine",
            "scheduler_warmup_updates": 1000,
            "scheduler_min_lr_ratio": 0.1,
            "continuation_start_update": 50000,
            "continuation_start_epoch": 2,
        },
    }
    child = {
        "experiment_name": "uniform_with_t_same_continue_100k_to_200k_16gpu",
        "training": {
            **parent["training"],
            "max_updates": 200000,
            "continuation_start_update": 99999,
            "continuation_start_epoch": 4,
        },
    }
    with pytest.raises(ValueError, match="start at the parent max_updates"):
        validate_weight_continuation_configuration(parent, child)

    child["training"]["continuation_start_update"] = 100000
    child["training"]["continuation_start_epoch"] = 1
    with pytest.raises(ValueError, match="advance the dataloader epoch"):
        validate_weight_continuation_configuration(parent, child)


def test_weight_continuation_dataset_identity_is_topology_and_relocation_neutral():
    parent = {
        "repo_id": "org/data",
        "revision": "abc123",
        "subset_manifest_sha256": "1" * 64,
        "sample_exclusion_manifest_sha256": "2" * 64,
        "selected_bytes": 198999476704,
        "eval_manifest": "/data-a/eval.jsonl",
        "local_root": "/data-a/audio",
        "subset_manifest": "/data-a/subset.json",
        "world_size": 8,
        "rank": 0,
        "epoch": 0,
        "samples_per_rank_after_trim": 52086,
        "samples_per_rank_before_trim": [52115] * 8,
        "sample_exclusion_manifest": "/snapshot-a/exclusions.json",
    }
    child = {
        **parent,
        "world_size": 16,
        "rank": 9,
        "epoch": 2,
        "samples_per_rank_after_trim": 26040,
        "samples_per_rank_before_trim": [26050] * 16,
        "sample_exclusion_manifest": "/snapshot-b/exclusions.json",
        "eval_manifest": "/data-b/eval.jsonl",
        "local_root": "/data-b/audio",
        "subset_manifest": "/data-b/subset.json",
    }
    assert weight_continuation_dataset_identity(parent) == (
        weight_continuation_dataset_identity(child)
    )
    child["subset_manifest_sha256"] = "f" * 64
    assert weight_continuation_dataset_identity(parent) != (
        weight_continuation_dataset_identity(child)
    )


def test_explicit_external_fork_checks_parent_output_delta_and_provenance(tmp_path):
    parent = _checkpoint(tmp_path / "parent")
    child_config = _parent_config()
    child_config["training"]["max_grad_norm"] = 1.0
    child_output = tmp_path / "child"
    fork = resolve_explicit_training_fork(
        parent,
        output_dir=child_output,
        child_config=child_config,
        allowed_changes=["training.max_grad_norm"],
        expected_parent_update=2530,
    )
    assert fork.checkpoint.path == parent.resolve()
    assert fork.provenance["parent_update"] == 2530
    assert len(fork.provenance["parent_manifest_sha256"]) == 64
    assert fork.provenance["config_changes"][0]["path"] == "training.max_grad_norm"

    validate_fork_immutable_identities(
        fork,
        dataset_identity_sha256=IDENTITIES["dataset_identity_sha256"],
        token_policy_sha256=IDENTITIES["token_policy_sha256"],
        trainable_parameters_sha256=IDENTITIES["trainable_parameters_sha256"],
        base_snapshot_sha256=IDENTITIES["base_snapshot_sha256"],
        strategy="lora",
        base_checkpoint="org/model",
        revision="user-provided-revision",
    )
    payload = load_fork_training_state(fork)
    assert payload["optimizer"] == {"moments": "preserved"}

    provenance = write_fork_provenance(
        child_output,
        fork,
        child_training_config_sha256="a" * 64,
        child_source_code_sha256="b" * 64,
    )
    written = json.loads(provenance.read_text(encoding="utf-8"))
    assert written["parent_update"] == 2530
    assert len(written["provenance_sha256"]) == 64
    assert read_fork_provenance(provenance) == written

    with pytest.raises(ValueError, match="absent or empty"):
        resolve_explicit_training_fork(
            parent,
            output_dir=child_output,
            child_config=child_config,
            allowed_changes=["training.max_grad_norm"],
        )
    with pytest.raises(ValueError, match="explicit checkpoint"):
        resolve_explicit_training_fork(
            "latest",
            output_dir=tmp_path / "other",
            child_config=child_config,
            allowed_changes=["training.max_grad_norm"],
        )


def test_fork_rejects_tampered_artifact_and_immutable_identity_change(tmp_path):
    parent = _checkpoint(tmp_path / "parent")
    child = _parent_config()
    child["training"]["max_grad_norm"] = 1.0
    fork = resolve_explicit_training_fork(
        parent,
        output_dir=tmp_path / "child",
        child_config=child,
        allowed_changes=["training.max_grad_norm"],
    )
    with pytest.raises(ValueError, match="immutable identity mismatch"):
        validate_fork_immutable_identities(
            fork,
            dataset_identity_sha256="f" * 64,
            token_policy_sha256=IDENTITIES["token_policy_sha256"],
            trainable_parameters_sha256=IDENTITIES["trainable_parameters_sha256"],
            base_snapshot_sha256=IDENTITIES["base_snapshot_sha256"],
            strategy="lora",
            base_checkpoint="org/model",
            revision="user-provided-revision",
        )

    (parent / "training_state.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        resolve_explicit_training_fork(
            parent,
            output_dir=tmp_path / "fresh",
            child_config=child,
            allowed_changes=["training.max_grad_norm"],
        )


def test_lr_fork_scales_loaded_optimizer_and_scheduler_without_reset():
    class Optimizer:
        param_groups = [{"lr": 8.0e-5, "initial_lr": 1.0e-4}]

    class Scheduler:
        base_lrs = [1.0e-4]
        _last_lr = [8.0e-5]

    changes = validate_fork_configuration(
        _parent_config(),
        {
            "experiment_name": "uniform_no_t_shifted",
            "training": {"learning_rate": 2.0e-5},
        },
        allowed_changes=["training.learning_rate"],
    )
    report = apply_learning_rate_fork(Optimizer, Scheduler, changes)
    assert report["scale"] == pytest.approx(0.2)
    assert Optimizer.param_groups[0]["lr"] == pytest.approx(1.6e-5)
    assert Optimizer.param_groups[0]["initial_lr"] == pytest.approx(2.0e-5)
    assert Scheduler.base_lrs == pytest.approx([2.0e-5])
    assert Scheduler._last_lr == pytest.approx([1.6e-5])


def test_historical_replay_fingerprint_detects_source_and_config_drift(tmp_path):
    repository = tmp_path / "snapshot"
    (repository / "audio" / "diffusion").mkdir(parents=True)
    (repository / "examples").mkdir()
    source = repository / "audio" / "diffusion" / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    trainer = repository / "examples" / "train_mawjais_diffusion.py"
    trainer.write_text("# trainer\n", encoding="utf-8")
    config = repository / "config.yaml"
    config.write_text("training: {}\n", encoding="utf-8")
    source_hash = training_source_identity(repository)
    result = validate_replay_snapshot(
        repository,
        expected_source_code_sha256=source_hash,
        config_path=config,
        expected_config_sha256=sha256_file(config),
    )
    assert result["source_code_sha256"] == source_hash

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source fingerprint mismatch"):
        validate_replay_snapshot(
            repository,
            expected_source_code_sha256=source_hash,
            config_path=config,
            expected_config_sha256=sha256_file(config),
        )
