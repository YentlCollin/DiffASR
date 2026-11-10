"""Auditable checkpoint forks for controlled training interventions.

A resume continues one run and therefore requires every trajectory identity to
match.  A fork deliberately changes a *small, declared* part of the trajectory
(for example enabling gradient clipping after an unstable update) and writes to
a new run directory.  Keeping those two operations separate prevents a rescue
experiment from being mislabeled as an exact resume.

This module is intentionally independent from the training entrypoint.  It can
be used by both the single-rank reference trainer and the DDP trainer.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .training_resume import (
    ResumeCheckpoint,
    inspect_resume_checkpoint,
    mapping_sha256,
    sha256_file,
)


FORK_SCHEMA_VERSION = 1
FORK_PROVENANCE_FILE = "fork_provenance.json"
WEIGHTS_ONLY_CONTINUATION = "weights_only_reset_optimizer_scheduler_rng"
OPTIMIZER_PRESERVING_CONTINUATION = (
    "preserve_optimizer_scaler_rng_reset_scheduler"
)

# These fields are the complete safety envelope supported by V1.  Callers must
# additionally opt in to every field that actually changes for a particular
# fork.  In particular, changing data, seeds, noise, alignment, model shape,
# trainable parameters, or the global batch size is never accepted here.
SAFE_FORK_CONFIG_PATHS = frozenset(
    {
        "experiment_name",
        "training.max_grad_norm",
        "training.learning_rate",
    }
)

# A topology-changing weight continuation is deliberately a different
# operation from the rescue fork above. It loads only trained model weights
# and resets optimizer, scheduler, RNG and dataloader cursor. The allowlist is
# narrow enough to reject changes to the data, diffusion objective, model
# shape, seeds, global batch size, or trainable parameter selection.
WEIGHT_CONTINUATION_CONFIG_PATHS = frozenset(
    {
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
)

# A same-topology phase extension must also extend the signed epoch plan and
# move checkpoint gates into the child phase.  This contract applies both to a
# first continuation from a finite cosine phase and to a later warm restart
# from an already completed continuation_cosine phase.
SAME_TOPOLOGY_CONTINUATION_CONFIG_PATHS = frozenset(
    {
        *WEIGHT_CONTINUATION_CONFIG_PATHS,
        "training.expected_epoch_updates",
        "training.save_at_updates",
        "training.scheduler_horizon_updates",
    }
)
SAME_TOPOLOGY_FIRST_CONTINUATION_REQUIRED_CONFIG_PATHS = frozenset(
    {
        "experiment_name",
        "training.max_updates",
        "training.scheduler_type",
        "training.scheduler_horizon_updates",
        "training.continuation_start_update",
        "training.continuation_start_epoch",
    }
)
SAME_TOPOLOGY_REPEATED_CONTINUATION_REQUIRED_CONFIG_PATHS = frozenset(
    {
        "experiment_name",
        "training.max_updates",
        "training.continuation_start_update",
        "training.continuation_start_epoch",
    }
)

# A deliberate warm restart may recover a peak LR above the immediately
# preceding phase, but it remains bounded and must start/end continuously at
# that phase's terminal LR.  Three is sufficient to recover the previously
# validated 3e-5 LoRA peak from the 1e-5 continuation phase.
MAX_WARM_RESTART_PEAK_RATIO = 3.0

# Every weights-only phase must identify a distinct child trajectory, extend
# the update budget and declare both the exact parent update and the new
# deterministic dataloader epoch. The initial 50k->100k topology transition
# must change every field in ``WEIGHT_CONTINUATION_CONFIG_PATHS``. Once the
# parent already uses the audited DDP16 continuation schedule, a repeated
# continuation is deliberately stricter: these are the only four permitted
# deltas and all four are required.
WEIGHT_CONTINUATION_REQUIRED_CONFIG_PATHS = frozenset(
    {
        "experiment_name",
        "training.max_updates",
        "training.continuation_start_update",
        "training.continuation_start_epoch",
    }
)

# Controlled objective interventions may be applied while loading only the
# trained weights from a completed continuation. They are deliberately kept
# separate from the required phase-boundary changes above: a control
# continuation need not change them, while an ablation that does change one is
# recorded explicitly in ``fork_provenance.json``. Do not add broad mappings
# here; every new field needs its own test and scientific justification.
WEIGHT_CONTINUATION_OPTIONAL_EXPERIMENT_PATHS = frozenset(
    {
        "diffusion.full_noise_probability",
    }
)

_IMMUTABLE_IDENTITY_FIELDS = (
    "dataset_identity_sha256",
    "token_policy_sha256",
    "trainable_parameters_sha256",
    "base_snapshot_sha256",
    "strategy",
    "base_checkpoint",
    "revision",
)
_STATE_IDENTITY_FIELDS = (
    "training_config_sha256",
    "dataset_identity_sha256",
    "token_policy_sha256",
    "trainable_parameters_sha256",
    "base_snapshot_sha256",
    "source_code_sha256",
)

_TOPOLOGY_DEPENDENT_DATASET_FIELDS = frozenset(
    {
        "epoch",
        "rank",
        "world_size",
        "samples_per_rank_after_trim",
        "samples_per_rank_before_trim",
        # Their content hashes stay in the identity. Absolute storage paths
        # can change when a source/data snapshot is mounted elsewhere.
        "eval_manifest",
        "local_root",
        "sample_exclusion_manifest",
        "subset_manifest",
    }
)


@dataclass(frozen=True)
class ConfigChange:
    """One leaf-level difference between parent and child configurations."""

    path: str
    parent_present: bool
    parent_value: Any
    child_present: bool
    child_value: Any

    def to_metadata(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "parent_present": self.parent_present,
            "parent_value": self.parent_value,
            "child_present": self.child_present,
            "child_value": self.child_value,
        }


@dataclass(frozen=True)
class TrainingFork:
    """A checksum-verified parent plus its explicitly allowed config delta."""

    checkpoint: ResumeCheckpoint
    changes: tuple[ConfigChange, ...]
    requested_allowed_changes: tuple[str, ...]
    provenance: dict[str, Any]


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - runtime dependency
        raise RuntimeError("PyYAML is required to inspect fork configurations") from error

    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise ValueError(f"fork configuration must not be a symlink: {requested}")
    resolved = requested.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"fork configuration must be a regular file: {resolved}")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"fork configuration must be a YAML mapping: {resolved}")
    return payload


def _canonical_yaml_value(value: Any) -> Any:
    """Normalize in-memory tuples to their lossless YAML representation."""

    if isinstance(value, Mapping):
        return {
            key: _canonical_yaml_value(nested)
            for key, nested in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_yaml_value(nested) for nested in value]
    return value


def _semantic_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Drop runtime observations and normalize YAML round-trip containers."""

    result = dict(config)
    result.pop("runtime", None)
    return _canonical_yaml_value(result)


def _configuration_changes(
    parent: Any,
    child: Any,
    *,
    path: tuple[str, ...] = (),
) -> list[ConfigChange]:
    if isinstance(parent, Mapping) and isinstance(child, Mapping):
        changes: list[ConfigChange] = []
        for key in sorted(set(parent) | set(child)):
            key_string = str(key)
            nested_path = (*path, key_string)
            in_parent = key in parent
            in_child = key in child
            if in_parent and in_child:
                changes.extend(
                    _configuration_changes(
                        parent[key], child[key], path=nested_path
                    )
                )
            else:
                changes.append(
                    ConfigChange(
                        path=".".join(nested_path),
                        parent_present=in_parent,
                        parent_value=parent.get(key),
                        child_present=in_child,
                        child_value=child.get(key),
                    )
                )
        return changes
    if parent == child:
        return []
    return [
        ConfigChange(
            path=".".join(path),
            parent_present=True,
            parent_value=parent,
            child_present=True,
            child_value=child,
        )
    ]


def validate_fork_configuration(
    parent_config: Mapping[str, Any],
    child_config: Mapping[str, Any],
    *,
    allowed_changes: Iterable[str],
) -> tuple[ConfigChange, ...]:
    """Return the exact allowed delta or reject the fork.

    ``allowed_changes`` is intentionally an exact declaration: listing a field
    that did not change is rejected as a likely typo or stale launch command.
    """

    requested = tuple(sorted(set(str(path) for path in allowed_changes)))
    unsafe = sorted(set(requested).difference(SAFE_FORK_CONFIG_PATHS))
    if unsafe:
        raise ValueError(
            "fork requested configuration fields outside the safety allowlist: "
            f"{unsafe}; safe={sorted(SAFE_FORK_CONFIG_PATHS)}"
        )
    changes = tuple(
        _configuration_changes(
            _semantic_config(parent_config), _semantic_config(child_config)
        )
    )
    actual = {change.path for change in changes}
    undeclared = sorted(actual.difference(requested))
    stale = sorted(set(requested).difference(actual))
    if undeclared or stale:
        raise ValueError(
            "fork configuration delta does not match the explicit declaration: "
            f"undeclared={undeclared}, declared_but_unchanged={stale}"
        )
    return changes


def validate_weight_continuation_configuration(
    parent_config: Mapping[str, Any],
    child_config: Mapping[str, Any],
) -> tuple[ConfigChange, ...]:
    """Validate the narrow delta permitted for a weights-only continuation."""

    changes = tuple(
        _configuration_changes(
            _semantic_config(parent_config), _semantic_config(child_config)
        )
    )
    actual = {change.path for change in changes}
    parent_training = parent_config.get("training")
    child_training = child_config.get("training")
    if not isinstance(parent_training, Mapping) or not isinstance(
        child_training, Mapping
    ):
        raise ValueError("weight-continuation configs require training mappings")
    repeated_continuation = (
        parent_training.get("scheduler_type") == "continuation_cosine"
    )
    topology_fields = (
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "global_batch_size",
        "supported_world_sizes",
    )
    same_topology = all(
        parent_training.get(field) == child_training.get(field)
        for field in topology_fields
    )
    same_topology_first_continuation = (
        not repeated_continuation
        and same_topology
        and parent_training.get("scheduler_type") == "cosine"
    )
    same_topology_repeated_continuation = repeated_continuation and same_topology
    same_topology_continuation = (
        same_topology_first_continuation or same_topology_repeated_continuation
    )
    if same_topology_first_continuation:
        required = SAME_TOPOLOGY_FIRST_CONTINUATION_REQUIRED_CONFIG_PATHS
        phase_permitted = SAME_TOPOLOGY_CONTINUATION_CONFIG_PATHS
    elif same_topology_repeated_continuation:
        required = SAME_TOPOLOGY_REPEATED_CONTINUATION_REQUIRED_CONFIG_PATHS
        phase_permitted = SAME_TOPOLOGY_CONTINUATION_CONFIG_PATHS
    elif repeated_continuation:
        required = WEIGHT_CONTINUATION_REQUIRED_CONFIG_PATHS
        phase_permitted = WEIGHT_CONTINUATION_REQUIRED_CONFIG_PATHS
    else:
        required = WEIGHT_CONTINUATION_CONFIG_PATHS
        phase_permitted = WEIGHT_CONTINUATION_CONFIG_PATHS
    permitted = (
        phase_permitted
        | WEIGHT_CONTINUATION_OPTIONAL_EXPERIMENT_PATHS
        | {"training.continuation_state_policy"}
    )
    unsafe = sorted(actual.difference(permitted))
    missing = sorted(required.difference(actual))
    if unsafe or missing:
        raise ValueError(
            "weight-continuation configuration delta is outside the audited "
            f"contract: unsafe={unsafe}, required_but_unchanged={missing}"
        )
    changes_by_path = {change.path: change for change in changes}
    full_noise_change = changes_by_path.get("diffusion.full_noise_probability")
    if full_noise_change is not None:
        valid_full_noise_intervention = (
            full_noise_change.parent_present
            and full_noise_change.child_present
            and isinstance(full_noise_change.parent_value, (int, float))
            and isinstance(full_noise_change.child_value, (int, float))
            and math.isclose(float(full_noise_change.parent_value), 0.0)
            and math.isclose(float(full_noise_change.child_value), 0.10)
        )
        if not valid_full_noise_intervention:
            raise ValueError(
                "the audited full-noise continuation requires exactly "
                "diffusion.full_noise_probability: 0.0 -> 0.10"
            )
    if same_topology_continuation:
        continuation_start = child_training.get("continuation_start_update")
        if continuation_start != parent_training.get("max_updates"):
            raise ValueError(
                "same-topology continuation must start at the parent max_updates"
            )
        if not isinstance(child_training.get("max_updates"), int) or child_training[
            "max_updates"
        ] <= continuation_start:
            raise ValueError(
                "same-topology continuation max_updates must exceed its start update"
            )

        parent_epoch_updates = parent_training.get("expected_epoch_updates")
        child_epoch_updates = child_training.get("expected_epoch_updates")
        continuation_epoch = child_training.get("continuation_start_epoch")
        if isinstance(parent_epoch_updates, list):
            if not isinstance(child_epoch_updates, list):
                raise ValueError(
                    "same-topology conditional continuation must extend "
                    "expected_epoch_updates"
                )
            if continuation_epoch != len(parent_epoch_updates):
                raise ValueError(
                    "same-topology continuation_start_epoch must equal the signed "
                    "parent epoch count"
                )
            if child_epoch_updates[:continuation_epoch] != parent_epoch_updates:
                raise ValueError(
                    "same-topology continuation must preserve the signed parent "
                    "epoch-update prefix"
                )
            if sum(child_epoch_updates) != child_training["max_updates"]:
                raise ValueError(
                    "same-topology continuation epoch plan must sum to max_updates"
                )
        elif repeated_continuation:
            parent_continuation_epoch = parent_training.get(
                "continuation_start_epoch"
            )
            if (
                not isinstance(parent_continuation_epoch, int)
                or not isinstance(continuation_epoch, int)
                or continuation_epoch <= parent_continuation_epoch
            ):
                raise ValueError(
                    "repeated continuation must advance the dataloader epoch"
                )

        child_save_gates = child_training.get("save_at_updates", [])
        if any(gate <= continuation_start for gate in child_save_gates):
            raise ValueError(
                "same-topology continuation save_at_updates must contain only "
                "child-phase updates"
            )

        if child_training.get("scheduler_type") != "continuation_cosine":
            raise ValueError(
                "same-topology phase extension requires continuation_cosine"
            )
        parent_terminal_lr = float(parent_training["learning_rate"]) * float(
            parent_training["scheduler_min_lr_ratio"]
        )
        child_base_lr = float(child_training["learning_rate"])
        child_minimum_ratio = float(child_training["scheduler_min_lr_ratio"])
        child_warmup = int(child_training["scheduler_warmup_updates"])
        child_initial_multiplier = (
            child_minimum_ratio
            + (1.0 - child_minimum_ratio) / child_warmup
            if child_warmup
            else 1.0
        )
        child_initial_lr = child_base_lr * child_initial_multiplier
        child_terminal_lr = child_base_lr * child_minimum_ratio
        tolerance = max(abs(parent_terminal_lr) * 0.01, 1.0e-12)
        if abs(child_initial_lr - parent_terminal_lr) > tolerance:
            raise ValueError(
                "same-topology continuation initial LR must be within 1% of "
                "the parent terminal LR"
            )
        if abs(child_terminal_lr - parent_terminal_lr) > tolerance:
            raise ValueError(
                "same-topology continuation terminal LR must be within 1% of "
                "the parent terminal LR"
            )
        maximum_peak_lr = float(parent_training["learning_rate"]) * (
            MAX_WARM_RESTART_PEAK_RATIO
        )
        if child_base_lr > maximum_peak_lr and not math.isclose(
            child_base_lr, maximum_peak_lr, rel_tol=1.0e-12
        ):
            raise ValueError(
                "same-topology continuation peak LR exceeds the audited "
                f"{MAX_WARM_RESTART_PEAK_RATIO:g}x warm-restart bound"
            )

    if repeated_continuation and not same_topology_repeated_continuation:
        expected_fixed_values = {
            "gradient_accumulation_steps": 1,
            "global_batch_size": 16,
            "supported_world_sizes": [1, 2, 4, 8, 16],
            "learning_rate": 1.0e-5,
            "scheduler_type": "continuation_cosine",
            "scheduler_warmup_updates": 1000,
            "scheduler_min_lr_ratio": 0.1,
        }
        incoherent = {
            key: child_training.get(key)
            for key, expected in expected_fixed_values.items()
            if child_training.get(key) != expected
        }
        if incoherent:
            raise ValueError(
                "repeated weight continuation must preserve the audited DDP16 "
                f"schedule: {incoherent}"
            )
        if child_training.get("continuation_start_update") != parent_training.get(
            "max_updates"
        ):
            raise ValueError(
                "repeated weight continuation must start at the parent max_updates"
            )
        if not isinstance(child_training.get("max_updates"), int) or child_training[
            "max_updates"
        ] <= child_training["continuation_start_update"]:
            raise ValueError(
                "repeated weight continuation max_updates must exceed its start update"
            )
        if not isinstance(child_training.get("continuation_start_epoch"), int) or (
            child_training["continuation_start_epoch"]
            <= int(parent_training.get("continuation_start_epoch", -1))
        ):
            raise ValueError(
                "repeated weight continuation must advance the dataloader epoch"
            )
    return changes


def _validate_empty_child_output(output_dir: str | Path) -> Path:
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        if output.is_symlink() or not output.is_dir():
            raise ValueError(f"fork output must be a real directory: {output}")
        entries = list(output.iterdir())
        if entries:
            raise ValueError(
                "fork output must be absent or empty; refusing to mix parent and child "
                f"artifacts: {output} contains {[item.name for item in entries[:8]]}"
            )
    return output


def resolve_explicit_training_fork(
    checkpoint_path: str | Path,
    *,
    output_dir: str | Path,
    child_config: Mapping[str, Any],
    allowed_changes: Sequence[str],
    expected_parent_update: int | None = None,
) -> TrainingFork:
    """Resolve an external parent without weakening normal resume semantics."""

    request = str(checkpoint_path).strip()
    if not request or request == "latest":
        raise ValueError("--fork-from-checkpoint requires one explicit checkpoint path")
    output = _validate_empty_child_output(output_dir)
    checkpoint = inspect_resume_checkpoint(request)
    if (
        output == checkpoint.path
        or output in checkpoint.path.parents
        or checkpoint.path in output.parents
    ):
        raise ValueError("fork output must be distinct from the parent checkpoint tree")
    if expected_parent_update is not None and checkpoint.update != int(
        expected_parent_update
    ):
        raise ValueError(
            "fork parent update mismatch: "
            f"expected={expected_parent_update}, actual={checkpoint.update}"
        )
    parent_config = _load_yaml_mapping(checkpoint.resolved_config)
    changes = validate_fork_configuration(
        parent_config,
        child_config,
        allowed_changes=allowed_changes,
    )
    requested = tuple(sorted(set(str(path) for path in allowed_changes)))
    manifest_path = checkpoint.path / "diffusion_checkpoint.json"
    provenance = {
        "schema_version": FORK_SCHEMA_VERSION,
        "operation": "training_fork",
        "parent_checkpoint": str(checkpoint.path),
        "parent_update": checkpoint.update,
        "parent_manifest_sha256": sha256_file(manifest_path),
        "parent_artifact_sha256": dict(checkpoint.manifest["artifact_sha256"]),
        "parent_training_config_sha256": checkpoint.manifest[
            "training_config_sha256"
        ],
        "parent_source_code_sha256": checkpoint.manifest["source_code_sha256"],
        "requested_allowed_changes": list(requested),
        "config_changes": [change.to_metadata() for change in changes],
    }
    provenance["provenance_sha256"] = mapping_sha256(provenance)
    return TrainingFork(
        checkpoint=checkpoint,
        changes=changes,
        requested_allowed_changes=requested,
        provenance=provenance,
    )


def resolve_weight_continuation(
    checkpoint_path: str | Path,
    *,
    output_dir: str | Path,
    child_config: Mapping[str, Any],
    expected_parent_update: int,
    verify_trained_model_hash: bool = True,
) -> TrainingFork:
    """Authenticate a parent for an explicitly controlled continuation."""

    request = str(checkpoint_path).strip()
    if not request or request == "latest":
        raise ValueError(
            "--continue-from-checkpoint requires one explicit checkpoint path"
        )
    output = _validate_empty_child_output(output_dir)
    checkpoint = inspect_resume_checkpoint(
        request,
        verify_trained_model_hash=verify_trained_model_hash,
    )
    if (
        output == checkpoint.path
        or output in checkpoint.path.parents
        or checkpoint.path in output.parents
    ):
        raise ValueError(
            "weight-continuation output must be distinct from the parent checkpoint tree"
        )
    if checkpoint.update != int(expected_parent_update):
        raise ValueError(
            "weight-continuation parent update mismatch: "
            f"expected={expected_parent_update}, actual={checkpoint.update}"
        )
    parent_config = _load_yaml_mapping(checkpoint.resolved_config)
    changes = validate_weight_continuation_configuration(
        parent_config,
        child_config,
    )
    requested_changes = tuple(sorted(change.path for change in changes))
    child_training = child_config.get("training")
    if not isinstance(child_training, Mapping):
        raise ValueError("continuation child config requires a training mapping")
    continuation_state_policy = str(
        child_training.get(
            "continuation_state_policy",
            WEIGHTS_ONLY_CONTINUATION,
        )
    )
    if continuation_state_policy == OPTIMIZER_PRESERVING_CONTINUATION:
        state_policy = {
            "name": continuation_state_policy,
            "trained_model_weights": "loaded",
            "diffusion_component_weights": "loaded",
            "optimizer": "loaded_from_parent",
            "scheduler": "reset_from_child_config",
            "gradient_scaler": "loaded_from_parent",
            "process_rng": "loaded_from_parent",
            "dataloader_cursor": "loaded_from_parent",
        }
    elif continuation_state_policy == WEIGHTS_ONLY_CONTINUATION:
        state_policy = {
            "name": continuation_state_policy,
            "trained_model_weights": "loaded",
            "diffusion_component_weights": "loaded",
            "optimizer": "reset",
            "scheduler": "reset",
            "gradient_scaler": "reset",
            "process_rng": "reset_from_child_seeds",
            "dataloader_cursor": "reset_at_declared_epoch",
        }
    else:
        raise ValueError(
            "unsupported training.continuation_state_policy: "
            f"{continuation_state_policy}"
        )
    manifest_path = checkpoint.path / "diffusion_checkpoint.json"
    provenance = {
        "schema_version": FORK_SCHEMA_VERSION,
        "operation": "weight_continuation",
        "parent_checkpoint": str(checkpoint.path),
        "parent_update": checkpoint.update,
        "parent_world_size": checkpoint.manifest.get("world_size"),
        "parent_manifest_sha256": sha256_file(manifest_path),
        "parent_artifact_sha256": dict(checkpoint.manifest["artifact_sha256"]),
        "parent_training_config_sha256": checkpoint.manifest[
            "training_config_sha256"
        ],
        "parent_source_code_sha256": checkpoint.manifest["source_code_sha256"],
        "requested_allowed_changes": list(requested_changes),
        "config_changes": [change.to_metadata() for change in changes],
        "state_policy": state_policy,
    }
    provenance["provenance_sha256"] = mapping_sha256(provenance)
    return TrainingFork(
        checkpoint=checkpoint,
        changes=changes,
        requested_allowed_changes=requested_changes,
        provenance=provenance,
    )


def validate_fork_immutable_identities(
    fork: TrainingFork,
    *,
    dataset_identity_sha256: str,
    token_policy_sha256: str,
    trainable_parameters_sha256: str,
    base_snapshot_sha256: str,
    strategy: str,
    base_checkpoint: str,
    revision: str,
) -> None:
    """Validate everything a clip/LR rescue is forbidden to change."""

    current = {
        "dataset_identity_sha256": dataset_identity_sha256,
        "token_policy_sha256": token_policy_sha256,
        "trainable_parameters_sha256": trainable_parameters_sha256,
        "base_snapshot_sha256": base_snapshot_sha256,
        "strategy": strategy,
        "base_checkpoint": base_checkpoint,
        "revision": revision,
    }
    mismatches = {
        name: {
            "parent": fork.checkpoint.manifest.get(name),
            "child": current[name],
        }
        for name in _IMMUTABLE_IDENTITY_FIELDS
        if fork.checkpoint.manifest.get(name) != current[name]
    }
    if mismatches:
        raise ValueError(f"fork immutable identity mismatch: {mismatches}")


def weight_continuation_dataset_identity(
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Return dataset content/provenance without rank-layout observations."""

    return {
        key: _canonical_yaml_value(value)
        for key, value in summary.items()
        if key not in _TOPOLOGY_DEPENDENT_DATASET_FIELDS
    }


def validate_weight_continuation_immutable_identities(
    fork: TrainingFork,
    *,
    child_dataset_summary: Mapping[str, Any],
    token_policy_sha256: str,
    trainable_parameters_sha256: str,
    base_snapshot_sha256: str,
    strategy: str,
    base_checkpoint: str,
    revision: str,
) -> dict[str, Any]:
    """Validate content identities while deliberately allowing a new DDP layout."""

    current = {
        "token_policy_sha256": token_policy_sha256,
        "trainable_parameters_sha256": trainable_parameters_sha256,
        "base_snapshot_sha256": base_snapshot_sha256,
        "strategy": strategy,
        "base_checkpoint": base_checkpoint,
        "revision": revision,
    }
    mismatches = {
        name: {"parent": fork.checkpoint.manifest.get(name), "child": value}
        for name, value in current.items()
        if fork.checkpoint.manifest.get(name) != value
    }
    if mismatches:
        raise ValueError(
            f"weight-continuation immutable identity mismatch: {mismatches}"
        )

    parent_config = _load_yaml_mapping(fork.checkpoint.resolved_config)
    parent_runtime = parent_config.get("runtime")
    parent_summary = (
        parent_runtime.get("dataset")
        if isinstance(parent_runtime, Mapping)
        else None
    )
    if not isinstance(parent_summary, Mapping):
        raise ValueError(
            "weight-continuation parent has no authenticated runtime.dataset summary"
        )
    parent_content = weight_continuation_dataset_identity(parent_summary)
    child_content = weight_continuation_dataset_identity(child_dataset_summary)
    if parent_content != child_content:
        changes = _configuration_changes(parent_content, child_content)
        raise ValueError(
            "weight-continuation dataset content/provenance mismatch: "
            f"{[change.to_metadata() for change in changes]}"
        )
    return {
        "content_identity_sha256": mapping_sha256(parent_content),
        "parent_world_size": parent_summary.get("world_size"),
        "child_world_size": child_dataset_summary.get("world_size"),
        "parent_samples_per_rank_after_trim": parent_summary.get(
            "samples_per_rank_after_trim"
        ),
        "child_samples_per_rank_after_trim": child_dataset_summary.get(
            "samples_per_rank_after_trim"
        ),
        "excluded_sample_count": child_content.get("excluded_sample_count"),
        "subset_manifest_sha256": child_content.get("subset_manifest_sha256"),
        "sample_exclusion_manifest_sha256": child_content.get(
            "sample_exclusion_manifest_sha256"
        ),
    }


def load_fork_training_state(fork: TrainingFork) -> dict[str, Any]:
    """Load state only after artifact checks and verify parent self-consistency.

    The state schema is owned by the trainer and may contain additional DDP
    fields.  This function validates the cross-file invariants required for a
    fork without discarding those fields.
    """

    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime dependency
        raise RuntimeError("PyTorch is required to load fork training state") from error

    payload = torch.load(
        fork.checkpoint.training_state,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise ValueError("fork parent training state must be a mapping")
    if payload.get("update") != fork.checkpoint.update:
        raise ValueError("fork training-state update does not match its manifest")
    state_identities = payload.get("identities")
    if not isinstance(state_identities, Mapping):
        raise ValueError("fork parent training state has no identities mapping")
    expected = {
        name: fork.checkpoint.manifest.get(name) for name in _STATE_IDENTITY_FIELDS
    }
    if dict(state_identities) != expected:
        raise ValueError(
            "fork parent training-state identities do not match its manifest"
        )
    return payload


def apply_learning_rate_fork(
    optimizer: Any,
    scheduler: Any,
    changes: Sequence[ConfigChange],
) -> dict[str, Any] | None:
    """Scale the loaded LR state while preserving optimizer moments/cursor/RNG.

    This is used only by the optional second rescue.  The primary clip-only
    fork returns ``None`` and leaves optimizer/scheduler state bit-identical.
    """

    lr_changes = [change for change in changes if change.path == "training.learning_rate"]
    if not lr_changes:
        return None
    if len(lr_changes) != 1:
        raise ValueError("fork contains more than one learning-rate change")
    change = lr_changes[0]
    if not change.parent_present or not change.child_present:
        raise ValueError("learning-rate fork requires values in parent and child")
    old = float(change.parent_value)
    new = float(change.child_value)
    if not math.isfinite(old) or not math.isfinite(new) or old <= 0 or new <= 0:
        raise ValueError("fork learning rates must be finite and > 0")
    ratio = new / old
    previous_group_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    for group in optimizer.param_groups:
        group["lr"] = float(group["lr"]) * ratio
        if "initial_lr" in group:
            group["initial_lr"] = float(group["initial_lr"]) * ratio
    if not hasattr(scheduler, "base_lrs") or not hasattr(scheduler, "_last_lr"):
        raise TypeError("fork LR override requires a PyTorch-style scheduler")
    scheduler.base_lrs = [float(value) * ratio for value in scheduler.base_lrs]
    scheduler._last_lr = [float(value) * ratio for value in scheduler._last_lr]
    return {
        "parent_base_learning_rate": old,
        "child_base_learning_rate": new,
        "scale": ratio,
        "optimizer_lr_before": previous_group_lrs,
        "optimizer_lr_after": [float(group["lr"]) for group in optimizer.param_groups],
    }


def write_fork_provenance(
    output_dir: str | Path,
    fork: TrainingFork,
    *,
    child_training_config_sha256: str,
    child_source_code_sha256: str,
    additional_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically write the immutable parent/child lineage record."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / FORK_PROVENANCE_FILE
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"fork provenance already exists: {path}")
    payload = {
        **{
            key: value
            for key, value in fork.provenance.items()
            if key != "provenance_sha256"
        },
        "child_training_config_sha256": str(child_training_config_sha256),
        "child_source_code_sha256": str(child_source_code_sha256),
    }
    if additional_metadata is not None:
        overlap = set(payload).intersection(additional_metadata)
        if overlap:
            raise ValueError(
                f"additional fork provenance would overwrite fields: {sorted(overlap)}"
            )
        payload.update(_canonical_yaml_value(dict(additional_metadata)))
    payload["provenance_sha256"] = mapping_sha256(
        {key: value for key, value in payload.items() if key != "provenance_sha256"}
    )
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    return path


def read_fork_provenance(path: str | Path) -> dict[str, Any]:
    """Read and authenticate lineage before carrying it through a resume."""

    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise ValueError(f"fork provenance must not be a symlink: {requested}")
    resolved = requested.resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != FORK_SCHEMA_VERSION:
        raise ValueError("fork provenance has an unsupported schema")
    saved_hash = payload.get("provenance_sha256")
    actual_hash = mapping_sha256(
        {key: value for key, value in payload.items() if key != "provenance_sha256"}
    )
    if saved_hash != actual_hash:
        raise ValueError(
            "fork provenance checksum mismatch: "
            f"saved={saved_hash}, actual={actual_hash}"
        )
    return payload


def training_source_identity(repository_root: str | Path) -> str:
    """Reproduce the trainer's source fingerprint without importing the trainer."""

    root = Path(repository_root).expanduser().resolve(strict=True)
    patterns = (
        "_mawjais_diffusion_repository/**/*.py",
        "audio/diffusion/**/*.py",
        "audio/diffusion_data/**/*.py",
        "audio/model/*.py",
        "huggingface/xllm/*.py",
        "examples/train_mawjais_diffusion.py",
    )
    selected = {
        path.resolve(strict=True)
        for pattern in patterns
        for path in root.glob(pattern)
        if path.is_file() and not path.is_symlink()
    }
    if not selected:
        raise ValueError(f"no critical training sources found below {root}")
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted(selected, key=lambda item: item.relative_to(root).as_posix())
    ]
    return mapping_sha256({"schema_version": 1, "files": files})


def validate_replay_snapshot(
    repository_root: str | Path,
    *,
    expected_source_code_sha256: str,
    config_path: str | Path,
    expected_config_sha256: str,
) -> dict[str, str]:
    """Fail before allocating model weights if the historical replay drifted."""

    actual_source = training_source_identity(repository_root)
    if actual_source != expected_source_code_sha256:
        raise ValueError(
            "historical replay source fingerprint mismatch: "
            f"expected={expected_source_code_sha256}, actual={actual_source}"
        )
    actual_config = sha256_file(config_path)
    if actual_config != expected_config_sha256:
        raise ValueError(
            "historical replay config checksum mismatch: "
            f"expected={expected_config_sha256}, actual={actual_config}"
        )
    return {
        "repository_root": str(Path(repository_root).expanduser().resolve()),
        "source_code_sha256": actual_source,
        "config_path": str(Path(config_path).expanduser().resolve()),
        "config_sha256": actual_config,
    }


__all__ = [
    "ConfigChange",
    "FORK_PROVENANCE_FILE",
    "FORK_SCHEMA_VERSION",
    "SAFE_FORK_CONFIG_PATHS",
    "TrainingFork",
    "WEIGHT_CONTINUATION_CONFIG_PATHS",
    "apply_learning_rate_fork",
    "load_fork_training_state",
    "read_fork_provenance",
    "resolve_explicit_training_fork",
    "resolve_weight_continuation",
    "training_source_identity",
    "validate_fork_configuration",
    "validate_weight_continuation_configuration",
    "validate_weight_continuation_immutable_identities",
    "validate_fork_immutable_identities",
    "validate_replay_snapshot",
    "weight_continuation_dataset_identity",
    "write_fork_provenance",
]
