#!/usr/bin/env python3
"""Reference PyTorch trainer for the first MawJais diffusion experiments.

The command is intentionally fail-fast.  In particular, ``--train`` requires a
machine-readable AR PyTorch parity artifact *before* it resolves/downloads the
4B checkpoint and before it constructs an optimizer.  ``--validate-only``
loads and composes YAML, optionally validates the bounded local streaming
dataset, and writes the resolved config without loading any model weights.

This is a correctness-oriented ROCm trainer for one to 128 ranks. Distributed
training requires a signed pre-sharding overflow manifest and equal per-rank
sample quotas so uneven IterableDataset exhaustion cannot deadlock a collective.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import random
import re
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from audio.diffusion.checkpointing import (  # noqa: E402
    load_diffusion_component_state,
    save_diffusion_component_state,
    write_diffusion_checkpoint_manifest,
)
from audio.diffusion.canvas import CanvasOverflowError  # noqa: E402
from audio.diffusion.collator import (  # noqa: E402
    MawJaisDiffusionCollator,
    NoisedCanvasBatch,
    corrupt_diffusion_batch,
)
from audio.diffusion.config import CanvasConfig, DiffusionConfig  # noqa: E402
from audio.diffusion.distributed import (  # noqa: E402
    collective_unexpected_overflow,
    environment_world_size,
    gather_rank_objects,
    reduce_additive_values,
    reduce_max_value,
    resolve_distributed_training_layout,
    step_raw_scheduler_for_synchronized_update,
    validate_rank_states,
)
from audio.diffusion.experiment_config import load_experiment_config  # noqa: E402
from audio.diffusion.metrics import (  # noqa: E402
    denoising_token_metrics,
    ratios_from_additive_metrics,
)
from audio.diffusion.noise_process import (  # noqa: E402
    MaskNoiseProcess,
    UniformNoiseProcess,
)
from audio.diffusion.peft_utils import PeftCompositeProxy  # noqa: E402
from audio.diffusion.token_policy import (  # noqa: E402
    build_diffusion_token_policy,
    validate_mawjais_token_contract,
)
from audio.diffusion.training_fork import (  # noqa: E402
    OPTIMIZER_PRESERVING_CONTINUATION,
    TrainingFork,
    resolve_weight_continuation,
    validate_weight_continuation_immutable_identities,
    write_fork_provenance,
)
from audio.diffusion.training_resume import (  # noqa: E402
    CHECKPOINT_DIRECTORY_PATTERN,
    RESUME_STATE_SCHEMA_VERSION,
    ResumeCheckpoint,
    dataset_identity,
    inspect_resume_checkpoint,
    mapping_sha256,
    resolve_resume_checkpoint,
    sha256_directory,
    sha256_file,
    validate_resume_identities,
)


PARITY_ARTIFACT_SCHEMA_VERSION = 1
PARITY_FAILURE_BANNER = "PYTORCH AR PARITY GATE CLOSED"
COLLATOR_METRIC_NAMES = (
    "examples_seen",
    "examples_emitted",
    "duration_bucket_overflow",
    "target_bucket_overflow",
    "batches_fully_skipped",
)
CONDITIONAL_BATCH_FIELDS = (
    "view_ids",
    "request_code_switching",
    "request_diacritics",
    "content_has_code_switching",
    "applied_code_switching",
)


class ParityGateError(RuntimeError):
    """The known-good AR PyTorch prerequisite was not demonstrated."""


class UnsupportedTrainingFeatureError(RuntimeError):
    """A YAML option is valid for the roadmap but absent from this trainer."""


def accumulate_collator_metrics(
    batch: Mapping[str, Any], counters: dict[str, int]
) -> bool:
    """Add one worker-produced overflow report and return whether to skip it."""

    metrics = batch.get("collator_metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("collator output is missing additive overflow metrics")
    if set(counters) != set(COLLATOR_METRIC_NAMES):
        raise ValueError("overflow counter mapping has an unexpected schema")
    for name in COLLATOR_METRIC_NAMES:
        value = metrics.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"invalid collator metric {name}: {value!r}")
        counters[name] += value
    skip_batch = batch.get("skip_batch")
    if not isinstance(skip_batch, bool):
        raise ValueError("collator output skip_batch must be boolean")
    return skip_batch


def validate_conditional_batch_fields(batch: Mapping[str, Any]) -> dict[str, int]:
    """Validate and summarize the conditional view contract from the collator."""

    sample_keys = batch.get("sample_keys")
    if not isinstance(sample_keys, (list, tuple)) or not sample_keys:
        raise ValueError("conditional batch requires non-empty sample_keys")
    batch_size = len(sample_keys)
    view_ids = batch.get("view_ids")
    if (
        not isinstance(view_ids, torch.Tensor)
        or view_ids.dtype != torch.long
        or view_ids.shape != (batch_size,)
        or bool(((view_ids < 0) | (view_ids > 3)).any())
    ):
        raise ValueError("conditional batch view_ids must be LongTensor[B] in [0, 3]")
    bool_fields: dict[str, torch.Tensor] = {}
    for name in CONDITIONAL_BATCH_FIELDS[1:]:
        value = batch.get(name)
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.bool
            or value.shape != (batch_size,)
        ):
            raise ValueError(f"conditional batch {name} must be BoolTensor[B]")
        bool_fields[name] = value

    expected_code_switch = view_ids.ge(2)
    expected_diacritics = view_ids.remainder(2).eq(1)
    if not torch.equal(bool_fields["request_code_switching"], expected_code_switch):
        raise ValueError("conditional request_code_switching disagrees with view_ids")
    if not torch.equal(bool_fields["request_diacritics"], expected_diacritics):
        raise ValueError("conditional request_diacritics disagrees with view_ids")

    return {
        **{
            f"condition/view_{view_id}": int(view_ids.eq(view_id).sum())
            for view_id in range(4)
        },
        "condition/content_code_switch": int(
            bool_fields["content_has_code_switching"].sum()
        ),
        "condition/applied_code_switch": int(
            bool_fields["applied_code_switching"].sum()
        ),
        "condition/examples": batch_size,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def verify_pytorch_parity_artifact(
    artifact_path: str | Path,
    *,
    expected_checkpoint: str,
    expected_revision: str,
) -> dict[str, Any]:
    """Validate the P0 artifact consumed by the hard training gate.

    Expected minimum JSON schema::

        {
          "schema_version": 1,
          "status": "passed",
          "pytorch_ar_parity": true,
          "checkpoint": "/path/to/authorized-base-model",
          "revision": "user-provided-revision"
        }

    The producing diagnostic, not this trainer, is responsible for comparing
    reference transcription and/or logits.  This function never manufactures a
    passing artifact from a command-line acknowledgement.
    """

    path = Path(artifact_path).expanduser().resolve()
    if not path.is_file():
        raise ParityGateError(f"{PARITY_FAILURE_BANNER}: artifact not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ParityGateError(
            f"{PARITY_FAILURE_BANNER}: artifact is not valid JSON: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise ParityGateError(f"{PARITY_FAILURE_BANNER}: artifact must be a JSON object")
    checks = {
        "schema_version": payload.get("schema_version") == PARITY_ARTIFACT_SCHEMA_VERSION,
        "status": payload.get("status") == "passed",
        "pytorch_ar_parity": payload.get("pytorch_ar_parity") is True,
        "checkpoint": payload.get("checkpoint") == expected_checkpoint,
        "revision": payload.get("revision") == expected_revision,
    }
    internal = payload.get("checks")
    checks["checks_object"] = isinstance(internal, Mapping)
    if isinstance(internal, Mapping):
        token_accuracy = internal.get("teacher_forced_token_accuracy")
        checks.update(
            {
                "teacher_forced_exact": internal.get("teacher_forced_exact") is True,
                # Older parity artifacts computed the mean in float32, which
                # can serialize an all-True vector as 0.9999999403953552. The
                # exact boolean and first-mismatch fields remain authoritative;
                # accept only the corresponding one-ULP accuracy rounding.
                "teacher_forced_token_accuracy": isinstance(
                    token_accuracy, (int, float)
                )
                and not isinstance(token_accuracy, bool)
                and math.isfinite(float(token_accuracy))
                and math.isclose(
                    float(token_accuracy), 1.0, rel_tol=0.0, abs_tol=1.0e-6
                ),
                "teacher_forced_first_mismatch": internal.get(
                    "teacher_forced_first_mismatch"
                )
                is None,
                "generation_checked": internal.get("generation_checked") is True,
                "generation_exact": internal.get("generation_exact") is True,
            }
        )
    provenance = payload.get("provenance")
    current_commit = _repository_commit()
    checks["provenance_object"] = isinstance(provenance, Mapping)
    checks["repository_commit_known"] = current_commit is not None
    if isinstance(provenance, Mapping) and current_commit is not None:
        checks["repository_commit"] = (
            provenance.get("repository_commit") == current_commit
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ParityGateError(
            f"{PARITY_FAILURE_BANNER}: failed fields {failed}; no model or optimizer will be created"
        )
    return {
        "status": "open",
        "artifact": str(path),
        "sha256": _sha256_file(path),
        "payload": payload,
    }


def parity_gate_report(
    artifact_path: str | Path | None,
    *,
    expected_checkpoint: str,
    expected_revision: str,
) -> dict[str, Any]:
    """Non-raising gate report used only by ``--validate-only``."""

    if artifact_path is None:
        return {
            "status": "closed",
            "reason": f"{PARITY_FAILURE_BANNER}: --parity-artifact was not provided",
        }
    try:
        return verify_pytorch_parity_artifact(
            artifact_path,
            expected_checkpoint=expected_checkpoint,
            expected_revision=expected_revision,
        )
    except ParityGateError as error:
        return {"status": "closed", "reason": str(error)}


def diffusion_config_from_experiment(config: Mapping[str, Any]) -> DiffusionConfig:
    diffusion = config["diffusion"]
    canvas = CanvasConfig(
        sizes=tuple(int(value) for value in diffusion["canvas_sizes"]),
        duration_tokens_per_second=float(diffusion["duration_tokens_per_second"]),
        duration_margin_tokens=float(diffusion["duration_margin_tokens"]),
        tail_fill_mode=diffusion["tail_fill_mode"],
        turn_end_token_id=int(diffusion["turn_end_token_id"]),
        pad_token_id=int(diffusion["pad_token_id"]),
        eos_tail_loss_weight=float(diffusion["eos_tail_loss_weight"]),
        pad_loss_weight=float(diffusion["pad_loss_weight"]),
    )
    return DiffusionConfig(
        noise_type=diffusion["noise_type"],
        use_time_conditioning=bool(diffusion["use_time_conditioning"]),
        denoiser_output_alignment=diffusion["denoiser_output_alignment"],
        min_diffusion_time=float(diffusion["min_diffusion_time"]),
        full_noise_probability=float(diffusion["full_noise_probability"]),
        mask_token_id=int(diffusion["mask_token_id"]),
        uniform_vocab_scope=diffusion["uniform_vocab_scope"],
        diffusion_objective=diffusion["diffusion_objective"],
        loss_positions=diffusion["loss_positions"],
        loss_weighting=diffusion["loss_weighting"],
        canvas=canvas,
    )


def assert_trainer_support(config: Mapping[str, Any]) -> None:
    """Reject roadmap options that this one-stage trainer does not execute."""

    data = config["data"]
    if data.get("exclude_eval_video_groups") is not True:
        raise ValueError(
            "data.exclude_eval_video_groups must be true; training/evaluation video "
            "groups are required to remain disjoint"
        )
    if not isinstance(data.get("dataset_revision"), str) or not data[
        "dataset_revision"
    ]:
        raise ValueError("data.dataset_revision must be a non-empty immutable revision")
    loader = data.get("loader", "legacy_local_tar")
    if loader == "legacy_local_tar":
        exclusion_manifest = data.get("target_overflow_exclusion_manifest")
        if not isinstance(exclusion_manifest, str) or not exclusion_manifest:
            raise ValueError(
                "data.target_overflow_exclusion_manifest must name the signed "
                "pre-sharding manifest"
            )
        if int(data.get("expected_target_overflow_exclusions", -1)) < 0:
            raise ValueError("data.expected_target_overflow_exclusions must be >= 0")
    elif loader == "conditional_local_tar":
        if not isinstance(data.get("conditional_metadata_index"), str) or not data[
            "conditional_metadata_index"
        ]:
            raise ValueError(
                "data.conditional_metadata_index must name the signed full-corpus index"
            )
        if data.get("balance_distributed_shards", True) is not True:
            raise ValueError(
                "conditional DDP training requires balance_distributed_shards=true"
            )
    else:
        raise ValueError(f"unsupported data.loader: {loader!r}")
    diffusion = config["diffusion"]
    if diffusion.get("use_self_conditioning", False):
        raise UnsupportedTrainingFeatureError(
            "self-conditioning is not implemented by this initial trainer"
        )
    if diffusion.get("use_iterative_self_correction", False):
        raise UnsupportedTrainingFeatureError(
            "ISCT is a separate two-stage refinement and is not implemented by "
            "examples/train_mawjais_diffusion.py; refusing to silently run one-stage training"
        )
    training = config["training"]
    positive_integer_fields = (
        "max_updates",
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "log_every_updates",
        "save_every_updates",
        "save_total_limit",
    )
    for field in positive_integer_fields:
        if int(training[field]) <= 0:
            raise ValueError(f"training.{field} must be > 0")
    if float(training["learning_rate"]) <= 0:
        raise ValueError("training.learning_rate must be > 0")
    if float(training["weight_decay"]) < 0:
        raise ValueError("training.weight_decay must be >= 0")
    if not 0 <= float(training["warmup_ratio"]) < 1:
        raise ValueError("training.warmup_ratio must be in [0, 1)")
    if training["mixed_precision"] not in ("bf16", "fp16"):
        raise ValueError("training.mixed_precision must be 'bf16' or 'fp16'")
    if training.get("attention_implementation", "eager") not in ("eager", "sdpa"):
        raise ValueError(
            "training.attention_implementation must be 'eager' or 'sdpa'"
        )
    save_at_updates = training.get("save_at_updates", [])
    if not isinstance(save_at_updates, list) or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in save_at_updates
    ):
        raise ValueError("training.save_at_updates must contain positive integers")
    if save_at_updates != sorted(set(save_at_updates)):
        raise ValueError(
            "training.save_at_updates must be strictly increasing without duplicates"
        )
    if save_at_updates and save_at_updates[-1] > int(training["max_updates"]):
        raise ValueError("training.save_at_updates must not exceed max_updates")
    if float(training.get("max_grad_norm", 0.0)) <= 0:
        raise ValueError("training.max_grad_norm must be > 0")
    supported_world_sizes = training.get("supported_world_sizes", [1, 2, 4, 8])
    if not isinstance(supported_world_sizes, list) or not supported_world_sizes:
        raise ValueError("training.supported_world_sizes must be a non-empty list")
    if supported_world_sizes != sorted(set(supported_world_sizes)):
        raise ValueError(
            "training.supported_world_sizes must be strictly increasing without duplicates"
        )
    if any(value not in (1, 2, 4, 8, 16, 32, 64, 128) for value in supported_world_sizes):
        raise ValueError("training.supported_world_sizes contains an unsupported topology")
    if int(training.get("global_batch_size", 0)) <= 0:
        raise ValueError("training.global_batch_size must be > 0")


def build_learning_rate_multiplier(
    training_config: Mapping[str, Any],
) -> tuple[Callable[[int], float], int]:
    """Build a topology-independent schedule indexed by optimizer updates."""

    max_updates = int(training_config["max_updates"])
    scheduler_type = str(training_config.get("scheduler_type", "linear"))
    if scheduler_type == "linear":
        warmup_updates = int(max_updates * float(training_config["warmup_ratio"]))

        def linear_multiplier(update: int) -> float:
            if warmup_updates and update < warmup_updates:
                return (update + 1) / warmup_updates
            remaining = max(max_updates - warmup_updates, 1)
            return max(0.0, (max_updates - update) / remaining)

        return linear_multiplier, warmup_updates

    if scheduler_type == "cosine":
        warmup_updates = int(training_config["scheduler_warmup_updates"])
        minimum_ratio = float(training_config["scheduler_min_lr_ratio"])
        horizon_updates = int(training_config["scheduler_horizon_updates"])
        if not 0 <= minimum_ratio < 1:
            raise ValueError("scheduler_min_lr_ratio must be in [0, 1)")
        if not 0 <= warmup_updates < horizon_updates:
            raise ValueError(
                "scheduler_warmup_updates must be in [0, scheduler_horizon_updates)"
            )
        if horizon_updates != max_updates:
            raise ValueError(
                "fresh cosine scheduler horizon must equal configured max_updates"
            )

        def cosine_multiplier(update: int) -> float:
            if warmup_updates and update < warmup_updates:
                return (update + 1) / warmup_updates
            decay_updates = max(horizon_updates - warmup_updates, 1)
            decay_progress = min(
                max((update - warmup_updates) / decay_updates, 0.0),
                1.0,
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            return minimum_ratio + (1.0 - minimum_ratio) * cosine

        return cosine_multiplier, warmup_updates

    if scheduler_type == "continuation_cosine":
        continuation_start = int(training_config["continuation_start_update"])
        continuation_updates = max_updates - continuation_start
        warmup_updates = int(training_config["scheduler_warmup_updates"])
        minimum_ratio = float(training_config["scheduler_min_lr_ratio"])

        def continuation_multiplier(local_update: int) -> float:
            if warmup_updates and local_update < warmup_updates:
                warmup_progress = (local_update + 1) / warmup_updates
                return minimum_ratio + (1.0 - minimum_ratio) * warmup_progress
            decay_updates = max(continuation_updates - warmup_updates, 1)
            decay_progress = min(
                max((local_update - warmup_updates) / decay_updates, 0.0),
                1.0,
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            return minimum_ratio + (1.0 - minimum_ratio) * cosine

        return continuation_multiplier, warmup_updates

    raise ValueError(f"unsupported scheduler_type: {scheduler_type}")


def capture_fresh_scheduler_state(
    optimizer: Any,
    scheduler: Any,
) -> tuple[dict[str, Any], tuple[dict[str, float], ...]]:
    """Capture the child schedule before a parent optimizer restore.

    Loading Adam state also restores the parent's param-group learning rate.
    FSDP's collective loader additionally restores every registered scheduler.
    A state-preserving continuation wants neither: it keeps Adam's moments but
    starts the newly configured continuation schedule at its continuous floor.
    """

    scheduler_state = deepcopy(scheduler.state_dict())
    group_state: list[dict[str, float]] = []
    for group in optimizer.param_groups:
        values = {"lr": float(group["lr"])}
        if "initial_lr" in group:
            values["initial_lr"] = float(group["initial_lr"])
        group_state.append(values)
    return scheduler_state, tuple(group_state)


def restore_fresh_scheduler_after_optimizer_load(
    optimizer: Any,
    scheduler: Any,
    scheduler_state: Mapping[str, Any],
    group_state: Sequence[Mapping[str, float]],
) -> None:
    """Restore only the child scheduler/LR after loading parent Adam state."""

    if len(optimizer.param_groups) != len(group_state):
        raise ValueError(
            "parent optimizer param-group count changed during continuation load"
        )
    scheduler.load_state_dict(dict(scheduler_state))
    for group, saved in zip(optimizer.param_groups, group_state, strict=True):
        group["lr"] = float(saved["lr"])
        if "initial_lr" in saved:
            group["initial_lr"] = float(saved["initial_lr"])


def optimizer_state_entry_count(optimizer: Any) -> int:
    """Return the number of parameters carrying optimizer state."""

    inner = getattr(optimizer, "optimizer", optimizer)
    state = getattr(inner, "state", None)
    if not isinstance(state, Mapping):
        raise TypeError("optimizer does not expose a mapping state")
    return len(state)


def materialize_adam_state_for_fsdp_load(optimizer: Any) -> int:
    """Create the empty Adam tensor template required by sharded FSDP load.

    Adam initializes its moments lazily on the first optimizer step. The
    PyTorch distributed-checkpoint loader used by Accelerate loads into an
    existing state-dict template, so a freshly constructed optimizer exposes
    no moment keys and silently has nothing into which it can load the saved
    FSDP shards. Materializing zero-valued entries before ``load_state`` makes
    those keys visible without applying an update or changing any parameter.
    The checkpoint loader overwrites every value immediately afterwards.
    """

    inner = getattr(optimizer, "optimizer", optimizer)
    if not isinstance(inner, (torch.optim.Adam, torch.optim.AdamW)):
        raise TypeError(
            "FSDP optimizer-state materialization currently supports Adam/AdamW"
        )
    if optimizer_state_entry_count(optimizer) > 0:
        return optimizer_state_entry_count(optimizer)

    for group in inner.param_groups:
        for parameter in group["params"]:
            if not parameter.requires_grad:
                continue
            state = inner.state[parameter]
            step_device = (
                parameter.device
                if bool(group.get("capturable")) or bool(group.get("fused"))
                else torch.device("cpu")
            )
            state["step"] = torch.zeros((), dtype=torch.float32, device=step_device)
            state["exp_avg"] = torch.zeros_like(
                parameter, memory_format=torch.preserve_format
            )
            state["exp_avg_sq"] = torch.zeros_like(
                parameter, memory_format=torch.preserve_format
            )
            if bool(group.get("amsgrad")):
                state["max_exp_avg_sq"] = torch.zeros_like(
                    parameter, memory_format=torch.preserve_format
                )
    count = optimizer_state_entry_count(optimizer)
    if count <= 0:
        raise RuntimeError("failed to materialize the FSDP Adam state template")
    return count


def checkpoint_due(update: int, training_config: Mapping[str, Any]) -> bool:
    """Return whether an update is a periodic or explicitly pinned save gate."""

    if not isinstance(update, int) or isinstance(update, bool) or update <= 0:
        raise ValueError("update must be a positive integer")
    periodic = int(training_config["save_every_updates"])
    pinned = training_config.get("save_at_updates", [])
    return update % periodic == 0 or update in pinned


def checkpoint_epoch_cursor(
    *,
    update: int,
    epoch: int,
    raw_batches_consumed_in_epoch: int,
    usable_batches_consumed_in_epoch: int,
    raw_batch_cursor_sha256: str,
    epoch_dataloader_generator_state: torch.Tensor,
    validated_epoch_updates: Sequence[int] | None,
    gradient_accumulation_steps: int,
    next_epoch_dataloader_generator_state: torch.Tensor,
) -> tuple[int, int, int, str, torch.Tensor, bool]:
    """Normalize a signed epoch boundary to the next zero-cost resume cursor.

    A checkpoint is written immediately after an optimizer update, before the
    Python iterator gets its final ``StopIteration``.  At a signed cumulative
    epoch boundary we nevertheless know that the just-consumed batch is the
    final one.  Persisting the old epoch/full cursor would force every sample
    to be decoded again during resume replay.  Mid-epoch cursors are returned
    byte-for-byte unchanged.
    """

    current = (
        int(epoch),
        int(raw_batches_consumed_in_epoch),
        int(usable_batches_consumed_in_epoch),
        str(raw_batch_cursor_sha256),
        epoch_dataloader_generator_state,
        False,
    )
    if validated_epoch_updates is None or epoch >= len(validated_epoch_updates):
        return current
    boundary_update = sum(int(value) for value in validated_epoch_updates[: epoch + 1])
    if update != boundary_update:
        return current
    expected_raw_batches = (
        int(validated_epoch_updates[epoch]) * int(gradient_accumulation_steps)
    )
    if raw_batches_consumed_in_epoch != expected_raw_batches:
        raise RuntimeError(
            "signed epoch boundary has an unexpected raw batch cursor: "
            f"epoch={epoch}, update={update}, expected={expected_raw_batches}, "
            f"observed={raw_batches_consumed_in_epoch}"
        )
    if usable_batches_consumed_in_epoch != expected_raw_batches:
        raise RuntimeError(
            "signed conditional epoch boundary contains skipped batches"
        )
    return (
        epoch + 1,
        0,
        0,
        hashlib.sha256().hexdigest(),
        next_epoch_dataloader_generator_state.clone(),
        True,
    )


def training_run_identity(
    effective_config: Mapping[str, Any],
    *,
    overfit_indices: Sequence[int],
    fixed_noise_level: float | None,
    overfit_mode: str = "standard",
) -> dict[str, Any]:
    """Return every semantic training input that must match on resume.

    Runtime-only stop targets and output paths are deliberately absent. Overfit
    selection is not part of the streaming dataset summary, so it must be
    signed explicitly to prevent resuming sample A on sample B.
    """

    return {
        "schema_version": 1,
        "effective_config": dict(effective_config),
        "overfit_mode": str(overfit_mode),
        "overfit_sample_indices": [int(index) for index in overfit_indices],
        "fixed_noise_level": (
            None if fixed_noise_level is None else float(fixed_noise_level)
        ),
    }


def validate_dataset_provenance(
    config: Mapping[str, Any], summary: Mapping[str, Any]
) -> None:
    """Bind the streaming loader to the exact configured dataset revision."""

    data = config["data"]
    if data.get("exclude_eval_video_groups") is not True:
        raise ValueError("data.exclude_eval_video_groups must be true")
    expected_revision = str(data["dataset_revision"])
    actual_revision = summary.get("revision")
    if actual_revision != expected_revision:
        raise ValueError(
            "streaming dataset revision mismatch: "
            f"config={expected_revision!r}, manifest={actual_revision!r}"
        )
    loader = data.get("loader", "legacy_local_tar")
    if loader == "conditional_local_tar":
        if summary.get("loader") not in (
            "conditional_local_tar",
            "ConditionalLocalTarDataset",
        ):
            raise ValueError("conditional dataset summary reports the wrong loader")
        if summary.get("balance_distributed_shards") is not True:
            raise ValueError(
                "conditional dataset did not confirm balanced distributed shards"
            )
        for digest_name in (
            "metadata_index_sha256",
            "source_metadata_sha256",
            "split_manifest_sha256",
            "tokenizer_sha256",
            "preprocessing_code_sha256",
        ):
            digest = summary.get(digest_name)
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(
                    f"conditional dataset summary has invalid {digest_name}"
                )
        total_train_rows = summary.get("total_train_rows")
        any_valid_rows = summary.get("any_valid_rows")
        if (
            not isinstance(total_train_rows, int)
            or isinstance(total_train_rows, bool)
            or total_train_rows <= 0
            or not isinstance(any_valid_rows, int)
            or isinstance(any_valid_rows, bool)
            or not 0 < any_valid_rows <= total_train_rows
        ):
            raise ValueError("conditional dataset summary has invalid train row counts")
        return
    if loader != "legacy_local_tar":
        raise ValueError(f"unsupported data.loader: {loader!r}")
    excluded_groups = summary.get("excluded_eval_video_groups")
    if not isinstance(excluded_groups, int) or excluded_groups <= 0:
        raise ValueError(
            "streaming dataset did not confirm any excluded evaluation video groups"
        )


def enable_verified_gradient_checkpointing(autoregressive_model: nn.Module) -> None:
    """Enable and verify Transformers' layer-level checkpointing contract.

    ``XllmDecoderLayer`` inherits ``GradientCheckpointingLayer``. In current
    Transformers versions its overridden ``__call__`` invokes the assigned
    checkpoint function; the parent model's Python loop therefore should *not*
    call ``_gradient_checkpointing_func`` a second time.
    """

    base = _base_autoregressive_model(autoregressive_model)
    language_model = base.model.language_model
    base.gradient_checkpointing_enable()
    if not getattr(language_model, "gradient_checkpointing", False):
        raise UnsupportedTrainingFeatureError(
            "gradient_checkpointing_enable() did not activate the language model"
        )
    inactive_layers = [
        index
        for index, layer in enumerate(language_model.layers)
        if not getattr(layer, "gradient_checkpointing", False)
        or not hasattr(layer, "_gradient_checkpointing_func")
    ]
    if inactive_layers:
        raise UnsupportedTrainingFeatureError(
            f"gradient checkpointing is inactive on decoder layers {inactive_layers[:8]}"
        )


def _resolve_repository_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def build_streaming_dataset(
    config: Mapping[str, Any],
    *,
    rank: int = 0,
    world_size: int = 1,
    validate_file_sizes: bool = True,
    shuffle_buffer_size: int | None = None,
):
    """Instantiate the bounded local TAR/Parquet IterableDataset."""

    data = config["data"]
    loader = data.get("loader", "legacy_local_tar")
    if loader == "conditional_local_tar":
        from audio.diffusion_data import ConditionalLocalTarDataset

        optional_kwargs: dict[str, Any] = {}
        for name in (
            "validate_sidecar_hashes",
            "validate_tar_files",
            "balance_distributed_shards",
            "max_audio_bytes",
        ):
            if name in data:
                optional_kwargs[name] = data[name]
        # ``validate_file_sizes`` is the legacy CLI spelling. For the signed
        # conditional index it controls TAR stat validation while sidecar hash
        # validation remains independently enabled by default.
        optional_kwargs.setdefault("validate_tar_files", validate_file_sizes)
        return ConditionalLocalTarDataset(
            local_root=_resolve_repository_path(data["local_root"]),
            metadata_index_path=_resolve_repository_path(
                data["conditional_metadata_index"]
            ),
            view_seed=int(data["view_seed"]),
            seed=int(data["seed"]),
            shuffle_buffer_size=(
                int(data["shuffle_buffer_size"])
                if shuffle_buffer_size is None
                else int(shuffle_buffer_size)
            ),
            parquet_batch_size=int(data["parquet_batch_size"]),
            target_sample_rate=int(data["target_sample_rate"]),
            rank=rank,
            world_size=world_size,
            strict_missing_members=True,
            error_policy="raise",
            **optional_kwargs,
        )
    if loader != "legacy_local_tar":
        raise ValueError(f"unsupported data.loader: {loader!r}")

    from audio.diffusion_data import LocalTarParquetASRDataset

    return LocalTarParquetASRDataset(
        local_root=_resolve_repository_path(data["local_root"]),
        subset_manifest_path=_resolve_repository_path(data["subset_manifest"]),
        eval_manifest_path=_resolve_repository_path(data["eval_manifest"]),
        seed=int(data["seed"]),
        shuffle_buffer_size=(
            int(data["shuffle_buffer_size"])
            if shuffle_buffer_size is None
            else int(shuffle_buffer_size)
        ),
        parquet_batch_size=int(data["parquet_batch_size"]),
        target_sample_rate=int(data["target_sample_rate"]),
        rank=rank,
        world_size=world_size,
        validate_file_sizes=validate_file_sizes,
        strict_missing_members=True,
        error_policy="raise",
        sample_exclusion_manifest_path=_resolve_repository_path(
            data["target_overflow_exclusion_manifest"]
        ),
        expected_excluded_samples=int(data["expected_target_overflow_exclusions"]),
    )


def validate_conditional_epoch_update_plan(
    dataset: Any,
    training_config: Mapping[str, Any],
    layout: Any,
) -> list[int] | None:
    """Bind the frozen two-epoch scheduler to signed per-epoch data quotas."""

    expected = training_config.get("expected_epoch_updates")
    if expected is None:
        return None
    supported_world_sizes = training_config.get("supported_world_sizes")
    required_world_size = (
        None if supported_world_sizes is None else max(supported_world_sizes)
    )
    if required_world_size is not None and int(layout.world_size) != int(required_world_size):
        # DDP32/DDP64 are bounded communication/I/O gates. TAR-level trimming
        # can yield a slightly different full-epoch length at those topologies;
        # only the declared production topology is bound to the two-epoch plan.
        return None
    if not isinstance(expected, list) or not expected:
        raise ValueError("training.expected_epoch_updates must be a non-empty list")
    if not hasattr(dataset, "set_epoch") or not hasattr(
        dataset, "assigned_sample_quota"
    ):
        raise TypeError(
            "expected_epoch_updates requires dataset.set_epoch and "
            "dataset.assigned_sample_quota"
        )
    samples_per_update = (
        int(layout.per_device_batch_size)
        * int(layout.gradient_accumulation_steps)
    )
    observed: list[int] = []
    try:
        for epoch, expected_updates in enumerate(expected):
            dataset.set_epoch(epoch)
            quota = dataset.assigned_sample_quota(worker_id=0, num_workers=1)
            if not isinstance(quota, int) or isinstance(quota, bool) or quota <= 0:
                raise ValueError(
                    f"conditional dataset returned invalid epoch-{epoch} quota: {quota!r}"
                )
            if quota % samples_per_update:
                raise ValueError(
                    "conditional per-rank quota must end on a synchronized optimizer "
                    f"boundary: epoch={epoch}, quota={quota}, "
                    f"samples_per_update_per_rank={samples_per_update}"
                )
            updates = quota // samples_per_update
            observed.append(updates)
            if updates != expected_updates:
                raise ValueError(
                    "signed conditional data no longer matches the frozen epoch plan: "
                    f"epoch={epoch}, expected_updates={expected_updates}, "
                    f"observed_updates={updates}, quota={quota}"
                )
    finally:
        dataset.set_epoch(0)
    return observed


class RepeatedOverfitExamplesDataset(IterableDataset):
    """Repeat one example, or alternate two examples, entirely in memory."""

    def __init__(
        self,
        samples: Sequence[Mapping[str, Any]],
        sample_indices: Sequence[int],
    ) -> None:
        super().__init__()
        if len(samples) not in (1, 2) or len(samples) != len(sample_indices):
            raise ValueError("overfit repetition requires one or two indexed examples")
        indices = tuple(int(index) for index in sample_indices)
        if any(index < 0 for index in indices):
            raise ValueError("overfit sample indices must be >= 0")
        if len(set(indices)) != len(indices):
            raise ValueError("overfit sample indices must be distinct")
        self.samples = tuple(dict(sample) for sample in samples)
        self.sample_indices = indices

    def __iter__(self):
        while True:
            for sample in self.samples:
                # The collator treats records as read-only. A shallow copy
                # protects scalar metadata while reusing bounded waveforms.
                yield dict(sample)


class RepeatedSingleExampleDataset(RepeatedOverfitExamplesDataset):
    """Backward-compatible one-example specialization."""

    def __init__(self, sample: Mapping[str, Any], sample_index: int) -> None:
        super().__init__((sample,), (sample_index,))
        self.sample = self.samples[0]
        self.sample_index = self.sample_indices[0]


def conditional_four_view_records(
    sample: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Expand one signed conditional record into its deterministic 2x2 views.

    The streaming loader selects one view per epoch, but retains all four
    signed targets and their token/canvas metadata.  P0 intentionally cycles
    those four targets for one frozen waveform in the canonical view order
    ``plain, diacritics, code_switch, code_switch_diacritics``.
    """

    from audio.diffusion_data import TARGET_COLUMNS, VIEW_NAMES, VIEW_REQUESTS

    valid_view_mask = sample.get("valid_view_mask")
    applied_mask = sample.get("applied_code_switching_mask")
    token_counts = sample.get("target_token_counts")
    canvas_sizes = sample.get("canvas_sizes")
    vectors = {
        "valid_view_mask": valid_view_mask,
        "applied_code_switching_mask": applied_mask,
        "target_token_counts": token_counts,
        "canvas_sizes": canvas_sizes,
    }
    for name, value in vectors.items():
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError(
                f"conditional four-view overfit requires signed {name}[4]"
            )
    if not all(isinstance(value, (bool, np.bool_)) for value in valid_view_mask):
        raise TypeError("conditional valid_view_mask entries must be bool")
    if not all(bool(value) for value in valid_view_mask):
        raise ValueError(
            "conditional four-view overfit requires a sample with all four valid views"
        )
    if not all(isinstance(value, (bool, np.bool_)) for value in applied_mask):
        raise TypeError("conditional applied_code_switching_mask entries must be bool")

    content = sample.get("content_has_code_switching")
    if not isinstance(content, (bool, np.bool_)):
        raise TypeError("conditional content_has_code_switching must be bool")
    country_code = sample.get("country_code")
    if not isinstance(country_code, str) or not country_code.strip():
        raise ValueError("conditional four-view sample requires country_code")

    records: list[dict[str, Any]] = []
    for view_id, (target_field, view_name, requests) in enumerate(
        zip(TARGET_COLUMNS, VIEW_NAMES, VIEW_REQUESTS, strict=True)
    ):
        transcription = sample.get(target_field)
        if not isinstance(transcription, str) or not transcription.strip():
            raise ValueError(
                f"conditional four-view sample has empty target {target_field}"
            )
        request_code_switching, request_diacritics = requests
        applied_code_switching = bool(applied_mask[view_id])
        if applied_code_switching and not (
            bool(content) and request_code_switching
        ):
            raise ValueError(
                "signed applied_code_switching_mask is inconsistent with the view"
            )
        record = dict(sample)
        record.update(
            {
                "target_transcription": transcription,
                "transcription": transcription,
                "request_code_switching": request_code_switching,
                "request_diacritics": request_diacritics,
                "applied_code_switching": applied_code_switching,
                "view_id": view_id,
                "view_name": view_name,
                "target_token_count": int(token_counts[view_id]),
                "canvas_size": int(canvas_sizes[view_id]),
            }
        )
        nested_target = dict(sample.get("target") or {})
        nested_target.update(
            {
                "target_transcription": transcription,
                "transcription": transcription,
                "country_code": country_code,
                "content_has_code_switching": bool(content),
                "has_code_switching": bool(content),
                "request_code_switching": request_code_switching,
                "request_diacritics": request_diacritics,
                "applied_code_switching": applied_code_switching,
                "view_id": view_id,
                "view_name": view_name,
            }
        )
        record["target"] = nested_target
        records.append(record)
    return tuple(records)  # type: ignore[return-value]


class RepeatedConditionalFourViewDataset(IterableDataset):
    """Cycle all four signed targets of one waveform forever, in fixed order."""

    def __init__(self, sample: Mapping[str, Any], sample_index: int) -> None:
        super().__init__()
        if int(sample_index) < 0:
            raise ValueError("overfit sample index must be >= 0")
        self.sample_index = int(sample_index)
        self.views = conditional_four_view_records(sample)

    def __iter__(self):
        while True:
            for view in self.views:
                yield dict(view)


def select_overfit_samples(
    dataset: Iterable[Mapping[str, Any]],
    sample_indices: Sequence[int],
) -> tuple[Mapping[str, Any], ...]:
    """Select one or two requested records during one deterministic traversal."""

    indices = tuple(int(index) for index in sample_indices)
    if len(indices) not in (1, 2):
        raise ValueError("overfit selection requires one or two sample indices")
    if any(index < 0 for index in indices):
        raise ValueError("overfit sample indices must be >= 0")
    if len(set(indices)) != len(indices):
        raise ValueError("overfit sample indices must be distinct")

    requested = set(indices)
    selected: dict[int, Mapping[str, Any]] = {}
    for index, sample in enumerate(dataset):
        if index in requested:
            selected[index] = sample
            if len(selected) == len(requested):
                return tuple(selected[index] for index in indices)
    missing = [index for index in indices if index not in selected]
    raise IndexError(f"stream ended before overfit sample indices {missing}")


def select_overfit_sample(
    dataset: Iterable[Mapping[str, Any]], sample_index: int
) -> Mapping[str, Any]:
    """Backward-compatible single-record selector."""

    return select_overfit_samples(dataset, (sample_index,))[0]


def requested_overfit_sample_indices(args: argparse.Namespace) -> tuple[int, ...]:
    """Normalize standard and conditional-four-view overfit selections."""

    single = getattr(args, "overfit_sample_index", None)
    pair = getattr(args, "overfit_sample_indices", None)
    four_view = getattr(args, "overfit_four_view_sample_index", None)
    if sum(value is not None for value in (single, pair, four_view)) > 1:
        raise ValueError(
            "overfit sample selection options are mutually exclusive"
        )
    if four_view is not None:
        indices = (int(four_view),)
    else:
        indices = (
            (int(single),)
            if single is not None
            else tuple(int(i) for i in pair or ())
        )
    if len(indices) not in (0, 1, 2):
        raise ValueError("overfit mode requires one index or exactly two indices")
    if any(index < 0 for index in indices):
        raise ValueError("overfit sample indices must be >= 0")
    if len(set(indices)) != len(indices):
        raise ValueError("overfit sample indices must be distinct")
    return indices


def requested_overfit_mode(args: argparse.Namespace) -> str:
    """Return the resume-signed repetition semantics selected by the CLI."""

    indices = requested_overfit_sample_indices(args)
    if getattr(args, "overfit_four_view_sample_index", None) is not None:
        return "conditional_four_view"
    return "standard" if indices else "none"


def overfit_sample_metadata(
    samples: Sequence[Mapping[str, Any]],
    sample_indices: Sequence[int],
) -> list[dict[str, Any]]:
    """Return the non-transcript provenance saved in resolved run configs."""

    return [
        {
            "index": int(index),
            "sample_key": sample.get("sample_key"),
            "duration_sec": sample.get("duration_sec"),
            "country_code": sample.get("country_code"),
        }
        for index, sample in zip(sample_indices, samples, strict=True)
    ]


def validate_tokenized_overfit_samples(
    collator: MawJaisDiffusionCollator,
    samples: Sequence[Mapping[str, Any]],
    sample_indices: Sequence[int],
) -> list[dict[str, Any]]:
    """Validate overfit records after tokenizer/model loading.

    A paired audio-permutation test must not be identifiable from the canvas
    size alone, so paired samples are required to share the same bucket.
    """

    metadata = overfit_sample_metadata(samples, sample_indices)
    target_token_ids: list[list[int]] = []
    canvas_sizes: list[int] = []
    prompt_tags: list[str | None] = []
    for sample, index, item in zip(samples, sample_indices, metadata, strict=True):
        probe = collator([sample])
        if probe.get("skip_batch") is True:
            raise CanvasOverflowError(
                f"overfit sample {index} does not fit its duration-selected canvas; "
                "choose another deterministic index"
            )
        target_result = collator._target_ids_and_text(sample)
        token_ids, target_text = target_result[:2]
        target_token_ids.append(token_ids)
        canvas_size = int(probe["canvas_sizes"][0].item())
        canvas_sizes.append(canvas_size)
        probe_prompt_tags = probe.get("prompt_tags")
        prompt_tag = (
            str(probe_prompt_tags[0])
            if isinstance(probe_prompt_tags, Sequence) and probe_prompt_tags
            else None
        )
        prompt_tags.append(prompt_tag)
        item.update(
            {
                "target_text": target_text,
                "target_token_ids": token_ids,
                "target_token_count": len(token_ids),
                "canvas_size": canvas_size,
                "prompt_tag": prompt_tag,
            }
        )

    if len(samples) == 2:
        if target_token_ids[0] == target_token_ids[1]:
            raise ValueError(
                "paired overfit samples must have different tokenized targets; "
                f"indices {list(sample_indices)} are indistinguishable"
            )
        if canvas_sizes[0] != canvas_sizes[1]:
            raise ValueError(
                "paired overfit samples must use the same duration-selected canvas "
                "bucket so an audio permutation cannot be identified from canvas size; "
                f"indices {list(sample_indices)} use buckets {canvas_sizes}"
            )
        if prompt_tags[0] != prompt_tags[1]:
            raise ValueError(
                "paired overfit samples must use the same prompt tag so an audio "
                "permutation cannot be identified from target-derived prompt metadata; "
                f"indices {list(sample_indices)} use prompt tags {prompt_tags}"
            )
    return metadata


def validate_tokenized_conditional_four_view_sample(
    collator: MawJaisDiffusionCollator,
    sample: Mapping[str, Any],
    sample_index: int,
) -> list[dict[str, Any]]:
    """Fail fast on every signed P0 target before the first optimizer update."""

    if not collator.conditional_mode:
        raise ValueError("conditional four-view overfit requires conditional collator mode")
    views = conditional_four_view_records(sample)
    view_metadata: list[dict[str, Any]] = []
    for expected_view_id, view in enumerate(views):
        probe = collator([view])
        if probe.get("skip_batch") is True:
            raise CanvasOverflowError(
                "conditional four-view overfit sample does not fit its signed canvas: "
                f"index={sample_index}, view_id={expected_view_id}"
            )
        validate_conditional_batch_fields(probe)
        observed_view_id = int(probe["view_ids"][0].item())
        if observed_view_id != expected_view_id:
            raise ValueError(
                "conditional four-view cycle changed order: "
                f"expected={expected_view_id}, observed={observed_view_id}"
            )
        token_ids, target_text, transcription = collator._target_ids_and_text(view)
        prompt_tags = probe.get("prompt_tags")
        view_metadata.append(
            {
                "view_id": expected_view_id,
                "view_name": view["view_name"],
                "target_text": target_text,
                "target_transcription": transcription,
                "target_token_ids": token_ids,
                "target_token_count": len(token_ids),
                "canvas_size": int(probe["canvas_sizes"][0].item()),
                "prompt_tag": str(prompt_tags[0]),
            }
        )

    metadata = overfit_sample_metadata((sample,), (sample_index,))[0]
    metadata.update(
        {
            "cycle": [0, 1, 2, 3],
            "views": view_metadata,
        }
    )
    return [metadata]


def save_resolved_config(config: Mapping[str, Any], output_dir: str | Path) -> Path:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("PyYAML is required to save the resolved run config") from error
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "resolved_config.yaml"
    temporary = output / "resolved_config.yaml.tmp"
    temporary.write_text(
        yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def optional_dependency_report(strategy: str) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for package in ("accelerate", "peft"):
        present = importlib.util.find_spec(package) is not None
        version = None
        if present:
            try:
                version = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        report[package] = {"present": present, "version": version}
    report["ready_for_strategy"] = bool(
        report["accelerate"]["present"]
        and (strategy != "lora" or report["peft"]["present"])
    )
    return report


def require_training_dependencies(strategy: str):
    try:
        from accelerate import Accelerator
    except ImportError as error:
        raise RuntimeError(
            "Training requires Accelerate. Install it in the ROCm training environment."
        ) from error
    if strategy == "lora":
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as error:
            raise RuntimeError(
                "llm_train_strategy='lora' requires PEFT, but PEFT is not installed. "
                "Install a Transformers-compatible `peft` package before loading the 4B model."
            ) from error
        return Accelerator, LoraConfig, TaskType, get_peft_model
    return Accelerator, None, None, None


def uses_fsdp_hybrid_shard(training_config: Mapping[str, Any]) -> bool:
    """Return whether this run uses the audited full-finetuning sharding path."""

    return training_config.get("distributed_strategy", "ddp") == "fsdp_hybrid_shard"


def create_training_accelerator(
    Accelerator: type,
    training_config: Mapping[str, Any],
):
    """Create DDP or FSDP without relying on mutable Accelerate CLI state.

    Full fine-tuning keeps FP32 master shards while FSDP casts forward/backward
    parameters to BF16.  The two frozen audio modules are explicitly ignored:
    they stay replicated in BF16, eval-only and outside optimizer sharding.
    ``HYBRID_SHARD`` shards within each eight-MI210 node and replicates those
    shards across nodes, avoiding a full-parameter all-gather over InfiniBand.
    """

    kwargs: dict[str, Any] = {
        "mixed_precision": str(training_config["mixed_precision"]),
        "gradient_accumulation_steps": int(
            training_config["gradient_accumulation_steps"]
        ),
    }
    if uses_fsdp_hybrid_shard(training_config):
        try:
            from accelerate import FullyShardedDataParallelPlugin
        except ImportError as error:
            raise RuntimeError(
                "fsdp_hybrid_shard requires Accelerate's FSDP plugin"
            ) from error
        kwargs["fsdp_plugin"] = FullyShardedDataParallelPlugin(
            fsdp_version=1,
            sharding_strategy="HYBRID_SHARD",
            backward_prefetch="BACKWARD_PRE",
            auto_wrap_policy="size_based_wrap",
            min_num_params=int(training_config["fsdp_min_num_params"]),
            state_dict_type="SHARDED_STATE_DICT",
            use_orig_params=True,
            sync_module_states=False,
            cpu_offload=False,
            forward_prefetch=False,
            activation_checkpointing=False,
            limit_all_gathers=True,
            ignored_modules=(
                r"autoregressive_model\.model\.(audio_tower|embed_audio)"
            ),
        )
    return Accelerator(**kwargs)


def resolve_resume_checkpoint_distributed(
    request: str | Path | None,
    *,
    output_dir: Path,
    accelerator: Any,
) -> ResumeCheckpoint | None:
    """Authenticate a large FSDP checkpoint once, then share its path.

    Rank zero performs the full content hash (about 58 GiB for the current
    4B full-finetuning state). Other ranks validate the manifest and small
    control artifacts only. PyTorch DCP subsequently reads each rank's shard.
    """

    if accelerator.num_processes == 1:
        return resolve_resume_checkpoint(request, output_dir=output_dir)
    selected: ResumeCheckpoint | None = None
    payload: list[Any] = [None, None]
    if accelerator.is_main_process:
        try:
            selected = resolve_resume_checkpoint(request, output_dir=output_dir)
            payload[0] = None if selected is None else str(selected.path)
        except Exception as error:  # broadcast a uniform failure to all ranks
            payload[1] = f"{type(error).__name__}: {error}"
    torch.distributed.broadcast_object_list(
        payload,
        src=0,
        device=accelerator.device,
    )
    if payload[1] is not None:
        raise ValueError(f"distributed resume resolution failed: {payload[1]}")
    if payload[0] is None:
        return None
    if accelerator.is_main_process:
        if selected is None or str(selected.path) != payload[0]:
            raise RuntimeError("rank-zero resume selection changed before broadcast")
        return selected
    return inspect_resume_checkpoint(
        payload[0],
        verify_trained_model_hash=False,
    )


def resolve_weight_continuation_distributed(
    request: str | Path,
    *,
    output_dir: Path,
    child_config: Mapping[str, Any],
    expected_parent_update: int,
    accelerator: Any,
) -> TrainingFork:
    """Authenticate a large continuation parent once across all ranks."""

    if accelerator.num_processes == 1:
        return resolve_weight_continuation(
            request,
            output_dir=output_dir,
            child_config=child_config,
            expected_parent_update=expected_parent_update,
        )
    selected: TrainingFork | None = None
    payload: list[Any] = [None, None]
    if accelerator.is_main_process:
        try:
            selected = resolve_weight_continuation(
                request,
                output_dir=output_dir,
                child_config=child_config,
                expected_parent_update=expected_parent_update,
            )
            payload[0] = str(selected.checkpoint.path)
        except Exception as error:
            payload[1] = f"{type(error).__name__}: {error}"
    torch.distributed.broadcast_object_list(
        payload,
        src=0,
        device=accelerator.device,
    )
    if payload[1] is not None:
        raise ValueError(f"distributed continuation resolution failed: {payload[1]}")
    if not isinstance(payload[0], str):
        raise RuntimeError("rank zero did not publish a continuation checkpoint")
    if accelerator.is_main_process:
        if selected is None or str(selected.checkpoint.path) != payload[0]:
            raise RuntimeError("rank-zero continuation selection changed before broadcast")
        return selected
    return resolve_weight_continuation(
        payload[0],
        output_dir=output_dir,
        child_config=child_config,
        expected_parent_update=expected_parent_update,
        verify_trained_model_hash=False,
    )


def load_fsdp_weight_continuation_model(
    *,
    accelerator: Any,
    prepared_model: nn.Module,
    continuation: TrainingFork,
    loader: Callable[..., Any] | None = None,
) -> Any:
    """Load only FSDP model shards, leaving the new optimizer untouched."""

    if continuation.checkpoint.manifest.get("checkpoint_format") != (
        "accelerate_fsdp_sharded_state_dict"
    ):
        raise ValueError("weight-continuation parent is not an FSDP sharded checkpoint")
    fsdp_plugin = getattr(accelerator.state, "fsdp_plugin", None)
    if fsdp_plugin is None:
        raise RuntimeError("FSDP weight continuation requires an active FSDP plugin")
    if loader is None:
        from accelerate.utils import load_fsdp_model

        loader = load_fsdp_model
    load_result = loader(
        fsdp_plugin,
        accelerator,
        prepared_model,
        str(continuation.checkpoint.trained_model),
    )
    missing = list(getattr(load_result, "missing_keys", ()) or ())
    unexpected = list(getattr(load_result, "unexpected_keys", ()) or ())
    if missing or unexpected:
        raise RuntimeError(
            "FSDP continuation model-state mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    accelerator.wait_for_everyone()
    return load_result


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainingRNGStreams:
    """Independent RNGs so the eight experiments share t/corruption masks."""

    time: torch.Generator
    corruption: torch.Generator
    uniform_prior: torch.Generator
    sampler: torch.Generator
    dataloader: torch.Generator
    seeds: dict[str, int]

    @classmethod
    def create(
        cls,
        training_config: Mapping[str, Any],
        *,
        device: torch.device,
        rank: int = 0,
    ) -> "TrainingRNGStreams":
        # A rank offset avoids identical random examples across ranks while
        # remaining identical across the eight experiment variants.
        seeds = {
            "time": int(training_config["time_seed"]) + rank,
            "corruption": int(training_config["corruption_seed"]) + rank,
            "uniform_prior": int(training_config["uniform_prior_seed"]) + rank,
            "sampler": int(training_config["sampler_seed"]) + rank,
            "dataloader": int(training_config["dataloader_seed"]) + rank,
        }
        generator_device = device if device.type != "mps" else torch.device("cpu")
        return cls(
            time=torch.Generator(device=generator_device).manual_seed(seeds["time"]),
            corruption=torch.Generator(device=generator_device).manual_seed(seeds["corruption"]),
            uniform_prior=torch.Generator(device=generator_device).manual_seed(
                seeds["uniform_prior"]
            ),
            sampler=torch.Generator(device=generator_device).manual_seed(seeds["sampler"]),
            # DataLoader only accepts a CPU generator for worker base seeds.
            dataloader=torch.Generator(device="cpu").manual_seed(seeds["dataloader"]),
            seeds=seeds,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "seeds": dict(self.seeds),
            "time": self.time.get_state().cpu(),
            "corruption": self.corruption.get_state().cpu(),
            "uniform_prior": self.uniform_prior.get_state().cpu(),
            "sampler": self.sampler.get_state().cpu(),
            "dataloader": self.dataloader.get_state().cpu(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected_keys = {
            "seeds",
            "time",
            "corruption",
            "uniform_prior",
            "sampler",
            "dataloader",
        }
        if set(state) != expected_keys:
            raise ValueError(
                "training RNG state keys mismatch: "
                f"expected={sorted(expected_keys)}, actual={sorted(state)}"
            )
        if state["seeds"] != self.seeds:
            raise ValueError(
                f"training RNG seeds mismatch: {state['seeds']} != {self.seeds}"
            )
        for name in expected_keys.difference({"seeds"}):
            saved = state[name]
            if not isinstance(saved, torch.Tensor) or saved.dtype != torch.uint8:
                raise ValueError(f"invalid generator state for {name}")
            getattr(self, name).set_state(saved.cpu())


def capture_global_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def restore_global_rng_state(state: Mapping[str, Any]) -> None:
    expected_keys = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != expected_keys:
        raise ValueError("global RNG state has an unexpected schema")
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, Mapping) or set(numpy_state) != {
        "bit_generator",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian",
    }:
        raise ValueError("NumPy RNG state has an unexpected schema")
    numpy_keys = numpy_state["keys"]
    if not isinstance(numpy_keys, torch.Tensor):
        raise ValueError("NumPy RNG keys must be a tensor")
    cpu_state = state["torch_cpu"]
    cuda_states = state["torch_cuda"]
    if not isinstance(cpu_state, torch.Tensor) or not isinstance(cuda_states, list):
        raise ValueError("torch RNG state has an unexpected schema")
    if cuda_states:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint contains CUDA/HIP RNG state but no device is visible")
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "CUDA/HIP RNG device-count mismatch: "
                f"checkpoint={len(cuda_states)}, current={torch.cuda.device_count()}"
            )

    random.setstate(state["python"])
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            numpy_keys.cpu().numpy().astype(np.uint32, copy=False),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(cpu_state.cpu())
    if cuda_states:
        torch.cuda.set_rng_state_all([saved.cpu() for saved in cuda_states])


def _save_training_loop_state(
    path: Path,
    *,
    optimizer: Any,
    scheduler: Any,
    gradient_scaler: Any | None,
    rng: TrainingRNGStreams,
    update: int,
    epoch: int,
    raw_batches_consumed_in_epoch: int,
    usable_batches_consumed_in_epoch: int,
    raw_batch_cursor_sha256: str,
    epoch_dataloader_generator_state: torch.Tensor,
    overflow_counters: Mapping[str, int],
    last_reported_loss: float | None,
    elapsed_seconds: float,
    identities: Mapping[str, str],
    rank_states: Sequence[Mapping[str, Any]] | None = None,
    max_memory_allocated_gib: float | None = None,
    optimizer_state_external: bool = False,
) -> Path:
    if rank_states is None:
        rank_states = [
            {
                "rank": 0,
                "world_size": 1,
                "epoch": int(epoch),
                "raw_batches_consumed_in_epoch": int(raw_batches_consumed_in_epoch),
                "usable_batches_consumed_in_epoch": int(usable_batches_consumed_in_epoch),
                "raw_batch_cursor_sha256": str(raw_batch_cursor_sha256),
                "epoch_dataloader_generator_state": epoch_dataloader_generator_state.cpu(),
                "training_rng": rng.state_dict(),
                "global_rng": capture_global_rng_state(),
                "overflow_counters": dict(overflow_counters),
            }
        ]
    rank_states = [dict(state) for state in rank_states]
    world_size = len(rank_states)
    validate_rank_states(rank_states, world_size=world_size)
    payload = {
        "schema_version": RESUME_STATE_SCHEMA_VERSION,
        "update": int(update),
        "world_size": world_size,
        "rank_states": rank_states,
        # FSDP uses Accelerate/PyTorch distributed checkpointing. Calling the
        # ordinary optimizer state_dict on rank zero would either materialize
        # every shard or save an incomplete state. Keep the cursor schema
        # stable while recording that these three states live in full_model/.
        "optimizer": None if optimizer_state_external else optimizer.state_dict(),
        "scheduler": None if optimizer_state_external else scheduler.state_dict(),
        "gradient_scaler": (
            None
            if optimizer_state_external or gradient_scaler is None
            else gradient_scaler.state_dict()
        ),
        "last_reported_loss": last_reported_loss,
        "elapsed_seconds": float(elapsed_seconds),
        "max_memory_allocated_gib": (
            None
            if max_memory_allocated_gib is None
            else float(max_memory_allocated_gib)
        ),
        "identities": dict(identities),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def _load_training_loop_state(
    checkpoint: ResumeCheckpoint,
    *,
    identities: Mapping[str, str],
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, Any]:
    payload = torch.load(
        checkpoint.training_state,
        map_location="cpu",
        weights_only=True,
    )
    legacy_required = {
        "schema_version",
        "update",
        "epoch",
        "raw_batches_consumed_in_epoch",
        "usable_batches_consumed_in_epoch",
        "raw_batch_cursor_sha256",
        "epoch_dataloader_generator_state",
        "optimizer",
        "scheduler",
        "gradient_scaler",
        "training_rng",
        "global_rng",
        "overflow_counters",
        "last_reported_loss",
        "elapsed_seconds",
        "identities",
    }
    distributed_required = {
        "schema_version",
        "update",
        "world_size",
        "rank_states",
        "optimizer",
        "scheduler",
        "gradient_scaler",
        "last_reported_loss",
        "elapsed_seconds",
        "max_memory_allocated_gib",
        "identities",
    }
    if not isinstance(payload, dict):
        raise ValueError("training loop state has an unexpected schema")
    if payload["schema_version"] == 1 and set(payload) == legacy_required:
        if (rank, world_size) != (0, 1):
            raise ValueError("legacy checkpoint can only resume with world_size=1")
        local_payload = payload
    elif (
        payload["schema_version"] == RESUME_STATE_SCHEMA_VERSION
        and set(payload) == distributed_required
    ):
        if payload["world_size"] != world_size:
            raise ValueError(
                "checkpoint world_size must match exactly on resume: "
                f"{payload['world_size']} != {world_size}"
            )
        rank_states = payload["rank_states"]
        if not isinstance(rank_states, list):
            raise ValueError("distributed checkpoint rank_states must be a list")
        validate_rank_states(rank_states, world_size=world_size)
        local_payload = {
            **payload,
            **dict(rank_states[rank]),
        }
        local_payload.pop("rank_states", None)
    else:
        raise ValueError("unsupported training loop state schema")
    if payload["update"] != checkpoint.update:
        raise ValueError("training state update does not match checkpoint manifest")
    if payload["identities"] != dict(identities):
        raise ValueError("training state identities do not match the current run")
    for name in (
        "update",
        "epoch",
        "raw_batches_consumed_in_epoch",
        "usable_batches_consumed_in_epoch",
    ):
        value = local_payload[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"training state {name} must be a non-negative integer")
    if not isinstance(local_payload["epoch_dataloader_generator_state"], torch.Tensor):
        raise ValueError("training state lacks the epoch DataLoader generator state")
    counters = local_payload["overflow_counters"]
    if not isinstance(counters, Mapping) or set(counters) != set(COLLATOR_METRIC_NAMES):
        raise ValueError("training state overflow counters have an unexpected schema")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in counters.values()
    ):
        raise ValueError("training state overflow counters must be non-negative integers")
    if (
        local_payload["usable_batches_consumed_in_epoch"]
        > local_payload["raw_batches_consumed_in_epoch"]
    ):
        raise ValueError("training state usable batch cursor exceeds raw cursor")
    if not isinstance(local_payload["raw_batch_cursor_sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", local_payload["raw_batch_cursor_sha256"]
    ):
        raise ValueError("training state raw batch cursor hash is invalid")
    return local_payload


def restore_gradient_scaler_state(
    gradient_scaler: Any | None,
    saved_state: Mapping[str, Any] | None,
) -> None:
    """Restore fp16 loss scaling without weakening bf16 resume checks."""

    if gradient_scaler is None:
        if saved_state is not None:
            raise ValueError(
                "checkpoint contains a gradient scaler but the current run has none"
            )
        return
    if saved_state is None:
        raise ValueError(
            "current run requires a gradient scaler but the checkpoint has none"
        )
    gradient_scaler.load_state_dict(dict(saved_state))


def update_raw_batch_cursor(cursor: Any, batch: Mapping[str, Any]) -> bool:
    """Hash batch order/overflow metadata and return whether it is usable."""

    scratch = {name: 0 for name in COLLATOR_METRIC_NAMES}
    skipped = accumulate_collator_metrics(batch, scratch)
    key_field = "skipped_sample_keys" if skipped else "sample_keys"
    sample_keys = batch.get(key_field)
    if not isinstance(sample_keys, (list, tuple)) or not all(
        isinstance(value, str) for value in sample_keys
    ):
        raise ValueError(f"collator output has invalid {key_field}")
    conditional_cursor: dict[str, list[int | bool]] | None = None
    if not skipped and "view_ids" in batch:
        validate_conditional_batch_fields(batch)
        conditional_cursor = {}
        for name in CONDITIONAL_BATCH_FIELDS:
            value = batch[name]
            conditional_cursor[name] = value.detach().cpu().tolist()
    payload = json.dumps(
        {
            "skip_batch": skipped,
            "sample_keys": list(sample_keys),
            "collator_metrics": scratch,
            "conditional": conditional_cursor,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    cursor.update(len(payload).to_bytes(8, "big"))
    cursor.update(payload)
    return not skipped


def replay_raw_dataloader_batches(
    iterator: Any,
    count: int,
    *,
    expected_usable_batches: int | None = None,
    cursor: Any | None = None,
) -> tuple[int, Any]:
    """Replay and verify exactly ``count`` collated batches without recounting."""

    if count < 0:
        raise ValueError("raw DataLoader replay count must be non-negative")
    if cursor is None:
        cursor = hashlib.sha256()
    usable = 0
    for index in range(count):
        try:
            batch = next(iterator)
        except StopIteration as error:
            raise RuntimeError(
                "DataLoader exhausted while replaying resume cursor: "
                f"replayed={index}, requested={count}"
            ) from error
        if expected_usable_batches is not None:
            if not isinstance(batch, Mapping):
                raise ValueError("resume replay requires collated mapping batches")
            usable += int(update_raw_batch_cursor(cursor, batch))
    if expected_usable_batches is not None and usable != expected_usable_batches:
        raise RuntimeError(
            "resume usable-batch cursor mismatch: "
            f"replayed={usable}, checkpoint={expected_usable_batches}"
        )
    return usable, cursor


def seed_dataloader_worker(_worker_id: int) -> None:
    """Seed Python/NumPy from PyTorch's deterministic per-worker base seed."""

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _base_autoregressive_model(model: nn.Module) -> nn.Module:
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def assert_model_attention_implementation(
    autoregressive_model: nn.Module,
    expected: str,
) -> None:
    """Fail if the loaded text-attention modules ignored the selected backend."""

    if expected not in ("eager", "sdpa"):
        raise ValueError(f"unsupported attention implementation: {expected!r}")
    base = _base_autoregressive_model(autoregressive_model)
    language_model = base.model.language_model
    implementations: set[str | None] = set()
    for module in language_model.modules():
        if not all(
            hasattr(module, name)
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        ):
            continue
        config = getattr(module, "config", None)
        implementations.add(
            None if config is None else getattr(config, "_attn_implementation", None)
        )
    if implementations != {expected}:
        raise RuntimeError(
            "loaded text-attention backend differs from training config: "
            f"expected={expected!r}, observed={sorted(map(str, implementations))}"
        )


def composite_attention_implementation(text_backend: str) -> dict[str, str]:
    """Select the gated text backend while preserving frozen audio features.

    The numerical gate compares eager/SDPA in the trainable text decoder using
    identical projected audio embeddings. Keeping the audio tower eager makes
    the production smoke exercise exactly that validated change.
    """

    if text_backend not in ("eager", "sdpa"):
        raise ValueError(f"unsupported attention implementation: {text_backend!r}")
    return {
        "": "eager",
        "text_config": text_backend,
        "audio_config": "eager",
    }


def _freeze_and_assert_audio(
    autoregressive_model: nn.Module, *, freeze: bool = True
) -> None:
    base = _base_autoregressive_model(autoregressive_model)
    if not hasattr(base, "freeze_audio_components"):
        raise TypeError("loaded model does not expose freeze_audio_components()")
    if freeze:
        base.freeze_audio_components()
    base.assert_audio_components_frozen()
    for name, parameter in base.named_parameters():
        if ".audio_tower." in f".{name}." or ".embed_audio." in f".{name}.":
            if parameter.requires_grad or parameter.grad is not None:
                raise AssertionError(f"audio parameter is not strictly frozen: {name}")


def enter_diffusion_training_mode(
    diffusion_model: nn.Module,
    *,
    unwrap_model: Callable[[nn.Module], nn.Module] | None = None,
) -> nn.Module:
    """Enter train mode and return the model that owns diffusion attributes.

    ``Accelerator.prepare`` may replace the outer diffusion module with an
    opaque DDP wrapper. Training mode must be set on that prepared wrapper so
    it propagates normally, while model-specific attributes must be accessed
    only after an explicit ``Accelerator.unwrap_model`` call.
    """

    diffusion_model.train()
    unwrapped_model = (
        diffusion_model if unwrap_model is None else unwrap_model(diffusion_model)
    )
    if not hasattr(unwrapped_model, "autoregressive_model"):
        raise TypeError(
            "unwrapped diffusion model does not expose autoregressive_model; "
            "pass accelerator.unwrap_model after accelerator.prepare"
        )
    _freeze_and_assert_audio(unwrapped_model.autoregressive_model, freeze=True)
    return unwrapped_model


def collect_language_model_lora_targets(
    autoregressive_model: nn.Module,
    target_suffixes: Sequence[str],
) -> list[str]:
    """Resolve exact text-decoder module names, excluding every audio match."""

    suffixes = set(target_suffixes)
    targets = [
        name
        for name, _module in autoregressive_model.named_modules()
        if name.startswith("model.language_model.") and name.rsplit(".", 1)[-1] in suffixes
    ]
    missing = suffixes.difference(name.rsplit(".", 1)[-1] for name in targets)
    if missing:
        raise ValueError(f"LoRA targets not found below model.language_model: {sorted(missing)}")
    if not targets:
        raise ValueError("no LoRA targets resolved below model.language_model")
    if any("audio_tower" in name or "embed_audio" in name for name in targets):
        raise AssertionError("audio module leaked into the exact LoRA target list")
    return targets


def configure_trainable_parameters(
    diffusion_model: nn.Module,
    model_config: Mapping[str, Any],
    *,
    peft_symbols: tuple[Any, Any, Any] | None = None,
    resume_adapter_path: Path | None = None,
) -> nn.Module:
    """Apply full or text-decoder-only LoRA training and reassert audio freeze."""

    strategy = model_config["llm_train_strategy"]
    autoregressive_model = diffusion_model.autoregressive_model
    if strategy == "full":
        autoregressive_model.requires_grad_(True)
    elif strategy == "lora":
        if peft_symbols is None:
            raise RuntimeError(
                "LoRA strategy requested without PEFT symbols; call require_training_dependencies first"
            )
        LoraConfig, TaskType, get_peft_model = peft_symbols
        exact_targets = collect_language_model_lora_targets(
            autoregressive_model,
            model_config["lora_target_modules"],
        )
        if resume_adapter_path is None:
            modules_to_save = (
                ["lm_head"] if model_config.get("train_lm_head", False) else None
            )
            lora_config = LoraConfig(
                r=int(model_config["lora_rank"]),
                lora_alpha=int(model_config["lora_alpha"]),
                lora_dropout=float(model_config.get("lora_dropout", 0.0)),
                target_modules=exact_targets,
                modules_to_save=modules_to_save,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            peft_model = get_peft_model(autoregressive_model, lora_config)
        else:
            from peft import PeftModel

            peft_model = PeftModel.from_pretrained(
                autoregressive_model,
                resume_adapter_path,
                is_trainable=True,
            )
        diffusion_model.autoregressive_model = PeftCompositeProxy(
            peft_model
        )
        autoregressive_model = diffusion_model.autoregressive_model
    else:
        raise ValueError(f"unsupported llm_train_strategy: {strategy!r}")

    diffusion_model.diffusion_mask_embedding.requires_grad_(
        bool(model_config.get("train_mask_embedding", True))
    )
    if getattr(diffusion_model, "time_mlp", None) is not None:
        diffusion_model.time_mlp.requires_grad_(True)
    _freeze_and_assert_audio(autoregressive_model)

    bad_audio = []
    bad_lora = []
    trainable_lora = []
    for name, parameter in diffusion_model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "audio_tower" in name or "embed_audio" in name:
            bad_audio.append(name)
        if "lora_" in name and "language_model" not in name:
            bad_lora.append(name)
        if "lora_" in name:
            trainable_lora.append(name)
    if bad_audio:
        raise AssertionError(f"trainable audio parameters after strategy setup: {bad_audio[:8]}")
    if bad_lora:
        raise AssertionError(f"LoRA adapters outside model.language_model: {bad_lora[:8]}")
    if strategy == "lora" and not trainable_lora:
        raise AssertionError("PEFT created no trainable LoRA parameters")
    return diffusion_model


def parameter_report(model: nn.Module) -> dict[str, Any]:
    total = 0
    trainable = 0
    modules: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
            module = name.rsplit(".", 1)[0] if "." in name else name
            modules[module] = modules.get(module, 0) + count
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total if total else 0.0,
        "trainable_modules": dict(sorted(modules.items())),
        "audio_tower_frozen": True,
        "audio_projector_frozen": True,
    }


def frozen_audio_sha256(autoregressive_model: nn.Module) -> str:
    """Hash every audio-tower/projector tensor for bit-exact freeze audits."""

    base = _base_autoregressive_model(autoregressive_model)
    base.assert_audio_components_frozen()
    digest = hashlib.sha256()
    for module_name, module in (
        ("model.audio_tower", base.model.audio_tower),
        ("model.embed_audio", base.model.embed_audio),
    ):
        for tensor_name, tensor in sorted(module.state_dict().items()):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"non-tensor audio state: {module_name}.{tensor_name}")
            value = tensor.detach().contiguous().cpu()
            name = f"{module_name}.{tensor_name}".encode("utf-8")
            digest.update(len(name).to_bytes(8, "big"))
            digest.update(name)
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def trainable_parameter_identity(model: nn.Module) -> str:
    return mapping_sha256(
        {
            "parameters": [
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                }
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ]
        }
    )


def checkpoint_snapshot_identity(snapshot: str | Path) -> str:
    """Fingerprint the immutable HF commit, or exact files for a local model."""

    root = Path(snapshot).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"checkpoint snapshot is not a directory: {root}")
    if root.parent.name == "snapshots" and re.fullmatch(r"[0-9a-f]{40}", root.name):
        return mapping_sha256(
            {"kind": "huggingface_commit", "commit": root.name}
        )

    files: list[dict[str, Any]] = []
    for item in sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix()):
        if item.is_dir():
            if item.is_symlink():
                raise ValueError(
                    f"local checkpoint contains a directory symlink: {item}"
                )
            continue
        if item.is_symlink():
            target = item.resolve(strict=True)
            if not target.is_file():
                raise ValueError(f"local checkpoint symlink is not a file: {item}")
            content_path = target
        elif item.is_file():
            content_path = item
        else:
            raise ValueError(f"local checkpoint entry is not a regular file: {item}")
        files.append(
            {
                "path": item.relative_to(root).as_posix(),
                "sha256": sha256_file(content_path),
            }
        )
    if not files:
        raise ValueError(f"checkpoint snapshot contains no files: {root}")
    return mapping_sha256({"kind": "local_content", "files": files})


def training_source_identity(repository_root: str | Path) -> str:
    """Hash the dirty-worktree Python sources that determine the trajectory."""

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


def resolve_checkpoint_snapshot(
    checkpoint: str,
    revision: str,
    *,
    local_files_only: bool,
) -> Path:
    """Resolve the exact revision before repository classes load its tensors."""

    local = Path(checkpoint).expanduser()
    if local.exists():
        return local.resolve()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("huggingface_hub is required to resolve the checkpoint revision") from error
    return Path(
        snapshot_download(
            repo_id=checkpoint,
            revision=revision,
            local_files_only=local_files_only,
        )
    ).resolve()


def fixed_or_random_corruption(
    clean_canvas_ids: torch.Tensor,
    canvas_valid_mask: torch.Tensor,
    noise_process,
    diffusion_config: DiffusionConfig,
    rng: TrainingRNGStreams,
    *,
    fixed_noise_level: float | None,
) -> NoisedCanvasBatch:
    if fixed_noise_level is None:
        return corrupt_diffusion_batch(
            clean_canvas_ids,
            canvas_valid_mask,
            noise_process,
            min_diffusion_time=diffusion_config.min_diffusion_time,
            full_noise_probability=diffusion_config.full_noise_probability,
            pad_token_id=diffusion_config.canvas.pad_token_id,
            time_generator=rng.time,
            corruption_generator=rng.corruption,
            token_noise_generator=rng.uniform_prior,
        )
    if not 0 < fixed_noise_level <= 1:
        raise ValueError("fixed_noise_level must be in (0, 1]")
    times = torch.full(
        (clean_canvas_ids.shape[0],),
        fixed_noise_level,
        device=clean_canvas_ids.device,
        dtype=torch.float32,
    )
    noisy, corrupt = noise_process.q_sample(
        clean_canvas_ids,
        times,
        generator=rng.corruption,
        noise_generator=rng.uniform_prior,
    )
    corrupt &= canvas_valid_mask
    noisy = noisy.masked_fill(
        ~canvas_valid_mask, diffusion_config.canvas.pad_token_id
    )
    return NoisedCanvasBatch(
        noisy_canvas_ids=noisy,
        clean_canvas_ids=clean_canvas_ids,
        canvas_valid_mask=canvas_valid_mask,
        corrupt_mask=corrupt,
        noise_level=times,
    )


def _move_tensor_fields(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device=device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _save_rng_state(rng: TrainingRNGStreams | Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    state = rng.state_dict() if hasattr(rng, "state_dict") else rng
    torch.save(state, temporary)
    temporary.replace(path)


def build_local_rank_checkpoint_state(
    *,
    rank: int,
    world_size: int,
    epoch: int,
    raw_batches_consumed_in_epoch: int,
    usable_batches_consumed_in_epoch: int,
    raw_batch_cursor_sha256: str,
    epoch_dataloader_generator_state: torch.Tensor,
    rng: TrainingRNGStreams,
    overflow_counters: Mapping[str, int],
) -> dict[str, Any]:
    """Capture the exact replay cursor and RNG streams owned by one rank."""

    return {
        "rank": int(rank),
        "world_size": int(world_size),
        "epoch": int(epoch),
        "raw_batches_consumed_in_epoch": int(raw_batches_consumed_in_epoch),
        "usable_batches_consumed_in_epoch": int(usable_batches_consumed_in_epoch),
        "raw_batch_cursor_sha256": str(raw_batch_cursor_sha256),
        "epoch_dataloader_generator_state": epoch_dataloader_generator_state.cpu(),
        "training_rng": rng.state_dict(),
        "global_rng": capture_global_rng_state(),
        "overflow_counters": dict(overflow_counters),
    }


def prune_old_training_checkpoints(
    *, output_dir: Path, save_total_limit: int
) -> tuple[Path, ...]:
    """Delete only old, direct ``checkpoint-NNNNNN`` directories.

    Symlinks, files, nested paths and names outside the exact six-digit
    checkpoint contract are deliberately ignored.  The caller invokes this
    only after the new checkpoint manifest has been written successfully.
    """

    if (
        not isinstance(save_total_limit, int)
        or isinstance(save_total_limit, bool)
        or save_total_limit < 1
    ):
        raise ValueError("training.save_total_limit must be a positive integer")

    root = output_dir.resolve(strict=True)
    candidates: list[tuple[int, Path]] = []
    for entry in output_dir.iterdir():
        match = CHECKPOINT_DIRECTORY_PATTERN.fullmatch(entry.name)
        if match is None or entry.is_symlink() or not entry.is_dir():
            continue
        resolved_entry = entry.resolve(strict=True)
        if resolved_entry.parent != root:
            raise RuntimeError(
                f"refusing to prune checkpoint outside output directory: {entry}"
            )
        candidates.append((int(match.group(1)), entry))

    candidates.sort(key=lambda item: (item[0], item[1].name))
    to_delete = candidates[:-save_total_limit]
    deleted: list[Path] = []
    for _, candidate in to_delete:
        # Revalidate immediately before the destructive operation so a changed
        # directory entry cannot broaden the deletion target.
        if (
            candidate.parent != output_dir
            or CHECKPOINT_DIRECTORY_PATTERN.fullmatch(candidate.name) is None
            or candidate.is_symlink()
            or not candidate.is_dir()
            or candidate.resolve(strict=True).parent != root
        ):
            raise RuntimeError(f"checkpoint changed during retention: {candidate}")
        shutil.rmtree(candidate)
        deleted.append(candidate)
    return tuple(deleted)


def save_training_checkpoint(
    *,
    unwrapped_model: nn.Module,
    output_dir: Path,
    resolved_config_path: Path,
    config: Mapping[str, Any],
    update: int,
    epoch: int,
    rng: TrainingRNGStreams,
    optimizer: Any,
    scheduler: Any,
    gradient_scaler: Any | None,
    raw_batches_consumed_in_epoch: int,
    usable_batches_consumed_in_epoch: int,
    raw_batch_cursor_sha256: str,
    epoch_dataloader_generator_state: torch.Tensor,
    overflow_counters: Mapping[str, int],
    last_reported_loss: float | None,
    elapsed_seconds: float,
    resume_identities: Mapping[str, str],
    rank_states: Sequence[Mapping[str, Any]] | None = None,
    max_memory_allocated_gib: float | None = None,
) -> Path:
    """Save the documented base/adapter + diffusion-component checkpoint."""

    checkpoint_dir = output_dir / f"checkpoint-{update:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    diffusion_state = checkpoint_dir / "diffusion_state.pt"
    save_diffusion_component_state(unwrapped_model, diffusion_state)
    strategy = str(config["model"]["llm_train_strategy"])
    autoregressive_model = unwrapped_model.autoregressive_model
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "strategy": strategy,
        "base_checkpoint": config["model"]["checkpoint"],
        "revision": config["model"]["revision"],
        "update": int(update),
        "diffusion_state": diffusion_state.name,
        "resolved_config": "resolved_config.yaml",
        "token_policy": config["runtime"]["token_policy"],
        "uniform_noise_ids_sha256": config["runtime"]["token_policy"][
            "uniform_noise_ids_sha256"
        ],
        "resume_schema_version": RESUME_STATE_SCHEMA_VERSION,
        "training_state": "training_state.pt",
        "world_size": 1 if rank_states is None else len(rank_states),
        **dict(resume_identities),
    }
    if strategy == "lora":
        adapter_dir = checkpoint_dir / "adapter"
        autoregressive_model.save_pretrained(adapter_dir, safe_serialization=True)
        manifest["adapter_dir"] = adapter_dir.name
    else:
        full_model_dir = checkpoint_dir / "full_model"
        autoregressive_model.save_pretrained(
            full_model_dir,
            safe_serialization=True,
            max_shard_size="4GB",
        )
        manifest["full_model_dir"] = full_model_dir.name
    checkpoint_config_path = checkpoint_dir / "resolved_config.yaml"
    shutil.copy2(resolved_config_path, checkpoint_config_path)
    manifest["resolved_config_sha256"] = _sha256_file(checkpoint_config_path)
    _save_rng_state(
        rng
        if rank_states is None
        else {
            "world_size": len(rank_states),
            "rank_rng_states": [state["training_rng"] for state in rank_states],
        },
        checkpoint_dir / "rng_state.pt",
    )
    training_state_path = checkpoint_dir / manifest["training_state"]
    _save_training_loop_state(
        training_state_path,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_scaler=gradient_scaler,
        rng=rng,
        update=update,
        epoch=epoch,
        raw_batches_consumed_in_epoch=raw_batches_consumed_in_epoch,
        usable_batches_consumed_in_epoch=usable_batches_consumed_in_epoch,
        raw_batch_cursor_sha256=raw_batch_cursor_sha256,
        epoch_dataloader_generator_state=epoch_dataloader_generator_state,
        overflow_counters=overflow_counters,
        last_reported_loss=last_reported_loss,
        elapsed_seconds=elapsed_seconds,
        identities=resume_identities,
        rank_states=rank_states,
        max_memory_allocated_gib=max_memory_allocated_gib,
    )
    manifest["artifact_sha256"] = {
        "resolved_config": _sha256_file(checkpoint_config_path),
        "diffusion_state": _sha256_file(diffusion_state),
        "training_state": _sha256_file(training_state_path),
        "trained_model": sha256_directory(
            checkpoint_dir
            / (
                manifest["adapter_dir"]
                if strategy == "lora"
                else manifest["full_model_dir"]
            )
        ),
    }
    (checkpoint_dir / "trainer_state.json").write_text(
        json.dumps(
            {
                "update": int(update),
                "epoch": int(epoch),
                "raw_batches_consumed_in_epoch": int(
                    raw_batches_consumed_in_epoch
                ),
                "usable_batches_consumed_in_epoch": int(
                    usable_batches_consumed_in_epoch
                ),
                "resume_supported": True,
                "world_size": 1 if rank_states is None else len(rank_states),
                "training_state": manifest["training_state"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    write_diffusion_checkpoint_manifest(checkpoint_dir, manifest)
    prune_old_training_checkpoints(
        output_dir=output_dir,
        save_total_limit=config["training"]["save_total_limit"],
    )
    return checkpoint_dir


def save_fsdp_training_checkpoint(
    *,
    accelerator: Any,
    prepared_model: nn.Module,
    unwrapped_model: nn.Module,
    output_dir: Path,
    resolved_config_path: Path,
    config: Mapping[str, Any],
    update: int,
    epoch: int,
    rng: TrainingRNGStreams,
    optimizer: Any,
    scheduler: Any,
    gradient_scaler: Any | None,
    raw_batches_consumed_in_epoch: int,
    usable_batches_consumed_in_epoch: int,
    raw_batch_cursor_sha256: str,
    epoch_dataloader_generator_state: torch.Tensor,
    overflow_counters: Mapping[str, int],
    last_reported_loss: float | None,
    elapsed_seconds: float,
    resume_identities: Mapping[str, str],
    rank_states: Sequence[Mapping[str, Any]],
    expected_audio_sha256: str,
    max_memory_allocated_gib: float | None = None,
) -> Path:
    """Collectively save one resumable FSDP sharded checkpoint.

    The checkpoint manifest remains the final atomic completion marker. Model,
    optimizer, registered scheduler and scaler states are written collectively
    below ``full_model`` by Accelerate/PyTorch DCP; small semantic data cursors
    remain in the existing rank-aware training-state contract.
    """

    if str(config["model"]["llm_train_strategy"]) != "full":
        raise ValueError("FSDP checkpointing is only valid for full fine-tuning")
    if not uses_fsdp_hybrid_shard(config["training"]):
        raise ValueError("FSDP checkpoint requested for a non-FSDP configuration")
    validate_rank_states(rank_states, world_size=accelerator.num_processes)

    checkpoint_dir = output_dir / f"checkpoint-{update:06d}"
    full_model_dir = checkpoint_dir / "full_model"
    if accelerator.is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
    accelerator.wait_for_everyone()
    # Every rank must enter this call for sharded model and optimizer state.
    accelerator.save_state(str(full_model_dir), safe_serialization=False)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        observed_audio_sha256 = frozen_audio_sha256(
            unwrapped_model.autoregressive_model
        )
        if observed_audio_sha256 != expected_audio_sha256:
            raise AssertionError(
                "frozen audio weights changed during full fine-tuning: "
                f"initial={expected_audio_sha256}, current={observed_audio_sha256}"
            )
        diffusion_state = checkpoint_dir / "diffusion_state.json"
        diffusion_state.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "storage": "full_model",
                    "note": "diffusion parameters are part of the FSDP distributed state",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        checkpoint_config_path = checkpoint_dir / "resolved_config.yaml"
        shutil.copy2(resolved_config_path, checkpoint_config_path)
        _save_rng_state(
            {
                "world_size": len(rank_states),
                "rank_rng_states": [state["training_rng"] for state in rank_states],
            },
            checkpoint_dir / "rng_state.pt",
        )
        training_state_path = checkpoint_dir / "training_state.pt"
        _save_training_loop_state(
            training_state_path,
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_scaler=gradient_scaler,
            rng=rng,
            update=update,
            epoch=epoch,
            raw_batches_consumed_in_epoch=raw_batches_consumed_in_epoch,
            usable_batches_consumed_in_epoch=usable_batches_consumed_in_epoch,
            raw_batch_cursor_sha256=raw_batch_cursor_sha256,
            epoch_dataloader_generator_state=epoch_dataloader_generator_state,
            overflow_counters=overflow_counters,
            last_reported_loss=last_reported_loss,
            elapsed_seconds=elapsed_seconds,
            identities=resume_identities,
            rank_states=rank_states,
            max_memory_allocated_gib=max_memory_allocated_gib,
            optimizer_state_external=True,
        )
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "strategy": "full",
            "distributed_strategy": "fsdp_hybrid_shard",
            "checkpoint_format": "accelerate_fsdp_sharded_state_dict",
            "base_checkpoint": config["model"]["checkpoint"],
            "revision": config["model"]["revision"],
            "update": int(update),
            "diffusion_state": diffusion_state.name,
            "resolved_config": checkpoint_config_path.name,
            "full_model_dir": full_model_dir.name,
            "training_state": training_state_path.name,
            "resume_schema_version": RESUME_STATE_SCHEMA_VERSION,
            "world_size": len(rank_states),
            "audio_frozen_sha256": observed_audio_sha256,
            "token_policy": config["runtime"]["token_policy"],
            "uniform_noise_ids_sha256": config["runtime"]["token_policy"][
                "uniform_noise_ids_sha256"
            ],
            **dict(resume_identities),
        }
        manifest["artifact_sha256"] = {
            "resolved_config": _sha256_file(checkpoint_config_path),
            "diffusion_state": _sha256_file(diffusion_state),
            "training_state": _sha256_file(training_state_path),
            "trained_model": sha256_directory(full_model_dir),
        }
        (checkpoint_dir / "trainer_state.json").write_text(
            json.dumps(
                {
                    "update": int(update),
                    "epoch": int(epoch),
                    "raw_batches_consumed_in_epoch": int(
                        raw_batches_consumed_in_epoch
                    ),
                    "usable_batches_consumed_in_epoch": int(
                        usable_batches_consumed_in_epoch
                    ),
                    "resume_supported": True,
                    "world_size": len(rank_states),
                    "distributed_strategy": "fsdp_hybrid_shard",
                    "training_state": training_state_path.name,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        write_diffusion_checkpoint_manifest(checkpoint_dir, manifest)
        prune_old_training_checkpoints(
            output_dir=output_dir,
            save_total_limit=config["training"]["save_total_limit"],
        )
    accelerator.wait_for_everyone()
    return checkpoint_dir


def run_validate_only(
    config: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> int:
    model_config = config["model"]
    overfit_indices = requested_overfit_sample_indices(args)
    overfit_mode = requested_overfit_mode(args)
    gate = parity_gate_report(
        args.parity_artifact,
        expected_checkpoint=model_config["checkpoint"],
        expected_revision=model_config["revision"],
    )
    support_error = None
    try:
        assert_trainer_support(config)
    except UnsupportedTrainingFeatureError as error:
        support_error = str(error)

    runtime: dict[str, Any] = {
        "validation_mode": True,
        "model_weights_loaded": False,
        "optimizer_created": False,
        "parity_gate": gate,
        "trainer_supported": support_error is None,
        "trainer_support_error": support_error,
        "optional_dependencies": optional_dependency_report(
            model_config["llm_train_strategy"]
        ),
    }
    if args.skip_data_validation:
        if overfit_indices:
            raise ValueError(
                "--skip-data-validation cannot validate overfit sample indices"
            )
        runtime["dataset"] = {"validated": False, "reason": "--skip-data-validation"}
    else:
        dataset = build_streaming_dataset(
            config,
            validate_file_sizes=not args.skip_file_size_validation,
            shuffle_buffer_size=1 if overfit_indices else None,
        )
        dataset_summary = dataset.configuration_summary()
        validate_dataset_provenance(config, dataset_summary)
        runtime["dataset"] = {
            "validated": True,
            **dataset_summary,
        }
        if overfit_indices:
            samples = select_overfit_samples(dataset, overfit_indices)
            if overfit_mode == "conditional_four_view":
                if config["data"].get("loader") != "conditional_local_tar":
                    raise ValueError(
                        "--overfit-four-view-sample-index requires conditional_local_tar"
                    )
                # Validate all four signed fields while the selected waveform is
                # already resident; tokenizer validation occurs in --train.
                conditional_four_view_records(samples[0])
            metadata = overfit_sample_metadata(samples, overfit_indices)
            for item in metadata:
                item.update(
                    {
                        "fixed_noise_level": args.fixed_noise_level,
                        "waveform_loaded_in_memory": True,
                    }
                )
            runtime["overfit_sample_indices"] = list(overfit_indices)
            runtime["overfit_samples"] = metadata
            runtime["overfit_mode"] = overfit_mode
            runtime["deterministic_view_cycle"] = (
                [0, 1, 2, 3]
                if overfit_mode == "conditional_four_view"
                else None
            )
            if len(metadata) == 1:
                # Preserve the original resolved-config contract.
                runtime["overfit_sample"] = metadata[0]
    resolved = deepcopy(config)
    resolved["runtime"] = runtime
    path = save_resolved_config(resolved, output_dir)
    print(json.dumps({"resolved_config": str(path), **runtime}, indent=2, default=str))
    if gate["status"] != "open":
        print(f"\n*** {gate['reason']} ***\n", file=sys.stderr)
    if support_error is not None:
        print(f"\n*** TRAINER SUPPORT CLOSED: {support_error} ***\n", file=sys.stderr)
    # Structural validation remains useful even when the separate P0 gate, a
    # runtime feature, or an optional dependency is pending.
    return 0


def run_training(
    config: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> int:
    model_config = config["model"]
    overfit_indices = requested_overfit_sample_indices(args)
    overfit_mode = requested_overfit_mode(args)

    # Hard ordering guarantee: these checks happen before snapshot resolution,
    # model loading, Accelerator construction, and optimizer construction.
    parity = verify_pytorch_parity_artifact(
        args.parity_artifact,
        expected_checkpoint=model_config["checkpoint"],
        expected_revision=model_config["revision"],
    )
    assert_trainer_support(config)
    Accelerator, LoraConfig, TaskType, get_peft_model = require_training_dependencies(
        model_config["llm_train_strategy"]
    )

    training_config = dict(config["training"])
    if args.max_updates is not None:
        training_config["max_updates"] = int(args.max_updates)
    if training_config["max_updates"] <= 0:
        raise ValueError("max_updates must be > 0")
    requested_world_size = environment_world_size()
    distributed_layout = resolve_distributed_training_layout(
        training_config,
        world_size=requested_world_size,
    )
    training_config["gradient_accumulation_steps"] = (
        distributed_layout.gradient_accumulation_steps
    )
    stop_after_updates = (
        int(training_config["max_updates"])
        if args.stop_after_updates is None
        else int(args.stop_after_updates)
    )
    if not 0 < stop_after_updates <= int(training_config["max_updates"]):
        raise ValueError(
            "--stop-after-updates must be in (0, configured max_updates]"
        )
    if args.fixed_noise_level is not None and not overfit_indices:
        raise ValueError("--fixed-noise-level is restricted to explicit overfit mode")
    effective_config = deepcopy(config)
    effective_config["training"] = training_config
    checkpoint_snapshot = resolve_checkpoint_snapshot(
        model_config["checkpoint"],
        model_config["revision"],
        local_files_only=args.local_files_only,
    )
    base_snapshot_sha256 = checkpoint_snapshot_identity(checkpoint_snapshot)
    source_code_sha256 = training_source_identity(REPOSITORY_ROOT)
    training_config_sha256 = mapping_sha256(
        training_run_identity(
            effective_config,
            overfit_indices=overfit_indices,
            fixed_noise_level=args.fixed_noise_level,
            overfit_mode=overfit_mode,
        )
    )
    mixed_precision = str(training_config["mixed_precision"])
    accelerator = create_training_accelerator(Accelerator, training_config)
    if accelerator.num_processes != requested_world_size:
        raise RuntimeError(
            "Accelerate/torchrun world-size mismatch: "
            f"accelerator={accelerator.num_processes}, environment={requested_world_size}"
        )
    if accelerator.device.type != "cuda" or torch.version.hip is None:
        raise RuntimeError(
            "MawJais diffusion training requires a ROCm CUDA-compatible device (MI210); "
            "CPU/CUDA-non-ROCm training is intentionally not started"
        )
    resume_checkpoint = resolve_resume_checkpoint_distributed(
        args.resume_from_checkpoint,
        output_dir=output_dir,
        accelerator=accelerator,
    )
    weight_continuation = None
    if args.continue_from_checkpoint not in (None, "", "-"):
        if resume_checkpoint is not None:
            raise ValueError(
                "--resume-from-checkpoint and --continue-from-checkpoint are "
                "mutually exclusive"
            )
        if training_config.get("scheduler_type") != "continuation_cosine":
            raise ValueError(
                "--continue-from-checkpoint requires the audited "
                "continuation_cosine training configuration"
            )
        weight_continuation = resolve_weight_continuation_distributed(
            args.continue_from_checkpoint,
            output_dir=output_dir,
            child_config=effective_config,
            expected_parent_update=int(
                training_config["continuation_start_update"]
            ),
            accelerator=accelerator,
        )
    preserve_parent_optimizer = bool(
        weight_continuation is not None
        and weight_continuation.provenance["state_policy"].get("name")
        == OPTIMIZER_PRESERVING_CONTINUATION
    )
    initial_checkpoint = (
        resume_checkpoint
        if resume_checkpoint is not None
        else (
            None
            if weight_continuation is None
            else weight_continuation.checkpoint
        )
    )
    if resume_checkpoint is not None:
        early_expected = {
            "training_config_sha256": training_config_sha256,
            "strategy": model_config["llm_train_strategy"],
            "base_checkpoint": model_config["checkpoint"],
            "revision": model_config["revision"],
            "base_snapshot_sha256": base_snapshot_sha256,
            "source_code_sha256": source_code_sha256,
        }
        early_mismatches = {
            name: {
                "checkpoint": resume_checkpoint.manifest.get(name),
                "current": value,
            }
            for name, value in early_expected.items()
            if resume_checkpoint.manifest.get(name) != value
        }
        if early_mismatches:
            raise ValueError(f"resume checkpoint identity mismatch: {early_mismatches}")

    if overfit_indices and accelerator.num_processes != 1:
        raise ValueError("explicit overfit mode is restricted to world_size=1")
    set_global_seed(int(training_config["seed"]))
    rng = TrainingRNGStreams.create(
        training_config,
        device=accelerator.device,
        rank=accelerator.process_index,
    )
    dataset = build_streaming_dataset(
        config,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        validate_file_sizes=not args.skip_file_size_validation,
        shuffle_buffer_size=1 if overfit_indices else None,
    )
    dataset_summary = dataset.configuration_summary()
    validate_dataset_provenance(config, dataset_summary)
    conditional_epoch_updates = (
        None
        if overfit_indices
        else validate_conditional_epoch_update_plan(
            dataset,
            training_config,
            distributed_layout,
        )
    )
    dataset_identity_sha256 = mapping_sha256(dataset_identity(dataset_summary))
    if (
        resume_checkpoint is not None
        and resume_checkpoint.manifest.get("dataset_identity_sha256")
        != dataset_identity_sha256
    ):
        raise ValueError(
            "resume checkpoint dataset identity does not match current data"
        )
    overfit_samples: tuple[Mapping[str, Any], ...] = ()
    if overfit_indices:
        overfit_samples = select_overfit_samples(dataset, overfit_indices)
        if overfit_mode == "conditional_four_view":
            if config["data"].get("loader") != "conditional_local_tar":
                raise ValueError(
                    "--overfit-four-view-sample-index requires conditional_local_tar"
                )
            dataset = RepeatedConditionalFourViewDataset(
                overfit_samples[0], overfit_indices[0]
            )
        elif len(overfit_samples) == 1:
            dataset = RepeatedSingleExampleDataset(
                overfit_samples[0], overfit_indices[0]
            )
        else:
            dataset = RepeatedOverfitExamplesDataset(
                overfit_samples, overfit_indices
            )

    from audio.diffusion.loading import (
        load_mawjais_r2_for_diffusion,
        load_repository_classes,
    )
    from audio.diffusion.model import MawJaisDiffusionASRModel

    dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
    attention_implementation = str(training_config["attention_implementation"])
    loader_attention_implementation = composite_attention_implementation(
        attention_implementation
    )
    if (
        initial_checkpoint is not None
        and model_config["llm_train_strategy"] == "full"
        and not uses_fsdp_hybrid_shard(training_config)
    ):
        classes = load_repository_classes(REPOSITORY_ROOT)
        processor = classes.processor_class.from_pretrained(checkpoint_snapshot)
        resumed_config = classes.config_class.from_pretrained(
            initial_checkpoint.trained_model
        )
        autoregressive_model = classes.model_class.from_pretrained(
            initial_checkpoint.trained_model,
            config=resumed_config,
            dtype=dtype,
            attn_implementation=loader_attention_implementation,
            device_map=None,
            low_cpu_mem_usage=True,
        )
    else:
        autoregressive_model, processor = load_mawjais_r2_for_diffusion(
            checkpoint_snapshot,
            repository_root=REPOSITORY_ROOT,
            dtype=dtype,
            attn_implementation=loader_attention_implementation,
            device_map=None,
            low_cpu_mem_usage=True,
        )
    assert_model_attention_implementation(
        autoregressive_model,
        attention_implementation,
    )
    diffusion_config = diffusion_config_from_experiment(config)
    diffusion_model = MawJaisDiffusionASRModel(
        autoregressive_model,
        diffusion_config,
        normal_text_token_count=int(processor.tokenizer.vocab_size),
        time_embedding_seed=int(training_config["time_embedding_seed"]),
    )
    peft_symbols = (
        (LoraConfig, TaskType, get_peft_model)
        if model_config["llm_train_strategy"] == "lora"
        else None
    )
    # PEFT initializes LoRA A/B matrices. Reset immediately before adapter
    # creation so the optional time module cannot perturb their shared seed.
    set_global_seed(int(training_config["seed"]))
    diffusion_model = configure_trainable_parameters(
        diffusion_model,
        model_config,
        peft_symbols=peft_symbols,
        resume_adapter_path=(
            initial_checkpoint.trained_model
            if initial_checkpoint is not None
            and model_config["llm_train_strategy"] == "lora"
            else None
        ),
    )
    if uses_fsdp_hybrid_shard(training_config):
        # FSDP1 flat parameters require one dtype per wrapped unit. The loaded
        # checkpoint is BF16 while newly constructed diffusion/time parameters
        # intentionally start in FP32. Normalize before wrapping; Accelerate's
        # FSDP mixed-precision path then upcasts trainable flat shards back to
        # FP32 master storage and uses BF16 only for forward/backward compute.
        diffusion_model.to(dtype=dtype)
    if initial_checkpoint is not None and not uses_fsdp_hybrid_shard(training_config):
        load_diffusion_component_state(
            diffusion_model,
            initial_checkpoint.diffusion_state,
        )
    if training_config.get("gradient_checkpointing", False):
        enable_verified_gradient_checkpointing(diffusion_model.autoregressive_model)
    params = parameter_report(diffusion_model)
    if params["trainable_parameters"] == 0:
        raise RuntimeError("no trainable parameters were selected")
    initial_audio_sha256: str | None = None
    if accelerator.is_main_process:
        initial_audio_sha256 = frozen_audio_sha256(
            diffusion_model.autoregressive_model
        )
    if accelerator.num_processes > 1:
        payload = [initial_audio_sha256]
        torch.distributed.broadcast_object_list(
            payload, src=0, device=accelerator.device
        )
        initial_audio_sha256 = payload[0]
    if not isinstance(initial_audio_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", initial_audio_sha256
    ):
        raise RuntimeError("could not establish the initial frozen-audio checksum")

    token_policy = build_diffusion_token_policy(
        processor.tokenizer,
        model_vocab_size=int(autoregressive_model.lm_head.out_features),
        uniform_vocab_scope=diffusion_config.uniform_vocab_scope,
        allowed_added_output_token_ids=config["diffusion"][
            "allowed_added_output_token_ids"
        ],
        extra_forbidden_output_token_ids=config["diffusion"][
            "extra_forbidden_output_token_ids"
        ],
        mask_token_id=diffusion_config.mask_token_id,
        tail_fill_mode=diffusion_config.canvas.tail_fill_mode,
        pad_token_id=diffusion_config.canvas.pad_token_id,
        turn_end_token_id=diffusion_config.canvas.turn_end_token_id,
    )
    token_contract = validate_mawjais_token_contract(
        processor.tokenizer,
        processor=processor,
        model_config=_base_autoregressive_model(autoregressive_model).config,
        mask_token_text=config["diffusion"]["mask_token_text"],
        mask_token_id=diffusion_config.mask_token_id,
        pad_token_id=diffusion_config.canvas.pad_token_id,
        turn_end_token_id=diffusion_config.canvas.turn_end_token_id,
        forbidden_output_token_ids=token_policy.sampling_forbidden_ids,
    )
    token_policy_sha256 = mapping_sha256(token_policy.to_metadata())
    resume_identities = {
        "training_config_sha256": training_config_sha256,
        "dataset_identity_sha256": dataset_identity_sha256,
        "token_policy_sha256": token_policy_sha256,
        "trainable_parameters_sha256": trainable_parameter_identity(
            diffusion_model
        ),
        "base_snapshot_sha256": base_snapshot_sha256,
        "source_code_sha256": source_code_sha256,
    }
    if resume_checkpoint is not None:
        validate_resume_identities(
            resume_checkpoint,
            **resume_identities,
            strategy=model_config["llm_train_strategy"],
            base_checkpoint=model_config["checkpoint"],
            revision=model_config["revision"],
        )
    continuation_dataset_identity: dict[str, Any] | None = None
    if weight_continuation is not None:
        continuation_dataset_identity = validate_weight_continuation_immutable_identities(
            weight_continuation,
            child_dataset_summary=dataset_summary,
            token_policy_sha256=token_policy_sha256,
            trainable_parameters_sha256=resume_identities[
                "trainable_parameters_sha256"
            ],
            base_snapshot_sha256=base_snapshot_sha256,
            strategy=model_config["llm_train_strategy"],
            base_checkpoint=model_config["checkpoint"],
            revision=model_config["revision"],
        )
    noise_process = (
        MaskNoiseProcess(diffusion_config.mask_token_id)
        if diffusion_config.noise_type == "mask"
        else UniformNoiseProcess(token_policy.uniform_noise_ids)
    )

    conditional_training = (
        config["data"].get("loader", "legacy_local_tar")
        == "conditional_local_tar"
    )
    collator = MawJaisDiffusionCollator(
        processor,
        diffusion_config,
        # Bucket choice remains duration-only. A target that does not fit that
        # chosen bucket is excluded explicitly and counted; it is never
        # truncated or promoted using reference length. This is deterministic
        # across all eight runs and lets a full epoch cross known outliers.
        overflow_policy="raise" if conditional_training else "skip",
        conditional_mode=conditional_training,
    )
    overfit_metadata: list[dict[str, Any]] = []
    if overfit_samples:
        # The repeated dataset is intentionally infinite. Reject overflowing or
        # scientifically ambiguous choices before the first optimizer update.
        if overfit_mode == "conditional_four_view":
            overfit_metadata = validate_tokenized_conditional_four_view_sample(
                collator, overfit_samples[0], overfit_indices[0]
            )
        else:
            overfit_metadata = validate_tokenized_overfit_samples(
                collator, overfit_samples, overfit_indices
            )
    num_workers = 0 if overfit_indices else int(config["data"]["num_workers"])
    dataloader = DataLoader(
        dataset,
        batch_size=int(training_config["per_device_batch_size"]),
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=False,
        generator=rng.dataloader,
        worker_init_fn=seed_dataloader_worker if num_workers else None,
    )
    trainable_parameters = [
        parameter for parameter in diffusion_model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    max_updates = int(training_config["max_updates"])
    lr_multiplier, warmup_updates = build_learning_rate_multiplier(training_config)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    fresh_scheduler_state, fresh_optimizer_group_state = (
        capture_fresh_scheduler_state(optimizer, scheduler)
    )
    # Keep the native scheduler out of ``prepare``. AcceleratedScheduler steps
    # by world_size when batches are not split, but our rank-aware accumulation
    # already holds the semantic global batch constant across all admitted
    # topologies up to 128 ranks.
    diffusion_model, optimizer = accelerator.prepare(diffusion_model, optimizer)
    if uses_fsdp_hybrid_shard(training_config):
        # The scheduler intentionally remains outside prepare so one semantic
        # optimizer update advances it exactly once. Registering preserves it
        # in Accelerate's collective FSDP checkpoint.
        accelerator.register_for_checkpointing(scheduler)

    resume_state: dict[str, Any] | None = None
    loaded_optimizer_state_entries: int | None = None
    if resume_checkpoint is not None:
        resume_state = _load_training_loop_state(
            resume_checkpoint,
            identities=resume_identities,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
        )
        if int(resume_state["update"]) > max_updates:
            raise ValueError(
                "resume checkpoint update exceeds configured max_updates: "
                f"{resume_state['update']} > {max_updates}"
            )
        if int(resume_state["update"]) > stop_after_updates:
            raise ValueError(
                "resume checkpoint update exceeds --stop-after-updates: "
                f"{resume_state['update']} > {stop_after_updates}"
            )
        if uses_fsdp_hybrid_shard(training_config):
            if resume_checkpoint.manifest.get("checkpoint_format") != (
                "accelerate_fsdp_sharded_state_dict"
            ):
                raise ValueError("resume checkpoint is not an FSDP sharded checkpoint")
            materialize_adam_state_for_fsdp_load(optimizer)
            accelerator.load_state(str(resume_checkpoint.trained_model))
        else:
            optimizer.load_state_dict(resume_state["optimizer"])
            scheduler.load_state_dict(resume_state["scheduler"])
            restore_gradient_scaler_state(
                getattr(accelerator, "scaler", None),
                resume_state["gradient_scaler"],
            )
    elif weight_continuation is not None and preserve_parent_optimizer:
        parent_identities = {
            name: weight_continuation.checkpoint.manifest.get(name)
            for name in (
                "training_config_sha256",
                "dataset_identity_sha256",
                "token_policy_sha256",
                "trainable_parameters_sha256",
                "base_snapshot_sha256",
                "source_code_sha256",
            )
        }
        if not all(isinstance(value, str) for value in parent_identities.values()):
            raise ValueError(
                "optimizer-preserving parent manifest lacks training identities"
            )
        resume_state = _load_training_loop_state(
            weight_continuation.checkpoint,
            identities=parent_identities,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
        )
        if int(resume_state["update"]) != int(
            training_config["continuation_start_update"]
        ):
            raise ValueError(
                "optimizer-preserving continuation state/update mismatch: "
                f"{resume_state['update']} != "
                f"{training_config['continuation_start_update']}"
            )
        if int(resume_state["epoch"]) != int(
            training_config["continuation_start_epoch"]
        ):
            raise ValueError(
                "optimizer-preserving continuation state/epoch mismatch: "
                f"{resume_state['epoch']} != "
                f"{training_config['continuation_start_epoch']}"
            )
        if uses_fsdp_hybrid_shard(training_config):
            if weight_continuation.checkpoint.manifest.get("checkpoint_format") != (
                "accelerate_fsdp_sharded_state_dict"
            ):
                raise ValueError(
                    "optimizer-preserving full parent is not an FSDP checkpoint"
                )
            # This collective operation restores model, Adam moments, scaler,
            # RNG and the registered parent scheduler. The scheduler is reset
            # to the child schedule immediately below; Adam is left untouched.
            materialize_adam_state_for_fsdp_load(optimizer)
            accelerator.load_state(
                str(weight_continuation.checkpoint.trained_model)
            )
        else:
            if resume_state["optimizer"] is None:
                raise ValueError(
                    "optimizer-preserving parent has no serialized optimizer"
                )
            optimizer.load_state_dict(resume_state["optimizer"])
            restore_gradient_scaler_state(
                getattr(accelerator, "scaler", None),
                resume_state["gradient_scaler"],
            )
        restore_fresh_scheduler_after_optimizer_load(
            optimizer,
            scheduler,
            fresh_scheduler_state,
            fresh_optimizer_group_state,
        )
        loaded_optimizer_state_entries = optimizer_state_entry_count(optimizer)
        if loaded_optimizer_state_entries <= 0:
            raise RuntimeError(
                "optimizer-preserving continuation restored no Adam state entries"
            )
    elif weight_continuation is not None and uses_fsdp_hybrid_shard(training_config):
        # The child phase intentionally owns a fresh optimizer and scheduler.
        # Load only the authenticated parent model shards; accelerator.load_state
        # would also restore the parent's optimizer and defeat the LR restart.
        load_fsdp_weight_continuation_model(
            accelerator=accelerator,
            prepared_model=diffusion_model,
            continuation=weight_continuation,
        )
        if accelerator.num_processes > 1:
            # The parent shards must be loaded with identical construction RNGs,
            # but dropout during the child phase must remain rank-specific.
            set_global_seed(int(training_config["seed"]) + accelerator.process_index)
    elif accelerator.num_processes > 1:
        # Parameter construction is intentionally identical across ranks; use
        # rank-specific process RNGs only after DDP has synchronized weights.
        set_global_seed(int(training_config["seed"]) + accelerator.process_index)
    unwrapped_diffusion_model = enter_diffusion_training_mode(
        diffusion_model,
        unwrap_model=accelerator.unwrap_model,
    )

    runtime_config = deepcopy(config)
    runtime_config["training"] = training_config
    runtime_config["runtime"] = {
        "validation_mode": False,
        "parity_gate": parity,
        "checkpoint_snapshot": str(checkpoint_snapshot),
        "dataset": dataset_summary,
        "token_policy": token_policy.to_metadata(),
        "resolved_special_token_ids": token_contract.to_metadata(),
        "rng_seeds": rng.seeds,
        "parameter_report": params,
        "distributed_strategy": training_config.get(
            "distributed_strategy", "ddp"
        ),
        "audio_frozen_sha256": initial_audio_sha256,
        "world_size": accelerator.num_processes,
        "rank": accelerator.process_index,
        "global_batch_size": distributed_layout.global_batch_size,
        "per_device_batch_size": distributed_layout.per_device_batch_size,
        "gradient_accumulation_steps": distributed_layout.gradient_accumulation_steps,
        "attention_implementation": attention_implementation,
        "audio_attention_implementation": "eager",
        "overfit_sample_index": (
            overfit_indices[0] if len(overfit_indices) == 1 else None
        ),
        "overfit_sample_indices": list(overfit_indices) or None,
        "overfit_mode": overfit_mode,
        "deterministic_view_cycle": (
            [0, 1, 2, 3]
            if overfit_mode == "conditional_four_view"
            else None
        ),
        "overfit_samples": overfit_metadata or None,
        "fixed_noise_level": args.fixed_noise_level,
        "overflow_policy": (
            "signed_conditional_sidecar_raise_on_drift"
            if conditional_training
            else "signed_pre_shard_exclusion_fail_unexpected"
        ),
        "conditional_training": conditional_training,
        "validated_epoch_updates": conditional_epoch_updates,
        "resume": {
            "requested": args.resume_from_checkpoint,
            "checkpoint": (
                None if resume_checkpoint is None else str(resume_checkpoint.path)
            ),
            "resumed_update": (
                None if resume_checkpoint is None else resume_checkpoint.update
            ),
            "identities": resume_identities,
        },
        "weight_continuation": (
            None
            if weight_continuation is None
            else {
                "checkpoint": str(weight_continuation.checkpoint.path),
                "parent_update": weight_continuation.checkpoint.update,
                "parent_world_size": weight_continuation.checkpoint.manifest.get(
                    "world_size"
                ),
                "child_world_size": accelerator.num_processes,
                "state_policy": weight_continuation.provenance["state_policy"],
                "start_epoch": int(training_config["continuation_start_epoch"]),
                "dataset_content_identity": continuation_dataset_identity,
                "loaded_optimizer_state_entries": (
                    loaded_optimizer_state_entries
                    if preserve_parent_optimizer
                    else None
                ),
            }
        ),
        "stop_after_updates": stop_after_updates,
    }
    resolved_config_path = output_dir / "resolved_config.yaml"
    if accelerator.is_main_process:
        resolved_config_path = save_resolved_config(runtime_config, output_dir)
        if weight_continuation is not None:
            write_fork_provenance(
                output_dir,
                weight_continuation,
                child_training_config_sha256=training_config_sha256,
                child_source_code_sha256=source_code_sha256,
                additional_metadata={
                    "dataset_content_identity": continuation_dataset_identity
                },
            )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(
            json.dumps(
                {
                    "total_parameters": params["total_parameters"],
                    "trainable_parameters": params["trainable_parameters"],
                    "trainable_fraction": params["trainable_fraction"],
                    "audio_tower_frozen": True,
                    "audio_projector_frozen": True,
                    "trainable_module_count": len(params["trainable_modules"]),
                    "uniform_prior_hash": token_policy.uniform_noise_ids_sha256,
                },
                indent=2,
            ),
            flush=True,
        )

    def persist_training_checkpoint(
        *,
        checkpoint_update: int,
        checkpoint_epoch: int,
        checkpoint_rank_states: Sequence[Mapping[str, Any]],
        checkpoint_raw_batches: int,
        checkpoint_usable_batches: int,
        checkpoint_cursor_sha256: str,
        checkpoint_dataloader_state: torch.Tensor,
        checkpoint_last_loss: float | None,
        checkpoint_elapsed_seconds: float,
        checkpoint_peak_memory_gib: float,
    ) -> None:
        common = {
            "output_dir": output_dir,
            "resolved_config_path": resolved_config_path,
            "config": runtime_config,
            "update": checkpoint_update,
            "epoch": checkpoint_epoch,
            "rng": rng,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "gradient_scaler": getattr(accelerator, "scaler", None),
            "raw_batches_consumed_in_epoch": checkpoint_raw_batches,
            "usable_batches_consumed_in_epoch": checkpoint_usable_batches,
            "raw_batch_cursor_sha256": checkpoint_cursor_sha256,
            "epoch_dataloader_generator_state": checkpoint_dataloader_state,
            "overflow_counters": overflow_counters,
            "last_reported_loss": checkpoint_last_loss,
            "elapsed_seconds": checkpoint_elapsed_seconds,
            "resume_identities": resume_identities,
            "rank_states": checkpoint_rank_states,
            "max_memory_allocated_gib": checkpoint_peak_memory_gib,
        }
        if uses_fsdp_hybrid_shard(training_config):
            save_fsdp_training_checkpoint(
                accelerator=accelerator,
                prepared_model=diffusion_model,
                unwrapped_model=unwrapped_diffusion_model,
                expected_audio_sha256=initial_audio_sha256,
                **common,
            )
        elif accelerator.is_main_process:
            save_training_checkpoint(
                unwrapped_model=unwrapped_diffusion_model,
                **common,
            )

    update = (
        int(resume_state["update"])
        if resume_state is not None
        else (
            weight_continuation.checkpoint.update
            if weight_continuation is not None
            else 0
        )
    )
    epoch = (
        int(resume_state["epoch"])
        if resume_state is not None
        else (
            int(training_config["continuation_start_epoch"])
            if weight_continuation is not None
            else 0
        )
    )
    raw_batches_consumed_in_epoch = (
        0
        if resume_state is None
        else int(resume_state["raw_batches_consumed_in_epoch"])
    )
    usable_batches_consumed_in_epoch = (
        0
        if resume_state is None
        else int(resume_state["usable_batches_consumed_in_epoch"])
    )
    last_saved_update = update if resume_state is not None else -1
    optimizer.zero_grad(set_to_none=True)
    additive_metrics: dict[str, torch.Tensor] = {}
    log_loss = 0.0
    log_microbatches = 0
    log_grad_norm = 0.0
    log_clipped_updates = 0
    log_optimizer_updates = 0
    run_peak_memory_gib = reduce_max_value(
        torch.cuda.max_memory_allocated(accelerator.device) / float(1 << 30),
        device=accelerator.device,
        world_size=accelerator.num_processes,
    )
    if resume_state is not None and resume_state.get("max_memory_allocated_gib") is not None:
        run_peak_memory_gib = max(
            run_peak_memory_gib,
            float(resume_state["max_memory_allocated_gib"]),
        )
    last_reported_loss: float | None = (
        None if resume_state is None else resume_state["last_reported_loss"]
    )
    overflow_counters = (
        {name: 0 for name in COLLATOR_METRIC_NAMES}
        if resume_state is None
        else {
            name: int(resume_state["overflow_counters"][name])
            for name in COLLATOR_METRIC_NAMES
        }
    )
    elapsed_before_resume = (
        0.0 if resume_state is None else float(resume_state["elapsed_seconds"])
    )
    start_time = time.monotonic()

    resume_replay_pending = bool(
        resume_state is not None
        and int(resume_state["raw_batches_consumed_in_epoch"]) > 0
    )
    if resume_state is not None and not resume_replay_pending:
        empty_cursor_sha256 = hashlib.sha256().hexdigest()
        if (
            int(resume_state["usable_batches_consumed_in_epoch"]) != 0
            or resume_state["raw_batch_cursor_sha256"] != empty_cursor_sha256
        ):
            raise ValueError(
                "zero-batch resume cursor must use the empty SHA-256 and zero usable batches"
            )
        # Epoch-boundary checkpoints are already normalized to the next epoch.
        # Start that iterator directly instead of constructing the prior epoch
        # and replaying its complete waveform/TAR stream.
        rng.load_state_dict(resume_state["training_rng"])
        restore_global_rng_state(resume_state["global_rng"])
    reached_max_updates = update >= stop_after_updates
    while update < stop_after_updates:
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        if resume_replay_pending:
            epoch_dataloader_generator_state = resume_state[
                "epoch_dataloader_generator_state"
            ].clone()
            rng.dataloader.set_state(epoch_dataloader_generator_state)
        else:
            raw_batches_consumed_in_epoch = 0
            usable_batches_consumed_in_epoch = 0
            epoch_dataloader_generator_state = rng.dataloader.get_state().clone()
        epoch_batch_cursor = hashlib.sha256()

        # Recreate the iterator each epoch. Its worker base seed and all dataset
        # ordering are replayed from the state captured before the original
        # iterator was created.
        batch_iterator = iter(dataloader)
        if resume_replay_pending:
            _, epoch_batch_cursor = replay_raw_dataloader_batches(
                batch_iterator,
                raw_batches_consumed_in_epoch,
                expected_usable_batches=usable_batches_consumed_in_epoch,
                cursor=epoch_batch_cursor,
            )
            if (
                epoch_batch_cursor.hexdigest()
                != resume_state["raw_batch_cursor_sha256"]
            ):
                raise RuntimeError(
                    "DataLoader batch order/content diverged while replaying the resume epoch"
                )
            expected_dataloader_state = resume_state["training_rng"]["dataloader"]
            if not torch.equal(
                rng.dataloader.get_state().cpu(),
                expected_dataloader_state.cpu(),
            ):
                raise RuntimeError(
                    "DataLoader generator diverged while replaying the resume epoch"
                )
            # Model/optimizer construction and replay may touch process-global
            # RNGs. Restore every stream only after the cursor has been rebuilt.
            rng.load_state_dict(resume_state["training_rng"])
            restore_global_rng_state(resume_state["global_rng"])
            resume_replay_pending = False

        reached_max_updates = False
        for cpu_batch in batch_iterator:
            update_raw_batch_cursor(epoch_batch_cursor, cpu_batch)
            raw_batches_consumed_in_epoch += 1
            skipped = accumulate_collator_metrics(cpu_batch, overflow_counters)
            collator_metrics = cpu_batch["collator_metrics"]
            local_unexpected_overflow = skipped or any(
                int(collator_metrics[name]) > 0
                for name in ("duration_bucket_overflow", "target_bucket_overflow")
            )
            if collective_unexpected_overflow(
                local_unexpected_overflow,
                device=accelerator.device,
                world_size=accelerator.num_processes,
            ):
                local_keys = cpu_batch.get(
                    "skipped_sample_keys" if skipped else "sample_keys", []
                )
                raise RuntimeError(
                    "unexpected canvas overflow after the signed pre-sharding exclusion; "
                    "all ranks abort without discarding a synchronized batch; "
                    f"rank={accelerator.process_index}, local_keys={local_keys}, "
                    f"local_metrics={dict(collator_metrics)}"
                )
            usable_batches_consumed_in_epoch += 1
            batch = _move_tensor_fields(cpu_batch, accelerator.device)
            noised = fixed_or_random_corruption(
                batch["clean_canvas_ids"],
                batch["canvas_valid_mask"],
                noise_process,
                diffusion_config,
                rng,
                fixed_noise_level=args.fixed_noise_level,
            )
            with accelerator.accumulate(diffusion_model):
                output = diffusion_model(
                    prefix_input_ids=batch["prefix_input_ids"],
                    prefix_attention_mask=batch["prefix_attention_mask"],
                    audio_position_mask=batch["audio_position_mask"],
                    input_features=batch["input_features"],
                    feature_attention_mask=batch["feature_attention_mask"],
                    noisy_canvas_ids=noised.noisy_canvas_ids,
                    clean_canvas_ids=noised.clean_canvas_ids,
                    corrupt_mask=noised.corrupt_mask,
                    canvas_valid_mask=noised.canvas_valid_mask,
                    noise_level=noised.noise_level,
                    forbidden_output_token_ids=token_policy.training_forbidden_ids,
                )
                if output.loss is None or not torch.isfinite(output.loss):
                    raise FloatingPointError(f"non-finite diffusion loss: {output.loss}")
                accelerator.backward(output.loss)
                current_grad_norm: float | None = None
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        trainable_parameters,
                        float(training_config["max_grad_norm"]),
                    )
                    current_grad_norm = float(torch.as_tensor(grad_norm).detach())
                _freeze_and_assert_audio(
                    unwrapped_diffusion_model.autoregressive_model,
                    # Do not call freeze_audio_components() here: it clears
                    # parameter.grad and would hide an accidental audio
                    # gradient instead of failing the guard.
                    freeze=False,
                )
                sync_gradients = bool(accelerator.sync_gradients)
                optimizer.step()
                optimizer_step_was_skipped = bool(
                    accelerator.optimizer_step_was_skipped
                )
                scheduler_stepped = step_raw_scheduler_for_synchronized_update(
                    scheduler,
                    sync_gradients=sync_gradients,
                    optimizer_step_was_skipped=optimizer_step_was_skipped,
                )
                optimizer.zero_grad(set_to_none=True)

            log_loss += float(output.loss.detach())
            log_microbatches += 1
            # Backward has completed. Masking this now-dead logits buffer in
            # place avoids a second [B,C,250624] allocation merely for argmax.
            safe_metric_logits = output.logits.detach()
            safe_metric_logits[..., list(token_policy.sampling_forbidden_ids)] = -torch.inf
            metrics = denoising_token_metrics(
                safe_metric_logits,
                noised.clean_canvas_ids,
                noised.corrupt_mask,
                noised.canvas_valid_mask,
                noised.noise_level,
                turn_end_token_id=diffusion_config.canvas.turn_end_token_id,
                pad_token_id=diffusion_config.canvas.pad_token_id,
            )
            metrics["examples/full_noise"] = noised.noise_level.eq(1.0).sum(
                dtype=torch.long
            )
            metrics["examples/total"] = torch.tensor(
                noised.noise_level.numel(),
                device=noised.noise_level.device,
                dtype=torch.long,
            )
            if conditional_training:
                conditional_metrics = validate_conditional_batch_fields(cpu_batch)
                metrics.update(
                    {
                        name: torch.tensor(
                            value,
                            device=noised.noise_level.device,
                            dtype=torch.long,
                        )
                        for name, value in conditional_metrics.items()
                    }
                )
            for name, value in metrics.items():
                additive_metrics[name] = additive_metrics.get(
                    name, torch.zeros_like(value)
                ) + value

            if not sync_gradients:
                continue
            if optimizer_step_was_skipped:
                # fp16 overflow: GradScaler skipped the optimizer update and
                # the raw scheduler was deliberately not stepped. Do not
                # advance the semantic optimizer-update counter either.
                continue
            if not scheduler_stepped:
                raise RuntimeError(
                    "successful synchronized optimizer update did not step scheduler"
                )
            if current_grad_norm is None:
                raise RuntimeError("synchronized optimizer update has no gradient norm")
            log_grad_norm += current_grad_norm
            log_clipped_updates += int(
                current_grad_norm > float(training_config["max_grad_norm"])
            )
            log_optimizer_updates += 1
            update += 1
            should_checkpoint = checkpoint_due(update, training_config)
            checkpoint_cursor_values = None
            if should_checkpoint or update >= stop_after_updates:
                checkpoint_cursor_values = checkpoint_epoch_cursor(
                    update=update,
                    epoch=epoch,
                    raw_batches_consumed_in_epoch=raw_batches_consumed_in_epoch,
                    usable_batches_consumed_in_epoch=usable_batches_consumed_in_epoch,
                    raw_batch_cursor_sha256=epoch_batch_cursor.hexdigest(),
                    epoch_dataloader_generator_state=epoch_dataloader_generator_state,
                    validated_epoch_updates=conditional_epoch_updates,
                    gradient_accumulation_steps=int(
                        training_config["gradient_accumulation_steps"]
                    ),
                    next_epoch_dataloader_generator_state=(
                        rng.dataloader.get_state()
                    ),
                )
            if (
                update % int(training_config["log_every_updates"]) == 0
                or should_checkpoint
                or update >= stop_after_updates
            ):
                elapsed = elapsed_before_resume + time.monotonic() - start_time
                reduced_metrics = reduce_additive_values(
                    additive_metrics,
                    device=accelerator.device,
                    world_size=accelerator.num_processes,
                )
                reduced_window = reduce_additive_values(
                    {
                        "loss_sum": log_loss,
                        "microbatches": log_microbatches,
                        "grad_norm_sum": log_grad_norm,
                        "clipped_updates": log_clipped_updates,
                        "optimizer_updates": log_optimizer_updates,
                        **{
                            f"data/{name}": value
                            for name, value in overflow_counters.items()
                        },
                    },
                    device=accelerator.device,
                    world_size=accelerator.num_processes,
                )
                ratios = ratios_from_additive_metrics(reduced_metrics)
                full_noise_examples = int(reduced_metrics["examples/full_noise"])
                total_noise_examples = int(reduced_metrics["examples/total"])
                last_reported_loss = reduced_window["loss_sum"] / max(
                    reduced_window["microbatches"], 1.0
                )
                window_peak_memory_gib = reduce_max_value(
                    torch.cuda.max_memory_allocated(accelerator.device)
                    / float(1 << 30),
                    device=accelerator.device,
                    world_size=accelerator.num_processes,
                )
                run_peak_memory_gib = max(
                    run_peak_memory_gib, window_peak_memory_gib
                )
                accelerator.print(
                    json.dumps(
                        {
                            "update": update,
                            "loss": last_reported_loss,
                            "learning_rate": scheduler.get_last_lr()[0],
                            "elapsed_seconds": elapsed,
                            "gradient_norm": reduced_window["grad_norm_sum"]
                            / max(reduced_window["optimizer_updates"], 1.0),
                            "gradient_clipped_fraction": reduced_window[
                                "clipped_updates"
                            ]
                            / max(reduced_window["optimizer_updates"], 1.0),
                            "noise/full_examples": full_noise_examples,
                            "noise/examples": total_noise_examples,
                            "noise/full_fraction": full_noise_examples
                            / max(total_noise_examples, 1),
                            "max_memory_allocated_gib": window_peak_memory_gib,
                            **{
                                key: int(value)
                                for key, value in reduced_window.items()
                                if key.startswith("data/")
                            },
                            **{
                                key: int(value)
                                for key, value in reduced_metrics.items()
                                if key.startswith("condition/")
                            },
                            **ratios,
                        },
                        sort_keys=True,
                    )
                )
                additive_metrics.clear()
                log_loss = 0.0
                log_microbatches = 0
                log_grad_norm = 0.0
                log_clipped_updates = 0
                log_optimizer_updates = 0
                torch.cuda.reset_peak_memory_stats(accelerator.device)
            if should_checkpoint:
                if additive_metrics or log_microbatches or log_loss:
                    raise RuntimeError(
                        "checkpoint reached outside a synchronized log boundary"
                    )
                accelerator.wait_for_everyone()
                if checkpoint_cursor_values is None:  # pragma: no cover - invariant
                    raise RuntimeError("checkpoint cursor was not prepared")
                (
                    checkpoint_epoch,
                    checkpoint_raw_batches,
                    checkpoint_usable_batches,
                    checkpoint_cursor_sha256,
                    checkpoint_generator_state,
                    checkpoint_at_epoch_boundary,
                ) = checkpoint_cursor_values
                local_rank_state = build_local_rank_checkpoint_state(
                    rank=accelerator.process_index,
                    world_size=accelerator.num_processes,
                    epoch=checkpoint_epoch,
                    raw_batches_consumed_in_epoch=checkpoint_raw_batches,
                    usable_batches_consumed_in_epoch=checkpoint_usable_batches,
                    raw_batch_cursor_sha256=checkpoint_cursor_sha256,
                    epoch_dataloader_generator_state=checkpoint_generator_state,
                    rng=rng,
                    overflow_counters=overflow_counters,
                )
                rank_states = gather_rank_objects(
                    local_rank_state,
                    world_size=accelerator.num_processes,
                )
                validate_rank_states(
                    rank_states,
                    world_size=accelerator.num_processes,
                )
                persist_training_checkpoint(
                    checkpoint_update=update,
                    checkpoint_epoch=checkpoint_epoch,
                    checkpoint_rank_states=rank_states,
                    checkpoint_raw_batches=checkpoint_raw_batches,
                    checkpoint_usable_batches=checkpoint_usable_batches,
                    checkpoint_cursor_sha256=checkpoint_cursor_sha256,
                    checkpoint_dataloader_state=checkpoint_generator_state,
                    checkpoint_last_loss=last_reported_loss,
                    checkpoint_elapsed_seconds=(
                        elapsed_before_resume + time.monotonic() - start_time
                    ),
                    checkpoint_peak_memory_gib=run_peak_memory_gib,
                )
                accelerator.wait_for_everyone()
                # Checkpoint serialization may touch process-global RNG on the
                # writer. Continue every rank from exactly the state that was
                # persisted, so uninterrupted and resumed trajectories agree.
                rng.load_state_dict(local_rank_state["training_rng"])
                restore_global_rng_state(local_rank_state["global_rng"])
                last_saved_update = update
            if update >= stop_after_updates:
                if checkpoint_cursor_values is None:  # pragma: no cover - invariant
                    raise RuntimeError("stop cursor was not prepared")
                (
                    stopped_epoch,
                    stopped_raw_batches,
                    stopped_usable_batches,
                    stopped_cursor_sha256,
                    stopped_generator_state,
                    stopped_at_epoch_boundary,
                ) = checkpoint_cursor_values
                if stopped_at_epoch_boundary:
                    epoch = stopped_epoch
                    raw_batches_consumed_in_epoch = stopped_raw_batches
                    usable_batches_consumed_in_epoch = stopped_usable_batches
                    epoch_batch_cursor = hashlib.sha256()
                    if epoch_batch_cursor.hexdigest() != stopped_cursor_sha256:
                        raise RuntimeError("normalized empty cursor hash changed")
                    epoch_dataloader_generator_state = stopped_generator_state
                reached_max_updates = True
                break
        if reached_max_updates:
            break
        if raw_batches_consumed_in_epoch == 0:
            raise RuntimeError(f"streaming dataset produced no batches for epoch {epoch}")
        if usable_batches_consumed_in_epoch == 0:
            raise RuntimeError(
                f"every batch in epoch {epoch} overflowed its duration-selected canvas"
            )
        epoch += 1
        raw_batches_consumed_in_epoch = 0
        usable_batches_consumed_in_epoch = 0

    accelerator.wait_for_everyone()
    if last_saved_update != update:
        local_rank_state = build_local_rank_checkpoint_state(
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
            epoch=epoch,
            raw_batches_consumed_in_epoch=raw_batches_consumed_in_epoch,
            usable_batches_consumed_in_epoch=usable_batches_consumed_in_epoch,
            raw_batch_cursor_sha256=epoch_batch_cursor.hexdigest(),
            epoch_dataloader_generator_state=epoch_dataloader_generator_state,
            rng=rng,
            overflow_counters=overflow_counters,
        )
        rank_states = gather_rank_objects(
            local_rank_state,
            world_size=accelerator.num_processes,
        )
        validate_rank_states(
            rank_states,
            world_size=accelerator.num_processes,
        )
        persist_training_checkpoint(
            checkpoint_update=update,
            checkpoint_epoch=epoch,
            checkpoint_rank_states=rank_states,
            checkpoint_raw_batches=raw_batches_consumed_in_epoch,
            checkpoint_usable_batches=usable_batches_consumed_in_epoch,
            checkpoint_cursor_sha256=epoch_batch_cursor.hexdigest(),
            checkpoint_dataloader_state=epoch_dataloader_generator_state,
            checkpoint_last_loss=last_reported_loss,
            checkpoint_elapsed_seconds=(
                elapsed_before_resume + time.monotonic() - start_time
            ),
            checkpoint_peak_memory_gib=run_peak_memory_gib,
        )
    accelerator.wait_for_everyone()
    global_overflow_counters = reduce_additive_values(
        overflow_counters,
        device=accelerator.device,
        world_size=accelerator.num_processes,
    )
    if accelerator.is_main_process:
        summary_path = output_dir / "training_summary.json"
        temporary = summary_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "updates": update,
                    "configured_max_updates": max_updates,
                    "stop_after_updates": stop_after_updates,
                    # ``epoch`` is the zero-based resume cursor and must remain
                    # unchanged in checkpoints. Human-facing summaries count
                    # the current partially consumed epoch as started.
                    "epochs_started": epoch
                    + int(raw_batches_consumed_in_epoch > 0),
                    "elapsed_seconds": (
                        elapsed_before_resume + time.monotonic() - start_time
                    ),
                    "last_reported_loss": last_reported_loss,
                    "world_size": accelerator.num_processes,
                    "global_batch_size": distributed_layout.global_batch_size,
                    "max_memory_allocated_gib": run_peak_memory_gib,
                    "resumed_from": (
                        None
                        if resume_checkpoint is None
                        else str(resume_checkpoint.path)
                    ),
                    "initialized_from_weights": (
                        None
                        if weight_continuation is None
                        else str(weight_continuation.checkpoint.path)
                    ),
                    "raw_batches_consumed_in_epoch": (
                        raw_batches_consumed_in_epoch
                    ),
                    "overflow_policy": (
                        "signed_conditional_sidecar_raise_on_drift"
                        if conditional_training
                        else "signed_pre_shard_exclusion_fail_unexpected"
                    ),
                    **{
                        name: int(value)
                        for name, value in global_overflow_counters.items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(summary_path)
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Composable experiment YAML")
    parser.add_argument("--output-dir", type=Path, default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="Resolve config/data only; never load the 4B checkpoint",
    )
    mode.add_argument("--train", action="store_true", help="Start gated ROCm training")
    parser.add_argument(
        "--parity-artifact",
        type=Path,
        default=None,
        help="Passing P0 PyTorch AR parity JSON; mandatory for --train",
    )
    parser.add_argument(
        "--skip-data-validation",
        action="store_true",
        help="Only for light config validation; ignored by --train",
    )
    parser.add_argument(
        "--skip-file-size-validation",
        action="store_true",
        help="Check manifest paths but not every declared byte size",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Resolve the exact checkpoint revision only from the HF cache",
    )
    parser.add_argument(
        "--max-updates",
        type=int,
        default=None,
        help="Explicit saved override, mainly for overfit experiments",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        metavar="PATH_OR_LATEST",
        help=(
            "Explicit checksum-verified training resume. 'latest' starts fresh "
            "only when output is absent/empty, otherwise resumes the highest "
            "complete checkpoint."
        ),
    )
    parser.add_argument(
        "--continue-from-checkpoint",
        type=str,
        default=None,
        metavar="EXPLICIT_PARENT_CHECKPOINT",
        help=(
            "Checksum-verified continuation in a new output. The child config's "
            "continuation_state_policy selects either a weights-only restart or "
            "an optimizer/scaler/RNG-preserving continuation with a new scheduler."
        ),
    )
    parser.add_argument(
        "--stop-after-updates",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Runtime-only absolute synchronized update at which to checkpoint "
            "and exit successfully; scheduler still uses configured max_updates."
        ),
    )
    overfit_selection = parser.add_mutually_exclusive_group()
    overfit_selection.add_argument(
        "--overfit-sample-index",
        type=int,
        default=None,
        help="Select one deterministic streaming sample (shuffle disabled) and repeat it",
    )
    overfit_selection.add_argument(
        "--overfit-sample-indices",
        type=int,
        nargs=2,
        metavar=("FIRST", "SECOND"),
        default=None,
        help=(
            "Select two distinct deterministic streaming samples in one pass and "
            "repeat them in alternating order"
        ),
    )
    overfit_selection.add_argument(
        "--overfit-four-view-sample-index",
        type=int,
        default=None,
        help=(
            "Select one all-valid conditional sample and cycle its four signed "
            "targets deterministically in canonical view order"
        ),
    )
    parser.add_argument(
        "--fixed-noise-level",
        type=float,
        default=None,
        help="Fixed t in (0,1], allowed only with an explicit overfit selection",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    config = load_experiment_config(args.config)
    if args.max_updates is not None:
        if args.max_updates <= 0:
            raise ValueError("--max-updates must be > 0")
        config = deepcopy(config)
        config["training"]["max_updates"] = args.max_updates
    overfit_indices = requested_overfit_sample_indices(args)
    if args.fixed_noise_level is not None:
        if not overfit_indices:
            raise ValueError(
                "--fixed-noise-level is restricted to explicit overfit mode"
            )
        if not 0 < args.fixed_noise_level <= 1:
            raise ValueError("--fixed-noise-level must be in (0, 1]")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (REPOSITORY_ROOT / "runs" / config["experiment_name"]).resolve()
    )
    if args.validate_only:
        if args.resume_from_checkpoint not in (None, "", "-"):
            raise ValueError("--resume-from-checkpoint is only valid with --train")
        if args.continue_from_checkpoint not in (None, "", "-"):
            raise ValueError("--continue-from-checkpoint is only valid with --train")
        if args.stop_after_updates is not None:
            raise ValueError("--stop-after-updates is only valid with --train")
        return run_validate_only(config, args, output_dir)
    if args.skip_data_validation:
        raise ValueError("--skip-data-validation is not allowed with --train")
    if args.parity_artifact is None:
        raise ParityGateError(
            f"{PARITY_FAILURE_BANNER}: --parity-artifact is mandatory; "
            "no model or optimizer was created"
        )
    if (
        args.resume_from_checkpoint not in (None, "", "-")
        and args.continue_from_checkpoint not in (None, "", "-")
    ):
        raise ValueError(
            "--resume-from-checkpoint and --continue-from-checkpoint are mutually exclusive"
        )
    return run_training(config, args, output_dir)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ParityGateError, UnsupportedTrainingFeatureError) as error:
        print(f"\n*** {error} ***\n", file=sys.stderr)
        raise SystemExit(2) from error
