"""Composable YAML experiment configs with fail-fast diffusion validation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path
from typing import Any, Mapping

from .config import CanvasConfig, ConfidenceSamplerConfig, DiffusionConfig
from .distributed import SUPPORTED_WORLD_SIZES


DEFAULT_SAVE_TOTAL_LIMIT = 2


_TOP_LEVEL_KEYS = {
    "experiment_name",
    "refinement_profile",
    "model",
    "data",
    "diffusion",
    "sampling",
    "training",
    # Appended only to the immutable resolved config saved by the trainer.
    "runtime",
}
_MODEL_KEYS = {
    "checkpoint",
    "revision",
    "mode",
    "preserve_ar_forward",
    "freeze_audio_tower",
    "freeze_audio_projector",
    "llm_train_strategy",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "lora_module_prefix",
    "lora_target_modules",
    "train_lm_head",
    "train_mask_embedding",
}
_DATA_KEYS = {
    "loader",
    "local_root",
    "subset_manifest",
    "eval_manifest",
    "target_overflow_exclusion_manifest",
    "expected_target_overflow_exclusions",
    "conditional_metadata_index",
    "view_seed",
    "validate_sidecar_hashes",
    "validate_tar_files",
    "balance_distributed_shards",
    "max_audio_bytes",
    "dataset_revision",
    "stream_tar_archives",
    "extract_tar_archives",
    "exclude_eval_video_groups",
    "seed",
    "shuffle_buffer_size",
    "parquet_batch_size",
    "num_workers",
    "target_sample_rate",
}
_DIFFUSION_KEYS = {
    "noise_type",
    "use_time_conditioning",
    "denoiser_output_alignment",
    "noise_schedule",
    "min_diffusion_time",
    "full_noise_probability",
    "mask_token_id",
    "mask_token_text",
    "uniform_vocab_scope",
    "canvas_sizes",
    "duration_tokens_per_second",
    "duration_margin_tokens",
    "tail_fill_mode",
    "turn_end_token_id",
    "pad_token_id",
    "eos_tail_loss_weight",
    "pad_loss_weight",
    "diffusion_objective",
    "loss_positions",
    "loss_weighting",
    "added_token_output_policy",
    "allowed_added_output_token_ids",
    "extra_forbidden_output_token_ids",
    "use_self_conditioning",
    "use_iterative_self_correction",
    "isct_num_stages",
    "isct_loss_weight",
    "isct_start_update",
    "isct_prediction_mode",
    "isct_stop_gradient",
    # Materialized by this validator and accepted when re-reading checkpoints.
    "resolved_objective",
    "resolved_canvas",
}
_SAMPLING_KEYS = {
    "sampler",
    "max_nfe",
    "prediction_mode",
    "temperature_start",
    "temperature_end",
    "adaptive_stopping",
    "entropy_budget_gamma",
    "position_bias_lambda",
}
_TRAINING_KEYS = {
    "seed",
    "time_seed",
    "corruption_seed",
    "uniform_prior_seed",
    "sampler_seed",
    "dataloader_seed",
    "time_embedding_seed",
    "max_updates",
    "per_device_batch_size",
    "gradient_accumulation_steps",
    "global_batch_size",
    "supported_world_sizes",
    "max_grad_norm",
    "learning_rate",
    "weight_decay",
    "warmup_ratio",
    "attention_implementation",
    "gradient_checkpointing",
    "mixed_precision",
    "log_every_updates",
    "save_every_updates",
    "save_at_updates",
    "save_total_limit",
    "scheduler_type",
    "scheduler_warmup_updates",
    "scheduler_min_lr_ratio",
    "scheduler_horizon_updates",
    "expected_epoch_updates",
    "distributed_strategy",
    "fsdp_min_num_params",
    "continuation_start_update",
    "continuation_start_epoch",
    "continuation_state_policy",
}


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = set(mapping).difference(allowed)
    if unknown:
        raise ValueError(f"unknown {section} configuration keys: {sorted(unknown)}")


def _required_bool(mapping: Mapping[str, Any], key: str, section: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{section}.{key} must be a YAML boolean")
    return value


def _required_int(
    mapping: Mapping[str, Any],
    key: str,
    section: str,
    *,
    minimum: int | None = None,
) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{section}.{key} must be a YAML integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{section}.{key} must be >= {minimum}")
    return value


def _required_real(
    mapping: Mapping[str, Any],
    key: str,
    section: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{section}.{key} must be a YAML number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{section}.{key} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{section}.{key} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{section}.{key} must be <= {maximum}")
    return value


def _required_strictly_increasing_int_list(
    mapping: Mapping[str, Any],
    key: str,
    section: str,
    *,
    minimum: int,
) -> list[int]:
    values = mapping.get(key)
    if not isinstance(values, list):
        raise ValueError(f"{section}.{key} must be a YAML integer list")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        raise ValueError(f"{section}.{key} must contain YAML integers")
    if any(value < minimum for value in values):
        raise ValueError(f"{section}.{key} values must be >= {minimum}")
    if values != sorted(set(values)):
        raise ValueError(
            f"{section}.{key} must be strictly increasing without duplicates"
        )
    return values


def _required_positive_int_list(
    mapping: Mapping[str, Any], key: str, section: str
) -> list[int]:
    values = mapping.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{section}.{key} must be a non-empty YAML integer list")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in values
    ):
        raise ValueError(f"{section}.{key} must contain positive YAML integers")
    return values


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings; lists and scalar values replace the base."""

    merged = deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - depends on the runtime image
        raise RuntimeError("PyYAML is required to load diffusion experiment configs") from error

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"experiment config must be a YAML mapping: {path}")
    return payload


def _load_with_inheritance(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        cycle = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"cyclic experiment config inheritance: {cycle}")
    if not path.is_file():
        raise FileNotFoundError(path)

    payload = _read_yaml(path)
    parent = payload.pop("extends", None)
    if parent is None:
        return payload
    if not isinstance(parent, str) or not parent.strip():
        raise ValueError(f"extends must be one non-empty relative path in {path}")
    parent_path = (path.parent / parent).resolve()
    # Inheritance is intentionally local to the config tree. This also avoids an
    # accidentally submitted job reading an unrelated YAML file elsewhere.
    if path.parent not in parent_path.parents and parent_path.parent != path.parent:
        raise ValueError(f"extends escapes the config directory: {parent!r}")
    base = _load_with_inheritance(parent_path, (*stack, path))
    return _deep_merge(base, payload)


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"config section {key!r} must be a mapping")
    return value


def validate_experiment_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a resolved config and materialize its automatic objective.

    The returned mapping is a deep copy suitable for writing next to a run. It
    contains no inherited ``extends`` key and adds the resolved objective fields.
    """

    resolved = deepcopy(dict(config))
    _reject_unknown(resolved, _TOP_LEVEL_KEYS, "top-level")
    experiment_name = resolved.get("experiment_name")
    if not isinstance(experiment_name, str) or not experiment_name:
        raise ValueError("experiment_name must be a non-empty string")

    model = _required_mapping(resolved, "model")
    _reject_unknown(model, _MODEL_KEYS, "model")
    if model.get("mode") != "diffusion":
        raise ValueError("model.mode must be 'diffusion'")
    if _required_bool(model, "freeze_audio_tower", "model") is not True:
        raise ValueError("phase 1 requires freeze_audio_tower=true")
    if _required_bool(model, "freeze_audio_projector", "model") is not True:
        raise ValueError("phase 1 requires freeze_audio_projector=true")
    if _required_bool(model, "preserve_ar_forward", "model") is not True:
        raise ValueError("preserve_ar_forward must remain true")
    _required_bool(model, "train_lm_head", "model")
    _required_bool(model, "train_mask_embedding", "model")
    if model.get("llm_train_strategy") not in ("lora", "full"):
        raise ValueError("llm_train_strategy must be 'lora' or 'full'")
    if model.get("llm_train_strategy") == "lora":
        if model.get("lora_module_prefix") != "model.language_model":
            raise ValueError(
                "phase-1 LoRA targets must be restricted to model.language_model; "
                "unscoped projection names would also modify the audio tower"
            )
        targets = model.get("lora_target_modules")
        if not isinstance(targets, list) or not targets:
            raise ValueError("model.lora_target_modules must be a non-empty list")
        if not all(isinstance(target, str) and target for target in targets):
            raise ValueError("model.lora_target_modules must contain non-empty strings")
        _required_int(model, "lora_rank", "model", minimum=1)
        _required_int(model, "lora_alpha", "model", minimum=1)
        dropout = _required_real(model, "lora_dropout", "model", minimum=0.0)
        if dropout >= 1.0:
            raise ValueError("model.lora_dropout must be < 1")

    data = dict(_required_mapping(resolved, "data"))
    resolved["data"] = data
    _reject_unknown(data, _DATA_KEYS, "data")
    loader = data.get("loader", "legacy_local_tar")
    if loader not in ("legacy_local_tar", "conditional_local_tar"):
        raise ValueError(
            "data.loader must be 'legacy_local_tar' or 'conditional_local_tar'"
        )
    if _required_bool(data, "stream_tar_archives", "data") is not True:
        raise ValueError("data.stream_tar_archives must be true")
    if _required_bool(data, "extract_tar_archives", "data") is not False:
        raise ValueError("data.extract_tar_archives must be false")
    if _required_bool(data, "exclude_eval_video_groups", "data") is not True:
        raise ValueError("data.exclude_eval_video_groups must be true")
    if data.get("exclude_eval_video_groups") is not True:
        raise ValueError("phase 1 requires data.exclude_eval_video_groups=true")
    if not isinstance(data.get("dataset_revision"), str) or not data[
        "dataset_revision"
    ]:
        raise ValueError("data.dataset_revision must be a non-empty immutable revision")
    required_paths = ["local_root"]
    if loader == "conditional_local_tar":
        required_paths.append("conditional_metadata_index")
    else:
        required_paths.extend(
            (
                "subset_manifest",
                "eval_manifest",
                "target_overflow_exclusion_manifest",
            )
        )
    for key in required_paths:
        if not isinstance(data.get(key), str) or not data[key]:
            raise ValueError(f"data.{key} must be a non-empty path")
    common_integer_fields = [
        ("seed", 0),
        ("shuffle_buffer_size", 1),
        ("parquet_batch_size", 1),
        ("num_workers", 0),
        ("target_sample_rate", 1),
    ]
    if loader == "conditional_local_tar":
        common_integer_fields.append(("view_seed", 0))
        if "max_audio_bytes" in data:
            common_integer_fields.append(("max_audio_bytes", 1))
        for key in (
            "validate_sidecar_hashes",
            "validate_tar_files",
            "balance_distributed_shards",
        ):
            if key in data:
                _required_bool(data, key, "data")
        if data.get("balance_distributed_shards", True) is not True:
            raise ValueError(
                "conditional DDP training requires data.balance_distributed_shards=true"
            )
    else:
        common_integer_fields.append(("expected_target_overflow_exclusions", 0))
    for key, minimum in common_integer_fields:
        _required_int(data, key, "data", minimum=minimum)

    diffusion = _required_mapping(resolved, "diffusion")
    _reject_unknown(diffusion, _DIFFUSION_KEYS, "diffusion")
    if diffusion.get("noise_schedule") != "linear":
        raise ValueError("diffusion.noise_schedule must be 'linear' in V1")
    if diffusion.get("added_token_output_policy") != "forbid_all_except":
        raise ValueError(
            "diffusion.added_token_output_policy must be 'forbid_all_except' in V1"
        )
    mask_token_text = diffusion.get("mask_token_text")
    if not isinstance(mask_token_text, str) or not mask_token_text:
        raise ValueError("diffusion.mask_token_text must name the reserved checkpoint token")
    if not isinstance(diffusion.get("mask_token_text"), str) or not diffusion[
        "mask_token_text"
    ]:
        raise ValueError("diffusion.mask_token_text must be a non-empty tokenizer token")
    canvas_sizes = diffusion.get("canvas_sizes")
    if not isinstance(canvas_sizes, list) or not canvas_sizes:
        raise ValueError("diffusion.canvas_sizes must be a non-empty YAML list")
    if any(not isinstance(size, int) or isinstance(size, bool) for size in canvas_sizes):
        raise ValueError("diffusion.canvas_sizes must contain YAML integers")
    for key, minimum in (
        ("mask_token_id", 0),
        ("turn_end_token_id", 0),
        ("pad_token_id", 0),
        ("isct_num_stages", 1),
    ):
        _required_int(diffusion, key, "diffusion", minimum=minimum)
    for key in ("allowed_added_output_token_ids", "extra_forbidden_output_token_ids"):
        values = diffusion.get(key)
        if not isinstance(values, list) or any(
            not isinstance(value, int) or isinstance(value, bool) for value in values
        ):
            raise ValueError(f"diffusion.{key} must be a YAML integer list")
    _required_real(diffusion, "min_diffusion_time", "diffusion", minimum=0.0, maximum=1.0)
    _required_real(diffusion, "full_noise_probability", "diffusion", minimum=0.0, maximum=1.0)
    _required_real(diffusion, "duration_tokens_per_second", "diffusion", minimum=0.0)
    _required_real(diffusion, "duration_margin_tokens", "diffusion", minimum=0.0)
    _required_real(diffusion, "eos_tail_loss_weight", "diffusion", minimum=0.0)
    _required_real(diffusion, "pad_loss_weight", "diffusion", minimum=0.0)
    _required_real(diffusion, "isct_loss_weight", "diffusion", minimum=0.0)
    canvas = CanvasConfig(
        sizes=tuple(diffusion.get("canvas_sizes", (64, 128, 256, 384, 512))),
        duration_tokens_per_second=float(diffusion.get("duration_tokens_per_second", 9.0)),
        duration_margin_tokens=float(diffusion.get("duration_margin_tokens", 100.0)),
        tail_fill_mode=diffusion.get("tail_fill_mode", "eos_repeat"),
        turn_end_token_id=int(diffusion.get("turn_end_token_id", 250019)),
        pad_token_id=int(diffusion.get("pad_token_id", 1)),
        eos_tail_loss_weight=float(diffusion.get("eos_tail_loss_weight", 1.0)),
        pad_loss_weight=float(diffusion.get("pad_loss_weight", 0.1)),
    )
    core = DiffusionConfig(
        noise_type=diffusion.get("noise_type", "mask"),
        use_time_conditioning=_required_bool(
            diffusion, "use_time_conditioning", "diffusion"
        ),
        denoiser_output_alignment=diffusion.get(
            "denoiser_output_alignment", "same_position"
        ),
        min_diffusion_time=float(diffusion.get("min_diffusion_time", 1.0e-3)),
        full_noise_probability=float(diffusion.get("full_noise_probability", 0.0)),
        mask_token_id=int(diffusion.get("mask_token_id", 250064)),
        uniform_vocab_scope=diffusion.get("uniform_vocab_scope", "safe_vocab"),
        diffusion_objective=diffusion.get("diffusion_objective", "auto"),
        loss_positions=diffusion.get("loss_positions", "auto"),
        loss_weighting=diffusion.get("loss_weighting", "auto"),
        canvas=canvas,
    )
    use_self_conditioning = _required_bool(
        diffusion, "use_self_conditioning", "diffusion"
    )
    use_isct = _required_bool(
        diffusion, "use_iterative_self_correction", "diffusion"
    )
    _required_bool(diffusion, "isct_stop_gradient", "diffusion")
    if use_self_conditioning:
        raise ValueError("self-conditioning must stay disabled in the initial 2x2x2 matrix")
    refinement_profile = resolved.get("refinement_profile")
    if not isinstance(refinement_profile, bool):
        raise ValueError("refinement_profile must be a YAML boolean")
    if use_isct:
        if not refinement_profile:
            raise ValueError("ISCT is only valid in an explicit refinement_profile")
        if core.noise_type != "mask":
            raise ValueError("the initial ISCT refinement profile requires mask noise")

    sampling = _required_mapping(resolved, "sampling")
    _reject_unknown(sampling, _SAMPLING_KEYS, "sampling")
    adaptive_stopping = _required_bool(sampling, "adaptive_stopping", "sampling")
    if adaptive_stopping:
        raise ValueError(
            "sampling.adaptive_stopping=true is not implemented in the reference V1; "
            "set it to false instead of silently ignoring it"
        )
    sampler_name = sampling.get("sampler", "confidence_schedule")
    supported_samplers = {
        "confidence_schedule",
        "argmax_udm",
        "udm_marginalization",
        "entropy_bounded",
        "position_biased_entropy_bounded",
    }
    if sampler_name not in supported_samplers:
        raise ValueError(f"unsupported sampler: {sampler_name!r}")
    if sampler_name in {"argmax_udm", "udm_marginalization"} and core.noise_type != "uniform":
        raise ValueError("UDM samplers require uniform noise")
    if sampler_name in {
        "entropy_bounded",
        "position_biased_entropy_bounded",
    } and (core.noise_type != "mask" or core.use_time_conditioning):
        raise ValueError(
            "entropy-bounded samplers require Mask noise with "
            "use_time_conditioning=false in V1"
        )
    _required_int(sampling, "max_nfe", "sampling", minimum=1)
    _required_real(sampling, "temperature_start", "sampling", minimum=0.0)
    _required_real(sampling, "temperature_end", "sampling", minimum=0.0)
    _required_real(sampling, "entropy_budget_gamma", "sampling", minimum=0.0)
    _required_real(sampling, "position_bias_lambda", "sampling", minimum=0.0)
    ConfidenceSamplerConfig(
        sampler=sampler_name,
        max_nfe=int(sampling.get("max_nfe", 64)),
        prediction_mode=sampling.get("prediction_mode", "argmax"),
        temperature_start=float(sampling.get("temperature_start", 0.8)),
        temperature_end=float(sampling.get("temperature_end", 0.4)),
        entropy_budget_gamma=float(sampling.get("entropy_budget_gamma", 0.05)),
        position_bias_lambda=float(sampling.get("position_bias_lambda", 0.2)),
        forbidden_output_token_ids=tuple(
            int(token_id)
            for token_id in diffusion.get("extra_forbidden_output_token_ids", ())
        ),
    )

    training = dict(_required_mapping(resolved, "training"))
    # Older resolved configs predate these controls. Materialize their
    # documented defaults so re-reading them remains deterministic, while
    # keeping explicitly present values subject to strict validation below.
    training.setdefault("save_total_limit", DEFAULT_SAVE_TOTAL_LIMIT)
    training.setdefault("attention_implementation", "eager")
    training.setdefault("distributed_strategy", "ddp")
    training.setdefault("fsdp_min_num_params", 100_000_000)
    resolved["training"] = training
    _reject_unknown(training, _TRAINING_KEYS, "training")
    if training["attention_implementation"] not in ("eager", "sdpa"):
        raise ValueError(
            "training.attention_implementation must be 'eager' or 'sdpa'"
        )
    _required_bool(training, "gradient_checkpointing", "training")
    distributed_strategy = training["distributed_strategy"]
    if distributed_strategy not in ("ddp", "fsdp_hybrid_shard"):
        raise ValueError(
            "training.distributed_strategy must be 'ddp' or 'fsdp_hybrid_shard'"
        )
    _required_int(training, "fsdp_min_num_params", "training", minimum=1)
    if distributed_strategy == "fsdp_hybrid_shard":
        if model.get("llm_train_strategy") != "full":
            raise ValueError("fsdp_hybrid_shard is reserved for full fine-tuning")
        if training["mixed_precision"] != "bf16":
            raise ValueError("fsdp_hybrid_shard currently requires mixed_precision='bf16'")
    for seed_name in (
        "seed",
        "time_seed",
        "corruption_seed",
        "uniform_prior_seed",
        "sampler_seed",
        "dataloader_seed",
        "time_embedding_seed",
    ):
        _required_int(training, seed_name, "training", minimum=0)
    for key in (
        "max_updates",
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "log_every_updates",
        "save_every_updates",
        "save_total_limit",
        "global_batch_size",
    ):
        _required_int(training, key, "training", minimum=1)
    save_at_updates = training.get("save_at_updates", [])
    if "save_at_updates" in training:
        save_at_updates = _required_strictly_increasing_int_list(
            training,
            "save_at_updates",
            "training",
            minimum=1,
        )
        if save_at_updates and save_at_updates[-1] > training["max_updates"]:
            raise ValueError(
                "training.save_at_updates values must not exceed max_updates"
            )
    if "expected_epoch_updates" in training:
        expected_epoch_updates = _required_positive_int_list(
            training,
            "expected_epoch_updates",
            "training",
        )
        if sum(expected_epoch_updates) != training["max_updates"]:
            raise ValueError(
                "sum(training.expected_epoch_updates) must equal max_updates"
            )
    _required_real(training, "learning_rate", "training", minimum=0.0)
    _required_real(training, "weight_decay", "training", minimum=0.0)
    _required_real(training, "warmup_ratio", "training", minimum=0.0, maximum=1.0)
    max_grad_norm = _required_real(training, "max_grad_norm", "training", minimum=0.0)
    if max_grad_norm <= 0:
        raise ValueError("training.max_grad_norm must be > 0")
    configured_world_sizes = _required_strictly_increasing_int_list(
        training,
        "supported_world_sizes",
        "training",
        minimum=1,
    )
    unsupported_world_sizes = [
        value
        for value in configured_world_sizes
        if value not in SUPPORTED_WORLD_SIZES
    ]
    if unsupported_world_sizes:
        raise ValueError(
            "training.supported_world_sizes contains unsupported topologies "
            f"{unsupported_world_sizes}; allowed values are {list(SUPPORTED_WORLD_SIZES)}"
        )
    scheduler_type = training.get("scheduler_type", "linear")
    if scheduler_type not in ("linear", "cosine", "continuation_cosine"):
        raise ValueError(
            "training.scheduler_type must be 'linear', 'cosine', or "
            "'continuation_cosine'"
        )
    continuation_fields = (
        "scheduler_warmup_updates",
        "scheduler_min_lr_ratio",
        "continuation_start_update",
        "continuation_start_epoch",
    )
    cosine_fields = (
        "scheduler_warmup_updates",
        "scheduler_min_lr_ratio",
        "scheduler_horizon_updates",
    )
    if scheduler_type == "cosine":
        missing = [name for name in cosine_fields if name not in training]
        if missing:
            raise ValueError(f"cosine requires training fields: {missing}")
        warmup_updates = _required_int(
            training, "scheduler_warmup_updates", "training", minimum=0
        )
        horizon_updates = _required_int(
            training, "scheduler_horizon_updates", "training", minimum=1
        )
        minimum_ratio = _required_real(
            training,
            "scheduler_min_lr_ratio",
            "training",
            minimum=0.0,
            maximum=1.0,
        )
        if minimum_ratio >= 1.0:
            raise ValueError("training.scheduler_min_lr_ratio must be < 1")
        if horizon_updates != training["max_updates"]:
            raise ValueError(
                "fresh cosine runs require scheduler_horizon_updates=max_updates"
            )
        if warmup_updates >= horizon_updates:
            raise ValueError(
                "training.scheduler_warmup_updates must be smaller than the horizon"
            )
        if any(name in training for name in ("continuation_start_update", "continuation_start_epoch")):
            raise ValueError(
                "continuation scheduler fields are invalid for a fresh cosine run"
            )
        # A reusable run may use an update budget instead of the original
        # campaign's two-epoch plan. If expected_epoch_updates is supplied,
        # its types, total and signed dataset quotas are still validated.
    elif scheduler_type == "continuation_cosine":
        missing = [name for name in continuation_fields if name not in training]
        if missing:
            raise ValueError(
                "continuation_cosine requires training fields: "
                f"{missing}"
            )
        _required_int(
            training, "scheduler_warmup_updates", "training", minimum=0
        )
        _required_int(
            training, "continuation_start_update", "training", minimum=1
        )
        _required_int(
            training, "continuation_start_epoch", "training", minimum=0
        )
        minimum_ratio = _required_real(
            training,
            "scheduler_min_lr_ratio",
            "training",
            minimum=0.0,
            maximum=1.0,
        )
        if minimum_ratio >= 1.0:
            raise ValueError("training.scheduler_min_lr_ratio must be < 1")
        if training["continuation_start_update"] >= training["max_updates"]:
            raise ValueError(
                "training.continuation_start_update must be below max_updates"
            )
        continuation_updates = (
            training["max_updates"] - training["continuation_start_update"]
        )
        if training["scheduler_warmup_updates"] >= continuation_updates:
            raise ValueError(
                "training.scheduler_warmup_updates must be smaller than the "
                "continuation phase"
            )
        if "scheduler_horizon_updates" in training:
            raise ValueError(
                "training.scheduler_horizon_updates is only valid for scheduler_type=cosine"
            )
    elif any(name in training for name in (*continuation_fields, "scheduler_horizon_updates")):
        raise ValueError(
            "explicit scheduler fields require training.scheduler_type=cosine or "
            "continuation_cosine"
        )
    continuation_state_policy = training.get(
        "continuation_state_policy",
        "weights_only_reset_optimizer_scheduler_rng",
    )
    if continuation_state_policy not in (
        "weights_only_reset_optimizer_scheduler_rng",
        "preserve_optimizer_scaler_rng_reset_scheduler",
    ):
        raise ValueError(
            "training.continuation_state_policy must be "
            "'weights_only_reset_optimizer_scheduler_rng' or "
            "'preserve_optimizer_scaler_rng_reset_scheduler'"
        )
    if (
        "continuation_state_policy" in training
        and scheduler_type != "continuation_cosine"
    ):
        raise ValueError(
            "training.continuation_state_policy requires "
            "training.scheduler_type=continuation_cosine"
        )
    if training.get("mixed_precision") not in ("bf16", "fp16"):
        raise ValueError("training.mixed_precision must be 'bf16' or 'fp16'")

    objective = core.resolved_objective
    resolved["diffusion"] = dict(diffusion)
    resolved["diffusion"]["resolved_objective"] = asdict(objective)
    resolved["diffusion"]["resolved_canvas"] = asdict(canvas)
    return resolved


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load inheritance, validate semantics, and return a resolved run config."""

    return validate_experiment_config(_load_with_inheritance(Path(path), ()))


__all__ = ["load_experiment_config", "validate_experiment_config"]
