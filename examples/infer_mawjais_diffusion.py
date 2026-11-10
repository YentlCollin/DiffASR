#!/usr/bin/env python3
"""Reference PyTorch inference for a *trained* MawJais diffusion checkpoint.

This entry point intentionally refuses a bare the user-supplied base model AR directory. A valid
``diffusion_checkpoint.json`` produced by ``train_mawjais_diffusion.py`` is
required before any 4B model is loaded.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from itertools import islice
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from audio.diffusion.canvas import (  # noqa: E402
    DurationBucketSelector,
    truncate_at_first_turn_end,
)
from audio.diffusion.checkpointing import (  # noqa: E402
    DIFFUSION_CHECKPOINT_SCHEMA_VERSION,
    load_diffusion_component_state,
    read_diffusion_checkpoint_manifest,
)
from audio.diffusion.collator import (  # noqa: E402
    build_mawjais_asr_prefix,
    target_prompt_tag,
)
from audio.diffusion.config import (  # noqa: E402
    CanvasConfig,
    ConfidenceSamplerConfig,
    DiffusionConfig,
)
from audio.diffusion.experiment_config import (  # noqa: E402
    load_experiment_config,
    validate_experiment_config,
)
from audio.diffusion.loading import (  # noqa: E402
    load_mawjais_r2_for_diffusion,
    load_repository_classes,
)
from audio.diffusion.model import MawJaisDiffusionASRModel  # noqa: E402
from audio.diffusion.noise_process import (  # noqa: E402
    MaskNoiseProcess,
    UniformNoiseProcess,
)
from audio.diffusion.peft_utils import PeftCompositeProxy  # noqa: E402
from audio.diffusion.sampler import sample_diffusion  # noqa: E402
from audio.diffusion.token_policy import (  # noqa: E402
    build_diffusion_token_policy,
    validate_mawjais_token_contract,
)
from audio.diffusion_data import LocalTarParquetASRDataset  # noqa: E402


CHECKPOINT_MANIFEST = "diffusion_checkpoint.json"
FSDP_SHARDED_CHECKPOINT_FORMAT = "accelerate_fsdp_sharded_state_dict"
FSDP_EXPORT_MANIFEST = "fsdp_export_manifest.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DiffusionCheckpointLayout:
    root: Path
    metadata: Mapping[str, Any]
    strategy: str
    base_checkpoint: str
    revision: str
    resolved_config: Path
    diffusion_state: Path
    trained_model_path: Path


def _checkpoint_child(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"checkpoint metadata field {field!r} must be a relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"checkpoint metadata field {field!r} escapes its directory")
    return root / relative


def read_diffusion_checkpoint_layout(path: str | Path) -> DiffusionCheckpointLayout:
    """Validate the trainer checkpoint marker without loading model weights."""

    root = Path(path).expanduser().resolve()
    manifest = root / CHECKPOINT_MANIFEST
    if not root.is_dir() or not manifest.is_file():
        raise ValueError(
            f"{root} is not a trained MawJais diffusion checkpoint: missing "
            f"{CHECKPOINT_MANIFEST}. A bare AR/the user-supplied base model checkpoint cannot be sampled as diffusion."
        )
    metadata = read_diffusion_checkpoint_manifest(manifest)
    if metadata.get("schema_version") != DIFFUSION_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported diffusion checkpoint schema")
    strategy = metadata.get("strategy")
    if strategy not in ("lora", "full"):
        raise ValueError("diffusion checkpoint strategy must be 'lora' or 'full'")
    base_checkpoint = metadata.get("base_checkpoint")
    revision = metadata.get("revision")
    if not isinstance(base_checkpoint, str) or not base_checkpoint.strip():
        raise ValueError("diffusion checkpoint has no base_checkpoint")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("diffusion checkpoint has no checkpoint revision")

    resolved_config = _checkpoint_child(
        root, metadata.get("resolved_config", "resolved_config.yaml"), "resolved_config"
    )
    diffusion_state = _checkpoint_child(
        root, metadata.get("diffusion_state"), "diffusion_state"
    )
    model_field = "adapter_dir" if strategy == "lora" else "full_model_dir"
    trained_model_path = _checkpoint_child(root, metadata.get(model_field), model_field)
    for required in (resolved_config, diffusion_state):
        if not required.is_file():
            raise FileNotFoundError(f"incomplete diffusion checkpoint; missing {required}")
    expected_config_hash = metadata.get("resolved_config_sha256")
    if expected_config_hash is None:
        artifact_hashes = metadata.get("artifact_sha256")
        if isinstance(artifact_hashes, Mapping):
            expected_config_hash = artifact_hashes.get("resolved_config")
    if expected_config_hash is not None:
        actual_config_hash = _sha256_file(resolved_config)
        if expected_config_hash != actual_config_hash:
            raise ValueError(
                "checkpoint resolved_config.yaml hash mismatch: the saved experiment "
                "semantics were modified after checkpoint creation"
            )
    saved_policy = metadata.get("token_policy")
    saved_prior_hash = metadata.get("uniform_noise_ids_sha256")
    if saved_prior_hash is not None:
        if not isinstance(saved_policy, Mapping) or saved_policy.get(
            "uniform_noise_ids_sha256"
        ) != saved_prior_hash:
            raise ValueError("checkpoint Uniform-prior hash disagrees with token policy")
    if not trained_model_path.is_dir():
        raise FileNotFoundError(
            f"incomplete diffusion checkpoint; missing model directory {trained_model_path}"
        )
    return DiffusionCheckpointLayout(
        root=root,
        metadata=metadata,
        strategy=strategy,
        base_checkpoint=base_checkpoint,
        revision=revision,
        resolved_config=resolved_config,
        diffusion_state=diffusion_state,
        trained_model_path=trained_model_path,
    )


def _load_resolved_config(
    layout: DiffusionCheckpointLayout, override: str | None
) -> dict[str, Any]:
    def load_one(config_path: Path) -> dict[str, Any]:
        if config_path.suffix.lower() != ".json":
            return load_experiment_config(config_path)
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON experiment config must contain an object")
        return validate_experiment_config(payload)

    embedded = load_one(layout.resolved_config)
    if override is None:
        return embedded
    requested = load_one(Path(override).expanduser().resolve())
    # Runtime provenance is appended by the trainer and is absent from the
    # source matrix YAML. All experiment semantics must nevertheless match;
    # --config is therefore an assertion/alias, never a way to reinterpret an
    # adapter under different diffusion settings.
    semantic_sections = (
        "experiment_name",
        "refinement_profile",
        "model",
        "data",
        "diffusion",
        "sampling",
        "training",
    )
    mismatches = [
        section
        for section in semantic_sections
        if requested.get(section) != embedded.get(section)
    ]
    if mismatches:
        raise ValueError(
            "--config does not match the configuration embedded in the trained "
            f"checkpoint (different sections: {mismatches})"
        )
    return embedded


def _resolve_semantics(config: dict[str, Any], args: argparse.Namespace) -> DiffusionConfig:
    values = config["diffusion"]
    trained_noise = values["noise_type"]
    requested_noise = trained_noise if args.noise_type is None else args.noise_type
    if requested_noise != trained_noise:
        raise ValueError(
            f"checkpoint was trained with noise_type={trained_noise!r}, not "
            f"{requested_noise!r}; select a matching trained checkpoint"
        )
    trained_time = bool(values["use_time_conditioning"])
    requested_time = trained_time if args.time_conditioning is None else args.time_conditioning
    if requested_time != trained_time:
        raise ValueError(
            f"checkpoint was trained with use_time_conditioning={trained_time}, not "
            f"{requested_time}; time conditioning is a trained module, not a sampling switch"
        )
    alignment = (
        values["denoiser_output_alignment"]
        if args.alignment is None
        else args.alignment
    )
    canvas = CanvasConfig(
        sizes=tuple(int(size) for size in values["canvas_sizes"]),
        duration_tokens_per_second=float(values["duration_tokens_per_second"]),
        duration_margin_tokens=float(values["duration_margin_tokens"]),
        tail_fill_mode=values["tail_fill_mode"],
        turn_end_token_id=int(values["turn_end_token_id"]),
        pad_token_id=int(values["pad_token_id"]),
        eos_tail_loss_weight=float(values["eos_tail_loss_weight"]),
        pad_loss_weight=float(values["pad_loss_weight"]),
    )
    return DiffusionConfig(
        noise_type=requested_noise,
        use_time_conditioning=requested_time,
        denoiser_output_alignment=alignment,
        min_diffusion_time=float(values["min_diffusion_time"]),
        full_noise_probability=float(values["full_noise_probability"]),
        mask_token_id=int(values["mask_token_id"]),
        uniform_vocab_scope=values["uniform_vocab_scope"],
        diffusion_objective=values["diffusion_objective"],
        loss_positions=values["loss_positions"],
        loss_weighting=values["loss_weighting"],
        canvas=canvas,
    )


def _resolve_base_checkpoint(layout: DiffusionCheckpointLayout, override: str | None) -> str:
    checkpoint = layout.base_checkpoint if override is None else override
    local = Path(checkpoint).expanduser()
    if local.exists():
        return str(local.resolve())
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover - runtime environment dependent
        raise RuntimeError(
            "huggingface_hub is required to resolve a remote base checkpoint"
        ) from error
    return snapshot_download(repo_id=checkpoint, revision=layout.revision)


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _fsdp_merged_weights_path(
    layout: DiffusionCheckpointLayout, value: str | None
) -> Path:
    """Validate a temporary merged export against its source checkpoint."""

    if value is None:
        raise ValueError(
            "this FSDP checkpoint requires --fsdp-merged-weights; first merge "
            "full_model/pytorch_model_fsdp_0 with "
            "examples/export_fsdp_diffusion_checkpoint_for_inference.py"
        )
    requested = Path(value).expanduser().resolve()
    weights = requested / "model.safetensors" if requested.is_dir() else requested
    if not weights.is_file() or weights.name != "model.safetensors":
        raise FileNotFoundError(
            f"merged FSDP weights must point to model.safetensors (got {weights})"
        )
    export_manifest_path = weights.parent / FSDP_EXPORT_MANIFEST
    if not export_manifest_path.is_file():
        raise FileNotFoundError(
            f"merged FSDP export is missing {export_manifest_path.name}: "
            f"{export_manifest_path}"
        )
    try:
        export_manifest = json.loads(export_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid FSDP export manifest: {export_manifest_path}") from error
    if not isinstance(export_manifest, Mapping):
        raise ValueError("FSDP export manifest must contain a JSON object")
    if Path(str(export_manifest.get("checkpoint", ""))).resolve() != layout.root:
        raise ValueError("merged FSDP export belongs to a different checkpoint")
    if export_manifest.get("checkpoint_update") != layout.metadata.get("update"):
        raise ValueError("merged FSDP export update does not match the checkpoint")
    artifact_hashes = layout.metadata.get("artifact_sha256")
    expected_trained_hash = (
        artifact_hashes.get("trained_model")
        if isinstance(artifact_hashes, Mapping)
        else None
    )
    if (
        expected_trained_hash is not None
        and export_manifest.get("source_trained_model_sha256") != expected_trained_hash
    ):
        raise ValueError("merged FSDP export source hash does not match the checkpoint")
    if export_manifest.get("weights_file") != weights.name:
        raise ValueError("FSDP export manifest names a different weights file")
    return weights


def _load_fsdp_merged_wrapper_state(
    wrapper: MawJaisDiffusionASRModel,
    layout: DiffusionCheckpointLayout,
    weights_value: str | None,
) -> None:
    """Load the consolidated full wrapper state without a second HF checkpoint."""

    weights = _fsdp_merged_weights_path(layout, weights_value)
    try:
        from safetensors import safe_open
        from safetensors.torch import load_file, load_model
    except ImportError as error:  # pragma: no cover - runtime environment dependent
        raise RuntimeError("safetensors is required to load merged FSDP weights") from error
    # Loading through the CPU mmap backend avoids materializing another complete
    # FP32 copy on every evaluation GPU. load_state_dict copies each tensor to
    # the device/dtype already selected for the wrapper.
    with safe_open(weights, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    has_outer_model_prefix = bool(keys) and all(key.startswith("model.") for key in keys)
    if has_outer_model_prefix:
        state = load_file(weights, device="cpu", backend="mmap")
        state = {key.removeprefix("model."): value for key, value in state.items()}
        incompatibilities = wrapper.load_state_dict(state, strict=True)
        missing = list(incompatibilities.missing_keys)
        unexpected = list(incompatibilities.unexpected_keys)
    else:
        missing, unexpected = load_model(
            wrapper,
            weights,
            strict=True,
            device="cpu",
            backend="mmap",
        )
    if missing or unexpected:  # strict=True already raises; keep the contract explicit.
        raise RuntimeError(
            f"incomplete merged FSDP state: missing={missing}, unexpected={unexpected}"
        )


def _load_trained_model(
    layout: DiffusionCheckpointLayout,
    diffusion_config: DiffusionConfig,
    args: argparse.Namespace,
):
    """Load trained AR/LoRA tensors, then the diffusion-only component state."""

    base_checkpoint = _resolve_base_checkpoint(layout, args.base_checkpoint)
    dtype = _dtype(args.dtype)
    device_map: Any = None if args.device == "cpu" else {"": args.device}
    is_fsdp_sharded = (
        layout.metadata.get("checkpoint_format") == FSDP_SHARDED_CHECKPOINT_FORMAT
    )
    if layout.strategy == "lora":
        autoregressive_model, processor = load_mawjais_r2_for_diffusion(
            base_checkpoint,
            repository_root=REPOSITORY_ROOT,
            dtype=dtype,
            attn_implementation=args.attention_backend,
            device_map=device_map,
        )
        try:
            from peft import PeftModel
        except ImportError as error:  # pragma: no cover - runtime environment dependent
            raise RuntimeError(
                "PEFT is required to load this LoRA diffusion checkpoint"
            ) from error
        peft_model = PeftModel.from_pretrained(
            autoregressive_model, layout.trained_model_path, is_trainable=False
        )
        autoregressive_model = PeftCompositeProxy(peft_model)
    elif is_fsdp_sharded:
        autoregressive_model, processor = load_mawjais_r2_for_diffusion(
            base_checkpoint,
            repository_root=REPOSITORY_ROOT,
            dtype=dtype,
            attn_implementation=args.attention_backend,
            device_map=device_map,
        )
    else:
        classes = load_repository_classes(REPOSITORY_ROOT)
        processor = classes.processor_class.from_pretrained(base_checkpoint)
        full_config = classes.config_class.from_pretrained(layout.trained_model_path)
        autoregressive_model = classes.model_class.from_pretrained(
            layout.trained_model_path,
            config=full_config,
            dtype=dtype,
            attn_implementation=args.attention_backend,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )

    if args.device == "cpu":
        autoregressive_model.to(device=torch.device("cpu"), dtype=dtype)
    wrapper = MawJaisDiffusionASRModel(autoregressive_model, diffusion_config)
    if is_fsdp_sharded:
        _load_fsdp_merged_wrapper_state(
            wrapper, layout, getattr(args, "fsdp_merged_weights", None)
        )
    else:
        load_diffusion_component_state(wrapper, layout.diffusion_state)
    wrapper.eval()
    return wrapper, processor


def _load_wav(path: Path, target_sample_rate: int = 16_000) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError:
        # The immutable MI210 runtime used by this project has no Python
        # soundfile binding. Keep PCM WAV inference dependency-free, while
        # failing explicitly for formats that genuinely need libsndfile.
        import wave

        if path.suffix.lower() != ".wav":
            raise RuntimeError(
                "soundfile is required for non-WAV inference; provide a PCM WAV "
                "or install soundfile in the runtime image"
            )
        with wave.open(str(path), "rb") as stream:
            if stream.getcomptype() != "NONE":
                raise ValueError(
                    f"unsupported compressed WAV type: {stream.getcomptype()}"
                )
            channels = stream.getnchannels()
            sample_width = stream.getsampwidth()
            sample_rate = stream.getframerate()
            payload = stream.readframes(stream.getnframes())
        if channels <= 0:
            raise ValueError("WAV has no channels")
        if sample_width == 1:
            decoded = np.frombuffer(payload, dtype=np.uint8).astype(np.float32)
            decoded = (decoded - 128.0) / 128.0
        elif sample_width == 2:
            decoded = np.frombuffer(payload, dtype=np.dtype("<i2")).astype(np.float32)
            decoded /= 32768.0
        elif sample_width == 3:
            octets = np.frombuffer(payload, dtype=np.uint8)
            if octets.size % 3:
                raise ValueError("invalid 24-bit PCM WAV payload")
            octets = octets.reshape(-1, 3).astype(np.int32)
            integer = octets[:, 0] | (octets[:, 1] << 8) | (octets[:, 2] << 16)
            integer = (integer ^ 0x800000) - 0x800000
            decoded = integer.astype(np.float32) / 8388608.0
        elif sample_width == 4:
            decoded = np.frombuffer(payload, dtype=np.dtype("<i4")).astype(np.float32)
            decoded /= 2147483648.0
        else:
            raise ValueError(f"unsupported PCM WAV sample width: {sample_width}")
        if decoded.size % channels:
            raise ValueError("WAV payload is not divisible by its channel count")
        waveform = decoded.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    else:
        samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        waveform = samples.mean(axis=1, dtype=np.float32)
    if waveform.size == 0:
        raise ValueError(f"audio file is empty: {path}")
    sample_rate = int(sample_rate)
    if sample_rate != target_sample_rate:
        from scipy.signal import resample_poly
        import math

        divisor = math.gcd(sample_rate, target_sample_rate)
        waveform = resample_poly(
            waveform, target_sample_rate // divisor, sample_rate // divisor
        ).astype(np.float32, copy=False)
        sample_rate = target_sample_rate
    return np.ascontiguousarray(waveform), sample_rate


def _input_sample(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.wav is not None:
        path = Path(args.wav).expanduser().resolve()
        waveform, sample_rate = _load_wav(path)
        sample = {
            "sample_key": path.stem,
            "audio": waveform,
            "sampling_rate": sample_rate,
            "duration_sec": waveform.size / sample_rate,
        }
        return sample, args.prompt_tag or "[Diacritics]"

    dataset = LocalTarParquetASRDataset(
        local_root=args.local_root,
        subset_manifest_path=args.subset_manifest,
        eval_manifest_path=args.eval_manifest,
        seed=args.dataset_seed,
        shuffle_buffer_size=1,
        target_sample_rate=16_000,
    )
    sample = next(islice(iter(dataset), args.dataset_sample_index, None), None)
    if sample is None:
        raise IndexError(f"dataset has no sample at index {args.dataset_sample_index}")
    prompt = args.prompt_tag
    if prompt is None:
        prompt = target_prompt_tag(sample.get("target", sample))
    return sample, prompt


def _noise_process(
    diffusion: DiffusionConfig,
    *,
    uniform_noise_ids: Sequence[int],
):
    if diffusion.noise_type == "mask":
        return MaskNoiseProcess(diffusion.mask_token_id)
    return UniformNoiseProcess(uniform_noise_ids)


def _resolve_sampling_controls(
    resolved_config: Mapping[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    saved = resolved_config.get(
        "sampling",
        {
            "sampler": "confidence_schedule",
            "max_nfe": 64,
            "prediction_mode": "argmax",
            "temperature_start": 0.8,
            "temperature_end": 0.4,
            "entropy_budget_gamma": 0.05,
            "position_bias_lambda": 0.2,
            "adaptive_stopping": False,
        },
    )
    # When changing sampler families, do not inherit incompatible settings
    # from an older Confidence checkpoint. Explicit CLI overrides still win.
    saved = dict(saved)
    if args.sampler is not None and args.sampler != saved["sampler"]:
        if args.sampler in ("argmax_udm", "udm_marginalization"):
            saved.update(
                prediction_mode=("argmax" if args.sampler == "argmax_udm" else "categorical"),
                temperature_start=1.0,
                temperature_end=1.0,
            )
        elif args.sampler == "confidence_schedule":
            saved.update(prediction_mode="argmax", temperature_start=0.8, temperature_end=0.4)
    controls = {
        "sampler": saved["sampler"] if args.sampler is None else args.sampler,
        "max_nfe": int(saved["max_nfe"] if args.nfe is None else args.nfe),
        "prediction_mode": (
            saved["prediction_mode"]
            if args.prediction_mode is None
            else args.prediction_mode
        ),
        "temperature_start": float(
            saved["temperature_start"]
            if args.temperature_start is None
            else args.temperature_start
        ),
        "temperature_end": float(
            saved["temperature_end"]
            if args.temperature_end is None
            else args.temperature_end
        ),
        "entropy_budget_gamma": float(
            saved["entropy_budget_gamma"]
            if args.entropy_budget_gamma is None
            else args.entropy_budget_gamma
        ),
        "position_bias_lambda": float(
            saved["position_bias_lambda"]
            if args.position_bias_lambda is None
            else args.position_bias_lambda
        ),
        "adaptive_stopping": bool(
            saved.get("adaptive_stopping", False)
            if getattr(args, "adaptive_stopping", None) is None
            else args.adaptive_stopping
        ),
    }
    if controls["adaptive_stopping"]:
        raise NotImplementedError(
            "adaptive stopping is requested but is not implemented by the reference sampler; "
            "use --no-adaptive-stopping rather than silently ignoring it"
        )
    return controls


def _validate_sampling_semantics(
    diffusion: DiffusionConfig, sampling: Mapping[str, Any]
) -> None:
    ConfidenceSamplerConfig(**{key: value for key, value in sampling.items()
                               if key != "adaptive_stopping"})
    if sampling["sampler"] in ("argmax_udm", "udm_marginalization") and diffusion.noise_type != "uniform":
        raise ValueError("UDM samplers require a checkpoint trained with uniform noise")
    if sampling["sampler"] in (
        "entropy_bounded",
        "position_biased_entropy_bounded",
    ) and (diffusion.noise_type != "mask" or diffusion.use_time_conditioning):
        raise ValueError(
            "EB/PBEB sampling is restricted to Mask + no_t in V1: its entropy "
            "budget does not enforce a masked fraction matching the explicit t schedule"
        )


def resolve_runtime_token_policy(
    model: torch.nn.Module,
    processor: Any,
    resolved_config: Mapping[str, Any],
    diffusion: DiffusionConfig,
    layout: DiffusionCheckpointLayout,
):
    """Rebuild and verify the exact token policy saved by the trainer.

    This is shared by single-file inference and the two-example overfit
    evaluator.  Keeping one runtime check prevents an evaluation from silently
    sampling with a different MASK/control-token policy than its checkpoint.
    """

    model_vocab_size = int(model.autoregressive_model.get_output_embeddings().weight.shape[0])
    values = resolved_config["diffusion"]
    token_policy = build_diffusion_token_policy(
        processor.tokenizer,
        model_vocab_size=model_vocab_size,
        uniform_vocab_scope=diffusion.uniform_vocab_scope,
        allowed_added_output_token_ids=values.get("allowed_added_output_token_ids", ()),
        extra_forbidden_output_token_ids=values.get(
            "extra_forbidden_output_token_ids", ()
        ),
        mask_token_id=diffusion.mask_token_id,
        tail_fill_mode=diffusion.canvas.tail_fill_mode,
        pad_token_id=diffusion.canvas.pad_token_id,
        turn_end_token_id=diffusion.canvas.turn_end_token_id,
    )
    token_contract = validate_mawjais_token_contract(
        processor.tokenizer,
        processor=processor,
        model_config=model.autoregressive_model.config,
        mask_token_text=values["mask_token_text"],
        mask_token_id=diffusion.mask_token_id,
        pad_token_id=diffusion.canvas.pad_token_id,
        turn_end_token_id=diffusion.canvas.turn_end_token_id,
        forbidden_output_token_ids=token_policy.sampling_forbidden_ids,
    )
    saved_contract = resolved_config.get("runtime", {}).get(
        "resolved_special_token_ids"
    )
    if saved_contract is not None and saved_contract != token_contract.to_metadata():
        raise ValueError(
            "loaded tokenizer/model control-token contract differs from training"
        )
    saved_policy = layout.metadata.get("token_policy")
    if saved_policy is None:
        saved_policy = resolved_config.get("runtime", {}).get("token_policy")
    actual_policy = token_policy.to_metadata()
    policy_fields = (
        "vocab_size",
        "base_vocab_size",
        "uniform_vocab_scope",
        "uniform_noise_id_count",
        "uniform_noise_ids_sha256",
        "training_forbidden_ids",
        "sampling_forbidden_ids",
        "allowed_added_output_ids",
    )
    if saved_policy is None:
        warnings.warn(
            "legacy schema-v1 checkpoint has no token-policy provenance; current "
            "trainer checkpoints record and validate it",
            RuntimeWarning,
        )
    elif not isinstance(saved_policy, Mapping):
        raise ValueError("checkpoint token-policy provenance must be a mapping")
    else:
        mismatched_policy = [
            field for field in policy_fields if saved_policy.get(field) != actual_policy[field]
        ]
        if mismatched_policy:
            raise ValueError(
                "loaded tokenizer/config does not match the checkpoint token policy: "
                f"{mismatched_policy}"
            )
    return token_policy


@torch.no_grad()
def run_inference(
    args: argparse.Namespace,
    layout: DiffusionCheckpointLayout,
    resolved_config: dict[str, Any],
    diffusion: DiffusionConfig,
) -> dict[str, Any]:
    sampling = _resolve_sampling_controls(resolved_config, args)
    _validate_sampling_semantics(diffusion, sampling)
    model, processor = _load_trained_model(layout, diffusion, args)
    sample, prompt_tag = _input_sample(args)
    duration = float(sample["duration_sec"])
    if args.canvas_size is None:
        selector = DurationBucketSelector(
            diffusion.canvas.sizes,
            tokens_per_second=diffusion.canvas.duration_tokens_per_second,
            margin_tokens=diffusion.canvas.duration_margin_tokens,
        )
        canvas_size = selector.select_or_raise(duration)
    else:
        canvas_size = args.canvas_size
        if canvas_size not in diffusion.canvas.sizes:
            raise ValueError(
                f"canvas size {canvas_size} is not one of trained buckets "
                f"{diffusion.canvas.sizes}"
            )

    processor_output = processor(
        text=[build_mawjais_asr_prefix(prompt_tag)],
        audio=[sample["audio"]],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    required = {
        "input_ids",
        "attention_mask",
        "audio_position_mask",
        "input_features",
        "feature_attention_mask",
    }
    missing = required.difference(processor_output)
    if missing:
        raise ValueError(f"processor output is missing fields: {sorted(missing)}")
    if processor_output["audio_position_mask"].dtype != torch.bool:
        raise TypeError("processor audio_position_mask must be torch.bool")

    device = next(model.parameters()).device
    prefix_ids = processor_output["input_ids"].to(device)
    prefix_valid = processor_output["attention_mask"].to(device=device, dtype=torch.bool)
    audio_positions = processor_output["audio_position_mask"].to(device)
    input_features = processor_output["input_features"].to(device)
    feature_mask = processor_output["feature_attention_mask"].to(device)
    projected_audio = model.encode_audio(input_features, feature_mask)
    canvas_valid = torch.ones((1, canvas_size), device=device, dtype=torch.bool)
    token_policy = resolve_runtime_token_policy(
        model,
        processor,
        resolved_config,
        diffusion,
        layout,
    )
    forbidden = token_policy.sampling_forbidden_ids
    process = _noise_process(
        diffusion,
        uniform_noise_ids=token_policy.uniform_noise_ids,
    )

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

    sampler_config = ConfidenceSamplerConfig(
        sampler=sampling["sampler"],
        max_nfe=sampling["max_nfe"],
        prediction_mode=sampling["prediction_mode"],
        temperature_start=sampling["temperature_start"],
        temperature_end=sampling["temperature_end"],
        entropy_budget_gamma=sampling["entropy_budget_gamma"],
        position_bias_lambda=sampling["position_bias_lambda"],
        forbidden_output_token_ids=forbidden,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    sampled = sample_diffusion(
        denoiser,
        process,
        (1, canvas_size),
        device=device,
        config=sampler_config,
        generator=generator,
        canvas_valid_mask=canvas_valid,
        padding_token_id=diffusion.canvas.pad_token_id,
    )
    full_ids = sampled.canvas[0].tolist()
    output_ids = truncate_at_first_turn_end(
        full_ids, turn_end_token_id=diffusion.canvas.turn_end_token_id
    )
    result = {
        "checkpoint": str(layout.root),
        "checkpoint_update": layout.metadata.get("update"),
        "sample_key": sample.get("sample_key"),
        "duration_seconds": duration,
        "prompt_tag": prompt_tag,
        "noise_type": diffusion.noise_type,
        "use_time_conditioning": diffusion.use_time_conditioning,
        "denoiser_output_alignment": diffusion.denoiser_output_alignment,
        "seed": args.seed,
        "canvas_size": canvas_size,
        "sampler": sampling["sampler"],
        "nfe": sampled.nfe,
        "terminated": diffusion.canvas.turn_end_token_id in full_ids,
        "token_ids": output_ids,
        "transcript": processor.tokenizer.decode(output_ids, skip_special_tokens=False).strip(),
    }
    if "transcription" in sample:
        result["reference"] = sample["transcription"]
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PyTorch sampling from a trained MawJais discrete-diffusion checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, help="Trainer checkpoint directory.")
    parser.add_argument("--config", help="Optional matching experiment YAML/JSON override.")
    parser.add_argument("--base-checkpoint", help="Optional local the user-supplied base model snapshot override.")
    parser.add_argument(
        "--fsdp-merged-weights",
        help=(
            "Temporary model.safetensors (or its directory) exported from an "
            "Accelerate FSDP sharded checkpoint. Required only for that format."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wav", help="Local WAV/FLAC/OGG input path.")
    source.add_argument(
        "--dataset-sample-index",
        type=int,
        help="Zero-based item from deterministic local TAR streaming order.",
    )
    parser.add_argument(
        "--local-root",
        default=str(REPOSITORY_ROOT / "data/legal-ytb"),
    )
    parser.add_argument(
        "--subset-manifest",
        default=str(REPOSITORY_ROOT / "data/subset-manifest.json"),
    )
    parser.add_argument(
        "--eval-manifest",
        default=str(REPOSITORY_ROOT / "data/validation-manifest.jsonl"),
    )
    parser.add_argument("--dataset-seed", type=int, default=3407)
    parser.add_argument("--prompt-tag", help="System tag; dataset metadata is used when omitted.")
    parser.add_argument("--noise-type", choices=("mask", "uniform"))
    parser.add_argument(
        "--alignment", choices=("same_position", "previous_position")
    )
    time = parser.add_mutually_exclusive_group()
    time.add_argument("--with-time", dest="time_conditioning", action="store_true")
    time.add_argument("--no-time", dest="time_conditioning", action="store_false")
    parser.set_defaults(time_conditioning=None)
    parser.add_argument("--nfe", type=int, default=None)
    parser.add_argument("--seed", type=int, default=3421)
    parser.add_argument("--canvas-size", type=int)
    parser.add_argument(
        "--sampler",
        choices=(
            "confidence_schedule",
            "argmax_udm",
            "udm_marginalization",
            "entropy_bounded",
            "position_biased_entropy_bounded",
        ),
        default=None,
    )
    parser.add_argument("--prediction-mode", choices=("argmax", "categorical"), default=None)
    parser.add_argument("--temperature-start", type=float, default=None)
    parser.add_argument("--temperature-end", type=float, default=None)
    parser.add_argument("--entropy-budget-gamma", type=float, default=None)
    parser.add_argument("--position-bias-lambda", type=float, default=None)
    adaptive = parser.add_mutually_exclusive_group()
    adaptive.add_argument(
        "--adaptive-stopping", dest="adaptive_stopping", action="store_true"
    )
    adaptive.add_argument(
        "--no-adaptive-stopping", dest="adaptive_stopping", action="store_false"
    )
    parser.set_defaults(adaptive_stopping=None)
    parser.add_argument("--device", default="cuda", help="ROCm PyTorch uses the CUDA device API.")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--attention-backend", choices=("eager", "sdpa"), default="eager")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate checkpoint/config/arguments without loading the 4B model or opening TARs.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.nfe is not None and args.nfe <= 0:
        raise ValueError("--nfe must be positive")
    if args.seed < 0 or args.dataset_seed < 0:
        raise ValueError("seeds must be non-negative")
    if args.dataset_sample_index is not None and args.dataset_sample_index < 0:
        raise ValueError("--dataset-sample-index must be non-negative")
    if args.canvas_size is not None and args.canvas_size <= 0:
        raise ValueError("--canvas-size must be positive")
    if (args.temperature_start is not None and args.temperature_start <= 0) or (
        args.temperature_end is not None and args.temperature_end <= 0
    ):
        raise ValueError("temperatures must be positive")
    if (args.entropy_budget_gamma is not None and args.entropy_budget_gamma < 0) or (
        args.position_bias_lambda is not None and args.position_bias_lambda < 0
    ):
        raise ValueError("entropy budget and position bias must be non-negative")
    if args.wav is not None and not Path(args.wav).expanduser().is_file():
        raise FileNotFoundError(args.wav)
    if args.dataset_sample_index is not None:
        for path in (args.local_root, args.subset_manifest, args.eval_manifest):
            if not Path(path).expanduser().exists():
                raise FileNotFoundError(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
        layout = read_diffusion_checkpoint_layout(args.checkpoint)
        resolved = _load_resolved_config(layout, args.config)
        diffusion = _resolve_semantics(resolved, args)
        sampling = _resolve_sampling_controls(resolved, args)
        _validate_sampling_semantics(diffusion, sampling)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "status": "validated",
                        "checkpoint": str(layout.root),
                        "strategy": layout.strategy,
                        "noise_type": diffusion.noise_type,
                        "use_time_conditioning": diffusion.use_time_conditioning,
                        "denoiser_output_alignment": diffusion.denoiser_output_alignment,
                        "sampler": sampling["sampler"],
                        "nfe": sampling["max_nfe"],
                        "model_loaded": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        result = run_inference(args, layout, resolved, diffusion)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (FileNotFoundError, IndexError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
        return 2  # pragma: no cover - argparse exits


if __name__ == "__main__":
    raise SystemExit(main())
