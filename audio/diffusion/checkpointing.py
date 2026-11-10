"""Small, explicit checkpoint contract for diffusion-only MawJais parameters.

The 4B base weights (or a PEFT adapter) are stored separately.  This file keeps
the dedicated MASK embedding and optional time MLP portable between the trainer
and the PyTorch inference script without serializing the full wrapper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch


DIFFUSION_CHECKPOINT_SCHEMA_VERSION = 1


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def save_diffusion_component_state(model: torch.nn.Module, path: str | Path) -> Path:
    """Save only parameters owned by ``MawJaisDiffusionASRModel``.

    Exact payload keys are stable and intentionally few:

    ``schema_version``
        Integer contract version (currently 1).
    ``diffusion_mask_embedding``
        CPU tensor matching the wrapper parameter.
    ``time_mlp``
        CPU ``state_dict`` mapping when time conditioning exists, otherwise
        ``None``.
    """

    if not hasattr(model, "diffusion_mask_embedding"):
        raise TypeError("model does not expose diffusion_mask_embedding")
    time_mlp = getattr(model, "time_mlp", None)
    payload: dict[str, Any] = {
        "schema_version": DIFFUSION_CHECKPOINT_SCHEMA_VERSION,
        "diffusion_mask_embedding": model.diffusion_mask_embedding.detach().cpu().clone(),
        "time_mlp": None if time_mlp is None else _cpu_state_dict(time_mlp),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_diffusion_component_state(
    model: torch.nn.Module,
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Mapping[str, Any]:
    """Load diffusion-only state with strict schema and shape checks."""

    source = Path(path)
    payload = torch.load(source, map_location=map_location, weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"invalid diffusion component checkpoint: {source}")
    if payload.get("schema_version") != DIFFUSION_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported diffusion state schema {payload.get('schema_version')!r}; "
            f"expected {DIFFUSION_CHECKPOINT_SCHEMA_VERSION}"
        )
    expected_keys = {"schema_version", "diffusion_mask_embedding", "time_mlp"}
    extra_keys = set(payload).difference(expected_keys)
    missing_keys = expected_keys.difference(payload)
    if extra_keys or missing_keys:
        raise ValueError(
            f"invalid diffusion state keys: missing={sorted(missing_keys)}, "
            f"extra={sorted(extra_keys)}"
        )
    if not hasattr(model, "diffusion_mask_embedding"):
        raise TypeError("model does not expose diffusion_mask_embedding")
    saved_mask = payload["diffusion_mask_embedding"]
    current_mask = model.diffusion_mask_embedding
    if not isinstance(saved_mask, torch.Tensor) or saved_mask.shape != current_mask.shape:
        raise ValueError(
            "diffusion_mask_embedding shape mismatch: "
            f"saved={getattr(saved_mask, 'shape', None)}, expected={current_mask.shape}"
        )
    with torch.no_grad():
        current_mask.copy_(saved_mask.to(device=current_mask.device, dtype=current_mask.dtype))

    saved_time = payload["time_mlp"]
    current_time = getattr(model, "time_mlp", None)
    if current_time is None and saved_time is not None:
        raise ValueError("checkpoint contains time_mlp but the current experiment disables it")
    if current_time is not None and saved_time is None:
        raise ValueError("checkpoint has no time_mlp but the current experiment requires it")
    if current_time is not None:
        if not isinstance(saved_time, Mapping):
            raise ValueError("time_mlp must be a state_dict mapping")
        current_time.load_state_dict(dict(saved_time), strict=True)
    return payload


def write_diffusion_checkpoint_manifest(
    checkpoint_dir: str | Path,
    manifest: Mapping[str, Any],
) -> Path:
    """Write the human-readable checkpoint assembly manifest atomically."""

    payload = dict(manifest)
    payload.setdefault("schema_version", DIFFUSION_CHECKPOINT_SCHEMA_VERSION)
    if payload["schema_version"] != DIFFUSION_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint manifest schema_version must be 1")
    required = {
        "strategy",
        "base_checkpoint",
        "revision",
        "update",
        "diffusion_state",
        "resolved_config",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"checkpoint manifest is missing fields: {sorted(missing)}")
    if payload["strategy"] not in ("lora", "full"):
        raise ValueError("checkpoint strategy must be 'lora' or 'full'")
    strategy_key = "adapter_dir" if payload["strategy"] == "lora" else "full_model_dir"
    if strategy_key not in payload:
        raise ValueError(f"checkpoint manifest requires {strategy_key!r}")

    destination = Path(checkpoint_dir) / "diffusion_checkpoint.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def read_diffusion_checkpoint_manifest(path: str | Path) -> dict[str, Any]:
    """Read and validate a checkpoint manifest written by this module."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("diffusion checkpoint manifest must be a JSON object")
    # Reuse validation without leaving a file behind by checking fields here.
    if payload.get("schema_version") != DIFFUSION_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported diffusion checkpoint manifest schema")
    if payload.get("strategy") not in ("lora", "full"):
        raise ValueError("invalid diffusion checkpoint strategy")
    required = {
        "base_checkpoint",
        "revision",
        "update",
        "diffusion_state",
        "resolved_config",
    }
    missing = required.difference(payload)
    strategy_key = "adapter_dir" if payload.get("strategy") == "lora" else "full_model_dir"
    if missing or strategy_key not in payload:
        raise ValueError(
            f"incomplete diffusion checkpoint manifest: missing={sorted(missing)}, "
            f"strategy_key={strategy_key!r}"
        )
    return payload


__all__ = [
    "DIFFUSION_CHECKPOINT_SCHEMA_VERSION",
    "load_diffusion_component_state",
    "read_diffusion_checkpoint_manifest",
    "save_diffusion_component_state",
    "write_diffusion_checkpoint_manifest",
]
