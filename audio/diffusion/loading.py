"""Load the user-supplied base model weights with the versioned repository PyTorch implementation."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import torch


DEFAULT_REPOSITORY_PACKAGE = "_mawjais_diffusion_repository"


@dataclass(frozen=True)
class MawJaisRepositoryClasses:
    config_class: type
    processor_class: type
    model_class: type
    package_name: str


def load_repository_classes(
    repository_root: str | Path,
    *,
    package_name: str = DEFAULT_REPOSITORY_PACKAGE,
) -> MawJaisRepositoryClasses:
    """Import composite/backbone files through one repository package.

    The Hugging Face checkpoint contains a snapshot of its Python files. Loading
    it through ``trust_remote_code`` would silently execute those old copies and
    ignore the diffusion changes in this repository. This loader deliberately
    combines ``audio/model`` and ``huggingface/xllm`` under one package, so all
    relative imports resolve to the versioned working tree while tensor weights
    still come from the user-supplied base model. The default package is physically importable by a
    fresh worker; custom package names are synthetic and process-local.
    """

    root = Path(repository_root).resolve()
    model_dir = root / "audio" / "model"
    backbone_dir = root / "huggingface" / "xllm"
    required = (
        model_dir / "configuration_xllm_aut.py",
        model_dir / "processing_xllm_aut.py",
        model_dir / "modeling_xllm_aut.py",
        backbone_dir / "configuration_xllm.py",
        backbone_dir / "modeling_xllm.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Base-model implementation withheld pending redistribution authorization. "
                                "See docs/INTEGRATION.md. Missing local files: " + str(missing))

    existing = sys.modules.get(package_name)
    if existing is None:
        if package_name == DEFAULT_REPOSITORY_PACKAGE:
            # The default name has a physical top-level package in the working
            # tree.  Import it normally so a fresh spawn/forkserver worker can
            # repeat the import while unpickling the processor/collator.  The
            # package __init__ only exposes paths and never loads model weights.
            package = importlib.import_module(package_name)
        else:
            # Custom names remain useful for isolated loader tests and callers
            # that deliberately keep multiple worktrees in one interpreter.
            package = ModuleType(package_name)
            package.__path__ = [str(model_dir), str(backbone_dir)]
            package.__package__ = package_name
            sys.modules[package_name] = package
    else:
        package = existing

    package_path = getattr(package, "__path__", None)
    if package_path is None:
        raise RuntimeError(f"repository package {package_name!r} is not a package")
    package_paths = tuple(str(Path(path).resolve()) for path in package_path)
    expected_paths = (str(model_dir), str(backbone_dir))
    if package_paths != expected_paths:
        raise RuntimeError(
            f"repository package {package_name!r} points to {package_paths}, "
            f"expected {expected_paths}"
        )

    # Import dependencies first so the composite's relative imports cannot fall
    # back to an unrelated top-level module left by trust_remote_code.
    importlib.import_module(f"{package_name}.configuration_xllm")
    importlib.import_module(f"{package_name}.modeling_xllm")
    config_module = importlib.import_module(f"{package_name}.configuration_xllm_aut")
    processor_module = importlib.import_module(f"{package_name}.processing_xllm_aut")
    model_module = importlib.import_module(f"{package_name}.modeling_xllm_aut")
    return MawJaisRepositoryClasses(
        config_class=config_module.XllmAuTConfig,
        processor_class=processor_module.XllmAuTProcessor,
        model_class=model_module.XllmAuTForConditionalGeneration,
        package_name=package_name,
    )


def load_mawjais_r2_for_diffusion(
    checkpoint: str | Path,
    *,
    repository_root: str | Path,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str | Mapping[str, str] = "sdpa",
    device_map: Any = None,
    low_cpu_mem_usage: bool = True,
):
    """Return ``(model, processor)`` using repository code and checkpoint tensors."""

    classes = load_repository_classes(repository_root)
    checkpoint = str(Path(checkpoint).expanduser()) if Path(str(checkpoint)).expanduser().exists() else str(checkpoint)
    config = classes.config_class.from_pretrained(checkpoint)
    processor = classes.processor_class.from_pretrained(checkpoint)
    model = classes.model_class.from_pretrained(
        checkpoint,
        config=config,
        dtype=dtype,
        attn_implementation=attn_implementation,
        device_map=device_map,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    return model, processor


__all__ = [
    "DEFAULT_REPOSITORY_PACKAGE",
    "MawJaisRepositoryClasses",
    "load_mawjais_r2_for_diffusion",
    "load_repository_classes",
]
