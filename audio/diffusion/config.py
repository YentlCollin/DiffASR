"""Validated, model-independent configuration for discrete ASR diffusion.

The 4B MawJais wrapper deliberately does not depend on this module yet.  Keeping
the experiment semantics here makes them testable before changing the model
forward path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal


NoiseType = Literal["uniform", "mask"]
Alignment = Literal["same_position", "previous_position"]
TailFillMode = Literal["eos_repeat", "eos_then_pad"]
ObjectiveName = Literal["auto", "x0_ce_all", "mdm_absorbing"]
LossPositions = Literal["auto", "all", "corrupted_only"]
LossWeighting = Literal["auto", "none", "inverse_t"]
SamplerName = Literal[
    "confidence_schedule",
    "entropy_bounded",
    "position_biased_entropy_bounded",
    "full_argmax_refinement",
    "diffusion_gemma_entropy",
    "udm_marginalization",
    "udm_tempered_marginalization",
    "udm_marginalization_final_argmax",
    "argmax_udm",
    "mdlm_ancestral",
]


@dataclass(frozen=True)
class ResolvedObjective:
    """A fully resolved loss profile with no ``auto`` values left."""

    name: Literal["x0_ce_all", "mdm_absorbing"]
    positions: Literal["all", "corrupted_only"]
    weighting: Literal["none", "inverse_t"]


def resolve_objective(
    noise_type: NoiseType,
    objective: ObjectiveName = "auto",
    positions: LossPositions = "auto",
    weighting: LossWeighting = "auto",
) -> ResolvedObjective:
    """Resolve and validate the objective associated with a noise process.

    Uniform diffusion defaults to direct reconstruction of every allocated
    canvas position.  Absorbing-mask diffusion defaults to the MDM estimator:
    masked positions only, weighted by ``1/t``.  ``x0_ce_all`` remains valid for
    mask noise as an explicit cross-process ablation.
    """

    if noise_type not in ("uniform", "mask"):
        raise ValueError(f"unsupported noise_type: {noise_type!r}")
    if objective not in ("auto", "x0_ce_all", "mdm_absorbing"):
        raise ValueError(f"unsupported diffusion objective: {objective!r}")
    if positions not in ("auto", "all", "corrupted_only"):
        raise ValueError(f"unsupported loss_positions: {positions!r}")
    if weighting not in ("auto", "none", "inverse_t"):
        raise ValueError(f"unsupported loss_weighting: {weighting!r}")

    resolved_name = (
        "x0_ce_all" if noise_type == "uniform" else "mdm_absorbing"
    ) if objective == "auto" else objective

    if resolved_name == "mdm_absorbing" and noise_type != "mask":
        raise ValueError("mdm_absorbing requires noise_type='mask'")

    expected_positions = "all" if resolved_name == "x0_ce_all" else "corrupted_only"
    expected_weighting = "none" if resolved_name == "x0_ce_all" else "inverse_t"
    resolved_positions = expected_positions if positions == "auto" else positions
    resolved_weighting = expected_weighting if weighting == "auto" else weighting

    if resolved_positions != expected_positions:
        raise ValueError(
            f"{resolved_name} requires loss_positions={expected_positions!r}, "
            f"got {resolved_positions!r}"
        )
    if resolved_weighting != expected_weighting:
        raise ValueError(
            f"{resolved_name} requires loss_weighting={expected_weighting!r}, "
            f"got {resolved_weighting!r}"
        )

    return ResolvedObjective(
        name=resolved_name,
        positions=resolved_positions,
        weighting=resolved_weighting,
    )


@dataclass(frozen=True)
class CanvasConfig:
    """Duration-only canvas selection and semantic tail representation."""

    sizes: tuple[int, ...] = (64, 128, 256, 384, 512)
    duration_tokens_per_second: float = 9.0
    duration_margin_tokens: float = 100.0
    tail_fill_mode: TailFillMode = "eos_repeat"
    turn_end_token_id: int = 250019
    pad_token_id: int = 1
    eos_tail_loss_weight: float = 1.0
    pad_loss_weight: float = 0.1

    def __post_init__(self) -> None:
        if not self.sizes or any(size <= 0 for size in self.sizes):
            raise ValueError("canvas sizes must be non-empty and strictly positive")
        if tuple(sorted(set(self.sizes))) != self.sizes:
            raise ValueError("canvas sizes must be sorted and unique")
        if self.duration_tokens_per_second <= 0:
            raise ValueError("duration_tokens_per_second must be > 0")
        if self.duration_margin_tokens < 0:
            raise ValueError("duration_margin_tokens must be >= 0")
        if self.tail_fill_mode not in ("eos_repeat", "eos_then_pad"):
            raise ValueError(f"unsupported tail_fill_mode: {self.tail_fill_mode!r}")
        if self.turn_end_token_id == self.pad_token_id:
            raise ValueError("turn-end and PAD token IDs must be distinct")
        if self.eos_tail_loss_weight < 0 or self.pad_loss_weight < 0:
            raise ValueError("tail loss weights must be non-negative")


@dataclass(frozen=True)
class DiffusionConfig:
    """Core diffusion semantics shared by trainer and inference code."""

    noise_type: NoiseType = "mask"
    use_time_conditioning: bool = False
    denoiser_output_alignment: Alignment = "same_position"
    min_diffusion_time: float = 1.0e-3
    full_noise_probability: float = 0.0
    mask_token_id: int = 250064
    uniform_vocab_scope: Literal["full_vocab", "safe_vocab"] = "safe_vocab"
    diffusion_objective: ObjectiveName = "auto"
    loss_positions: LossPositions = "auto"
    loss_weighting: LossWeighting = "auto"
    canvas: CanvasConfig = field(default_factory=CanvasConfig)

    def __post_init__(self) -> None:
        if self.noise_type not in ("uniform", "mask"):
            raise ValueError(f"unsupported noise_type: {self.noise_type!r}")
        if self.denoiser_output_alignment not in ("same_position", "previous_position"):
            raise ValueError(
                "denoiser_output_alignment must be 'same_position' or 'previous_position'"
            )
        if not 0 < self.min_diffusion_time <= 1:
            raise ValueError("min_diffusion_time must be in (0, 1]")
        if not 0 <= self.full_noise_probability <= 1:
            raise ValueError("full_noise_probability must be in [0, 1]")
        if self.uniform_vocab_scope not in ("full_vocab", "safe_vocab"):
            raise ValueError(f"unsupported uniform_vocab_scope: {self.uniform_vocab_scope!r}")
        if self.mask_token_id in (
            self.canvas.turn_end_token_id,
            self.canvas.pad_token_id,
        ):
            raise ValueError("mask_token_id must differ from TURN_END and PAD")
        # Eager validation makes bad experiment overrides fail before a long job.
        resolve_objective(
            self.noise_type,
            self.diffusion_objective,
            self.loss_positions,
            self.loss_weighting,
        )

    @property
    def resolved_objective(self) -> ResolvedObjective:
        return resolve_objective(
            self.noise_type,
            self.diffusion_objective,
            self.loss_positions,
            self.loss_weighting,
        )


@dataclass(frozen=True)
class ConfidenceSamplerConfig:
    """Validated settings for both legacy and exact ancestral samplers.

    The class name is retained for checkpoint/config compatibility.  The two
    paper samplers deliberately require categorical sampling at temperature
    one; argmax or temperature scaling would define different algorithms.
    """

    sampler: SamplerName = "confidence_schedule"
    max_nfe: int = 64
    prediction_mode: Literal["argmax", "categorical"] = "argmax"
    temperature_start: float = 0.8
    temperature_end: float = 0.4
    entropy_budget_gamma: float = 0.05
    position_bias_lambda: float = 0.2
    forbidden_output_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.sampler not in (
            "confidence_schedule",
            "entropy_bounded",
            "position_biased_entropy_bounded",
            "full_argmax_refinement",
            "diffusion_gemma_entropy",
            "udm_marginalization",
            "udm_tempered_marginalization",
            "udm_marginalization_final_argmax",
            "argmax_udm",
            "mdlm_ancestral",
        ):
            raise ValueError(f"unsupported sampler: {self.sampler!r}")
        if self.max_nfe <= 0:
            raise ValueError("max_nfe must be > 0")
        if self.prediction_mode not in ("argmax", "categorical"):
            raise ValueError(f"unsupported prediction_mode: {self.prediction_mode!r}")
        if (
            not math.isfinite(self.temperature_start)
            or not math.isfinite(self.temperature_end)
            or self.temperature_start <= 0
            or self.temperature_end <= 0
        ):
            raise ValueError("sampling temperatures must be > 0")
        if self.entropy_budget_gamma < 0:
            raise ValueError("entropy_budget_gamma must be >= 0")
        if self.position_bias_lambda < 0:
            raise ValueError("position_bias_lambda must be >= 0")
        if any(token_id < 0 for token_id in self.forbidden_output_token_ids):
            raise ValueError("forbidden token IDs must be non-negative")
        if self.sampler == "full_argmax_refinement" and self.prediction_mode != "argmax":
            raise ValueError(
                "full_argmax_refinement requires prediction_mode='argmax'"
            )
        if self.sampler == "diffusion_gemma_entropy" and self.prediction_mode != "categorical":
            raise ValueError(
                "diffusion_gemma_entropy requires prediction_mode='categorical'"
            )
        if self.sampler in (
            "udm_marginalization",
            "udm_marginalization_final_argmax",
            "mdlm_ancestral",
        ):
            if self.prediction_mode != "categorical":
                raise ValueError(
                    f"{self.sampler} requires categorical intermediate transitions "
                    "and prediction_mode='categorical'"
                )
            if self.temperature_start != 1.0 or self.temperature_end != 1.0:
                raise ValueError(
                    f"{self.sampler} requires untempered intermediate transitions "
                    "with temperature_start=temperature_end=1.0"
                )
        if (
            self.sampler == "udm_tempered_marginalization"
            and self.prediction_mode != "categorical"
        ):
            raise ValueError(
                "udm_tempered_marginalization requires categorical bridge "
                "transitions and prediction_mode='categorical'"
            )
        if self.sampler == "argmax_udm":
            if self.prediction_mode != "argmax":
                raise ValueError(
                    "argmax_udm requires prediction_mode='argmax' for the clean "
                    "x0 hypothesis; its UDM bridge remains categorical"
                )
            if self.temperature_start != 1.0 or self.temperature_end != 1.0:
                raise ValueError(
                    "argmax_udm requires temperature_start=temperature_end=1.0"
                )


__all__ = [
    "Alignment",
    "CanvasConfig",
    "ConfidenceSamplerConfig",
    "DiffusionConfig",
    "LossPositions",
    "LossWeighting",
    "NoiseType",
    "ObjectiveName",
    "ResolvedObjective",
    "SamplerName",
    "TailFillMode",
    "resolve_objective",
]
