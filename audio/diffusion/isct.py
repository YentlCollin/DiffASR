"""Stop-gradient construction for two-stage iterative self-correction training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .noise_process import DiscreteNoiseProcess, sample_diffusion_times, validate_diffusion_time
from .sampler import mask_forbidden_logits


@dataclass(frozen=True)
class ISCTSecondStageBatch:
    """Inputs for stage two; ``clean_target_ids`` always remains the true x0."""

    predicted_x0_ids: torch.Tensor
    noisy_canvas_ids: torch.Tensor
    corrupt_mask: torch.Tensor
    noise_level: torch.Tensor
    clean_target_ids: torch.Tensor
    canvas_valid_mask: torch.Tensor


@torch.no_grad()
def build_isct_second_stage(
    first_stage_logits: torch.Tensor,
    clean_target_ids: torch.Tensor,
    canvas_valid_mask: torch.Tensor,
    noise_process: DiscreteNoiseProcess,
    *,
    forbidden_output_token_ids: Sequence[int],
    min_diffusion_time: float,
    full_noise_probability: float = 0.0,
    pad_token_id: int = 1,
    noise_level: torch.Tensor | None = None,
    time_generator: torch.Generator | None = None,
    corruption_generator: torch.Generator | None = None,
    token_noise_generator: torch.Generator | None = None,
    prediction_mode: str = "argmax",
    stop_gradient: bool = True,
) -> ISCTSecondStageBatch:
    """Create stage-two ``xt`` from the detached stage-one prediction.

    Noise is applied to the model prediction, but the second denoising loss must
    still use ``clean_target_ids``. Only the inexpensive argmax V1 is supported;
    this avoids materializing an additional full-vocabulary soft target.
    """

    if prediction_mode != "argmax":
        raise ValueError("the V1 ISCT implementation supports prediction_mode='argmax' only")
    if stop_gradient is not True:
        raise ValueError("ISCT stage-one prediction must use stop_gradient=true")
    if first_stage_logits.ndim != 3:
        raise ValueError("first_stage_logits must have shape [batch, canvas, vocab]")
    if clean_target_ids.shape != first_stage_logits.shape[:2]:
        raise ValueError("clean_target_ids must match the first two logit dimensions")
    if canvas_valid_mask.shape != clean_target_ids.shape or canvas_valid_mask.dtype != torch.bool:
        raise ValueError("canvas_valid_mask must be boolean and match clean_target_ids")
    if not 0 < min_diffusion_time <= 1:
        raise ValueError("min_diffusion_time must be in (0, 1]")

    filtered = mask_forbidden_logits(first_stage_logits.detach(), forbidden_output_token_ids)
    predicted = filtered.argmax(dim=-1).detach()
    predicted = predicted.masked_fill(~canvas_valid_mask, int(pad_token_id))

    if noise_level is None:
        noise_level = sample_diffusion_times(
            clean_target_ids.shape[0],
            device=clean_target_ids.device,
            generator=time_generator,
            full_noise_probability=full_noise_probability,
        ).clamp_min(min_diffusion_time)
    else:
        validate_diffusion_time(noise_level, clean_target_ids.shape[0], clean_target_ids.device)
        if (noise_level <= 0).any():
            raise ValueError("ISCT training noise levels must be strictly positive")

    noisy, corrupt = noise_process.q_sample(
        predicted,
        noise_level,
        generator=corruption_generator,
        noise_generator=token_noise_generator,
    )
    corrupt = corrupt & canvas_valid_mask
    noisy = noisy.masked_fill(~canvas_valid_mask, int(pad_token_id))
    return ISCTSecondStageBatch(
        predicted_x0_ids=predicted,
        noisy_canvas_ids=noisy,
        corrupt_mask=corrupt,
        noise_level=noise_level,
        clean_target_ids=clean_target_ids,
        canvas_valid_mask=canvas_valid_mask,
    )


__all__ = ["ISCTSecondStageBatch", "build_isct_second_stage"]
