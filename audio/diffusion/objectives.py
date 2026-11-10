"""Denoising objectives without any implicit causal label shift."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .canvas import MAWJAIS_PAD_TOKEN_ID, MAWJAIS_TURN_END_TOKEN_ID
from .config import ResolvedObjective, TailFillMode
from .noise_process import validate_diffusion_time


@dataclass(frozen=True)
class CanvasPositionMasks:
    lexical: torch.Tensor
    first_turn_end: torch.Tensor
    tail_turn_end: torch.Tensor
    pad: torch.Tensor


@dataclass(frozen=True)
class DenoisingLossOutput:
    """Loss plus tensors needed for decomposed training diagnostics."""

    loss: torch.Tensor
    per_sample_loss: torch.Tensor
    per_token_ce: torch.Tensor
    denominator: torch.Tensor
    effective_mask: torch.Tensor
    position_weight: torch.Tensor
    position_masks: CanvasPositionMasks
    corrupted_ce: torch.Tensor
    uncorrupted_ce: torch.Tensor
    lexical_ce: torch.Tensor
    first_turn_end_ce: torch.Tensor
    tail_turn_end_ce: torch.Tensor
    pad_ce: torch.Tensor


def classify_canvas_positions(
    clean_canvas_ids: torch.Tensor,
    *,
    canvas_valid_mask: torch.Tensor | None = None,
    turn_end_token_id: int = MAWJAIS_TURN_END_TOKEN_ID,
    pad_token_id: int = MAWJAIS_PAD_TOKEN_ID,
) -> CanvasPositionMasks:
    """Distinguish transcript, first turn-end, repeated tail, and semantic PAD."""

    if clean_canvas_ids.ndim != 2:
        raise ValueError("clean_canvas_ids must have shape [batch, canvas]")
    if clean_canvas_ids.is_floating_point() or clean_canvas_ids.dtype == torch.bool:
        raise TypeError("clean_canvas_ids must contain integer token IDs")
    valid = (
        torch.ones_like(clean_canvas_ids, dtype=torch.bool)
        if canvas_valid_mask is None
        else canvas_valid_mask
    )
    if valid.shape != clean_canvas_ids.shape or valid.dtype != torch.bool:
        raise ValueError("canvas_valid_mask must be boolean and match clean_canvas_ids")
    if valid.device != clean_canvas_ids.device:
        raise ValueError("canvas_valid_mask and clean_canvas_ids must share a device")

    is_turn_end = (clean_canvas_ids == turn_end_token_id) & valid
    turn_end_ordinal = is_turn_end.to(torch.int64).cumsum(dim=1)
    first_turn_end = is_turn_end & (turn_end_ordinal == 1)
    tail_turn_end = is_turn_end & (turn_end_ordinal > 1)
    pad = (clean_canvas_ids == pad_token_id) & valid
    lexical = valid & (turn_end_ordinal == 0) & ~is_turn_end & ~pad
    return CanvasPositionMasks(
        lexical=lexical,
        first_turn_end=first_turn_end,
        tail_turn_end=tail_turn_end,
        pad=pad,
    )


def build_position_weights(
    clean_canvas_ids: torch.Tensor,
    *,
    canvas_valid_mask: torch.Tensor | None = None,
    tail_fill_mode: TailFillMode = "eos_repeat",
    eos_tail_loss_weight: float = 1.0,
    pad_loss_weight: float = 0.1,
    turn_end_token_id: int = MAWJAIS_TURN_END_TOKEN_ID,
    pad_token_id: int = MAWJAIS_PAD_TOKEN_ID,
) -> tuple[torch.Tensor, CanvasPositionMasks]:
    if tail_fill_mode not in ("eos_repeat", "eos_then_pad"):
        raise ValueError(f"unsupported tail_fill_mode: {tail_fill_mode!r}")
    if eos_tail_loss_weight < 0 or pad_loss_weight < 0:
        raise ValueError("position loss weights must be non-negative")
    masks = classify_canvas_positions(
        clean_canvas_ids,
        canvas_valid_mask=canvas_valid_mask,
        turn_end_token_id=turn_end_token_id,
        pad_token_id=pad_token_id,
    )
    weights = torch.ones_like(clean_canvas_ids, dtype=torch.float32)
    if tail_fill_mode == "eos_repeat":
        weights = weights.masked_fill(masks.tail_turn_end, float(eos_tail_loss_weight))
    else:
        weights = weights.masked_fill(masks.pad, float(pad_loss_weight))
    return weights, masks


def _validate_loss_inputs(
    logits: torch.Tensor,
    clean_canvas_ids: torch.Tensor,
    corrupt_mask: torch.Tensor,
    canvas_valid_mask: torch.Tensor | None,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, canvas, vocab]")
    if clean_canvas_ids.shape != logits.shape[:2]:
        raise ValueError(
            f"clean canvas shape {tuple(clean_canvas_ids.shape)} does not match logits {tuple(logits.shape[:2])}"
        )
    if corrupt_mask.shape != clean_canvas_ids.shape or corrupt_mask.dtype != torch.bool:
        raise ValueError("corrupt_mask must be boolean and match clean_canvas_ids")
    if clean_canvas_ids.device != logits.device or corrupt_mask.device != logits.device:
        raise ValueError("logits, clean_canvas_ids, and corrupt_mask must share a device")
    if clean_canvas_ids.is_floating_point() or clean_canvas_ids.dtype == torch.bool:
        raise TypeError("clean_canvas_ids must contain integer token IDs")
    if (clean_canvas_ids < 0).any() or (clean_canvas_ids >= logits.shape[-1]).any():
        raise ValueError("clean target token IDs must fall inside the logits vocabulary")
    valid = (
        torch.ones_like(clean_canvas_ids, dtype=torch.bool)
        if canvas_valid_mask is None
        else canvas_valid_mask
    )
    if valid.shape != clean_canvas_ids.shape or valid.dtype != torch.bool:
        raise ValueError("canvas_valid_mask must be boolean and match clean_canvas_ids")
    if valid.device != logits.device:
        raise ValueError("canvas_valid_mask and logits must share a device")
    return valid


def _mask_forbidden_logits(logits: torch.Tensor, forbidden_ids: Sequence[int]) -> torch.Tensor:
    ids = tuple(sorted(set(int(token_id) for token_id in forbidden_ids)))
    if not ids:
        return logits
    if ids[0] < 0 or ids[-1] >= logits.shape[-1]:
        raise ValueError("a forbidden token ID is outside the logits vocabulary")
    masked = logits.clone()
    masked[..., list(ids)] = -torch.inf
    return masked


def _validate_targets_are_allowed(
    clean_canvas_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    forbidden_ids: Sequence[int],
) -> None:
    """Reject a contradictory objective before it produces an infinite CE.

    Sampling-only control IDs may be forbidden safely, but every semantic
    target in an allocated canvas must remain reachable. This matters in
    particular for ``eos_then_pad``: PAD is a real, weighted target in that
    profile and cannot simultaneously be removed from the training logits.
    """

    ids = tuple(sorted(set(int(token_id) for token_id in forbidden_ids)))
    if not ids:
        return
    forbidden = torch.as_tensor(ids, device=clean_canvas_ids.device)
    conflict = valid_mask & torch.isin(clean_canvas_ids, forbidden)
    if conflict.any():
        conflicting_ids = torch.unique(clean_canvas_ids[conflict]).tolist()
        raise ValueError(
            "forbidden_output_token_ids contain allocated clean targets: "
            f"{conflicting_ids}. Use separate training and sampling policies."
        )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    selected_weight = weights * mask.to(weights.dtype)
    numerator = (values * selected_weight).sum()
    denominator = selected_weight.sum()
    # Preserve a finite differentiable zero when a component is absent.
    return numerator / denominator.clamp_min(1.0)


def _finish_output(
    *,
    per_token_ce: torch.Tensor,
    per_sample_loss: torch.Tensor,
    denominator: torch.Tensor,
    effective_mask: torch.Tensor,
    position_weight: torch.Tensor,
    masks: CanvasPositionMasks,
    corrupt_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> DenoisingLossOutput:
    metric_weight = position_weight * valid_mask.to(position_weight.dtype)
    return DenoisingLossOutput(
        loss=per_sample_loss.mean(),
        per_sample_loss=per_sample_loss,
        per_token_ce=per_token_ce,
        denominator=denominator,
        effective_mask=effective_mask,
        position_weight=position_weight,
        position_masks=masks,
        corrupted_ce=_masked_mean(per_token_ce, corrupt_mask & valid_mask, metric_weight),
        uncorrupted_ce=_masked_mean(per_token_ce, ~corrupt_mask & valid_mask, metric_weight),
        lexical_ce=_masked_mean(per_token_ce, masks.lexical, metric_weight),
        first_turn_end_ce=_masked_mean(per_token_ce, masks.first_turn_end, metric_weight),
        tail_turn_end_ce=_masked_mean(per_token_ce, masks.tail_turn_end, metric_weight),
        pad_ce=_masked_mean(per_token_ce, masks.pad, metric_weight),
    )


def x0_ce_all(
    logits: torch.Tensor,
    clean_canvas_ids: torch.Tensor,
    corrupt_mask: torch.Tensor,
    *,
    canvas_valid_mask: torch.Tensor | None = None,
    tail_fill_mode: TailFillMode = "eos_repeat",
    eos_tail_loss_weight: float = 1.0,
    pad_loss_weight: float = 0.1,
    turn_end_token_id: int = MAWJAIS_TURN_END_TOKEN_ID,
    pad_token_id: int = MAWJAIS_PAD_TOKEN_ID,
    forbidden_output_token_ids: Sequence[int] = (),
) -> DenoisingLossOutput:
    """Direct x0 reconstruction on every allocated canvas position."""

    valid = _validate_loss_inputs(logits, clean_canvas_ids, corrupt_mask, canvas_valid_mask)
    _validate_targets_are_allowed(clean_canvas_ids, valid, forbidden_output_token_ids)
    logits = _mask_forbidden_logits(logits, forbidden_output_token_ids)
    # Non-allocated mixed-bucket padding is not a semantic target. Mark it with
    # ignore_index before CE so a forbidden PAD logit cannot create inf * 0 -> NaN.
    ce_targets = clean_canvas_ids.masked_fill(~valid, -100)
    per_token_ce = F.cross_entropy(
        logits.transpose(1, 2), ce_targets, reduction="none", ignore_index=-100
    )
    position_weight, masks = build_position_weights(
        clean_canvas_ids,
        canvas_valid_mask=valid,
        tail_fill_mode=tail_fill_mode,
        eos_tail_loss_weight=eos_tail_loss_weight,
        pad_loss_weight=pad_loss_weight,
        turn_end_token_id=turn_end_token_id,
        pad_token_id=pad_token_id,
    )
    valid_weight = position_weight * valid.to(position_weight.dtype)
    denominator = valid_weight.sum(dim=1).clamp_min(1.0)
    per_sample_loss = (per_token_ce * valid_weight).sum(dim=1) / denominator
    return _finish_output(
        per_token_ce=per_token_ce,
        per_sample_loss=per_sample_loss,
        denominator=denominator,
        effective_mask=valid,
        position_weight=position_weight,
        masks=masks,
        corrupt_mask=corrupt_mask,
        valid_mask=valid,
    )


def mdm_absorbing(
    logits: torch.Tensor,
    clean_canvas_ids: torch.Tensor,
    corrupt_mask: torch.Tensor,
    t: torch.Tensor,
    *,
    mask_token_id: int,
    min_diffusion_time: float = 1.0e-3,
    canvas_valid_mask: torch.Tensor | None = None,
    tail_fill_mode: TailFillMode = "eos_repeat",
    eos_tail_loss_weight: float = 1.0,
    pad_loss_weight: float = 0.1,
    turn_end_token_id: int = MAWJAIS_TURN_END_TOKEN_ID,
    pad_token_id: int = MAWJAIS_PAD_TOKEN_ID,
    forbidden_output_token_ids: Sequence[int] = (),
) -> DenoisingLossOutput:
    """Absorbing MDM loss with fixed denominator and ``1/t`` weighting.

    The denominator is the weight sum of *all allocated positions per example*,
    never ``corrupt_mask.sum()``.  An example with no corrupted position
    therefore contributes an exact finite zero.  ``t=0`` is rejected from this
    training objective even though it remains useful for deterministic q-sample
    tests.
    """

    if not 0 < min_diffusion_time <= 1:
        raise ValueError("min_diffusion_time must be in (0, 1]")
    valid = _validate_loss_inputs(logits, clean_canvas_ids, corrupt_mask, canvas_valid_mask)
    validate_diffusion_time(t, logits.shape[0], logits.device)
    if (t <= 0).any():
        raise ValueError("mdm_absorbing training requires strictly positive t")
    forbidden = tuple(forbidden_output_token_ids) + (int(mask_token_id),)
    _validate_targets_are_allowed(clean_canvas_ids, valid, forbidden)
    logits = _mask_forbidden_logits(logits, forbidden)
    ce_targets = clean_canvas_ids.masked_fill(~valid, -100)
    per_token_ce = F.cross_entropy(
        logits.transpose(1, 2), ce_targets, reduction="none", ignore_index=-100
    )
    position_weight, masks = build_position_weights(
        clean_canvas_ids,
        canvas_valid_mask=valid,
        tail_fill_mode=tail_fill_mode,
        eos_tail_loss_weight=eos_tail_loss_weight,
        pad_loss_weight=pad_loss_weight,
        turn_end_token_id=turn_end_token_id,
        pad_token_id=pad_token_id,
    )
    valid_weight = position_weight * valid.to(position_weight.dtype)
    denominator = valid_weight.sum(dim=1).clamp_min(1.0)
    inverse_t = t.float().clamp_min(min_diffusion_time).reciprocal()
    effective_mask = corrupt_mask & valid
    numerator = (
        per_token_ce
        * valid_weight
        * effective_mask.to(per_token_ce.dtype)
        * inverse_t[:, None]
    ).sum(dim=1)
    per_sample_loss = numerator / denominator
    return _finish_output(
        per_token_ce=per_token_ce,
        per_sample_loss=per_sample_loss,
        denominator=denominator,
        effective_mask=effective_mask,
        position_weight=position_weight,
        masks=masks,
        corrupt_mask=corrupt_mask,
        valid_mask=valid,
    )


def diffusion_loss(
    logits: torch.Tensor,
    clean_canvas_ids: torch.Tensor,
    corrupt_mask: torch.Tensor,
    *,
    objective: ResolvedObjective,
    t: torch.Tensor | None = None,
    mask_token_id: int | None = None,
    min_diffusion_time: float = 1.0e-3,
    **kwargs,
) -> DenoisingLossOutput:
    """Dispatch a resolved experiment objective without a causal LM shift."""

    if objective.name == "x0_ce_all":
        return x0_ce_all(logits, clean_canvas_ids, corrupt_mask, **kwargs)
    if objective.name == "mdm_absorbing":
        if t is None or mask_token_id is None:
            raise ValueError("mdm_absorbing requires both t and mask_token_id")
        return mdm_absorbing(
            logits,
            clean_canvas_ids,
            corrupt_mask,
            t,
            mask_token_id=mask_token_id,
            min_diffusion_time=min_diffusion_time,
            **kwargs,
        )
    raise ValueError(f"unsupported resolved objective: {objective.name!r}")


__all__ = [
    "CanvasPositionMasks",
    "DenoisingLossOutput",
    "build_position_weights",
    "classify_canvas_positions",
    "diffusion_loss",
    "mdm_absorbing",
    "x0_ce_all",
]
