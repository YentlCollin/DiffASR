"""Low-overhead token diagnostics for diffusion training and validation."""

from __future__ import annotations

from typing import Mapping

import torch

from .objectives import classify_canvas_positions


NOISE_BINS: tuple[tuple[str, float, float], ...] = (
    ("t_0.0_0.2", 0.0, 0.2),
    ("t_0.2_0.4", 0.2, 0.4),
    ("t_0.4_0.6", 0.4, 0.6),
    ("t_0.6_0.8", 0.6, 0.8),
    ("t_0.8_1.0", 0.8, 1.0),
)


def _accuracy_stats(
    correct: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = mask.sum(dtype=torch.long)
    correct_count = (correct & mask).sum(dtype=torch.long)
    return correct_count, count


@torch.no_grad()
def denoising_token_metrics(
    logits: torch.Tensor,
    clean_canvas_ids: torch.Tensor,
    corrupt_mask: torch.Tensor,
    canvas_valid_mask: torch.Tensor,
    noise_level: torch.Tensor,
    *,
    turn_end_token_id: int = 250019,
    pad_token_id: int = 1,
) -> dict[str, torch.Tensor]:
    """Return additive ``correct/*`` and ``count/*`` tensors.

    Callers aggregate counts across microbatches/ranks and divide only when
    reporting. This avoids averaging per-batch accuracies with different bucket
    sizes. Exact ``t=1`` examples have a dedicated bin and are excluded from
    the half-open ``[0.8, 1.0)`` interval.
    """

    if logits.ndim != 3 or clean_canvas_ids.shape != logits.shape[:2]:
        raise ValueError("logits/targets must have shapes [B,C,V] and [B,C]")
    for name, mask in (
        ("corrupt_mask", corrupt_mask),
        ("canvas_valid_mask", canvas_valid_mask),
    ):
        if mask.shape != clean_canvas_ids.shape or mask.dtype != torch.bool:
            raise ValueError(f"{name} must be boolean and match targets")
    if noise_level.shape != (clean_canvas_ids.shape[0],):
        raise ValueError("noise_level must have shape [batch]")
    if not noise_level.is_floating_point():
        raise TypeError("noise_level must be floating point")
    if any(tensor.device != logits.device for tensor in (clean_canvas_ids, corrupt_mask, canvas_valid_mask, noise_level)):
        raise ValueError("all metric tensors must share a device")

    prediction = logits.argmax(dim=-1)
    correct = prediction.eq(clean_canvas_ids)
    valid = canvas_valid_mask
    positions = classify_canvas_positions(
        clean_canvas_ids,
        canvas_valid_mask=valid,
        turn_end_token_id=turn_end_token_id,
        pad_token_id=pad_token_id,
    )
    first_token = torch.zeros_like(valid)
    first_token[:, 0] = valid[:, 0]
    categories: Mapping[str, torch.Tensor] = {
        "all": valid,
        "corrupted": valid & corrupt_mask,
        "uncorrupted": valid & ~corrupt_mask,
        "first_token": first_token,
        "lexical": positions.lexical,
        "first_turn_end": positions.first_turn_end,
        "tail_turn_end": positions.tail_turn_end,
        "pad": positions.pad,
    }
    output: dict[str, torch.Tensor] = {}
    for label, mask in categories.items():
        correct_count, count = _accuracy_stats(correct, mask)
        output[f"correct/{label}"] = correct_count
        output[f"count/{label}"] = count

    exact_one = noise_level.eq(1.0)
    for label, lower, upper in NOISE_BINS:
        sample_mask = (noise_level >= lower) & (noise_level < upper) & ~exact_one
        mask = valid & sample_mask[:, None]
        correct_count, count = _accuracy_stats(correct, mask)
        output[f"correct/{label}"] = correct_count
        output[f"count/{label}"] = count
    mask = valid & exact_one[:, None]
    correct_count, count = _accuracy_stats(correct, mask)
    output["correct/t_1.0"] = correct_count
    output["count/t_1.0"] = count
    return output


def ratios_from_additive_metrics(metrics: Mapping[str, torch.Tensor | int]) -> dict[str, float]:
    """Convert aggregated correct/count pairs to reporting accuracies."""

    ratios: dict[str, float] = {}
    for key, numerator in metrics.items():
        if not key.startswith("correct/"):
            continue
        label = key.removeprefix("correct/")
        denominator = metrics.get(f"count/{label}")
        if denominator is None:
            raise ValueError(f"missing count/{label}")
        numerator_value = int(numerator.item()) if isinstance(numerator, torch.Tensor) else int(numerator)
        denominator_value = int(denominator.item()) if isinstance(denominator, torch.Tensor) else int(denominator)
        ratios[f"accuracy/{label}"] = (
            numerator_value / denominator_value if denominator_value else float("nan")
        )
    return ratios


__all__ = ["NOISE_BINS", "denoising_token_metrics", "ratios_from_additive_metrics"]
