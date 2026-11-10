"""Centralized alignment between denoiser readouts and clean canvas targets."""

from __future__ import annotations

from typing import Literal

import torch


DenoiserOutputAlignment = Literal["same_position", "previous_position"]


def align_denoiser_logits(
    raw_logits: torch.Tensor,
    mode: DenoiserOutputAlignment,
) -> torch.Tensor:
    """Map ``[prefix boundary, canvas[0:C]]`` logits to ``x0[0:C]``.

    ``raw_logits`` has C+1 rows in this exact semantic order:

    * row 0 is the language-model readout at the last valid prefix position;
    * rows 1..C are readouts at the C noisy canvas positions.

    Same-position prediction uses rows 1..C.  Previous-position prediction
    uses rows 0..C-1, preserving the checkpoint's AR readout convention for
    the first token without inserting, rotating, or learning a BRIDGE query.
    The final canvas token remains present and visible to bidirectional canvas
    attention even though its own readout is unused in previous-position mode.
    """

    if raw_logits.ndim != 3:
        raise ValueError(
            f"raw_logits must have shape [batch, canvas+1, vocab], got {tuple(raw_logits.shape)}"
        )
    if raw_logits.shape[1] < 2:
        raise ValueError("raw_logits must contain one prefix boundary and at least one canvas row")
    if raw_logits.shape[2] <= 0:
        raise ValueError("vocabulary dimension must be positive")

    if mode == "same_position":
        return raw_logits[:, 1:, :]
    if mode == "previous_position":
        return raw_logits[:, :-1, :]
    raise ValueError(f"unsupported denoiser output alignment: {mode!r}")


def both_denoiser_alignments(raw_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return same/previous logits from one forward for pre-update diagnostics."""

    return (
        align_denoiser_logits(raw_logits, "same_position"),
        align_denoiser_logits(raw_logits, "previous_position"),
    )


__all__ = [
    "DenoiserOutputAlignment",
    "align_denoiser_logits",
    "both_denoiser_alignments",
]
