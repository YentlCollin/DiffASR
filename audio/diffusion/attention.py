"""Reference prefix-causal / canvas-bidirectional attention masks."""

from __future__ import annotations

from typing import Sequence

import torch


def _length_tensor(
    lengths: Sequence[int] | torch.Tensor,
    *,
    name: str,
    device: torch.device | str | None,
) -> torch.Tensor:
    tensor = torch.as_tensor(lengths, device=device)
    if tensor.ndim != 1 or tensor.numel() == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if tensor.is_floating_point() or tensor.is_complex() or tensor.dtype == torch.bool:
        raise TypeError(f"{name} must contain integers")
    return tensor.to(dtype=torch.long)


def build_prefix_canvas_allowed_mask(
    prefix_lengths: Sequence[int] | torch.Tensor,
    canvas_lengths: Sequence[int] | torch.Tensor,
    *,
    max_sequence_length: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return boolean allowed attention with shape ``[B, 1, S, S]``.

    Each example must be laid out contiguously as ``[real prefix][canvas]`` and
    then batch-padded.  The four semantic quadrants are:

    * prefix -> prefix: causal;
    * prefix -> canvas: forbidden;
    * canvas -> prefix: fully allowed;
    * canvas -> canvas: fully bidirectional.

    Real queries never see batch-padding keys.  A padded query sees only itself
    to avoid an all-masked row (which can otherwise yield NaNs in eager
    attention); those query outputs must still be discarded by the caller.
    """

    prefix = _length_tensor(prefix_lengths, name="prefix_lengths", device=device)
    if device is None:
        device = prefix.device
    canvas = _length_tensor(canvas_lengths, name="canvas_lengths", device=device)
    if prefix.shape != canvas.shape:
        raise ValueError("prefix_lengths and canvas_lengths must have the same shape")
    if (prefix < 0).any():
        raise ValueError("prefix lengths must be >= 0")
    if (canvas <= 0).any():
        raise ValueError("canvas lengths must be > 0")

    total = prefix + canvas
    required_length = int(total.max().item())
    sequence_length = required_length if max_sequence_length is None else int(max_sequence_length)
    if sequence_length < required_length:
        raise ValueError(
            f"max_sequence_length={sequence_length} is smaller than required {required_length}"
        )

    positions = torch.arange(sequence_length, device=device)
    query = positions[None, :, None]
    key = positions[None, None, :]
    prefix_boundary = prefix[:, None, None]
    total_boundary = total[:, None, None]

    prefix_query = query < prefix_boundary
    canvas_query = (query >= prefix_boundary) & (query < total_boundary)
    real_key = key < total_boundary

    prefix_to_prefix = prefix_query & (key < prefix_boundary) & (key <= query)
    canvas_to_real = canvas_query & real_key
    padding_self = (query >= total_boundary) & (query == key)
    allowed = prefix_to_prefix | canvas_to_real | padding_self
    return allowed[:, None, :, :]


def build_prefix_canvas_attention_mask(
    prefix_lengths: Sequence[int] | torch.Tensor,
    canvas_lengths: Sequence[int] | torch.Tensor,
    *,
    dtype: torch.dtype,
    max_sequence_length: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a 4D additive mask (0 allowed, finite minimum forbidden)."""

    if not dtype.is_floating_point:
        raise TypeError("attention mask dtype must be floating point")
    allowed = build_prefix_canvas_allowed_mask(
        prefix_lengths,
        canvas_lengths,
        max_sequence_length=max_sequence_length,
        device=device,
    )
    mask = torch.full(allowed.shape, torch.finfo(dtype).min, dtype=dtype, device=allowed.device)
    return mask.masked_fill(allowed, 0.0)


def build_prefix_canvas_position_ids(
    prefix_lengths: Sequence[int] | torch.Tensor,
    canvas_lengths: Sequence[int] | torch.Tensor,
    *,
    max_sequence_length: int | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build contiguous RoPE positions for packed ``prefix + canvas`` rows."""

    prefix = _length_tensor(prefix_lengths, name="prefix_lengths", device=device)
    if device is None:
        device = prefix.device
    canvas = _length_tensor(canvas_lengths, name="canvas_lengths", device=device)
    if prefix.shape != canvas.shape:
        raise ValueError("prefix_lengths and canvas_lengths must have the same shape")
    if (prefix < 0).any() or (canvas <= 0).any():
        raise ValueError("prefix lengths must be >= 0 and canvas lengths > 0")
    total = prefix + canvas
    required_length = int(total.max().item())
    sequence_length = required_length if max_sequence_length is None else int(max_sequence_length)
    if sequence_length < required_length:
        raise ValueError("max_sequence_length is too short")
    positions = torch.arange(sequence_length, device=device).expand(prefix.shape[0], -1)
    return torch.where(positions < total[:, None], positions, torch.zeros_like(positions))


__all__ = [
    "build_prefix_canvas_allowed_mask",
    "build_prefix_canvas_attention_mask",
    "build_prefix_canvas_position_ids",
]
