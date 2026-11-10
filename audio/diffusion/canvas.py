"""Transcript canvas construction and duration-only bucket selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from .config import TailFillMode


MAWJAIS_TURN_END_TOKEN_ID = 250019
MAWJAIS_PAD_TOKEN_ID = 1


class CanvasOverflowError(ValueError):
    """Raised when a target or duration estimate does not fit an allocated canvas."""


@dataclass(frozen=True)
class CanvasChoice:
    """Result of duration-only canvas selection.

    ``canvas_size`` is ``None`` on overflow.  This forces the trainer to count
    and explicitly skip the item (or choose a separately configured larger
    bucket), rather than silently truncating it.
    """

    duration_seconds: float
    estimated_tokens: int
    canvas_size: int | None

    @property
    def overflow(self) -> bool:
        return self.canvas_size is None


class DurationBucketSelector:
    """Choose a canvas from audio duration, never from reference token count."""

    def __init__(
        self,
        canvas_sizes: Sequence[int],
        *,
        tokens_per_second: float,
        margin_tokens: float,
    ) -> None:
        sizes = tuple(int(size) for size in canvas_sizes)
        if not sizes or any(size <= 0 for size in sizes):
            raise ValueError("canvas_sizes must be non-empty and positive")
        if tuple(sorted(set(sizes))) != sizes:
            raise ValueError("canvas_sizes must be sorted and unique")
        if tokens_per_second <= 0:
            raise ValueError("tokens_per_second must be > 0")
        if margin_tokens < 0:
            raise ValueError("margin_tokens must be >= 0")
        self.canvas_sizes = sizes
        self.tokens_per_second = float(tokens_per_second)
        self.margin_tokens = float(margin_tokens)

    def estimate_length(self, duration_seconds: float) -> int:
        if not math.isfinite(duration_seconds) or duration_seconds < 0:
            raise ValueError("duration_seconds must be finite and >= 0")
        return int(math.ceil(self.tokens_per_second * duration_seconds + self.margin_tokens))

    def select(self, duration_seconds: float) -> CanvasChoice:
        estimated = self.estimate_length(duration_seconds)
        canvas_size = next((size for size in self.canvas_sizes if size >= estimated), None)
        return CanvasChoice(float(duration_seconds), estimated, canvas_size)

    def select_or_raise(self, duration_seconds: float) -> int:
        choice = self.select(duration_seconds)
        if choice.overflow:
            raise CanvasOverflowError(
                f"duration estimate {choice.estimated_tokens} exceeds largest canvas "
                f"{self.canvas_sizes[-1]} (duration={duration_seconds:.3f}s)"
            )
        assert choice.canvas_size is not None
        return choice.canvas_size


@dataclass(frozen=True)
class CanvasEncoding:
    """A single semantic canvas and masks used by objectives/metrics."""

    input_ids: torch.Tensor
    valid_mask: torch.Tensor
    lexical_mask: torch.Tensor
    first_turn_end_mask: torch.Tensor
    tail_turn_end_mask: torch.Tensor
    pad_mask: torch.Tensor


def _as_token_tensor(token_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
    token_tensor = torch.as_tensor(token_ids, dtype=torch.long)
    if token_tensor.ndim != 1:
        raise ValueError(f"target_token_ids must be one-dimensional, got {tuple(token_tensor.shape)}")
    if (token_tensor < 0).any():
        raise ValueError("target_token_ids must be non-negative")
    return token_tensor


def build_clean_canvas(
    target_token_ids: Sequence[int] | torch.Tensor,
    canvas_size: int,
    *,
    tail_fill_mode: TailFillMode = "eos_repeat",
    turn_end_token_id: int = MAWJAIS_TURN_END_TOKEN_ID,
    pad_token_id: int = MAWJAIS_PAD_TOKEN_ID,
) -> CanvasEncoding:
    """Append MawJais turn-end and fill the remaining semantic tail.

    ``target_token_ids`` contains the assistant answer only and must not already
    contain ``<|ifm|im_end|>``.  The function never truncates a target.
    Hugging Face EOS/PAD (ID 1) is intentionally distinct from the MawJais
    assistant turn terminator (ID 250019).
    """

    if canvas_size <= 0:
        raise ValueError("canvas_size must be > 0")
    if tail_fill_mode not in ("eos_repeat", "eos_then_pad"):
        raise ValueError(f"unsupported tail_fill_mode: {tail_fill_mode!r}")
    if turn_end_token_id == pad_token_id:
        raise ValueError("turn-end and PAD token IDs must be distinct")

    target = _as_token_tensor(target_token_ids)
    if (target == turn_end_token_id).any():
        raise ValueError("target_token_ids must not already contain the turn-end token")
    required = target.numel() + 1
    if required > canvas_size:
        raise CanvasOverflowError(
            f"target requires {required} positions but canvas_size={canvas_size}; target was not truncated"
        )

    ids = torch.empty(canvas_size, dtype=torch.long, device=target.device)
    ids[: target.numel()] = target
    ids[target.numel()] = turn_end_token_id
    if tail_fill_mode == "eos_repeat":
        ids[target.numel() + 1 :] = turn_end_token_id
    else:
        ids[target.numel() + 1 :] = pad_token_id

    positions = torch.arange(canvas_size, device=ids.device)
    first_turn_end_mask = positions == target.numel()
    tail_turn_end_mask = (positions > target.numel()) & (ids == turn_end_token_id)
    pad_mask = ids == pad_token_id
    # PAD inside the source target is not expected, but lexical positions are
    # defined structurally so that any bad record remains observable.
    lexical_mask = positions < target.numel()
    valid_mask = torch.ones(canvas_size, dtype=torch.bool, device=ids.device)
    return CanvasEncoding(
        input_ids=ids,
        valid_mask=valid_mask,
        lexical_mask=lexical_mask,
        first_turn_end_mask=first_turn_end_mask,
        tail_turn_end_mask=tail_turn_end_mask,
        pad_mask=pad_mask,
    )


def pad_canvas_batch(
    canvases: Iterable[CanvasEncoding],
    *,
    pad_token_id: int = MAWJAIS_PAD_TOKEN_ID,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad mixed canvas buckets and mark non-allocated positions invalid.

    Semantic PAD positions *inside* an ``eos_then_pad`` canvas remain valid;
    only batch padding beyond that example's allocated bucket is False.
    """

    items = list(canvases)
    if not items:
        raise ValueError("canvases must be non-empty")
    device = items[0].input_ids.device
    if any(item.input_ids.device != device for item in items):
        raise ValueError("all canvases must be on the same device")
    max_size = max(item.input_ids.numel() for item in items)
    batch_ids = torch.full((len(items), max_size), pad_token_id, dtype=torch.long, device=device)
    valid_mask = torch.zeros((len(items), max_size), dtype=torch.bool, device=device)
    for row, item in enumerate(items):
        size = item.input_ids.numel()
        batch_ids[row, :size] = item.input_ids
        valid_mask[row, :size] = item.valid_mask
    return batch_ids, valid_mask


def truncate_at_first_turn_end(
    token_ids: Sequence[int] | torch.Tensor,
    *,
    turn_end_token_id: int = MAWJAIS_TURN_END_TOKEN_ID,
    include_turn_end: bool = False,
) -> list[int]:
    """Return one decoded sequence truncated at its first assistant turn-end."""

    ids = _as_token_tensor(token_ids).tolist()
    try:
        stop = ids.index(turn_end_token_id)
    except ValueError:
        return ids
    return ids[: stop + int(include_turn_end)]


__all__ = [
    "CanvasChoice",
    "CanvasEncoding",
    "CanvasOverflowError",
    "DurationBucketSelector",
    "MAWJAIS_PAD_TOKEN_ID",
    "MAWJAIS_TURN_END_TOKEN_ID",
    "build_clean_canvas",
    "pad_canvas_batch",
    "truncate_at_first_turn_end",
]
