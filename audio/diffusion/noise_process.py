"""Forward noise processes and their matching inference priors."""

from __future__ import annotations

from typing import Protocol, Sequence

import torch


def _is_integer_tensor(tensor: torch.Tensor) -> bool:
    return not tensor.is_floating_point() and not tensor.is_complex() and tensor.dtype != torch.bool


def _validate_canvas(canvas: torch.Tensor, name: str = "canvas") -> None:
    if not isinstance(canvas, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if canvas.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, canvas], got {tuple(canvas.shape)}")
    if not _is_integer_tensor(canvas):
        raise TypeError(f"{name} must contain integer token IDs, got {canvas.dtype}")


def validate_diffusion_time(t: torch.Tensor, batch_size: int, device: torch.device) -> None:
    """Validate the per-example scalar diffusion time used by ``q_sample``."""

    if not isinstance(t, torch.Tensor):
        raise TypeError("t must be a torch.Tensor")
    if t.ndim != 1 or t.shape[0] != batch_size:
        raise ValueError(f"t must have shape [{batch_size}], got {tuple(t.shape)}")
    if not t.is_floating_point():
        raise TypeError(f"t must be floating point, got {t.dtype}")
    if t.device != device:
        raise ValueError(f"t and tokens must be on the same device ({t.device} != {device})")
    if not torch.isfinite(t).all():
        raise ValueError("t must contain only finite values")
    if (t < 0).any() or (t > 1).any():
        raise ValueError("t must be in [0, 1]")


def sample_diffusion_times(
    batch_size: int,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
    full_noise_probability: float = 0.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample ``t ~ Uniform(0, 1)`` with an optional point mass at ``t=1``.

    The point mass is an experiment option, not part of the canonical uniform
    or MDM objectives.  A caller can use a dedicated generator to decouple time
    sampling from token-noise sampling.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if not 0 <= full_noise_probability <= 1:
        raise ValueError("full_noise_probability must be in [0, 1]")
    if not dtype.is_floating_point:
        raise TypeError("dtype must be floating point")

    times = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
    # Always consume the Bernoulli draw, including for the p=0 control. This
    # keeps every later continuous-time draw paired between a control run and
    # a full-noise intervention that starts from the same RNG state.
    force_full = (
        torch.rand(batch_size, device=device, generator=generator)
        < full_noise_probability
    )
    if full_noise_probability:
        times = torch.where(force_full, torch.ones_like(times), times)
    return times


class DiscreteNoiseProcess(Protocol):
    """Common interface used by training and iterative sampling."""

    is_absorbing: bool

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        noise_generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def initial_sample(
        self,
        shape: Sequence[int],
        *,
        device: torch.device | str,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.long,
    ) -> torch.Tensor: ...

    def renoise(
        self,
        candidate: torch.Tensor,
        keep_mask: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor: ...


class UniformNoiseProcess:
    """Independent token replacement from a configurable uniform prior.

    ``noise_ids`` is the exact prior support.  It can be the full vocabulary or
    a precomputed safe vocabulary; callers should log its size and hash.
    """

    is_absorbing = False

    def __init__(self, noise_ids: torch.Tensor | Sequence[int]) -> None:
        ids = torch.as_tensor(noise_ids, dtype=torch.long)
        if ids.ndim != 1 or ids.numel() == 0:
            raise ValueError("noise_ids must be a non-empty one-dimensional sequence")
        if (ids < 0).any():
            raise ValueError("noise_ids must be non-negative")
        if torch.unique(ids).numel() != ids.numel():
            raise ValueError("noise_ids must not contain duplicates")
        self.noise_ids = ids.detach().cpu().clone()

    def _draw(
        self,
        shape: Sequence[int],
        *,
        device: torch.device | str,
        generator: torch.Generator | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if len(tuple(shape)) != 2 or any(int(dimension) <= 0 for dimension in shape):
            raise ValueError(f"shape must be two positive dimensions, got {tuple(shape)}")
        ids = self.noise_ids.to(device=device)
        indices = torch.randint(
            ids.numel(), tuple(int(dimension) for dimension in shape), device=device, generator=generator
        )
        return ids[indices].to(dtype=dtype)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        noise_generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_canvas(x0, "x0")
        validate_diffusion_time(t, x0.shape[0], x0.device)
        corrupt_mask = torch.rand(x0.shape, device=x0.device, generator=generator) < t[:, None]
        noise = self._draw(
            x0.shape,
            device=x0.device,
            generator=generator if noise_generator is None else noise_generator,
            dtype=x0.dtype,
        )
        return torch.where(corrupt_mask, noise, x0), corrupt_mask

    def initial_sample(
        self,
        shape: Sequence[int],
        *,
        device: torch.device | str,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.long,
    ) -> torch.Tensor:
        return self._draw(shape, device=device, generator=generator, dtype=dtype)

    def renoise(
        self,
        candidate: torch.Tensor,
        keep_mask: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Keep the current selected set and redraw every other position.

        This operation does not imply permanent commitment.  The uniform
        sampler recomputes ``keep_mask`` at every step and may revise or renoise
        positions selected during an earlier step.
        """

        _validate_canvas(candidate, "candidate")
        if keep_mask.shape != candidate.shape or keep_mask.dtype != torch.bool:
            raise ValueError("keep_mask must be boolean and have candidate.shape")
        if keep_mask.device != candidate.device:
            raise ValueError("keep_mask and candidate must be on the same device")
        noise = self._draw(
            candidate.shape,
            device=candidate.device,
            generator=generator,
            dtype=candidate.dtype,
        )
        return torch.where(keep_mask, candidate, noise)


class MaskNoiseProcess:
    """Absorbing single-token mask process."""

    is_absorbing = True

    def __init__(self, mask_token_id: int) -> None:
        if mask_token_id < 0:
            raise ValueError("mask_token_id must be non-negative")
        self.mask_token_id = int(mask_token_id)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        noise_generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del noise_generator  # Mask replacement is deterministic once corruption is sampled.
        _validate_canvas(x0, "x0")
        validate_diffusion_time(t, x0.shape[0], x0.device)
        corrupt_mask = torch.rand(x0.shape, device=x0.device, generator=generator) < t[:, None]
        mask = torch.full_like(x0, self.mask_token_id)
        return torch.where(corrupt_mask, mask, x0), corrupt_mask

    def initial_sample(
        self,
        shape: Sequence[int],
        *,
        device: torch.device | str,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.long,
    ) -> torch.Tensor:
        del generator  # The absorbing prior is deterministic.
        if len(tuple(shape)) != 2 or any(int(dimension) <= 0 for dimension in shape):
            raise ValueError(f"shape must be two positive dimensions, got {tuple(shape)}")
        return torch.full(tuple(int(dimension) for dimension in shape), self.mask_token_id, device=device, dtype=dtype)

    def renoise(
        self,
        candidate: torch.Tensor,
        keep_mask: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Carry over selected tokens and mask every uncommitted position."""

        del generator
        _validate_canvas(candidate, "candidate")
        if keep_mask.shape != candidate.shape or keep_mask.dtype != torch.bool:
            raise ValueError("keep_mask must be boolean and have candidate.shape")
        if keep_mask.device != candidate.device:
            raise ValueError("keep_mask and candidate must be on the same device")
        mask = torch.full_like(candidate, self.mask_token_id)
        return torch.where(keep_mask, candidate, mask)


__all__ = [
    "DiscreteNoiseProcess",
    "MaskNoiseProcess",
    "UniformNoiseProcess",
    "sample_diffusion_times",
    "validate_diffusion_time",
]
