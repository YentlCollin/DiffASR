"""Correctness-first confidence sampler shared by Uniform and Mask noise."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import torch

from .config import ConfidenceSamplerConfig
from .noise_process import (
    DiscreteNoiseProcess,
    MaskNoiseProcess,
    UniformNoiseProcess,
)


Denoiser = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class SamplerStep:
    step: int
    noise_level: float
    canvas: torch.Tensor
    selected_mask: torch.Tensor
    newly_selected_mask: torch.Tensor
    prediction: torch.Tensor
    confidence: torch.Tensor
    entropy: torch.Tensor


@dataclass(frozen=True)
class SamplerOutput:
    canvas: torch.Tensor
    selected_mask: torch.Tensor
    nfe: int
    history: tuple[SamplerStep, ...]


def mask_forbidden_logits(
    logits: torch.Tensor,
    forbidden_output_token_ids: Sequence[int],
    *,
    inplace: bool = False,
) -> torch.Tensor:
    """Make invalid ASR outputs impossible, cloning unless explicitly safe."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, canvas, vocab]")
    ids = tuple(sorted(set(int(token_id) for token_id in forbidden_output_token_ids)))
    if not ids:
        return logits
    if ids[0] < 0 or ids[-1] >= logits.shape[-1]:
        raise ValueError("a forbidden token ID is outside the logits vocabulary")
    masked = logits if inplace else logits.clone()
    masked[..., list(ids)] = -torch.inf
    if torch.isneginf(masked).all(dim=-1).any():
        raise ValueError("forbidden_output_token_ids mask the entire vocabulary")
    return masked


def _predict(
    logits: torch.Tensor,
    *,
    temperature: float,
    prediction_mode: str,
    generator: torch.Generator | None,
    compute_entropy: bool,
    vocab_chunk_size: int = 16_384,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute prediction/statistics without a full FP32 probability tensor.

    At B=1,C=512 MawJais would otherwise materialize about 490 MiB merely for
    probabilities. Chunked log-sum-exp bounds scratch memory independently of
    the 250624-token vocabulary. Categorical sampling uses Gumbel-max, which is
    equivalent to drawing from the temperature-scaled categorical distribution.
    """

    if prediction_mode not in ("argmax", "categorical"):
        raise ValueError(f"unsupported prediction_mode: {prediction_mode!r}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if vocab_chunk_size <= 0:
        raise ValueError("vocab_chunk_size must be positive")

    output_shape = logits.shape[:2]
    log_normalizer = torch.full(
        output_shape, -torch.inf, dtype=torch.float32, device=logits.device
    )
    maximum = torch.full_like(log_normalizer, -torch.inf)
    prediction = torch.zeros(output_shape, dtype=torch.long, device=logits.device)
    sampled_maximum = torch.full_like(log_normalizer, -torch.inf)
    sampled_prediction = torch.zeros_like(prediction)

    for start in range(0, logits.shape[-1], vocab_chunk_size):
        stop = min(start + vocab_chunk_size, logits.shape[-1])
        chunk = logits[..., start:stop].float().div(float(temperature))
        chunk_maximum, chunk_argmax = chunk.max(dim=-1)
        better = chunk_maximum > maximum
        maximum = torch.where(better, chunk_maximum, maximum)
        prediction = torch.where(better, chunk_argmax + start, prediction)
        log_normalizer = torch.logaddexp(
            log_normalizer, torch.logsumexp(chunk, dim=-1)
        )
        if prediction_mode == "categorical":
            uniform = torch.rand(
                chunk.shape,
                device=chunk.device,
                dtype=torch.float32,
                generator=generator,
            ).clamp_(min=torch.finfo(torch.float32).tiny, max=1.0 - 1.0e-7)
            sampled_chunk_maximum, sampled_chunk_argmax = (
                chunk - torch.log(-torch.log(uniform))
            ).max(dim=-1)
            sampled_better = sampled_chunk_maximum > sampled_maximum
            sampled_maximum = torch.where(
                sampled_better, sampled_chunk_maximum, sampled_maximum
            )
            sampled_prediction = torch.where(
                sampled_better,
                sampled_chunk_argmax + start,
                sampled_prediction,
            )

    if not torch.isfinite(log_normalizer).all() or not torch.isfinite(maximum).all():
        raise ValueError("denoiser logits do not define a finite categorical distribution")
    confidence = torch.exp(maximum - log_normalizer)
    if prediction_mode == "categorical":
        prediction = sampled_prediction

    entropy = torch.zeros_like(log_normalizer)
    if compute_entropy:
        for start in range(0, logits.shape[-1], vocab_chunk_size):
            stop = min(start + vocab_chunk_size, logits.shape[-1])
            chunk = logits[..., start:stop].float().div(float(temperature))
            probabilities = torch.exp(chunk - log_normalizer[..., None])
            contribution = probabilities * (log_normalizer[..., None] - chunk)
            # A forbidden logit gives 0 * inf; its exact entropy contribution is 0.
            contribution = torch.where(
                probabilities == 0, torch.zeros_like(contribution), contribution
            )
            entropy += contribution.sum(dim=-1)
        if not torch.isfinite(entropy).all():
            raise ValueError("denoiser entropy contains NaN or infinity")
    return prediction, confidence, entropy


def _select_topk_per_example(
    scores: torch.Tensor,
    eligible: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    if scores.ndim != 2 or eligible.shape != scores.shape or eligible.dtype != torch.bool:
        raise ValueError("scores and boolean eligible must have shape [batch, canvas]")
    if counts.shape != (scores.shape[0],):
        raise ValueError("counts must have shape [batch]")
    selected = torch.zeros_like(eligible)
    for row in range(scores.shape[0]):
        count = int(counts[row].item())
        available = int(eligible[row].sum().item())
        if count < 0 or count > available:
            raise ValueError(f"cannot select {count} positions from {available} eligible positions")
        if count:
            row_scores = scores[row].masked_fill(~eligible[row], -torch.inf)
            indices = torch.topk(row_scores, count, sorted=False).indices
            selected[row, indices] = True
    return selected


def select_entropy_bounded_positions(
    confidence: torch.Tensor,
    entropy: torch.Tensor,
    eligible: torch.Tensor,
    *,
    gamma: float,
    position_bias_lambda: float = 0.0,
) -> torch.Tensor:
    """Select the largest confidence-ranked prefix under the EB-Conf budget.

    For a ranked prefix ``U`` the exact constraint is
    ``sum(H_i for i in U) - max(H_i for i in U) <= gamma``. Consequently a
    single eligible position is always selectable, including when ``gamma=0``.
    A positive position bias changes only the ranking score; the entropy budget
    itself is unchanged.
    """

    if confidence.ndim != 2 or entropy.shape != confidence.shape:
        raise ValueError("confidence and entropy must have shape [batch, canvas]")
    if eligible.shape != confidence.shape or eligible.dtype != torch.bool:
        raise ValueError("eligible must be boolean and match confidence")
    if confidence.device != entropy.device or confidence.device != eligible.device:
        raise ValueError("confidence, entropy, and eligible must share a device")
    if gamma < 0 or position_bias_lambda < 0:
        raise ValueError("gamma and position_bias_lambda must be non-negative")
    if not torch.isfinite(confidence).all() or not torch.isfinite(entropy).all():
        raise ValueError("confidence and entropy must be finite")

    positions = torch.arange(confidence.shape[1], device=confidence.device)
    bias = torch.exp(-float(position_bias_lambda) * positions.float())
    ranking_scores = confidence * bias[None, :]
    selected = torch.zeros_like(eligible)
    budget_tolerance = max(1.0e-7, abs(float(gamma)) * 1.0e-6)
    for row in range(confidence.shape[0]):
        indices = torch.nonzero(eligible[row], as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        order = torch.argsort(ranking_scores[row, indices], descending=True, stable=True)
        ranked = indices[order]
        ranked_entropy = entropy[row, ranked]
        cumulative_sum = ranked_entropy.cumsum(dim=0)
        cumulative_max = ranked_entropy.cummax(dim=0).values
        within_budget = (
            cumulative_sum - cumulative_max <= float(gamma) + budget_tolerance
        )
        # The budget expression is monotone as this prefix grows. Keep the
        # selection on-device instead of synchronizing once per canvas token.
        selected[row, ranked[within_budget]] = True
    return selected


def select_diffusion_gemma_entropy_positions(
    entropy: torch.Tensor,
    eligible: torch.Tensor,
    *,
    entropy_bound: float,
) -> torch.Tensor:
    """Reproduce DiffusionGemma's cumulative-entropy acceptance rule.

    Eligible positions are sorted by increasing categorical entropy.  The
    largest sorted prefix satisfying ``cumsum(H) - current_max(H) <= bound`` is
    accepted.  Since the entropies are sorted, ``current_max(H)`` is simply the
    current entropy.  The first eligible position is therefore always accepted.
    """

    if entropy.ndim != 2:
        raise ValueError("entropy must have shape [batch, canvas]")
    if eligible.shape != entropy.shape or eligible.dtype != torch.bool:
        raise ValueError("eligible must be boolean and match entropy")
    if eligible.device != entropy.device:
        raise ValueError("eligible and entropy must share a device")
    if entropy_bound <= 0:
        raise ValueError("DiffusionGemma entropy_bound must be > 0")
    if not torch.isfinite(entropy).all() or (entropy < 0).any():
        raise ValueError("entropy must be finite and non-negative")

    selected = torch.zeros_like(eligible)
    for row in range(entropy.shape[0]):
        indices = torch.nonzero(eligible[row], as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        sorted_entropy, order = torch.sort(
            entropy[row, indices], descending=False, stable=True
        )
        cumulative_entropy = sorted_entropy.cumsum(dim=0)
        sorted_selection = (
            cumulative_entropy - sorted_entropy <= float(entropy_bound)
        )
        selected[row, indices[order[sorted_selection]]] = True
    return selected


def _select_positions(
    *,
    config: ConfidenceSamplerConfig,
    confidence: torch.Tensor,
    entropy: torch.Tensor,
    eligible: torch.Tensor,
    target_counts: torch.Tensor,
    final_step: bool,
) -> torch.Tensor:
    if final_step:
        return eligible.clone()
    if config.sampler == "confidence_schedule":
        return _select_topk_per_example(confidence, eligible, target_counts)
    position_bias = (
        config.position_bias_lambda
        if config.sampler == "position_biased_entropy_bounded"
        else 0.0
    )
    return select_entropy_bounded_positions(
        confidence,
        entropy,
        eligible,
        gamma=config.entropy_budget_gamma,
        position_bias_lambda=position_bias,
    )


@torch.no_grad()
def confidence_schedule_sample(
    denoiser: Denoiser,
    noise_process: DiscreteNoiseProcess,
    shape: Sequence[int],
    *,
    device: torch.device | str,
    config: ConfidenceSamplerConfig | None = None,
    generator: torch.Generator | None = None,
    canvas_valid_mask: torch.Tensor | None = None,
    padding_token_id: int | None = None,
    return_history: bool = False,
) -> SamplerOutput:
    """Iteratively denoise using a deterministic confidence-count schedule.

    The two reverse-process semantics intentionally differ:

    * Mask noise carries committed tokens over forever.  Only newly selected
      positions are written, and uncommitted positions remain MASK.
    * Uniform noise has no absorbing state.  At every step, the selected set is
      recomputed from all positions, selected predictions are revised, and all
      other positions are redrawn from the uniform prior.  There is no
      persistent ``committed_mask`` for Uniform.

    This distinction avoids silently treating a convenient monotonic Uniform
    heuristic as the reverse process used by DiffusionGemma-style sampling.
    """

    cfg = ConfidenceSamplerConfig() if config is None else config
    shape = tuple(int(dimension) for dimension in shape)
    if len(shape) != 2 or any(dimension <= 0 for dimension in shape):
        raise ValueError(f"shape must be [batch, canvas] with positive values, got {shape}")
    batch_size, _ = shape
    entropy_sampler = cfg.sampler in (
        "entropy_bounded",
        "position_biased_entropy_bounded",
    )
    if entropy_sampler and not isinstance(noise_process, MaskNoiseProcess):
        raise ValueError(
            "entropy-bounded samplers are restricted to absorbing Mask noise in V1"
        )

    valid = (
        torch.ones(shape, dtype=torch.bool, device=device)
        if canvas_valid_mask is None
        else canvas_valid_mask
    )
    if valid.shape != shape or valid.dtype != torch.bool:
        raise ValueError("canvas_valid_mask must be boolean and match shape")
    if valid.device != torch.device(device):
        raise ValueError("canvas_valid_mask must be on the sampling device")
    if (~valid).any() and padding_token_id is None:
        raise ValueError("padding_token_id is required when canvas_valid_mask contains False")
    if not valid.any(dim=1).all():
        raise ValueError("every example must contain at least one valid canvas position")

    canvas = noise_process.initial_sample(shape, device=device, generator=generator)
    if padding_token_id is not None:
        canvas = canvas.masked_fill(~valid, int(padding_token_id))
    committed = torch.zeros(shape, dtype=torch.bool, device=device)
    selected = torch.zeros_like(committed)
    history: list[SamplerStep] = []
    forbidden = tuple(cfg.forbidden_output_token_ids)
    if isinstance(noise_process, MaskNoiseProcess):
        forbidden = forbidden + (noise_process.mask_token_id,)

    valid_counts = valid.sum(dim=1)
    for step in range(cfg.max_nfe):
        noise_level = 1.0 - step / cfg.max_nfe
        t = torch.full((batch_size,), noise_level, device=device, dtype=torch.float32)
        logits = denoiser(canvas, t)
        if logits.ndim != 3 or logits.shape[:2] != shape:
            raise ValueError(
                f"denoiser must return [batch, canvas, vocab], got {tuple(logits.shape)}"
            )
        if logits.device != canvas.device:
            raise ValueError("denoiser logits and canvas must be on the same device")
        logits = mask_forbidden_logits(logits, forbidden, inplace=True)
        interpolation = 0.0 if cfg.max_nfe == 1 else step / (cfg.max_nfe - 1)
        temperature = (
            cfg.temperature_start
            + interpolation * (cfg.temperature_end - cfg.temperature_start)
        )
        prediction, confidence, entropy = _predict(
            logits,
            temperature=temperature,
            prediction_mode=cfg.prediction_mode,
            generator=generator,
            compute_entropy=entropy_sampler or return_history,
        )
        target_counts = torch.ceil(valid_counts.float() * ((step + 1) / cfg.max_nfe)).long()

        final_step = step == cfg.max_nfe - 1
        if noise_process.is_absorbing:
            eligible = valid & ~committed
            desired_new_counts = (
                target_counts - (committed & valid).sum(dim=1)
            ).clamp_min(0)
            # EB-Conf uses its entropy budget rather than the count schedule;
            # the last NFE still commits every remaining position.
            newly_selected = _select_positions(
                config=cfg,
                confidence=confidence,
                entropy=entropy,
                eligible=eligible,
                target_counts=desired_new_counts,
                final_step=final_step,
            )
            candidate = canvas.clone()
            candidate[newly_selected] = prediction[newly_selected]
            committed |= newly_selected
            selected = committed.clone()
            canvas = noise_process.renoise(candidate, committed, generator=generator)
        else:
            # No carry-over: positions selected earlier may be revised or sent
            # back to the prior when confidence rankings change.
            newly_selected = _select_positions(
                config=cfg,
                confidence=confidence,
                entropy=entropy,
                eligible=valid,
                target_counts=target_counts,
                final_step=final_step,
            )
            selected = newly_selected
            canvas = noise_process.renoise(prediction, selected, generator=generator)

        if padding_token_id is not None:
            canvas = canvas.masked_fill(~valid, int(padding_token_id))
        if return_history:
            history.append(
                SamplerStep(
                    step=step,
                    noise_level=noise_level,
                    canvas=canvas.clone(),
                    selected_mask=selected.clone(),
                    newly_selected_mask=newly_selected.clone(),
                    prediction=prediction.clone(),
                    confidence=confidence.clone(),
                    entropy=entropy.clone(),
                )
            )

    if not torch.equal(selected, valid):
        raise RuntimeError("final confidence step failed to select every valid position")
    return SamplerOutput(
        canvas=canvas,
        selected_mask=selected,
        nfe=cfg.max_nfe,
        history=tuple(history),
    )


def udm_marginalized_reverse_probs(
    denoiser_probs: torch.Tensor,
    current_tokens: torch.Tensor,
    *,
    alpha_t: float,
    alpha_s: float,
) -> torch.Tensor:
    """Reference UDM marginalization over every possible clean token.

    ``current_tokens`` contains support-local indices.  This literal O(K)
    implementation is intended for tests and small vocabularies; the sampler
    below uses an algebraically equivalent chunked leave-one-out conversion so
    MawJais never materializes another full ``[B, C, 250624]`` FP32 tensor.
    """

    if denoiser_probs.ndim != 3:
        raise ValueError("denoiser_probs must have shape [batch, canvas, support]")
    if current_tokens.shape != denoiser_probs.shape[:2]:
        raise ValueError("current_tokens must match denoiser_probs[:2]")
    if current_tokens.dtype == torch.bool or current_tokens.is_floating_point():
        raise TypeError("current_tokens must contain integer support-local IDs")
    if current_tokens.device != denoiser_probs.device:
        raise ValueError("current_tokens and denoiser_probs must share a device")
    if not 0.0 <= float(alpha_t) <= float(alpha_s) <= 1.0:
        raise ValueError("reverse UDM step requires 0 <= alpha_t <= alpha_s <= 1")
    if float(alpha_t) >= 1.0:
        raise ValueError("reverse UDM step is undefined at alpha_t=1 (t=0)")
    if float(alpha_s) <= 0.0:
        raise ValueError("reverse UDM step requires alpha_s > 0")
    if not torch.isfinite(denoiser_probs).all() or (denoiser_probs < 0).any():
        raise ValueError("denoiser_probs must be finite and non-negative")

    support_size = denoiser_probs.shape[-1]
    if (current_tokens < 0).any() or (current_tokens >= support_size).any():
        raise ValueError("current_tokens contains an index outside the UDM support")
    probabilities = denoiser_probs.to(torch.float64)
    probability_mass = probabilities.sum(dim=-1, keepdim=True)
    if (probability_mass <= 0).any():
        raise ValueError("denoiser_probs has an empty categorical distribution")
    probabilities = probabilities / probability_mass

    uniform_t = (1.0 - float(alpha_t)) / support_size
    denominator = torch.full_like(probabilities, uniform_t)
    denominator.scatter_add_(
        -1,
        current_tokens[..., None],
        torch.full_like(current_tokens[..., None], float(alpha_t), dtype=torch.float64),
    )
    reweighted = probabilities / denominator
    reweighted_sum = reweighted.sum(dim=-1, keepdim=True)

    ratio = float(alpha_t) / float(alpha_s)
    transition_factor = torch.full_like(
        probabilities, (1.0 - ratio) / support_size
    )
    transition_factor.scatter_add_(
        -1,
        current_tokens[..., None],
        torch.full_like(current_tokens[..., None], ratio, dtype=torch.float64),
    )
    marginal = transition_factor * (
        float(alpha_s) * reweighted
        + ((1.0 - float(alpha_s)) / support_size) * reweighted_sum
    )
    mass = marginal.sum(dim=-1, keepdim=True)
    if not torch.isfinite(marginal).all() or (marginal < 0).any() or (mass <= 0).any():
        raise ValueError("UDM marginalization produced invalid reverse probabilities")
    return marginal / mass


def udm_argmax_reverse_probs(
    denoiser_probs: torch.Tensor,
    current_tokens: torch.Tensor,
    *,
    alpha_t: float,
    alpha_s: float,
) -> torch.Tensor:
    """Reference argmax-UDM law for tests and small vocabularies.

    The clean denoiser distribution is projected to a one-hot vector at its
    support-local argmax *before* the UDM bridge is evaluated.  Consequently
    the marginalization contains a single non-zero clean hypothesis and is
    exactly ``B(z; y_hat, x)``.  The production sampler below uses an
    algebraically equivalent four-component mixture without materializing this
    one-hot tensor for MawJais' 250k vocabulary.
    """

    if denoiser_probs.ndim != 3:
        raise ValueError("denoiser_probs must have shape [batch, canvas, support]")
    if not torch.isfinite(denoiser_probs).all() or (denoiser_probs < 0).any():
        raise ValueError("denoiser_probs must be finite and non-negative")
    if (denoiser_probs.sum(dim=-1) <= 0).any():
        raise ValueError("denoiser_probs has an empty categorical distribution")
    clean_argmax = denoiser_probs.argmax(dim=-1)
    one_hot = torch.zeros_like(denoiser_probs)
    one_hot.scatter_(-1, clean_argmax[..., None], 1)
    return udm_marginalized_reverse_probs(
        one_hot,
        current_tokens,
        alpha_t=alpha_t,
        alpha_s=alpha_s,
    )


def _prepare_exact_sampling_canvas(
    noise_process: DiscreteNoiseProcess,
    shape: Sequence[int],
    *,
    device: torch.device | str,
    generator: torch.Generator | None,
    canvas_valid_mask: torch.Tensor | None,
    padding_token_id: int | None,
) -> tuple[tuple[int, int], torch.Tensor, torch.Tensor]:
    resolved_shape = tuple(int(dimension) for dimension in shape)
    if len(resolved_shape) != 2 or any(dimension <= 0 for dimension in resolved_shape):
        raise ValueError(
            f"shape must be [batch, canvas] with positive values, got {resolved_shape}"
        )
    valid = (
        torch.ones(resolved_shape, dtype=torch.bool, device=device)
        if canvas_valid_mask is None
        else canvas_valid_mask
    )
    if valid.shape != resolved_shape or valid.dtype != torch.bool:
        raise ValueError("canvas_valid_mask must be boolean and match shape")
    if valid.device != torch.device(device):
        raise ValueError("canvas_valid_mask must be on the sampling device")
    if (~valid).any() and padding_token_id is None:
        raise ValueError("padding_token_id is required when canvas_valid_mask contains False")
    if not valid.any(dim=1).all():
        raise ValueError("every example must contain at least one valid canvas position")
    canvas = noise_process.initial_sample(
        resolved_shape, device=device, generator=generator
    )
    if padding_token_id is not None:
        canvas = canvas.masked_fill(~valid, int(padding_token_id))
    return resolved_shape, valid, canvas


def _validate_denoiser_output(
    logits: torch.Tensor, *, shape: tuple[int, int], canvas: torch.Tensor
) -> None:
    if logits.ndim != 3 or logits.shape[:2] != shape:
        raise ValueError(
            f"denoiser must return [batch, canvas, vocab], got {tuple(logits.shape)}"
        )
    if logits.device != canvas.device:
        raise ValueError("denoiser logits and canvas must be on the same device")


@torch.no_grad()
def full_argmax_refinement_sample(
    denoiser: Denoiser,
    noise_process: UniformNoiseProcess,
    shape: Sequence[int],
    *,
    device: torch.device | str,
    config: ConfidenceSamplerConfig | None = None,
    generator: torch.Generator | None = None,
    canvas_valid_mask: torch.Tensor | None = None,
    padding_token_id: int | None = None,
    return_history: bool = False,
) -> SamplerOutput:
    """Greedily replace the complete Uniform canvas at every model call.

    The first input is sampled from the Uniform training prior.  Every forward
    thereafter writes ``argmax p_theta(x_0 | x_t, t)`` at every valid position,
    and that full prediction is passed directly to the next forward.  Nothing
    is confidence-selected or renoised after initialization.
    """

    cfg = (
        ConfidenceSamplerConfig(sampler="full_argmax_refinement")
        if config is None
        else config
    )
    if cfg.sampler != "full_argmax_refinement":
        raise ValueError("full_argmax_refinement_sample requires its matching sampler config")
    if not isinstance(noise_process, UniformNoiseProcess):
        raise TypeError("full_argmax_refinement requires UniformNoiseProcess")
    resolved_shape, valid, canvas = _prepare_exact_sampling_canvas(
        noise_process,
        shape,
        device=device,
        generator=generator,
        canvas_valid_mask=canvas_valid_mask,
        padding_token_id=padding_token_id,
    )
    history: list[SamplerStep] = []
    forbidden = tuple(cfg.forbidden_output_token_ids)
    for step in range(cfg.max_nfe):
        t_value = 1.0 - step / cfg.max_nfe
        noise_level = torch.full(
            (resolved_shape[0],),
            t_value,
            device=device,
            dtype=torch.float32,
        )
        logits = denoiser(canvas, noise_level)
        _validate_denoiser_output(logits, shape=resolved_shape, canvas=canvas)
        logits = mask_forbidden_logits(logits, forbidden, inplace=True)
        maximum, prediction = logits.max(dim=-1)
        if not torch.isfinite(maximum[valid]).all():
            raise ValueError("denoiser logits have no finite argmax on a valid position")
        canvas = prediction
        if padding_token_id is not None:
            canvas = canvas.masked_fill(~valid, int(padding_token_id))
        if return_history:
            _, confidence, entropy = _predict(
                logits,
                temperature=1.0,
                prediction_mode="argmax",
                generator=None,
                compute_entropy=True,
            )
            history.append(
                SamplerStep(
                    step=step,
                    noise_level=t_value,
                    canvas=canvas.clone(),
                    selected_mask=valid.clone(),
                    newly_selected_mask=valid.clone(),
                    prediction=prediction.clone(),
                    confidence=confidence.clone(),
                    entropy=entropy.clone(),
                )
            )
    return SamplerOutput(
        canvas=canvas,
        selected_mask=valid.clone(),
        nfe=cfg.max_nfe,
        history=tuple(history),
    )


@torch.no_grad()
def diffusion_gemma_entropy_sample(
    denoiser: Denoiser,
    noise_process: UniformNoiseProcess,
    shape: Sequence[int],
    *,
    device: torch.device | str,
    config: ConfidenceSamplerConfig | None = None,
    generator: torch.Generator | None = None,
    canvas_valid_mask: torch.Tensor | None = None,
    padding_token_id: int | None = None,
    return_history: bool = False,
) -> SamplerOutput:
    """DiffusionGemma cumulative-entropy sampling for a Uniform canvas.

    This mirrors the released sampler's reverse-loop contract: categorical
    candidates under a linear temperature schedule, lowest-entropy cumulative
    acceptance, fresh Uniform noise on every rejected position, and the final
    denoiser argmax as the returned canvas.  The MawJais denoiser has no
    DiffusionGemma self-conditioning head, so no self-conditioning state is
    introduced here.
    """

    cfg = (
        ConfidenceSamplerConfig(
            sampler="diffusion_gemma_entropy",
            prediction_mode="categorical",
            temperature_start=0.8,
            temperature_end=0.4,
            entropy_budget_gamma=0.1,
        )
        if config is None
        else config
    )
    if cfg.sampler != "diffusion_gemma_entropy":
        raise ValueError("diffusion_gemma_entropy_sample requires its matching sampler config")
    if not isinstance(noise_process, UniformNoiseProcess):
        raise TypeError("diffusion_gemma_entropy requires UniformNoiseProcess")
    if cfg.entropy_budget_gamma <= 0:
        raise ValueError("diffusion_gemma_entropy requires entropy_budget_gamma > 0")
    resolved_shape, valid, canvas = _prepare_exact_sampling_canvas(
        noise_process,
        shape,
        device=device,
        generator=generator,
        canvas_valid_mask=canvas_valid_mask,
        padding_token_id=padding_token_id,
    )
    history: list[SamplerStep] = []
    forbidden = tuple(cfg.forbidden_output_token_ids)
    final_argmax: torch.Tensor | None = None
    for step in range(cfg.max_nfe):
        t_value = 1.0 - step / cfg.max_nfe
        noise_level = torch.full(
            (resolved_shape[0],),
            t_value,
            device=device,
            dtype=torch.float32,
        )
        logits = denoiser(canvas, noise_level)
        _validate_denoiser_output(logits, shape=resolved_shape, canvas=canvas)
        logits = mask_forbidden_logits(logits, forbidden, inplace=True)

        # DiffusionGemma counts cur_step down from max_nfe to one and applies
        # T = T_min + (T_max - T_min) * cur_step / max_nfe.
        remaining_steps = cfg.max_nfe - step
        temperature = cfg.temperature_end + (
            (cfg.temperature_start - cfg.temperature_end)
            * remaining_steps
            / cfg.max_nfe
        )
        candidate, confidence, entropy = _predict(
            logits,
            temperature=temperature,
            prediction_mode="categorical",
            generator=generator,
            compute_entropy=True,
        )
        final_argmax = logits.argmax(dim=-1)
        accepted = select_diffusion_gemma_entropy_positions(
            entropy,
            valid,
            entropy_bound=cfg.entropy_budget_gamma,
        )
        canvas = noise_process.renoise(candidate, accepted, generator=generator)
        if padding_token_id is not None:
            canvas = canvas.masked_fill(~valid, int(padding_token_id))
            final_argmax = final_argmax.masked_fill(~valid, int(padding_token_id))
        if return_history:
            history.append(
                SamplerStep(
                    step=step,
                    noise_level=t_value,
                    canvas=canvas.clone(),
                    selected_mask=accepted.clone(),
                    newly_selected_mask=accepted.clone(),
                    prediction=candidate.clone(),
                    confidence=confidence.clone(),
                    entropy=entropy.clone(),
                )
            )
    if final_argmax is None:
        raise AssertionError("positive max_nfe produced no DiffusionGemma denoiser call")
    return SamplerOutput(
        canvas=final_argmax,
        selected_mask=valid.clone(),
        nfe=cfg.max_nfe,
        history=tuple(history),
    )


def _udm_support_layout(
    noise_process: UniformNoiseProcess,
    *,
    vocab_size: int,
    device: torch.device,
    forbidden_output_token_ids: Sequence[int],
) -> tuple[torch.Tensor, bool, torch.Tensor | None]:
    cache_key = (
        str(device),
        int(vocab_size),
        tuple(sorted(set(int(token_id) for token_id in forbidden_output_token_ids))),
    )
    cache = getattr(noise_process, "_sampler_support_layout_cache", None)
    if isinstance(cache, dict) and cache_key in cache:
        return cache[cache_key]

    support_cpu = noise_process.noise_ids
    if (support_cpu >= vocab_size).any():
        raise ValueError("Uniform prior support contains an ID outside the model vocabulary")
    support_is_full = support_cpu.numel() == vocab_size and torch.equal(
        support_cpu, torch.arange(vocab_size, dtype=torch.long)
    )
    support = support_cpu.to(device=device)
    if support_is_full:
        layout = (support, True, None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(noise_process, "_sampler_support_layout_cache", cache)
        cache[cache_key] = layout
        return layout

    forbidden = set(int(token_id) for token_id in forbidden_output_token_ids)
    overlap = forbidden.intersection(int(token_id) for token_id in support_cpu.tolist())
    if overlap:
        raise ValueError(
            "a reduced safe Uniform prior must exclude every forbidden clean token; "
            f"overlap={sorted(overlap)[:8]}"
        )
    global_to_local = torch.full(
        (vocab_size,), -1, dtype=torch.long, device=device
    )
    global_to_local[support] = torch.arange(
        support.numel(), dtype=torch.long, device=device
    )
    layout = (support, False, global_to_local)
    if not isinstance(cache, dict):
        cache = {}
        setattr(noise_process, "_sampler_support_layout_cache", cache)
    cache[cache_key] = layout
    return layout


def _support_logits_chunk(
    logits: torch.Tensor,
    support: torch.Tensor,
    support_is_full: bool,
    start: int,
    stop: int,
) -> torch.Tensor:
    if support_is_full:
        return logits[..., start:stop].float().clone()
    return logits.index_select(-1, support[start:stop]).float()


def _argmax_udm_clean_support(
    logits: torch.Tensor,
    valid: torch.Tensor,
    *,
    support_layout: tuple[torch.Tensor, bool, torch.Tensor | None],
    forbidden_output_token_ids: Sequence[int],
    vocab_chunk_size: int = 16_384,
) -> torch.Tensor:
    """Return the clean-token argmax restricted to the trained UDM support.

    The final-argmax sampler uses this at ``alpha_s=1``; ``argmax_udm`` uses it
    before every bridge transition.  The helper projects the clean-token
    distribution to its mode without allocating another full FP32 vocabulary
    tensor.  A reduced ``safe_vocab`` remains authoritative: tokens outside
    the forward prior support cannot silently become a clean hypothesis.
    """

    if vocab_chunk_size <= 0:
        raise ValueError("vocab_chunk_size must be positive")
    support, support_is_full, _ = support_layout
    support_size = int(support.numel())
    clean_logits = (
        mask_forbidden_logits(logits, forbidden_output_token_ids, inplace=True)
        if support_is_full
        else logits
    )
    best_score = torch.full(
        valid.shape, -torch.inf, dtype=torch.float32, device=logits.device
    )
    best_local = torch.zeros(valid.shape, dtype=torch.long, device=logits.device)
    for start in range(0, support_size, vocab_chunk_size):
        stop = min(start + vocab_chunk_size, support_size)
        chunk = _support_logits_chunk(
            clean_logits, support, support_is_full, start, stop
        )
        chunk_score, chunk_index = chunk.max(dim=-1)
        better = chunk_score > best_score
        best_score = torch.where(better, chunk_score, best_score)
        best_local = torch.where(better, chunk_index + start, best_local)
    if not torch.isfinite(best_score[valid]).all():
        raise ValueError("UDM final clean-token distribution has no finite argmax")
    return best_local if support_is_full else support[best_local]


def _subtract_current_coordinate(
    chunk: torch.Tensor,
    current_local: torch.Tensor,
    valid: torch.Tensor,
    *,
    start: int,
    stop: int,
    amount: float,
) -> None:
    in_chunk = valid & (current_local >= start) & (current_local < stop)
    indices = (current_local - start).clamp(min=0, max=stop - start - 1)
    updates = torch.where(
        in_chunk,
        torch.full_like(current_local, -float(amount), dtype=torch.float32),
        torch.zeros_like(current_local, dtype=torch.float32),
    )
    chunk.scatter_add_(-1, indices[..., None], updates[..., None])


def _sample_udm_marginalized_transition(
    logits: torch.Tensor,
    canvas: torch.Tensor,
    valid: torch.Tensor,
    noise_process: UniformNoiseProcess,
    *,
    alpha_t: float,
    alpha_s: float,
    forbidden_output_token_ids: Sequence[int],
    generator: torch.Generator | None,
    clean_temperature: float = 1.0,
    vocab_chunk_size: int = 16_384,
    support_layout: tuple[torch.Tensor, bool, torch.Tensor | None] | None = None,
) -> torch.Tensor:
    """Sample the exact marginal UDM reverse law with bounded scratch memory.

    The ordinary clean-token denoiser is converted to the leave-one-out (LOO)
    posterior in logits space.  ``clean_temperature`` optionally calibrates
    the denoiser distribution *before* that LOO conversion: the computation is
    ``log d_T(y) - log q_t(x|y)``, not a temperature applied to the completed
    reverse bridge.  Applying the canonical bridge to that LOO posterior is
    algebraically identical to summing the normalized bridge over every clean
    token hypothesis; unlike ``B(z; d_theta, x)``, this is the true
    marginalization parameterization.
    """

    if vocab_chunk_size <= 0:
        raise ValueError("vocab_chunk_size must be positive")
    if not math.isfinite(clean_temperature) or clean_temperature <= 0.0:
        raise ValueError("clean_temperature must be finite and positive")
    if not 0.0 <= alpha_t <= alpha_s <= 1.0 or alpha_t >= 1.0 or alpha_s <= 0.0:
        raise ValueError("invalid UDM reverse alpha pair")
    support, support_is_full, global_to_local = (
        _udm_support_layout(
            noise_process,
            vocab_size=logits.shape[-1],
            device=logits.device,
            forbidden_output_token_ids=forbidden_output_token_ids,
        )
        if support_layout is None
        else support_layout
    )
    support_size = int(support.numel())
    if support_is_full:
        clean_logits = mask_forbidden_logits(
            logits, forbidden_output_token_ids, inplace=True
        )
        current_local = canvas
    else:
        clean_logits = logits
        assert global_to_local is not None
        safe_canvas = canvas.clamp(min=0, max=logits.shape[-1] - 1)
        current_local = global_to_local[safe_canvas]
        if (current_local[valid] < 0).any():
            raise ValueError("current Uniform canvas contains a token outside its prior support")
        current_local = current_local.masked_fill(~valid, 0)

    # d -> LOO conversion from Uniform Diffusion Models Revisited:
    # subtract log(q_t(x|y=x) / q_t(x|y!=x)) at the observed coordinate.
    delta = float(
        torch.log1p(
            torch.tensor(
                support_size * alpha_t / (1.0 - alpha_t), dtype=torch.float64
            )
        ).item()
    )
    loo_log_normalizer = torch.full(
        canvas.shape, -torch.inf, dtype=torch.float32, device=logits.device
    )
    for start in range(0, support_size, vocab_chunk_size):
        stop = min(start + vocab_chunk_size, support_size)
        chunk = _support_logits_chunk(
            clean_logits, support, support_is_full, start, stop
        )
        if clean_temperature != 1.0:
            chunk.div_(float(clean_temperature))
        _subtract_current_coordinate(
            chunk,
            current_local,
            valid,
            start=start,
            stop=stop,
            amount=delta,
        )
        loo_log_normalizer = torch.logaddexp(
            loo_log_normalizer, torch.logsumexp(chunk, dim=-1)
        )
    if not torch.isfinite(loo_log_normalizer[valid]).all():
        raise ValueError("UDM clean-token logits do not define a finite distribution")

    current_global = canvas.clamp(min=0, max=logits.shape[-1] - 1)
    current_logits = clean_logits.gather(-1, current_global[..., None]).squeeze(-1)
    current_logits = current_logits.float()
    if clean_temperature != 1.0:
        current_logits = current_logits.div(float(clean_temperature))
    loo_current = torch.exp(
        current_logits - float(delta) - loo_log_normalizer
    )
    loo_current = loo_current.masked_fill(~valid, 0.0)

    ratio = alpha_t / alpha_s
    clean_mass = alpha_s - alpha_t
    uniform_mass_per_token = (1.0 - ratio) * (1.0 - alpha_s) / support_size
    current_extra = (
        support_size * alpha_t * loo_current + (ratio - alpha_t)
    ).clamp_min(0.0)

    best_score = torch.full(
        canvas.shape, -torch.inf, dtype=torch.float32, device=logits.device
    )
    best_local = torch.zeros_like(canvas)
    for start in range(0, support_size, vocab_chunk_size):
        stop = min(start + vocab_chunk_size, support_size)
        chunk = _support_logits_chunk(
            clean_logits, support, support_is_full, start, stop
        )
        if clean_temperature != 1.0:
            chunk.div_(float(clean_temperature))
        _subtract_current_coordinate(
            chunk,
            current_local,
            valid,
            start=start,
            stop=stop,
            amount=delta,
        )
        loo_probabilities = torch.exp(chunk - loo_log_normalizer[..., None])
        weights = clean_mass * loo_probabilities + uniform_mass_per_token
        in_chunk = valid & (current_local >= start) & (current_local < stop)
        indices = (current_local - start).clamp(min=0, max=stop - start - 1)
        weights.scatter_add_(
            -1,
            indices[..., None],
            torch.where(in_chunk, current_extra, torch.zeros_like(current_extra))[
                ..., None
            ],
        )
        log_weights = torch.where(
            weights > 0,
            weights.log(),
            torch.full_like(weights, -torch.inf),
        )
        uniform = torch.rand(
            log_weights.shape,
            device=log_weights.device,
            dtype=torch.float32,
            generator=generator,
        ).clamp_(min=torch.finfo(torch.float32).tiny, max=1.0 - 1.0e-7)
        scores = log_weights - torch.log(-torch.log(uniform))
        chunk_score, chunk_index = scores.max(dim=-1)
        better = chunk_score > best_score
        best_score = torch.where(better, chunk_score, best_score)
        best_local = torch.where(better, chunk_index + start, best_local)
    if not torch.isfinite(best_score[valid]).all():
        raise ValueError("UDM reverse bridge has no finite categorical mass")
    return best_local if support_is_full else support[best_local]


def _sample_argmax_udm_transition(
    logits: torch.Tensor,
    canvas: torch.Tensor,
    valid: torch.Tensor,
    noise_process: UniformNoiseProcess,
    *,
    alpha_t: float,
    alpha_s: float,
    forbidden_output_token_ids: Sequence[int],
    generator: torch.Generator | None,
    support_layout: tuple[torch.Tensor, bool, torch.Tensor | None] | None = None,
) -> torch.Tensor:
    """Sample ``B(z; argmax_y d_theta(y), x)`` exactly in constant scratch.

    Expanding the Uniform bridge gives four non-negative components: a uniform
    draw over the prior support, a point mass on the clean argmax, a point mass
    on the current token, and an overlap mass when those two tokens agree.  The
    component weights sum to ``q_t(x | y_hat)``, so normalizing explicitly is
    unnecessary.  Only the clean argmax scans the vocabulary, in bounded
    chunks; sampling the bridge itself is O(B*C).
    """

    if not 0.0 <= alpha_t <= alpha_s <= 1.0 or alpha_t >= 1.0 or alpha_s <= 0.0:
        raise ValueError("invalid argmax-UDM reverse alpha pair")
    support, support_is_full, global_to_local = (
        _udm_support_layout(
            noise_process,
            vocab_size=logits.shape[-1],
            device=logits.device,
            forbidden_output_token_ids=forbidden_output_token_ids,
        )
        if support_layout is None
        else support_layout
    )
    clean_global = _argmax_udm_clean_support(
        logits,
        valid,
        support_layout=(support, support_is_full, global_to_local),
        forbidden_output_token_ids=forbidden_output_token_ids,
    )
    if support_is_full:
        clean_local = clean_global
        current_local = canvas
    else:
        assert global_to_local is not None
        safe_canvas = canvas.clamp(min=0, max=logits.shape[-1] - 1)
        current_local = global_to_local[safe_canvas]
        if (current_local[valid] < 0).any():
            raise ValueError("current Uniform canvas contains a token outside its prior support")
        current_local = current_local.masked_fill(~valid, 0)
        clean_local = global_to_local[clean_global]
        if (clean_local[valid] < 0).any():
            raise RuntimeError("argmax-UDM clean hypothesis escaped its prior support")

    support_size = int(support.numel())
    ratio = float(alpha_t) / float(alpha_s)
    uniform_component = (
        (1.0 - float(alpha_s)) * (1.0 - ratio) / support_size
    )
    clean_component = float(alpha_s) * (1.0 - ratio) / support_size
    current_component = (1.0 - float(alpha_s)) * ratio / support_size
    overlap_component = float(alpha_s) * ratio

    same = clean_local == current_local
    total_mass = torch.full(
        canvas.shape,
        uniform_component + clean_component + current_component,
        dtype=torch.float64,
        device=canvas.device,
    )
    total_mass = total_mass + same.to(torch.float64) * overlap_component
    if not torch.isfinite(total_mass[valid]).all() or (total_mass[valid] <= 0).any():
        raise ValueError("argmax-UDM bridge has no finite categorical mass")

    component_draw = torch.rand(
        canvas.shape,
        dtype=torch.float64,
        device=canvas.device,
        generator=generator,
    ) * total_mass
    uniform_local = torch.randint(
        low=0,
        high=support_size,
        size=canvas.shape,
        dtype=torch.long,
        device=canvas.device,
        generator=generator,
    )
    uniform_global = uniform_local if support_is_full else support[uniform_local]
    next_canvas = uniform_global

    clean_boundary = uniform_component + clean_component
    current_boundary = clean_boundary + current_component
    choose_clean = (component_draw >= uniform_component) & (
        component_draw < clean_boundary
    )
    choose_current = (component_draw >= clean_boundary) & (
        component_draw < current_boundary
    )
    # Any remaining mass is the overlap term and exists only when clean ==
    # current, so routing it to the clean token is exact.
    choose_overlap = same & (component_draw >= current_boundary)
    next_canvas = torch.where(choose_clean | choose_overlap, clean_global, next_canvas)
    next_canvas = torch.where(choose_current, canvas, next_canvas)
    return torch.where(valid, next_canvas, canvas)


@torch.no_grad()
def udm_marginalization_sample(
    denoiser: Denoiser,
    noise_process: UniformNoiseProcess,
    shape: Sequence[int],
    *,
    device: torch.device | str,
    config: ConfidenceSamplerConfig | None = None,
    generator: torch.Generator | None = None,
    canvas_valid_mask: torch.Tensor | None = None,
    padding_token_id: int | None = None,
    return_history: bool = False,
) -> SamplerOutput:
    """Sample UDM bridges, optionally projecting only the final law to argmax.

    ``udm_marginalization`` remains the exact categorical ancestral sampler.
    ``udm_tempered_marginalization`` changes only the clean denoiser
    distribution supplied to the same bridge, using a linear temperature
    schedule.  Temperature one is exactly the canonical computation.
    ``udm_marginalization_final_argmax`` uses the identical stochastic bridge
    for every intermediate transition and changes only the final ``s=0``
    transition, where the marginalized law equals the clean denoiser law.
    """

    cfg = (
        ConfidenceSamplerConfig(
            sampler="udm_marginalization",
            prediction_mode="categorical",
            temperature_start=1.0,
            temperature_end=1.0,
        )
        if config is None
        else config
    )
    if cfg.sampler not in (
        "udm_marginalization",
        "udm_tempered_marginalization",
        "udm_marginalization_final_argmax",
    ):
        raise ValueError(
            "udm_marginalization_sample requires an UDM marginalization sampler config"
        )
    if not isinstance(noise_process, UniformNoiseProcess):
        raise TypeError(f"{cfg.sampler} requires UniformNoiseProcess")
    use_final_argmax = cfg.sampler == "udm_marginalization_final_argmax"
    use_clean_temperature = cfg.sampler == "udm_tempered_marginalization"
    resolved_shape, valid, canvas = _prepare_exact_sampling_canvas(
        noise_process,
        shape,
        device=device,
        generator=generator,
        canvas_valid_mask=canvas_valid_mask,
        padding_token_id=padding_token_id,
    )
    history: list[SamplerStep] = []
    support_layout: tuple[torch.Tensor, bool, torch.Tensor | None] | None = None
    for step in range(cfg.max_nfe):
        t_value = 1.0 - step / cfg.max_nfe
        s_value = 1.0 - (step + 1) / cfg.max_nfe
        noise_level = torch.full(
            (resolved_shape[0],),
            t_value,
            device=device,
            dtype=torch.float32,
        )
        logits = denoiser(canvas, noise_level)
        _validate_denoiser_output(logits, shape=resolved_shape, canvas=canvas)
        if support_layout is None:
            support_layout = _udm_support_layout(
                noise_process,
                vocab_size=logits.shape[-1],
                device=logits.device,
                forbidden_output_token_ids=cfg.forbidden_output_token_ids,
            )
        if use_final_argmax and step == cfg.max_nfe - 1:
            next_canvas = _argmax_udm_clean_support(
                logits,
                valid,
                support_layout=support_layout,
                forbidden_output_token_ids=cfg.forbidden_output_token_ids,
            )
        else:
            transition_kwargs = {
                "alpha_t": 1.0 - t_value,
                "alpha_s": 1.0 - s_value,
                "forbidden_output_token_ids": cfg.forbidden_output_token_ids,
                "generator": generator,
                "support_layout": support_layout,
            }
            if use_clean_temperature:
                interpolation = (
                    0.0 if cfg.max_nfe == 1 else step / (cfg.max_nfe - 1)
                )
                transition_kwargs["clean_temperature"] = (
                    cfg.temperature_start
                    + interpolation
                    * (cfg.temperature_end - cfg.temperature_start)
                )
            next_canvas = _sample_udm_marginalized_transition(
                logits,
                canvas,
                valid,
                noise_process,
                **transition_kwargs,
            )
        if padding_token_id is not None:
            next_canvas = next_canvas.masked_fill(~valid, int(padding_token_id))
        changed = valid & (next_canvas != canvas)
        canvas = next_canvas
        if return_history:
            zeros = torch.zeros(resolved_shape, dtype=torch.float32, device=device)
            history.append(
                SamplerStep(
                    step=step,
                    noise_level=t_value,
                    canvas=canvas.clone(),
                    selected_mask=valid.clone(),
                    newly_selected_mask=changed.clone(),
                    prediction=canvas.clone(),
                    confidence=zeros.clone(),
                    entropy=zeros.clone(),
                )
            )
    return SamplerOutput(
        canvas=canvas,
        selected_mask=valid.clone(),
        nfe=cfg.max_nfe,
        history=tuple(history),
    )


@torch.no_grad()
def udm_tempered_marginalization_sample(
    denoiser: Denoiser,
    noise_process: UniformNoiseProcess,
    shape: Sequence[int],
    *,
    device: torch.device | str,
    config: ConfidenceSamplerConfig | None = None,
    generator: torch.Generator | None = None,
    canvas_valid_mask: torch.Tensor | None = None,
    padding_token_id: int | None = None,
    return_history: bool = False,
) -> SamplerOutput:
    """Sample UDM after temperature-calibrating each clean distribution."""

    cfg = (
        ConfidenceSamplerConfig(
            sampler="udm_tempered_marginalization",
            prediction_mode="categorical",
            temperature_start=1.0,
            temperature_end=1.0,
        )
        if config is None
        else config
    )
    if cfg.sampler != "udm_tempered_marginalization":
        raise ValueError(
            "udm_tempered_marginalization_sample requires "
            "sampler='udm_tempered_marginalization'"
        )
    return udm_marginalization_sample(
        denoiser,
        noise_process,
        shape,
        device=device,
        config=cfg,
        generator=generator,
        canvas_valid_mask=canvas_valid_mask,
        padding_token_id=padding_token_id,
        return_history=return_history,
    )


@torch.no_grad()
def argmax_udm_sample(
    denoiser: Denoiser,
    noise_process: UniformNoiseProcess,
    shape: Sequence[int],
    *,
    device: torch.device | str,
    config: ConfidenceSamplerConfig | None = None,
    generator: torch.Generator | None = None,
    canvas_valid_mask: torch.Tensor | None = None,
    padding_token_id: int | None = None,
    return_history: bool = False,
) -> SamplerOutput:
    """Run UDM with a one-hot clean argmax hypothesis at every transition.

    This is a plug-in bridge ablation, not exact UDM marginalization: each
    denoiser distribution is replaced by ``one_hot(argmax(d_theta))`` and the
    next state is sampled from the resulting single bridge.  The bridge remains
    stochastic until its final transition, which deterministically yields the
    current clean argmax.
    """

    cfg = (
        ConfidenceSamplerConfig(
            sampler="argmax_udm",
            prediction_mode="argmax",
            temperature_start=1.0,
            temperature_end=1.0,
        )
        if config is None
        else config
    )
    if cfg.sampler != "argmax_udm":
        raise ValueError("argmax_udm_sample requires sampler='argmax_udm'")
    if not isinstance(noise_process, UniformNoiseProcess):
        raise TypeError("argmax_udm requires UniformNoiseProcess")
    resolved_shape, valid, canvas = _prepare_exact_sampling_canvas(
        noise_process,
        shape,
        device=device,
        generator=generator,
        canvas_valid_mask=canvas_valid_mask,
        padding_token_id=padding_token_id,
    )
    history: list[SamplerStep] = []
    support_layout: tuple[torch.Tensor, bool, torch.Tensor | None] | None = None
    for step in range(cfg.max_nfe):
        t_value = 1.0 - step / cfg.max_nfe
        s_value = 1.0 - (step + 1) / cfg.max_nfe
        noise_level = torch.full(
            (resolved_shape[0],),
            t_value,
            device=device,
            dtype=torch.float32,
        )
        logits = denoiser(canvas, noise_level)
        _validate_denoiser_output(logits, shape=resolved_shape, canvas=canvas)
        if support_layout is None:
            support_layout = _udm_support_layout(
                noise_process,
                vocab_size=logits.shape[-1],
                device=logits.device,
                forbidden_output_token_ids=cfg.forbidden_output_token_ids,
            )
        next_canvas = _sample_argmax_udm_transition(
            logits,
            canvas,
            valid,
            noise_process,
            alpha_t=1.0 - t_value,
            alpha_s=1.0 - s_value,
            forbidden_output_token_ids=cfg.forbidden_output_token_ids,
            generator=generator,
            support_layout=support_layout,
        )
        if padding_token_id is not None:
            next_canvas = next_canvas.masked_fill(~valid, int(padding_token_id))
        changed = valid & (next_canvas != canvas)
        canvas = next_canvas
        if return_history:
            zeros = torch.zeros(resolved_shape, dtype=torch.float32, device=device)
            history.append(
                SamplerStep(
                    step=step,
                    noise_level=t_value,
                    canvas=canvas.clone(),
                    selected_mask=valid.clone(),
                    newly_selected_mask=changed.clone(),
                    prediction=canvas.clone(),
                    confidence=zeros.clone(),
                    entropy=zeros.clone(),
                )
            )
    return SamplerOutput(
        canvas=canvas,
        selected_mask=valid.clone(),
        nfe=cfg.max_nfe,
        history=tuple(history),
    )


@torch.no_grad()
def mdlm_ancestral_sample(
    denoiser: Denoiser,
    noise_process: MaskNoiseProcess,
    shape: Sequence[int],
    *,
    device: torch.device | str,
    config: ConfidenceSamplerConfig | None = None,
    generator: torch.Generator | None = None,
    canvas_valid_mask: torch.Tensor | None = None,
    padding_token_id: int | None = None,
    return_history: bool = False,
) -> SamplerOutput:
    """Classical MDLM SUBS sampler with independent Bernoulli unmasking.

    ``max_nfe`` is an upper bound.  Once every valid position is visible, SUBS
    carry-over makes all later states bit-identical, so returning immediately
    is an exact stopping rule rather than a confidence-based approximation.
    In a mixed batch the stop applies only when every example is complete.
    ``SamplerOutput.nfe`` records the number of denoiser calls actually made.
    """

    cfg = (
        ConfidenceSamplerConfig(
            sampler="mdlm_ancestral",
            prediction_mode="categorical",
            temperature_start=1.0,
            temperature_end=1.0,
        )
        if config is None
        else config
    )
    if cfg.sampler != "mdlm_ancestral":
        raise ValueError("mdlm_ancestral_sample requires its matching sampler config")
    if not isinstance(noise_process, MaskNoiseProcess):
        raise TypeError("mdlm_ancestral requires MaskNoiseProcess")
    resolved_shape, valid, canvas = _prepare_exact_sampling_canvas(
        noise_process,
        shape,
        device=device,
        generator=generator,
        canvas_valid_mask=canvas_valid_mask,
        padding_token_id=padding_token_id,
    )
    history: list[SamplerStep] = []
    actual_nfe = 0
    visible = valid & (canvas != noise_process.mask_token_id)
    forbidden = tuple(cfg.forbidden_output_token_ids) + (
        noise_process.mask_token_id,
    )
    for step in range(cfg.max_nfe):
        t_value = 1.0 - step / cfg.max_nfe
        s_value = 1.0 - (step + 1) / cfg.max_nfe
        noise_level = torch.full(
            (resolved_shape[0],),
            t_value,
            device=device,
            dtype=torch.float32,
        )
        logits = denoiser(canvas, noise_level)
        actual_nfe += 1
        _validate_denoiser_output(logits, shape=resolved_shape, canvas=canvas)
        logits = mask_forbidden_logits(logits, forbidden, inplace=True)

        eligible = valid & ~visible
        rho = (t_value - s_value) / t_value
        if step == cfg.max_nfe - 1:
            newly_visible = eligible.clone()
        else:
            newly_visible = eligible & (
                torch.rand(
                    resolved_shape,
                    device=device,
                    generator=generator,
                )
                < rho
            )
        next_canvas = canvas.clone()
        if newly_visible.any():
            selected_logits = logits[newly_visible].unsqueeze(0)
            sampled, _, _ = _predict(
                selected_logits,
                temperature=1.0,
                prediction_mode="categorical",
                generator=generator,
                compute_entropy=False,
            )
            next_canvas[newly_visible] = sampled.squeeze(0)
        visible |= newly_visible
        if padding_token_id is not None:
            next_canvas = next_canvas.masked_fill(~valid, int(padding_token_id))
        canvas = next_canvas
        if return_history:
            zeros = torch.zeros(resolved_shape, dtype=torch.float32, device=device)
            history.append(
                SamplerStep(
                    step=step,
                    noise_level=t_value,
                    canvas=canvas.clone(),
                    selected_mask=visible.clone(),
                    newly_selected_mask=newly_visible.clone(),
                    prediction=canvas.clone(),
                    confidence=zeros.clone(),
                    entropy=zeros.clone(),
                )
            )
        # Exact absorbing-state stop: a visible token is immutable under SUBS,
        # hence no subsequent time transition or model call can alter canvas.
        if step + 1 < cfg.max_nfe and torch.equal(visible, valid):
            break
    if not torch.equal(visible, valid):
        raise RuntimeError("final MDLM transition failed to unmask every valid position")
    if (canvas[valid] == noise_process.mask_token_id).any():
        raise RuntimeError("MDLM ancestral sampler returned a MASK token")
    return SamplerOutput(
        canvas=canvas,
        selected_mask=visible,
        nfe=actual_nfe,
        history=tuple(history),
    )


@torch.no_grad()
def sample_diffusion(
    denoiser: Denoiser,
    noise_process: DiscreteNoiseProcess,
    shape: Sequence[int],
    *,
    device: torch.device | str,
    config: ConfidenceSamplerConfig | None = None,
    generator: torch.Generator | None = None,
    canvas_valid_mask: torch.Tensor | None = None,
    padding_token_id: int | None = None,
    return_history: bool = False,
) -> SamplerOutput:
    """Dispatch to a named sampler without changing legacy semantics."""

    cfg = ConfidenceSamplerConfig() if config is None else config
    common = {
        "device": device,
        "config": cfg,
        "generator": generator,
        "canvas_valid_mask": canvas_valid_mask,
        "padding_token_id": padding_token_id,
        "return_history": return_history,
    }
    if cfg.sampler == "full_argmax_refinement":
        if not isinstance(noise_process, UniformNoiseProcess):
            raise ValueError("full_argmax_refinement requires noise_type='uniform'")
        return full_argmax_refinement_sample(
            denoiser, noise_process, shape, **common
        )
    if cfg.sampler == "diffusion_gemma_entropy":
        if not isinstance(noise_process, UniformNoiseProcess):
            raise ValueError("diffusion_gemma_entropy requires noise_type='uniform'")
        return diffusion_gemma_entropy_sample(
            denoiser, noise_process, shape, **common
        )
    if cfg.sampler in (
        "udm_marginalization",
        "udm_marginalization_final_argmax",
    ):
        if not isinstance(noise_process, UniformNoiseProcess):
            raise ValueError(f"{cfg.sampler} requires noise_type='uniform'")
        return udm_marginalization_sample(
            denoiser, noise_process, shape, **common
        )
    if cfg.sampler == "udm_tempered_marginalization":
        if not isinstance(noise_process, UniformNoiseProcess):
            raise ValueError(
                "udm_tempered_marginalization requires noise_type='uniform'"
            )
        return udm_tempered_marginalization_sample(
            denoiser, noise_process, shape, **common
        )
    if cfg.sampler == "argmax_udm":
        if not isinstance(noise_process, UniformNoiseProcess):
            raise ValueError("argmax_udm requires noise_type='uniform'")
        return argmax_udm_sample(denoiser, noise_process, shape, **common)
    if cfg.sampler == "mdlm_ancestral":
        if not isinstance(noise_process, MaskNoiseProcess):
            raise ValueError("mdlm_ancestral requires noise_type='mask'")
        return mdlm_ancestral_sample(denoiser, noise_process, shape, **common)
    return confidence_schedule_sample(denoiser, noise_process, shape, **common)


__all__ = [
    "Denoiser",
    "SamplerOutput",
    "SamplerStep",
    "argmax_udm_sample",
    "confidence_schedule_sample",
    "diffusion_gemma_entropy_sample",
    "full_argmax_refinement_sample",
    "mdlm_ancestral_sample",
    "mask_forbidden_logits",
    "sample_diffusion",
    "select_diffusion_gemma_entropy_positions",
    "select_entropy_bounded_positions",
    "udm_marginalization_sample",
    "udm_tempered_marginalization_sample",
    "udm_argmax_reverse_probs",
    "udm_marginalized_reverse_probs",
]
