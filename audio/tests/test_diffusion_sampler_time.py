import pytest
import torch

import audio.diffusion.sampler as sampler_module
from audio.diffusion.config import ConfidenceSamplerConfig
from audio.diffusion.noise_process import MaskNoiseProcess, UniformNoiseProcess
from audio.diffusion.sampler import (
    _predict,
    _sample_argmax_udm_transition,
    _sample_udm_marginalized_transition,
    argmax_udm_sample,
    confidence_schedule_sample,
    diffusion_gemma_entropy_sample,
    full_argmax_refinement_sample,
    mdlm_ancestral_sample,
    mask_forbidden_logits,
    sample_diffusion,
    select_diffusion_gemma_entropy_positions,
    select_entropy_bounded_positions,
    udm_argmax_reverse_probs,
    udm_marginalization_sample,
    udm_marginalized_reverse_probs,
    udm_tempered_marginalization_sample,
)
from audio.diffusion.time_embedding import SinusoidalTimeFeatures, TimeConditioningMLP


class ScriptedDenoiser:
    def __init__(self, stages, vocab_size=8):
        self.stages = stages
        self.vocab_size = vocab_size
        self.calls = 0

    def __call__(self, canvas, t):
        predictions, strengths = self.stages[self.calls]
        self.calls += 1
        assert t.shape == (canvas.shape[0],)
        logits = torch.zeros(canvas.shape[0], canvas.shape[1], self.vocab_size)
        for position, (token_id, strength) in enumerate(zip(predictions, strengths)):
            logits[:, position, token_id] = strength
        return logits


def test_mask_sampler_carries_committed_tokens_monotonically():
    denoiser = ScriptedDenoiser(
        [
            ([1, 2, 3], [12.0, 6.0, 5.0]),
            ([4, 2, 3], [12.0, 11.0, 5.0]),
            ([5, 6, 3], [12.0, 11.0, 10.0]),
        ]
    )
    output = confidence_schedule_sample(
        denoiser,
        MaskNoiseProcess(mask_token_id=7),
        (1, 3),
        device="cpu",
        config=ConfidenceSamplerConfig(
            max_nfe=3, temperature_start=1.0, temperature_end=1.0
        ),
        return_history=True,
    )
    assert output.canvas.tolist() == [[1, 2, 3]]
    assert [step.selected_mask.sum().item() for step in output.history] == [1, 2, 3]
    # Later predictions try to change committed positions 0/1, but carry-over wins.
    assert [step.canvas[0, 0].item() for step in output.history] == [1, 1, 1]
    assert output.selected_mask.all()


def test_uniform_sampler_recomputes_selection_and_renoises_old_positions():
    denoiser = ScriptedDenoiser(
        [
            ([1, 2, 3], [12.0, 6.0, 5.0]),
            ([4, 2, 3], [2.0, 12.0, 11.0]),
            ([4, 5, 6], [10.0, 10.0, 10.0]),
        ]
    )
    output = confidence_schedule_sample(
        denoiser,
        UniformNoiseProcess([7]),
        (1, 3),
        device="cpu",
        config=ConfidenceSamplerConfig(
            max_nfe=3, temperature_start=1.0, temperature_end=1.0
        ),
        generator=torch.Generator().manual_seed(0),
        return_history=True,
    )
    assert output.history[0].selected_mask.tolist() == [[True, False, False]]
    # Position 0 was selected at step 0, drops from the step-1 top-2 set, and
    # is redrawn from the prior.  This proves there is no persistent commitment.
    assert output.history[1].selected_mask.tolist() == [[False, True, True]]
    assert output.history[1].canvas[0, 0].item() == 7
    assert output.canvas.tolist() == [[4, 5, 6]]


def test_sampler_forbids_invalid_output_tokens():
    def denoiser(canvas, t):
        logits = torch.zeros(canvas.shape[0], canvas.shape[1], 5)
        logits[..., 4] = 100.0  # forbidden winner
        logits[..., 2] = 10.0   # best valid token
        return logits

    output = confidence_schedule_sample(
        denoiser,
        UniformNoiseProcess([0]),
        (1, 2),
        device="cpu",
        config=ConfidenceSamplerConfig(
            max_nfe=1,
            temperature_start=1.0,
            temperature_end=1.0,
            forbidden_output_token_ids=(4,),
        ),
    )
    assert output.canvas.tolist() == [[2, 2]]

    logits = torch.zeros(1, 1, 2)
    with pytest.raises(ValueError, match="entire vocabulary"):
        mask_forbidden_logits(logits, [0, 1])


def test_chunked_prediction_matches_full_softmax_statistics():
    logits = torch.tensor(
        [[[1.0, -2.0, 0.5, -torch.inf, 3.0], [0.2, 0.1, 0.0, -0.1, -0.2]]]
    )
    prediction, confidence, entropy = _predict(
        logits,
        temperature=0.7,
        prediction_mode="argmax",
        generator=None,
        compute_entropy=True,
        vocab_chunk_size=2,
    )
    probabilities = torch.softmax(logits / 0.7, dim=-1)
    expected_entropy = -(
        probabilities * probabilities.clamp_min(1.0e-12).log()
    ).sum(dim=-1)
    assert torch.equal(prediction, probabilities.argmax(dim=-1))
    assert torch.allclose(confidence, probabilities.amax(dim=-1), atol=1.0e-6)
    assert torch.allclose(entropy, expected_entropy, atol=1.0e-6)


def test_sampler_preserves_nonallocated_batch_padding():
    valid = torch.tensor([[True, True, False], [True, True, True]])

    def denoiser(canvas, t):
        logits = torch.zeros(2, 3, 6)
        logits[..., 2] = 10.0
        return logits

    output = confidence_schedule_sample(
        denoiser,
        MaskNoiseProcess(5),
        (2, 3),
        device="cpu",
        config=ConfidenceSamplerConfig(max_nfe=2),
        canvas_valid_mask=valid,
        padding_token_id=1,
    )
    assert output.canvas[0].tolist() == [2, 2, 1]
    assert output.selected_mask.equal(valid)


def test_time_mlp_zero_initialization_and_gradient():
    module = TimeConditioningMLP(hidden_size=16, feature_dim=8)
    t = torch.tensor([0.1, 0.9])
    output = module(t)
    assert output.shape == (2, 16)
    assert torch.equal(output, torch.zeros_like(output))
    output.sum().backward()
    assert module.out_proj.bias.grad is not None
    assert module.out_proj.bias.grad.abs().sum() > 0

    nonzero = TimeConditioningMLP(
        hidden_size=8, feature_dim=8, zero_init_output=False
    )
    assert not torch.allclose(nonzero(torch.tensor([0.1])), nonzero(torch.tensor([0.9])))


def test_time_features_validate_shape_and_range():
    features = SinusoidalTimeFeatures(feature_dim=7)
    assert features(torch.tensor([0.0, 1.0])).shape == (2, 7)
    with pytest.raises(ValueError):
        features(torch.tensor([[0.5]]))
    with pytest.raises(ValueError):
        features(torch.tensor([1.1]))


def test_entropy_bounded_exact_budget_and_single_token_case():
    confidence = torch.tensor([[0.9, 0.8, 0.7]])
    entropy = torch.tensor([[0.5, 0.2, 0.4]])
    eligible = torch.ones_like(confidence, dtype=torch.bool)
    # Prefix [0,1]: .7 - .5 = .2; adding 2 gives 1.1 - .5 = .6.
    selected = select_entropy_bounded_positions(
        confidence, entropy, eligible, gamma=0.2
    )
    assert selected.tolist() == [[True, True, False]]

    one = select_entropy_bounded_positions(
        confidence,
        entropy,
        torch.tensor([[False, False, True]]),
        gamma=0.0,
    )
    assert one.tolist() == [[False, False, True]]


def test_position_bias_prefers_earlier_token_and_zero_matches_eb():
    confidence = torch.tensor([[0.8, 0.9]])
    entropy = torch.tensor([[0.4, 0.4]])
    eligible = torch.ones_like(confidence, dtype=torch.bool)
    unbiased = select_entropy_bounded_positions(
        confidence, entropy, eligible, gamma=0.0, position_bias_lambda=0.0
    )
    biased = select_entropy_bounded_positions(
        confidence, entropy, eligible, gamma=0.0, position_bias_lambda=1.0
    )
    assert unbiased.tolist() == [[False, True]]
    assert biased.tolist() == [[True, False]]


def test_diffusion_gemma_entropy_selection_uses_lowest_entropy_cumulative_prefix():
    entropy = torch.tensor([[0.3, 0.1, 0.2, 0.9]])
    eligible = torch.ones_like(entropy, dtype=torch.bool)
    selected = select_diffusion_gemma_entropy_positions(
        entropy,
        eligible,
        entropy_bound=0.25,
    )
    # Sorted entropies: .1, .2, .3, .9.  The cumulative-minus-current
    # sequence is 0, .1, .3, .6, so exactly positions 1 and 2 are accepted.
    assert selected.tolist() == [[False, True, True, False]]


def test_full_argmax_refinement_reinjects_every_predicted_position():
    class RecordingDenoiser:
        def __init__(self):
            self.inputs = []
            self.times = []
            self.predictions = ([1, 2], [3, 4], [5, 6])

        def __call__(self, canvas, t):
            self.inputs.append(canvas.clone())
            self.times.append(t.clone())
            winners = self.predictions[len(self.inputs) - 1]
            logits = torch.full((*canvas.shape, 8), -20.0)
            for position, winner in enumerate(winners):
                logits[:, position, winner] = 20.0
            return logits

    denoiser = RecordingDenoiser()
    output = full_argmax_refinement_sample(
        denoiser,
        UniformNoiseProcess([7]),
        (1, 2),
        device="cpu",
        config=ConfidenceSamplerConfig(
            sampler="full_argmax_refinement",
            max_nfe=3,
        ),
        generator=torch.Generator().manual_seed(11),
    )
    assert [canvas.tolist() for canvas in denoiser.inputs] == [
        [[7, 7]],
        [[1, 2]],
        [[3, 4]],
    ]
    assert [float(t.item()) for t in denoiser.times] == pytest.approx(
        [1.0, 2.0 / 3.0, 1.0 / 3.0]
    )
    assert output.canvas.tolist() == [[5, 6]]
    assert output.selected_mask.all()


def test_diffusion_gemma_entropy_renoises_rejected_positions_and_returns_final_argmax():
    class RecordingDenoiser:
        def __init__(self):
            self.inputs = []

        def __call__(self, canvas, t):
            self.inputs.append(canvas.clone())
            logits = torch.zeros(*canvas.shape, 8)
            if len(self.inputs) == 1:
                logits[:, 0, 1] = 14.0
                logits[:, 1, 2] = 1.0
            else:
                logits[:, 0, 3] = 14.0
                logits[:, 1, 4] = 14.0
            return logits

    denoiser = RecordingDenoiser()
    output = diffusion_gemma_entropy_sample(
        denoiser,
        UniformNoiseProcess([7]),
        (1, 2),
        device="cpu",
        config=ConfidenceSamplerConfig(
            sampler="diffusion_gemma_entropy",
            max_nfe=2,
            prediction_mode="categorical",
            temperature_start=0.8,
            temperature_end=0.4,
            entropy_budget_gamma=1.0e-6,
        ),
        generator=torch.Generator().manual_seed(3),
        return_history=True,
    )
    assert output.history[0].selected_mask.tolist() == [[True, False]]
    assert denoiser.inputs[1].tolist() == [[1, 7]]
    # DiffusionGemma exposes the last denoiser argmax, not the categorical
    # acceptance canvas, as the completed block.
    assert output.canvas.tolist() == [[3, 4]]
    assert output.selected_mask.all()


def test_entropy_sampler_last_nfe_commits_every_position():
    denoiser = ScriptedDenoiser(
        [
            ([1, 2, 3], [3.0, 2.0, 1.0]),
            ([1, 2, 3], [3.0, 2.0, 1.0]),
        ]
    )
    output = confidence_schedule_sample(
        denoiser,
        MaskNoiseProcess(mask_token_id=7),
        (1, 3),
        device="cpu",
        config=ConfidenceSamplerConfig(
            sampler="entropy_bounded",
            max_nfe=2,
            temperature_start=1.0,
            temperature_end=1.0,
            entropy_budget_gamma=0.0,
        ),
        return_history=True,
    )
    assert output.history[0].selected_mask.sum() == 1
    assert output.history[-1].selected_mask.all()


def test_entropy_sampler_rejects_nonabsorbing_uniform_process():
    with pytest.raises(ValueError, match="Mask noise"):
        confidence_schedule_sample(
            ScriptedDenoiser([([1, 2], [2.0, 1.0])]),
            UniformNoiseProcess([0]),
            (1, 2),
            device="cpu",
            config=ConfidenceSamplerConfig(sampler="entropy_bounded", max_nfe=1),
        )


def test_removed_confidence_topk_alias_fails_explicitly():
    with pytest.raises(ValueError, match="unsupported sampler"):
        ConfidenceSamplerConfig(sampler="confidence_topk")


def _brute_force_udm_marginalization(denoiser_probs, current_tokens, alpha_t, alpha_s):
    probabilities = denoiser_probs.double()
    batch_size, canvas_size, support_size = probabilities.shape
    output = torch.zeros_like(probabilities)
    ratio = alpha_t / alpha_s
    for batch in range(batch_size):
        for position in range(canvas_size):
            observed = int(current_tokens[batch, position])
            for clean in range(support_size):
                denominator = (
                    alpha_t * float(observed == clean)
                    + (1.0 - alpha_t) / support_size
                )
                for candidate in range(support_size):
                    from_clean = (
                        alpha_s * float(candidate == clean)
                        + (1.0 - alpha_s) / support_size
                    )
                    to_observed = (
                        ratio * float(observed == candidate)
                        + (1.0 - ratio) / support_size
                    )
                    output[batch, position, candidate] += (
                        probabilities[batch, position, clean]
                        * from_clean
                        * to_observed
                        / denominator
                    )
    return output


def _paper_sampler_config(name, nfe):
    return ConfidenceSamplerConfig(
        sampler=name,
        max_nfe=nfe,
        prediction_mode="categorical",
        temperature_start=1.0,
        temperature_end=1.0,
    )


def _argmax_udm_config(nfe):
    return ConfidenceSamplerConfig(
        sampler="argmax_udm",
        max_nfe=nfe,
        prediction_mode="argmax",
        temperature_start=1.0,
        temperature_end=1.0,
    )


def _tempered_udm_config(nfe, temperature_start, temperature_end=None):
    return ConfidenceSamplerConfig(
        sampler="udm_tempered_marginalization",
        max_nfe=nfe,
        prediction_mode="categorical",
        temperature_start=temperature_start,
        temperature_end=(
            temperature_start if temperature_end is None else temperature_end
        ),
    )


def test_udm_marginalization_matches_explicit_sum_over_clean_hypotheses():
    denoiser_probs = torch.tensor(
        [[[0.60, 0.30, 0.10], [0.05, 0.15, 0.80]]], dtype=torch.float64
    )
    current = torch.tensor([[0, 2]])
    expected = _brute_force_udm_marginalization(
        denoiser_probs, current, alpha_t=0.25, alpha_s=0.65
    )
    actual = udm_marginalized_reverse_probs(
        denoiser_probs, current, alpha_t=0.25, alpha_s=0.65
    )
    assert torch.allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(actual.sum(dim=-1), torch.ones(1, 2, dtype=torch.float64))


def test_argmax_udm_is_exactly_onehot_clean_marginalization():
    denoiser_probs = torch.tensor(
        [[[0.60, 0.30, 0.10], [0.05, 0.15, 0.80]]], dtype=torch.float64
    )
    current = torch.tensor([[1, 2]])  # first differs from argmax; second agrees
    actual = udm_argmax_reverse_probs(
        denoiser_probs,
        current,
        alpha_t=0.25,
        alpha_s=0.65,
    )
    one_hot = torch.zeros_like(denoiser_probs)
    one_hot.scatter_(-1, denoiser_probs.argmax(dim=-1, keepdim=True), 1.0)
    expected = udm_marginalized_reverse_probs(
        one_hot,
        current,
        alpha_t=0.25,
        alpha_s=0.65,
    )
    brute_force = _brute_force_udm_marginalization(
        one_hot,
        current,
        alpha_t=0.25,
        alpha_s=0.65,
    )
    assert torch.allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(actual, brute_force, atol=1.0e-12, rtol=1.0e-12)


def test_argmax_udm_bridge_endpoints():
    denoiser_probs = torch.tensor([[[0.1, 0.7, 0.2]]], dtype=torch.float64)
    current = torch.tensor([[2]])
    initial = udm_argmax_reverse_probs(
        denoiser_probs, current, alpha_t=0.0, alpha_s=0.4
    )
    assert torch.allclose(
        initial,
        torch.tensor([[[0.2, 0.6, 0.2]]], dtype=torch.float64),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    final = udm_argmax_reverse_probs(
        denoiser_probs, current, alpha_t=0.6, alpha_s=1.0
    )
    assert torch.equal(final, torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float64))


def test_argmax_udm_transition_empirically_matches_onehot_bridge():
    repetitions = 30_000
    support = [1, 4, 7]
    local_probabilities = torch.tensor([0.60, 0.30, 0.10])
    logits = torch.full((1, repetitions, 8), -100.0)
    logits[..., support] = local_probabilities.log()
    canvas = torch.full((1, repetitions), 4, dtype=torch.long)
    valid = torch.ones_like(canvas, dtype=torch.bool)
    sampled = _sample_argmax_udm_transition(
        logits,
        canvas,
        valid,
        UniformNoiseProcess(support),
        alpha_t=0.25,
        alpha_s=0.65,
        forbidden_output_token_ids=(),
        generator=torch.Generator().manual_seed(2026),
    )
    expected = udm_argmax_reverse_probs(
        local_probabilities.view(1, 1, -1),
        torch.tensor([[1]]),
        alpha_t=0.25,
        alpha_s=0.65,
    )[0, 0]
    empirical = torch.tensor(
        [(sampled == token_id).float().mean() for token_id in support]
    )
    assert torch.allclose(empirical, expected.float(), atol=0.012, rtol=0.0)


def test_argmax_udm_optimized_bridge_handles_overlap_and_identity_transition():
    repetitions = 20_000
    support = [1, 4, 7]
    logits = torch.full((1, repetitions, 8), -100.0)
    logits[..., 4] = 10.0  # clean argmax agrees with the current token
    canvas = torch.full((1, repetitions), 4, dtype=torch.long)
    valid = torch.ones_like(canvas, dtype=torch.bool)
    sampled = _sample_argmax_udm_transition(
        logits,
        canvas,
        valid,
        UniformNoiseProcess(support),
        alpha_t=0.25,
        alpha_s=0.65,
        forbidden_output_token_ids=(),
        generator=torch.Generator().manual_seed(99),
    )
    expected = udm_argmax_reverse_probs(
        torch.tensor([[[0.1, 0.8, 0.1]]]),
        torch.tensor([[1]]),
        alpha_t=0.25,
        alpha_s=0.65,
    )[0, 0]
    empirical = torch.tensor(
        [(sampled == token_id).float().mean() for token_id in support]
    )
    assert torch.allclose(empirical, expected.float(), atol=0.012, rtol=0.0)

    # alpha_s == alpha_t means no reverse-time progress: X_s must equal X_t,
    # even when the denoiser's clean argmax disagrees with the current token.
    logits[..., 1] = 20.0
    identity = _sample_argmax_udm_transition(
        logits,
        canvas,
        valid,
        UniformNoiseProcess(support),
        alpha_t=0.65,
        alpha_s=0.65,
        forbidden_output_token_ids=(),
        generator=torch.Generator().manual_seed(100),
    )
    assert torch.equal(identity, canvas)


def test_argmax_udm_nfe1_matches_confidence_and_final_argmax():
    valid = torch.tensor([[True, True, False]])

    def denoiser(canvas, t):
        logits = torch.zeros(*canvas.shape, 6)
        logits[..., 5] = 100.0  # forbidden winner
        logits[:, 0, 2] = 90.0
        logits[:, 1, 3] = 90.0
        return logits

    common = {
        "denoiser": denoiser,
        "noise_process": UniformNoiseProcess(range(6)),
        "shape": (1, 3),
        "device": "cpu",
        "canvas_valid_mask": valid,
        "padding_token_id": 1,
    }
    confidence = sample_diffusion(
        **common,
        config=ConfidenceSamplerConfig(
            sampler="confidence_schedule",
            max_nfe=1,
            prediction_mode="argmax",
            temperature_start=0.8,
            temperature_end=0.4,
            forbidden_output_token_ids=(5,),
        ),
        generator=torch.Generator().manual_seed(3421),
    )
    final_only = sample_diffusion(
        **common,
        config=ConfidenceSamplerConfig(
            sampler="udm_marginalization_final_argmax",
            max_nfe=1,
            prediction_mode="categorical",
            temperature_start=1.0,
            temperature_end=1.0,
            forbidden_output_token_ids=(5,),
        ),
        generator=torch.Generator().manual_seed(3421),
    )
    plug_in = sample_diffusion(
        **common,
        config=ConfidenceSamplerConfig(
            sampler="argmax_udm",
            max_nfe=1,
            prediction_mode="argmax",
            temperature_start=1.0,
            temperature_end=1.0,
            forbidden_output_token_ids=(5,),
        ),
        generator=torch.Generator().manual_seed(3421),
    )
    assert torch.equal(plug_in.canvas, confidence.canvas)
    assert torch.equal(plug_in.canvas, final_only.canvas)
    assert plug_in.canvas.tolist() == [[2, 3, 1]]


def test_argmax_udm_recomputes_clean_argmax_and_preserves_safe_support_padding():
    class RecordingDenoiser:
        def __init__(self):
            self.inputs = []

        def __call__(self, canvas, t):
            self.inputs.append(canvas.clone())
            logits = torch.full((*canvas.shape, 8), -100.0)
            # Token 2 is outside support and must never be the clean hypothesis.
            logits[..., 2] = 100.0
            if len(self.inputs) == 1:
                logits[..., 4] = 90.0
            else:
                logits[..., 7] = 90.0
            return logits

    denoiser = RecordingDenoiser()
    valid = torch.tensor([[True, True, False]])
    output = argmax_udm_sample(
        denoiser,
        UniformNoiseProcess([1, 4, 7]),
        (1, 3),
        device="cpu",
        config=_argmax_udm_config(2),
        generator=torch.Generator().manual_seed(17),
        canvas_valid_mask=valid,
        padding_token_id=1,
        return_history=True,
    )
    assert len(denoiser.inputs) == output.nfe == 2
    assert output.canvas.tolist() == [[7, 7, 1]]
    assert all(set(step.canvas[0, :2].tolist()) <= {1, 4, 7} for step in output.history)
    assert all(step.canvas[0, 2].item() == 1 for step in output.history)


def test_udm_marginalization_endpoints_are_denoiser_identity_and_initial_mixture():
    denoiser_probs = torch.tensor([[[0.7, 0.2, 0.1]]], dtype=torch.float64)
    current = torch.tensor([[1]])

    final = udm_marginalized_reverse_probs(
        denoiser_probs, current, alpha_t=0.4, alpha_s=1.0
    )
    assert torch.allclose(final, denoiser_probs, atol=1.0e-12)

    identity = udm_marginalized_reverse_probs(
        denoiser_probs, current, alpha_t=0.4, alpha_s=0.4
    )
    assert torch.allclose(identity, torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float64))

    initial = udm_marginalized_reverse_probs(
        denoiser_probs, current, alpha_t=0.0, alpha_s=0.25
    )
    expected = 0.25 * denoiser_probs + 0.75 / 3
    assert torch.allclose(initial, expected, atol=1.0e-12)


def test_udm_sampler_uses_noncontiguous_safe_support_and_finishes_from_denoiser():
    def denoiser(canvas, t):
        logits = torch.full((*canvas.shape, 8), -100.0)
        logits[..., 4] = 100.0
        return logits

    output = udm_marginalization_sample(
        denoiser,
        UniformNoiseProcess([1, 4, 7]),
        (1, 5),
        device="cpu",
        config=_paper_sampler_config("udm_marginalization", 1),
        generator=torch.Generator().manual_seed(31),
        return_history=True,
    )
    assert output.canvas.tolist() == [[4, 4, 4, 4, 4]]
    assert output.nfe == 1
    assert output.selected_mask.all()
    assert output.history[0].noise_level == 1.0


def test_chunked_udm_transition_empirically_matches_literal_marginalization():
    repetitions = 20_000
    support = [1, 4, 7]
    local_probabilities = torch.tensor([0.60, 0.30, 0.10])
    logits = torch.full((1, repetitions, 8), -100.0)
    logits[..., support] = local_probabilities.log()
    canvas = torch.full((1, repetitions), 4, dtype=torch.long)
    valid = torch.ones_like(canvas, dtype=torch.bool)
    sampled = _sample_udm_marginalized_transition(
        logits,
        canvas,
        valid,
        UniformNoiseProcess(support),
        alpha_t=0.25,
        alpha_s=0.65,
        forbidden_output_token_ids=(),
        generator=torch.Generator().manual_seed(4321),
        vocab_chunk_size=2,
    )
    expected = udm_marginalized_reverse_probs(
        local_probabilities.view(1, 1, -1),
        torch.tensor([[1]]),
        alpha_t=0.25,
        alpha_s=0.65,
    )[0, 0]
    empirical = torch.tensor(
        [(sampled == token_id).float().mean() for token_id in support]
    )
    assert torch.allclose(empirical, expected.float(), atol=0.012, rtol=0.0)


def test_tempered_udm_transition_tempers_clean_distribution_before_loo():
    repetitions = 30_000
    support = [1, 4, 7]
    local_probabilities = torch.tensor([0.60, 0.30, 0.10])
    clean_temperature = 0.5
    logits = torch.full((1, repetitions, 8), -100.0)
    logits[..., support] = local_probabilities.log()
    canvas = torch.full((1, repetitions), 4, dtype=torch.long)
    valid = torch.ones_like(canvas, dtype=torch.bool)
    sampled = _sample_udm_marginalized_transition(
        logits,
        canvas,
        valid,
        UniformNoiseProcess(support),
        alpha_t=0.25,
        alpha_s=0.65,
        forbidden_output_token_ids=(),
        generator=torch.Generator().manual_seed(932),
        clean_temperature=clean_temperature,
        vocab_chunk_size=2,
    )
    tempered_probabilities = torch.softmax(
        local_probabilities.log() / clean_temperature, dim=-1
    )
    expected = udm_marginalized_reverse_probs(
        tempered_probabilities.view(1, 1, -1),
        torch.tensor([[1]]),
        alpha_t=0.25,
        alpha_s=0.65,
    )[0, 0]
    empirical = torch.tensor(
        [(sampled == token_id).float().mean() for token_id in support]
    )
    assert torch.allclose(empirical, expected.float(), atol=0.012, rtol=0.0)


def test_tempered_udm_temperature_one_is_canonical_bit_for_bit():
    def denoiser(canvas, t):
        logits = torch.zeros(*canvas.shape, 7)
        logits[..., 2] = 1.25 + t[:, None]
        logits[..., 5] = 0.75
        return logits

    common = {
        "denoiser": denoiser,
        "noise_process": UniformNoiseProcess(range(7)),
        "shape": (2, 11),
        "device": "cpu",
        "return_history": True,
    }
    canonical = udm_marginalization_sample(
        **common,
        config=_paper_sampler_config("udm_marginalization", 4),
        generator=torch.Generator().manual_seed(8128),
    )
    tempered = udm_tempered_marginalization_sample(
        **common,
        config=_tempered_udm_config(4, 1.0),
        generator=torch.Generator().manual_seed(8128),
    )
    assert torch.equal(tempered.canvas, canonical.canvas)
    assert torch.equal(tempered.selected_mask, canonical.selected_mask)
    assert tempered.nfe == canonical.nfe
    assert len(tempered.history) == len(canonical.history)
    for tempered_step, canonical_step in zip(tempered.history, canonical.history):
        assert tempered_step.noise_level == canonical_step.noise_level
        assert torch.equal(tempered_step.canvas, canonical_step.canvas)
        assert torch.equal(
            tempered_step.newly_selected_mask,
            canonical_step.newly_selected_mask,
        )


def test_tempered_udm_interpolates_clean_temperature_per_nfe(monkeypatch):
    observed_temperatures = []
    original_transition = sampler_module._sample_udm_marginalized_transition

    def recording_transition(*args, clean_temperature=1.0, **kwargs):
        observed_temperatures.append(clean_temperature)
        return original_transition(
            *args,
            clean_temperature=clean_temperature,
            **kwargs,
        )

    monkeypatch.setattr(
        sampler_module,
        "_sample_udm_marginalized_transition",
        recording_transition,
    )

    def denoiser(canvas, t):
        return torch.zeros(*canvas.shape, 5)

    sample_diffusion(
        denoiser,
        UniformNoiseProcess(range(5)),
        (1, 4),
        device="cpu",
        config=_tempered_udm_config(3, 0.9, 0.3),
        generator=torch.Generator().manual_seed(41),
    )
    assert observed_temperatures == pytest.approx([0.9, 0.6, 0.3])


def test_udm_full_vocab_forbidden_token_can_be_noise_but_not_clean_output():
    def denoiser(canvas, t):
        logits = torch.zeros(*canvas.shape, 5)
        logits[..., 4] = 100.0
        logits[..., 2] = 90.0
        return logits

    output = udm_marginalization_sample(
        denoiser,
        UniformNoiseProcess(range(5)),
        (1, 64),
        device="cpu",
        config=ConfidenceSamplerConfig(
            sampler="udm_marginalization",
            max_nfe=1,
            prediction_mode="categorical",
            temperature_start=1.0,
            temperature_end=1.0,
            forbidden_output_token_ids=(4,),
        ),
        generator=torch.Generator().manual_seed(9),
    )
    assert (output.canvas == 2).all()


def test_udm_final_argmax_nfe1_matches_confidence_bit_for_bit():
    valid = torch.tensor([[True, True, False]])

    def denoiser(canvas, t):
        logits = torch.zeros(*canvas.shape, 5)
        logits[..., 4] = 100.0  # forbidden winner
        logits[:, 0, 2] = 90.0
        logits[:, 1, 3] = 90.0
        return logits

    common = {
        "denoiser": denoiser,
        "noise_process": UniformNoiseProcess(range(5)),
        "shape": (1, 3),
        "device": "cpu",
        "canvas_valid_mask": valid,
        "padding_token_id": 1,
    }
    confidence = sample_diffusion(
        **common,
        config=ConfidenceSamplerConfig(
            sampler="confidence_schedule",
            max_nfe=1,
            prediction_mode="argmax",
            temperature_start=0.8,
            temperature_end=0.4,
            forbidden_output_token_ids=(4,),
        ),
        generator=torch.Generator().manual_seed(3421),
    )
    hybrid = sample_diffusion(
        **common,
        config=ConfidenceSamplerConfig(
            sampler="udm_marginalization_final_argmax",
            max_nfe=1,
            prediction_mode="categorical",
            temperature_start=1.0,
            temperature_end=1.0,
            forbidden_output_token_ids=(4,),
        ),
        generator=torch.Generator().manual_seed(3421),
    )
    assert torch.equal(hybrid.canvas, confidence.canvas)
    assert hybrid.canvas.tolist() == [[2, 3, 1]]


def test_udm_final_argmax_respects_noncontiguous_safe_support_and_padding():
    valid = torch.tensor([[True, True, False]])

    def denoiser(canvas, t):
        logits = torch.full((*canvas.shape, 8), -100.0)
        logits[..., 2] = 100.0  # globally best, but outside safe support
        logits[..., 4] = 90.0   # best token inside support [1, 4, 7]
        return logits

    output = udm_marginalization_sample(
        denoiser,
        UniformNoiseProcess([1, 4, 7]),
        (1, 3),
        device="cpu",
        config=_paper_sampler_config(
            "udm_marginalization_final_argmax", 1
        ),
        generator=torch.Generator().manual_seed(17),
        canvas_valid_mask=valid,
        padding_token_id=1,
    )
    assert output.canvas.tolist() == [[4, 4, 1]]


def test_udm_final_argmax_preserves_all_intermediate_udm_transitions():
    class RecordingDenoiser:
        def __init__(self):
            self.inputs = []

        def __call__(self, canvas, t):
            self.inputs.append(canvas.clone())
            logits = torch.zeros(*canvas.shape, 3)
            logits[..., 0] = 0.4
            logits[..., 1] = 0.2
            return logits

    exact_denoiser = RecordingDenoiser()
    hybrid_denoiser = RecordingDenoiser()
    exact = udm_marginalization_sample(
        exact_denoiser,
        UniformNoiseProcess(range(3)),
        (1, 12),
        device="cpu",
        config=_paper_sampler_config("udm_marginalization", 3),
        generator=torch.Generator().manual_seed(91),
        return_history=True,
    )
    hybrid = udm_marginalization_sample(
        hybrid_denoiser,
        UniformNoiseProcess(range(3)),
        (1, 12),
        device="cpu",
        config=_paper_sampler_config(
            "udm_marginalization_final_argmax", 3
        ),
        generator=torch.Generator().manual_seed(91),
        return_history=True,
    )
    assert len(exact_denoiser.inputs) == len(hybrid_denoiser.inputs) == 3
    assert all(
        torch.equal(exact_input, hybrid_input)
        for exact_input, hybrid_input in zip(
            exact_denoiser.inputs, hybrid_denoiser.inputs
        )
    )
    assert torch.equal(exact.history[0].canvas, hybrid.history[0].canvas)
    assert torch.equal(exact.history[1].canvas, hybrid.history[1].canvas)
    assert (hybrid.canvas == 0).all()
    assert hybrid.nfe == exact.nfe == 3


def test_mdlm_ancestral_uses_random_bernoulli_positions_and_strict_carry_over():
    class ChangingDenoiser:
        def __init__(self):
            self.calls = 0

        def __call__(self, canvas, t):
            logits = torch.full((*canvas.shape, 8), -100.0)
            logits[..., 1 + self.calls] = 100.0
            self.calls += 1
            return logits

    denoiser = ChangingDenoiser()
    output = mdlm_ancestral_sample(
        denoiser,
        MaskNoiseProcess(mask_token_id=7),
        (1, 600),
        device="cpu",
        config=_paper_sampler_config("mdlm_ancestral", 3),
        generator=torch.Generator().manual_seed(123),
        return_history=True,
    )
    first_visible = output.history[0].selected_mask
    # rho=1/3 on the first transition; this broad deterministic bound catches
    # accidental confidence top-k (which would reveal exactly 200 positions).
    assert 160 < int(first_visible.sum()) < 240
    assert int(first_visible.sum()) != 200
    assert torch.equal(
        output.history[0].canvas[first_visible],
        output.history[-1].canvas[first_visible],
    )
    assert (output.history[0].canvas[first_visible] == 1).all()
    assert output.selected_mask.all()
    assert not (output.canvas == 7).any()
    assert output.nfe == 3


def test_mdlm_ancestral_preserves_padding_and_last_transition_unmasks_all():
    valid = torch.tensor([[True, True, False], [True, True, True]])

    def denoiser(canvas, t):
        logits = torch.full((*canvas.shape, 7), -100.0)
        logits[..., 3] = 100.0
        return logits

    output = sample_diffusion(
        denoiser,
        MaskNoiseProcess(mask_token_id=6),
        (2, 3),
        device="cpu",
        config=_paper_sampler_config("mdlm_ancestral", 1),
        generator=torch.Generator().manual_seed(4),
        canvas_valid_mask=valid,
        padding_token_id=1,
    )
    assert output.canvas.tolist() == [[3, 3, 1], [3, 3, 3]]
    assert output.selected_mask.equal(valid)


def test_mdlm_ancestral_stops_exactly_when_every_mask_is_gone():
    class CountingDenoiser:
        def __init__(self):
            self.calls = 0

        def __call__(self, canvas, t):
            self.calls += 1
            logits = torch.full((*canvas.shape, 7), -100.0)
            logits[..., 3] = 100.0
            return logits

    denoiser = CountingDenoiser()
    output = mdlm_ancestral_sample(
        denoiser,
        MaskNoiseProcess(mask_token_id=6),
        (1, 1),
        device="cpu",
        config=_paper_sampler_config("mdlm_ancestral", 2),
        # The first Bernoulli draw is below rho=1/2, so the only MASK is
        # removed during transition 1 -> 0.5.
        generator=torch.Generator().manual_seed(3),
        return_history=True,
    )
    assert denoiser.calls == 1
    assert output.nfe == 1
    assert len(output.history) == 1
    assert output.canvas.tolist() == [[3]]
    assert output.selected_mask.all()


def test_paper_sampler_configuration_and_noise_process_mismatches_fail_loudly():
    with pytest.raises(ValueError, match="prediction_mode='argmax'"):
        ConfidenceSamplerConfig(
            sampler="full_argmax_refinement",
            prediction_mode="categorical",
        )
    with pytest.raises(ValueError, match="prediction_mode='categorical'"):
        ConfidenceSamplerConfig(sampler="diffusion_gemma_entropy")
    with pytest.raises(ValueError, match="prediction_mode='categorical'"):
        ConfidenceSamplerConfig(sampler="mdlm_ancestral")
    with pytest.raises(ValueError, match="temperature_start=temperature_end=1.0"):
        ConfidenceSamplerConfig(
            sampler="udm_marginalization",
            prediction_mode="categorical",
        )
    with pytest.raises(ValueError, match="temperature_start=temperature_end=1.0"):
        ConfidenceSamplerConfig(
            sampler="udm_marginalization_final_argmax",
            prediction_mode="categorical",
        )
    tempered = ConfidenceSamplerConfig(
        sampler="udm_tempered_marginalization",
        prediction_mode="categorical",
        temperature_start=0.9,
        temperature_end=0.2,
    )
    assert tempered.temperature_start == 0.9
    assert tempered.temperature_end == 0.2
    with pytest.raises(ValueError, match="prediction_mode='categorical'"):
        ConfidenceSamplerConfig(
            sampler="udm_tempered_marginalization",
            prediction_mode="argmax",
            temperature_start=0.9,
            temperature_end=0.2,
        )
    with pytest.raises(ValueError, match="finite and positive|must be > 0"):
        ConfidenceSamplerConfig(
            sampler="udm_tempered_marginalization",
            prediction_mode="categorical",
            temperature_start=float("nan"),
            temperature_end=0.2,
        )
    with pytest.raises(ValueError, match="prediction_mode='argmax'"):
        ConfidenceSamplerConfig(
            sampler="argmax_udm",
            prediction_mode="categorical",
            temperature_start=1.0,
            temperature_end=1.0,
        )
    with pytest.raises(ValueError, match="temperature_start=temperature_end=1.0"):
        ConfidenceSamplerConfig(
            sampler="argmax_udm",
            prediction_mode="argmax",
        )

    def denoiser(canvas, t):
        return torch.zeros(*canvas.shape, 8)

    with pytest.raises(ValueError, match="noise_type='uniform'"):
        sample_diffusion(
            denoiser,
            MaskNoiseProcess(7),
            (1, 2),
            device="cpu",
            config=_paper_sampler_config("udm_marginalization", 1),
        )
    with pytest.raises(ValueError, match="noise_type='uniform'"):
        sample_diffusion(
            denoiser,
            MaskNoiseProcess(7),
            (1, 2),
            device="cpu",
            config=_paper_sampler_config(
                "udm_marginalization_final_argmax", 1
            ),
        )
    with pytest.raises(ValueError, match="noise_type='uniform'"):
        sample_diffusion(
            denoiser,
            MaskNoiseProcess(7),
            (1, 2),
            device="cpu",
            config=_tempered_udm_config(1, 0.5),
        )
    with pytest.raises(ValueError, match="noise_type='uniform'"):
        sample_diffusion(
            denoiser,
            MaskNoiseProcess(7),
            (1, 2),
            device="cpu",
            config=_argmax_udm_config(1),
        )
    with pytest.raises(ValueError, match="noise_type='mask'"):
        sample_diffusion(
            denoiser,
            UniformNoiseProcess(range(8)),
            (1, 2),
            device="cpu",
            config=_paper_sampler_config("mdlm_ancestral", 1),
        )
