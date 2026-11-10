import pytest
import torch

from audio.diffusion.noise_process import (
    MaskNoiseProcess,
    UniformNoiseProcess,
    sample_diffusion_times,
)


def test_uniform_q_sample_endpoints_and_prior_support():
    process = UniformNoiseProcess([7, 8, 9])
    x0 = torch.arange(12).reshape(3, 4)

    unchanged, corrupt0 = process.q_sample(
        x0, torch.zeros(3), generator=torch.Generator().manual_seed(1)
    )
    assert torch.equal(unchanged, x0)
    assert not corrupt0.any()

    fully_noisy, corrupt1 = process.q_sample(
        x0, torch.ones(3), generator=torch.Generator().manual_seed(2)
    )
    assert corrupt1.all()
    assert set(fully_noisy.flatten().tolist()) <= {7, 8, 9}

    prior = process.initial_sample(
        (1000, 10), device="cpu", generator=torch.Generator().manual_seed(3)
    )
    counts = torch.stack([(prior == token_id).sum() for token_id in (7, 8, 9)]).float()
    assert torch.all((counts / prior.numel() - 1 / 3).abs() < 0.03)


def test_uniform_q_sample_is_generator_deterministic():
    process = UniformNoiseProcess([1, 2, 3, 4])
    x0 = torch.full((4, 20), 5)
    t = torch.tensor([0.1, 0.3, 0.7, 1.0])
    a = process.q_sample(x0, t, generator=torch.Generator().manual_seed(123))
    b = process.q_sample(x0, t, generator=torch.Generator().manual_seed(123))
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_uniform_supports_independent_corruption_and_token_noise_generators():
    process = UniformNoiseProcess([1, 2, 3, 4])
    x0 = torch.full((2, 20), 5)
    t = torch.tensor([0.4, 0.8])
    first = process.q_sample(
        x0,
        t,
        generator=torch.Generator().manual_seed(11),
        noise_generator=torch.Generator().manual_seed(21),
    )
    changed_noise = process.q_sample(
        x0,
        t,
        generator=torch.Generator().manual_seed(11),
        noise_generator=torch.Generator().manual_seed(22),
    )
    assert torch.equal(first[1], changed_noise[1])
    assert not torch.equal(first[0][first[1]], changed_noise[0][changed_noise[1]])


def test_q_sample_bernoulli_rate_tracks_each_examples_time():
    process = UniformNoiseProcess([7, 8])
    x0 = torch.zeros((2, 20_000), dtype=torch.long)
    _, corrupt = process.q_sample(
        x0,
        torch.tensor([0.2, 0.8]),
        generator=torch.Generator().manual_seed(987),
    )
    rates = corrupt.float().mean(dim=1)
    assert torch.allclose(rates, torch.tensor([0.2, 0.8]), atol=0.015, rtol=0)


def test_uniform_renoise_keeps_only_current_selection():
    process = UniformNoiseProcess([9])
    candidate = torch.tensor([[1, 2, 3, 4]])
    keep = torch.tensor([[True, False, True, False]])
    result = process.renoise(candidate, keep)
    assert result.tolist() == [[1, 9, 3, 9]]


def test_mask_q_sample_prior_and_absorbing_refill():
    process = MaskNoiseProcess(mask_token_id=42)
    x0 = torch.tensor([[1, 2, 3], [4, 5, 6]])

    unchanged, corrupt0 = process.q_sample(x0, torch.zeros(2))
    assert torch.equal(unchanged, x0)
    assert not corrupt0.any()

    masked, corrupt1 = process.q_sample(x0, torch.ones(2))
    assert corrupt1.all()
    assert torch.equal(masked, torch.full_like(x0, 42))
    assert torch.equal(process.initial_sample((2, 3), device="cpu"), masked)

    kept = process.renoise(x0, torch.tensor([[True, False, False], [False, True, False]]))
    assert kept.tolist() == [[1, 42, 42], [42, 5, 42]]


@pytest.mark.parametrize(
    "bad_t",
    [torch.tensor(0.5), torch.tensor([[0.5]]), torch.tensor([1.1]), torch.tensor([-0.1])],
)
def test_noise_rejects_invalid_time_shape_or_range(bad_t):
    with pytest.raises(ValueError):
        MaskNoiseProcess(9).q_sample(torch.ones((1, 2), dtype=torch.long), bad_t)


def test_noise_rejects_integer_time():
    with pytest.raises(TypeError):
        MaskNoiseProcess(9).q_sample(
            torch.ones((1, 2), dtype=torch.long), torch.ones(1, dtype=torch.long)
        )


def test_sample_diffusion_times_optional_full_noise_mass():
    all_full = sample_diffusion_times(
        8, device="cpu", full_noise_probability=1.0, generator=torch.Generator().manual_seed(0)
    )
    assert torch.equal(all_full, torch.ones(8))
    canonical = sample_diffusion_times(
        8, device="cpu", full_noise_probability=0.0, generator=torch.Generator().manual_seed(0)
    )
    assert ((canonical >= 0) & (canonical < 1)).all()


def test_full_noise_control_and_intervention_keep_time_rng_paired():
    control_generator = torch.Generator().manual_seed(1234)
    intervention_generator = torch.Generator().manual_seed(1234)
    control = sample_diffusion_times(
        128,
        device="cpu",
        full_noise_probability=0.0,
        generator=control_generator,
    )
    intervention = sample_diffusion_times(
        128,
        device="cpu",
        full_noise_probability=0.10,
        generator=intervention_generator,
    )

    forced = intervention.eq(1.0)
    assert forced.any()
    assert torch.equal(intervention[~forced], control[~forced])
    assert torch.equal(control_generator.get_state(), intervention_generator.get_state())
