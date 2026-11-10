import pytest
import torch

from audio.diffusion.isct import build_isct_second_stage
from audio.diffusion.noise_process import MaskNoiseProcess, UniformNoiseProcess


def test_isct_uses_filtered_detached_prediction_but_keeps_true_x0_target():
    logits = torch.zeros(1, 3, 8, requires_grad=True)
    logits.data[..., 7] = 100.0  # forbidden winner
    logits.data[..., 3] = 10.0   # best valid prediction
    clean = torch.tensor([[2, 4, 5]])
    valid = torch.tensor([[True, True, False]])

    stage = build_isct_second_stage(
        logits,
        clean,
        valid,
        MaskNoiseProcess(mask_token_id=6),
        forbidden_output_token_ids=(6, 7),
        min_diffusion_time=1.0e-3,
        pad_token_id=1,
        noise_level=torch.tensor([1.0]),
    )
    assert stage.predicted_x0_ids.tolist() == [[3, 3, 1]]
    assert stage.noisy_canvas_ids.tolist() == [[6, 6, 1]]
    assert stage.corrupt_mask.tolist() == [[True, True, False]]
    assert stage.clean_target_ids is clean
    assert logits.grad is None


def test_isct_uniform_has_independent_corruption_and_token_rngs():
    logits = torch.zeros(1, 8, 10)
    logits[..., 2] = 3.0
    clean = torch.arange(2, 10).reshape(1, 8)
    valid = torch.ones_like(clean, dtype=torch.bool)
    kwargs = dict(
        first_stage_logits=logits,
        clean_target_ids=clean,
        canvas_valid_mask=valid,
        noise_process=UniformNoiseProcess([4, 5, 6]),
        forbidden_output_token_ids=(),
        min_diffusion_time=1.0e-3,
        noise_level=torch.tensor([0.5]),
    )
    first = build_isct_second_stage(
        **kwargs,
        corruption_generator=torch.Generator().manual_seed(10),
        token_noise_generator=torch.Generator().manual_seed(20),
    )
    second = build_isct_second_stage(
        **kwargs,
        corruption_generator=torch.Generator().manual_seed(10),
        token_noise_generator=torch.Generator().manual_seed(21),
    )
    assert torch.equal(first.corrupt_mask, second.corrupt_mask)
    assert not torch.equal(first.noisy_canvas_ids, second.noisy_canvas_ids)


def test_isct_rejects_gradient_through_stage_one_and_non_argmax_mode():
    args = dict(
        first_stage_logits=torch.zeros(1, 2, 4),
        clean_target_ids=torch.tensor([[1, 2]]),
        canvas_valid_mask=torch.ones(1, 2, dtype=torch.bool),
        noise_process=MaskNoiseProcess(3),
        forbidden_output_token_ids=(3,),
        min_diffusion_time=1.0e-3,
        noise_level=torch.tensor([0.5]),
    )
    with pytest.raises(ValueError, match="stop_gradient"):
        build_isct_second_stage(**args, stop_gradient=False)
    with pytest.raises(ValueError, match="argmax"):
        build_isct_second_stage(**args, prediction_mode="categorical")
