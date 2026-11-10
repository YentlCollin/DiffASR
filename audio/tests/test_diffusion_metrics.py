import math

import torch

from audio.diffusion.metrics import denoising_token_metrics, ratios_from_additive_metrics


def test_metrics_are_additive_and_keep_t_one_separate():
    targets = torch.tensor([[2, 4, 4], [3, 4, 1]])
    valid = torch.tensor([[True, True, True], [True, True, False]])
    corrupt = torch.tensor([[True, False, True], [True, True, False]])
    logits = torch.full((2, 3, 6), -10.0)
    prediction = torch.tensor([[2, 5, 4], [0, 4, 0]])
    logits.scatter_(2, prediction[..., None], 10.0)
    metrics = denoising_token_metrics(
        logits,
        targets,
        corrupt,
        valid,
        torch.tensor([1.0, 0.9]),
        turn_end_token_id=4,
        pad_token_id=1,
    )
    assert metrics["count/all"].item() == 5
    assert metrics["correct/all"].item() == 3
    assert metrics["count/t_1.0"].item() == 3
    assert metrics["count/t_0.8_1.0"].item() == 2
    ratios = ratios_from_additive_metrics(metrics)
    assert ratios["accuracy/all"] == 3 / 5
    assert math.isnan(ratios["accuracy/pad"])


def test_corrupted_metrics_use_original_mask_not_xt_inequality():
    # Uniform noise can redraw the clean token. The metric API receives the
    # sampled corruption mask explicitly, so that position is still counted.
    targets = torch.tensor([[2]])
    logits = torch.tensor([[[0.0, 0.0, 5.0]]])
    metrics = denoising_token_metrics(
        logits,
        targets,
        torch.tensor([[True]]),
        torch.tensor([[True]]),
        torch.tensor([0.5]),
        turn_end_token_id=9,
        pad_token_id=1,
    )
    assert metrics["count/corrupted"].item() == 1
    assert metrics["correct/corrupted"].item() == 1
