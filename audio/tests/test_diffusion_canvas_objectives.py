import pytest
import torch
import torch.nn.functional as F

from audio.diffusion.canvas import (
    CanvasOverflowError,
    DurationBucketSelector,
    build_clean_canvas,
    pad_canvas_batch,
    truncate_at_first_turn_end,
)
from audio.diffusion.config import DiffusionConfig, resolve_objective
from audio.diffusion.objectives import (
    build_position_weights,
    classify_canvas_positions,
    mdm_absorbing,
    x0_ce_all,
)


def test_duration_bucket_selection_never_reads_target_length_and_reports_overflow():
    selector = DurationBucketSelector(
        [4, 8, 16], tokens_per_second=2.0, margin_tokens=1.0
    )
    choice = selector.select(2.1)  # ceil(2*2.1 + 1) = 6 -> bucket 8
    assert choice.estimated_tokens == 6
    assert choice.canvas_size == 8
    assert not choice.overflow
    overflow = selector.select(20.0)
    assert overflow.overflow and overflow.canvas_size is None
    with pytest.raises(CanvasOverflowError):
        selector.select_or_raise(20.0)


def test_canvas_tail_modes_use_turn_end_250019_not_hf_eos_pad_1():
    repeated = build_clean_canvas([10, 11], 5, tail_fill_mode="eos_repeat")
    assert repeated.input_ids.tolist() == [10, 11, 250019, 250019, 250019]
    assert repeated.first_turn_end_mask.tolist() == [False, False, True, False, False]
    assert repeated.tail_turn_end_mask.tolist() == [False, False, False, True, True]
    assert not repeated.pad_mask.any()

    padded = build_clean_canvas([10, 11], 5, tail_fill_mode="eos_then_pad")
    assert padded.input_ids.tolist() == [10, 11, 250019, 1, 1]
    assert padded.first_turn_end_mask.sum() == 1
    assert not padded.tail_turn_end_mask.any()
    assert padded.pad_mask.tolist() == [False, False, False, True, True]
    assert truncate_at_first_turn_end(repeated.input_ids) == [10, 11]


def test_canvas_overflow_is_never_silently_truncated():
    with pytest.raises(CanvasOverflowError, match="not truncated"):
        build_clean_canvas([1, 2, 3], 3)


def test_mixed_bucket_padding_is_not_semantic_pad():
    short = build_clean_canvas([10], 4, tail_fill_mode="eos_then_pad")
    long = build_clean_canvas([11, 12, 13], 6, tail_fill_mode="eos_repeat")
    ids, valid = pad_canvas_batch([short, long])
    assert ids.shape == valid.shape == (2, 6)
    assert valid[0].tolist() == [True, True, True, True, False, False]
    assert valid[1].all()


def test_objective_auto_resolution_and_ablation_validation():
    uniform = resolve_objective("uniform")
    assert (uniform.name, uniform.positions, uniform.weighting) == (
        "x0_ce_all",
        "all",
        "none",
    )
    mask = DiffusionConfig(noise_type="mask").resolved_objective
    assert (mask.name, mask.positions, mask.weighting) == (
        "mdm_absorbing",
        "corrupted_only",
        "inverse_t",
    )
    # Common CE is allowed as an explicit mask-vs-uniform ablation.
    assert resolve_objective("mask", "x0_ce_all").name == "x0_ce_all"
    with pytest.raises(ValueError):
        resolve_objective("uniform", "mdm_absorbing")
    with pytest.raises(ValueError):
        resolve_objective("mask", "mdm_absorbing", weighting="none")
    with pytest.raises(ValueError, match="objective"):
        resolve_objective("mask", "not_an_objective")
    with pytest.raises(ValueError, match="loss_positions"):
        resolve_objective("mask", positions="not_positions")
    with pytest.raises(ValueError, match="loss_weighting"):
        resolve_objective("mask", weighting="not_weighting")


def test_position_masks_distinguish_first_turn_end_tail_and_pad():
    repeated = torch.tensor([[2, 3, 4, 4, 4]])
    masks = classify_canvas_positions(repeated, turn_end_token_id=4, pad_token_id=5)
    assert masks.lexical.tolist() == [[True, True, False, False, False]]
    assert masks.first_turn_end.tolist() == [[False, False, True, False, False]]
    assert masks.tail_turn_end.tolist() == [[False, False, False, True, True]]

    padded = torch.tensor([[2, 3, 4, 5, 5]])
    weights, masks = build_position_weights(
        padded,
        tail_fill_mode="eos_then_pad",
        pad_loss_weight=0.1,
        turn_end_token_id=4,
        pad_token_id=5,
    )
    assert torch.allclose(weights, torch.tensor([[1.0, 1.0, 1.0, 0.1, 0.1]]))
    assert masks.first_turn_end.sum() == 1 and masks.pad.sum() == 2


def test_x0_ce_all_uses_same_position_without_causal_shift():
    clean = torch.tensor([[2, 3, 4, 4]])
    logits = torch.full((1, 4, 6), -20.0)
    logits.scatter_(2, clean[..., None], 20.0)
    result = x0_ce_all(
        logits,
        clean,
        torch.tensor([[False, True, False, True]]),
        turn_end_token_id=4,
        pad_token_id=5,
    )
    assert result.loss < 1.0e-6


def test_mdm_matches_manual_fixed_denominator_inverse_t_estimator():
    torch.manual_seed(7)
    clean = torch.tensor([[2, 3, 4, 4], [3, 2, 4, 4]])
    logits = torch.randn(2, 4, 6, dtype=torch.float64)
    corrupt = torch.tensor(
        [[True, False, False, True], [False, False, False, False]]
    )
    t = torch.tensor([0.5, 0.25], dtype=torch.float64)
    result = mdm_absorbing(
        logits,
        clean,
        corrupt,
        t,
        mask_token_id=0,
        eos_tail_loss_weight=0.5,
        turn_end_token_id=4,
        pad_token_id=5,
    )

    masked_logits = logits.clone()
    masked_logits[..., 0] = -torch.inf
    ce = F.cross_entropy(masked_logits.transpose(1, 2), clean, reduction="none")
    # weights=[1,1,1,.5], fixed denominator=3.5; second sample contributes 0.
    expected_sample0 = (ce[0, 0] + 0.5 * ce[0, 3]) * 2.0 / 3.5
    assert torch.allclose(result.per_sample_loss, torch.stack((expected_sample0, ce.new_zeros(()))))
    assert torch.allclose(result.loss, expected_sample0 / 2)
    assert result.effective_mask.equal(corrupt)

    lower_t = mdm_absorbing(
        logits[:1],
        clean[:1],
        corrupt[:1],
        torch.tensor([0.25], dtype=torch.float64),
        mask_token_id=0,
        eos_tail_loss_weight=0.5,
        turn_end_token_id=4,
        pad_token_id=5,
    )
    assert torch.allclose(lower_t.loss, result.per_sample_loss[0] * 2)


def test_mdm_ignores_uncorrupted_logits_and_handles_empty_corruption():
    clean = torch.tensor([[1, 2, 3]])
    corrupt = torch.tensor([[True, False, False]])
    logits = torch.zeros(1, 3, 5)
    baseline = mdm_absorbing(logits, clean, corrupt, torch.tensor([0.5]), mask_token_id=4)
    changed = logits.clone()
    changed[:, 1:, :] = torch.tensor([100.0, -100.0, -100.0, -100.0, -100.0])
    modified = mdm_absorbing(changed, clean, corrupt, torch.tensor([0.5]), mask_token_id=4)
    assert torch.allclose(baseline.loss, modified.loss)

    empty = mdm_absorbing(
        logits,
        clean,
        torch.zeros_like(corrupt),
        torch.tensor([0.5]),
        mask_token_id=4,
    )
    assert torch.isfinite(empty.loss) and empty.loss.item() == 0.0


def test_mdm_rejects_t_zero_in_training_loss():
    with pytest.raises(ValueError, match="strictly positive"):
        mdm_absorbing(
            torch.zeros(1, 2, 4),
            torch.tensor([[1, 2]]),
            torch.tensor([[False, False]]),
            torch.tensor([0.0]),
            mask_token_id=3,
        )


def test_nonallocated_forbidden_pad_does_not_create_nan():
    clean = torch.tensor([[2, 4, 1]])
    valid = torch.tensor([[True, True, False]])
    corrupt = torch.tensor([[True, True, True]])
    logits = torch.zeros(1, 3, 6, requires_grad=True)
    result = x0_ce_all(
        logits,
        clean,
        corrupt,
        canvas_valid_mask=valid,
        forbidden_output_token_ids=(1,),
        turn_end_token_id=4,
        pad_token_id=1,
    )
    assert torch.isfinite(result.loss)
    assert result.per_token_ce[0, 2].item() == 0.0
    result.loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_objective_rejects_forbidden_allocated_target():
    logits = torch.zeros(1, 3, 10)
    targets = torch.tensor([[2, 7, 7]])
    corrupt = torch.zeros_like(targets, dtype=torch.bool)

    with pytest.raises(ValueError, match="allocated clean targets"):
        x0_ce_all(
            logits,
            targets,
            corrupt,
            forbidden_output_token_ids=(7,),
        )
