import pytest
import torch

from audio.diffusion.alignment import align_denoiser_logits, both_denoiser_alignments
from audio.diffusion.attention import (
    build_prefix_canvas_allowed_mask,
    build_prefix_canvas_attention_mask,
    build_prefix_canvas_position_ids,
)


def test_alignment_uses_real_prefix_boundary_without_bridge():
    # Semantic raw rows are [last valid prefix, canvas_0, canvas_1, canvas_2].
    raw = torch.arange(2 * 4 * 5).reshape(2, 4, 5)
    same, previous = both_denoiser_alignments(raw)
    assert torch.equal(same, raw[:, 1:, :])
    assert torch.equal(previous, raw[:, :-1, :])
    assert same.shape == previous.shape == (2, 3, 5)
    assert torch.equal(align_denoiser_logits(raw, "same_position"), same)
    assert torch.equal(align_denoiser_logits(raw, "previous_position"), previous)


def test_alignment_rejects_bad_mode_or_shape():
    with pytest.raises(ValueError):
        align_denoiser_logits(torch.zeros(2, 3), "same_position")
    with pytest.raises(ValueError):
        align_denoiser_logits(torch.zeros(1, 2, 3), "bridge_position")


def test_prefix_canvas_mask_checks_every_quadrant_and_padding():
    # Layout: prefix=[0,1], canvas=[2,3,4], padding=[5].
    allowed = build_prefix_canvas_allowed_mask(
        [2], [3], max_sequence_length=6
    )[0, 0]

    # prefix -> prefix is causal
    assert allowed[0].tolist() == [True, False, False, False, False, False]
    assert allowed[1].tolist() == [True, True, False, False, False, False]
    # prefix -> canvas is forbidden (covered by the False columns above)
    assert not allowed[:2, 2:5].any()
    # every canvas query sees the full real prefix and canvas, but not padding
    assert allowed[2:5, :5].all()
    assert not allowed[2:5, 5].any()
    # padded query gets only a self edge, preventing an all-masked softmax row
    assert allowed[5].tolist() == [False, False, False, False, False, True]

    additive = build_prefix_canvas_attention_mask(
        [2], [3], dtype=torch.float32, max_sequence_length=6
    )[0, 0]
    assert torch.equal(additive == 0, allowed)
    assert (additive[~allowed] == torch.finfo(torch.float32).min).all()


def test_mask_and_position_ids_support_variable_prefix_lengths():
    allowed = build_prefix_canvas_allowed_mask([1, 3], [2, 1])
    assert allowed.shape == (2, 1, 4, 4)
    # Example 0 has one padding row/key at position 3.
    assert allowed[0, 0, 3, 3]
    assert not allowed[0, 0, :3, 3].any()
    # Example 1 prefix is causal through position 2, canvas query 3 sees all.
    assert allowed[1, 0, 2].tolist() == [True, True, True, False]
    assert allowed[1, 0, 3].all()

    positions = build_prefix_canvas_position_ids([1, 3], [2, 1])
    assert positions.tolist() == [[0, 1, 2, 0], [0, 1, 2, 3]]


def test_attention_rejects_short_output_length():
    with pytest.raises(ValueError):
        build_prefix_canvas_allowed_mask([2], [3], max_sequence_length=4)
