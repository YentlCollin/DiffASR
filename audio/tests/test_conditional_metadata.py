from __future__ import annotations

import json

import pytest

from audio.diffusion_data.conditional_metadata import (
    VIEW_NAMES,
    prepare_conditional_row,
    select_view_id,
    stable_view_permutation,
)


RAW_CS = "مَرْحَبًا فِي [يُوتْيُوبْ:CS] [EN: YouTube] الْيَوْمَ."


class CharacterTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in text]}


def _row(transcription: str = RAW_CS, **target_overrides: object) -> dict[str, object]:
    target = {
        "transcription": transcription,
        # Deliberately differs from the QA geography in the sample key.
        "country_code": "DZ",
        "has_code_switching": True,
    }
    target.update(target_overrides)
    return {
        "sample_key": "v3__QA__video-123__1-0",
        "audio_tar_path": "data/audio/a.tar",
        "audio_member": "sample.wav",
        "duration_sec": 1.0,
        "target_json": json.dumps(target, ensure_ascii=False),
    }


def test_prepares_four_distinct_views_and_separates_content_from_request() -> None:
    prepared = prepare_conditional_row(
        _row(),
        tokenizer=CharacterTokenizer(),
        canvas_sizes=[128],
        tokens_per_second=1.0,
        margin_tokens=0.0,
        view_seed=47,
    )

    assert prepared["country_code"] == "DZ"
    assert prepared["content_has_code_switching"] is True
    assert prepared["target_plain"] == "مرحبا في يوتيوب اليوم."
    assert prepared["target_diacritics"] == "مَرْحَبًا فِي يُوتْيُوبْ الْيَوْمَ."
    assert prepared["target_code_switch"] == "مرحبا في YouTube اليوم."
    assert prepared["target_code_switch_diacritics"] == "مَرْحَبًا فِي YouTube الْيَوْمَ."
    assert prepared["valid_view_mask"] == [True, True, True, True]
    assert prepared["applied_code_switching_mask"] == [False, False, True, True]
    assert prepared["view_permutation"] == list(
        stable_view_permutation(prepared["sample_key"], 47)
    )
    assert len(prepared["target_token_counts"]) == len(VIEW_NAMES) == 4
    assert prepared["canvas_sizes"] == [128, 128, 128, 128]


def test_no_code_switch_content_keeps_request_and_application_distinct() -> None:
    prepared = prepare_conditional_row(
        _row("مَرْحَبًا بِكُمْ", has_code_switching=False),
        tokenizer=CharacterTokenizer(),
        canvas_sizes=[64],
        tokens_per_second=1.0,
        margin_tokens=0.0,
    )
    assert prepared["content_has_code_switching"] is False
    assert prepared["valid_view_mask"] == [True, True, True, True]
    assert prepared["applied_code_switching_mask"] == [False] * 4
    assert prepared["target_plain"] == prepared["target_code_switch"]
    assert prepared["target_diacritics"] == prepared["target_code_switch_diacritics"]


def test_view_selection_filters_invalid_ids_but_preserves_hash_order() -> None:
    key = "v3__EG__video__0-0"
    permutation = stable_view_permutation(key, 91)
    mask = [True, True, False, False]
    expected = [view_id for view_id in permutation if mask[view_id]]
    observed = [select_view_id(key, 91, epoch, mask) for epoch in range(6)]
    assert observed == [expected[epoch % 2] for epoch in range(6)]
    assert set(observed) == {0, 1}
    with pytest.raises(ValueError, match="all-invalid"):
        select_view_id(key, 91, 0, [False] * 4)


def test_per_view_format_failures_do_not_invalidate_clean_views() -> None:
    prepared = prepare_conditional_row(
        _row("هَذَا [يُوتْيُوبْ:CS]"),
        tokenizer=CharacterTokenizer(),
        canvas_sizes=[64],
        tokens_per_second=1.0,
        margin_tokens=0.0,
    )
    assert prepared["valid_view_mask"] == [True, True, False, False]
    assert prepared["target_plain"] == "هذا يوتيوب"
    assert prepared["target_diacritics"] == "هَذَا يُوتْيُوبْ"
    assert prepared["target_code_switch"] is None
    assert "no paired Latin target" in prepared["view_errors"][2]


def test_target_overflow_is_recorded_per_view_without_truncation() -> None:
    prepared = prepare_conditional_row(
        _row(),
        tokenizer=CharacterTokenizer(),
        canvas_sizes=[32],
        tokens_per_second=1.0,
        margin_tokens=0.0,
    )
    assert not all(prepared["valid_view_mask"])
    for valid, error in zip(
        prepared["valid_view_mask"], prepared["view_errors"], strict=True
    ):
        if not valid:
            assert error.startswith("target_overflow:")
