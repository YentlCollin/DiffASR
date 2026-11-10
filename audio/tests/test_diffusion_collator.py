import numpy as np
import pytest
import torch

from audio.diffusion.canvas import CanvasOverflowError
from audio.diffusion.collator import (
    MawJaisDiffusionCollator,
    build_mawjais_asr_prefix,
    corrupt_diffusion_batch,
)
from audio.diffusion.conditional import CONDITIONAL_ASR_VIEWS
from audio.diffusion.config import CanvasConfig, DiffusionConfig
from audio.diffusion.noise_process import MaskNoiseProcess, UniformNoiseProcess


class _Tokenizer:
    def __call__(self, text, *, add_special_tokens=False, **kwargs):
        del add_special_tokens, kwargs
        if isinstance(text, list):
            return {"input_ids": [[2 + index for index, _ in enumerate(value.split())] for value in text]}
        return {"input_ids": [2 + index for index, _ in enumerate(text.split())]}


class _Processor:
    tokenizer = _Tokenizer()

    def __init__(self):
        self.seen_text = []

    def __call__(self, *, text, audio, return_tensors, padding, add_special_tokens):
        del return_tensors, padding, add_special_tokens
        self.seen_text.extend(text)
        assert len(text) == len(audio)
        lengths = [3 + index for index in range(len(text))]
        max_length = max(lengths)
        ids = torch.ones(len(text), max_length, dtype=torch.long)
        valid = torch.zeros_like(ids, dtype=torch.bool)
        audio_mask = torch.zeros_like(ids, dtype=torch.bool)
        for row, length in enumerate(lengths):
            ids[row, :length] = torch.arange(2, 2 + length)
            valid[row, :length] = True
            audio_mask[row, 1] = True
        return {
            "input_ids": ids,
            "attention_mask": valid,
            "audio_position_mask": audio_mask,
            "input_features": torch.zeros(len(text), 2, 3),
            "feature_attention_mask": torch.ones(len(text), 3, dtype=torch.long),
        }


def _config():
    return DiffusionConfig(
        noise_type="mask",
        mask_token_id=15,
        canvas=CanvasConfig(
            sizes=(4, 8),
            duration_tokens_per_second=1.0,
            duration_margin_tokens=1.0,
            turn_end_token_id=14,
            pad_token_id=1,
        ),
    )


def _sample(key, duration, transcription, *, code_switch=False):
    target = {
        "transcription": transcription,
        "country_code": "EG",
        "has_code_switching": code_switch,
    }
    return {
        "sample_key": key,
        "duration_sec": duration,
        "audio": np.zeros(160, dtype=np.float32),
        "transcription": transcription,
        "country_code": "EG",
        "target": target,
    }


def _conditional_sample(
    view_id,
    *,
    content_has_code_switching=True,
    target_transcription="target",
    country_code="EG",
):
    view = CONDITIONAL_ASR_VIEWS[view_id]
    applied = bool(content_has_code_switching and view.request_code_switching)
    target = {
        "transcription": target_transcription,
        "target_transcription": target_transcription,
        "country_code": country_code,
        # Deliberately the content flag, never the request flag.
        "has_code_switching": content_has_code_switching,
        "content_has_code_switching": content_has_code_switching,
        "request_code_switching": view.request_code_switching,
        "request_diacritics": view.request_diacritics,
        "applied_code_switching": applied,
    }
    return {
        "sample_key": f"conditional-{view_id}",
        "duration_sec": 2.0,
        "audio": np.zeros(160, dtype=np.float32),
        "transcription": target_transcription,
        "target_transcription": target_transcription,
        "country_code": country_code,
        "target": target,
        "view_id": view_id,
        "view_name": view.name,
        "request_code_switching": view.request_code_switching,
        "request_diacritics": view.request_diacritics,
        "content_has_code_switching": content_has_code_switching,
        "applied_code_switching": applied,
    }


def test_collator_builds_exact_prefix_duration_bucket_and_semantic_canvas():
    collator = MawJaisDiffusionCollator(_Processor(), _config())
    batch = collator(
        [
            _sample("a", 2.0, "one", code_switch=False),  # estimate 3 -> bucket 4
            _sample("b", 5.0, "two words", code_switch=True),  # estimate 6 -> bucket 8
        ]
    )
    assert batch["canvas_sizes"].tolist() == [4, 8]
    assert batch["clean_canvas_ids"].shape == (2, 8)
    assert batch["canvas_valid_mask"][0].tolist() == [True] * 4 + [False] * 4
    assert batch["clean_canvas_ids"][0, :4].tolist() == [2, 3, 14, 14]
    assert batch["prompt_tags"] == ["[Diacritics]", "[Code-switch] [Diacritics]"]
    assert batch["audio_position_mask"].dtype == torch.bool
    assert collator.counters.examples_seen == 2
    assert collator.counters.examples_emitted == 2
    assert batch["collator_metrics"] == {
        "examples_seen": 2,
        "examples_emitted": 2,
        "duration_bucket_overflow": 0,
        "target_bucket_overflow": 0,
        "batches_fully_skipped": 0,
    }


def test_target_overflow_is_reported_not_promoted_from_reference_length():
    collator = MawJaisDiffusionCollator(_Processor(), _config(), overflow_policy="skip")
    # Both have the same duration-selected bucket 4. The long target is skipped,
    # never promoted to bucket 8 by inspecting its reference length.
    batch = collator(
        [
            _sample("too-long", 1.0, "one two three four five"),
            _sample("fits", 1.0, "one"),
        ]
    )
    assert batch["sample_keys"] == ["fits"]
    assert batch["canvas_sizes"].tolist() == [4]
    assert collator.counters.target_bucket_overflow == 1
    assert batch["collator_metrics"]["target_bucket_overflow"] == 1
    assert batch["collator_metrics"]["examples_emitted"] == 1

    raising = MawJaisDiffusionCollator(_Processor(), _config(), overflow_policy="raise")
    with pytest.raises(CanvasOverflowError, match="not promoted or truncated"):
        raising([_sample("too-long", 1.0, "one two three four five")])

    skipped = collator([_sample("also-too-long", 1.0, "one two three four five")])
    assert skipped["skip_batch"] is True
    assert collator.counters.batches_fully_skipped == 1
    assert skipped["collator_metrics"]["target_bucket_overflow"] == 1
    assert skipped["collator_metrics"]["batches_fully_skipped"] == 1


def test_device_corruption_uses_separate_reproducible_rng_streams():
    clean = torch.tensor([[2, 3, 14, 14], [4, 14, 1, 1]])
    valid = torch.tensor([[True, True, True, True], [True, True, False, False]])

    def generators():
        return {
            "time_generator": torch.Generator().manual_seed(10),
            "corruption_generator": torch.Generator().manual_seed(20),
            "token_noise_generator": torch.Generator().manual_seed(30),
        }

    uniform = corrupt_diffusion_batch(
        clean,
        valid,
        UniformNoiseProcess([5, 6, 7]),
        min_diffusion_time=1.0e-3,
        **generators(),
    )
    mask = corrupt_diffusion_batch(
        clean,
        valid,
        MaskNoiseProcess(15),
        min_diffusion_time=1.0e-3,
        **generators(),
    )
    assert torch.equal(uniform.noise_level, mask.noise_level)
    assert torch.equal(uniform.corrupt_mask, mask.corrupt_mask)
    assert not uniform.corrupt_mask[1, 2:].any()
    assert uniform.noisy_canvas_ids[1, 2:].tolist() == [1, 1]
    assert mask.noisy_canvas_ids[1, 2:].tolist() == [1, 1]


def test_no_transcript_sentinel_is_not_prefixed_with_country():
    collator = MawJaisDiffusionCollator(_Processor(), _config())
    sample = _sample("silence", 1.0, "[IFM-NoTranscript]")
    sample["country_code"] = None
    sample["target"]["country_code"] = None
    batch = collator([sample])
    assert batch["target_texts"] == ["[IFM-NoTranscript]"]


def test_conditional_collator_uses_four_per_record_requests_and_country_targets():
    processor = _Processor()
    collator = MawJaisDiffusionCollator(
        processor, _config(), conditional_mode=True
    )
    samples = [
        _conditional_sample(0, target_transcription="plain"),
        _conditional_sample(1, target_transcription="vocalized"),
        _conditional_sample(2, target_transcription="english"),
        _conditional_sample(3, target_transcription="vocalized english"),
    ]
    batch = collator(samples)

    assert batch["view_ids"].tolist() == [0, 1, 2, 3]
    assert batch["view_names"] == [view.name for view in CONDITIONAL_ASR_VIEWS]
    assert batch["prompt_tags"] == [view.prompt_tag for view in CONDITIONAL_ASR_VIEWS]
    assert processor.seen_text == [
        build_mawjais_asr_prefix(view.prompt_tag) for view in CONDITIONAL_ASR_VIEWS
    ]
    assert batch["request_code_switching"].tolist() == [False, False, True, True]
    assert batch["request_diacritics"].tolist() == [False, True, False, True]
    assert batch["content_has_code_switching"].tolist() == [True] * 4
    assert batch["applied_code_switching"].tolist() == [False, False, True, True]
    assert batch["target_transcriptions"] == [
        "plain",
        "vocalized",
        "english",
        "vocalized english",
    ]
    assert batch["target_texts"] == [
        "[EG] plain",
        "[EG] vocalized",
        "[EG] english",
        "[EG] vocalized english",
    ]


def test_conditional_prompt_never_leaks_content_code_switch_metadata():
    processor = _Processor()
    collator = MawJaisDiffusionCollator(
        processor, _config(), conditional_mode=True
    )
    positive_content_without_request = _conditional_sample(
        0, content_has_code_switching=True
    )
    negative_content_with_request = _conditional_sample(
        2, content_has_code_switching=False
    )
    batch = collator([positive_content_without_request, negative_content_with_request])

    assert batch["prompt_tags"] == ["", "[Code-switch]"]
    assert batch["content_has_code_switching"].tolist() == [True, False]
    assert batch["request_code_switching"].tolist() == [False, True]
    assert batch["applied_code_switching"].tolist() == [False, False]


def test_conditional_collator_rejects_ambiguous_or_inconsistent_records():
    collator = MawJaisDiffusionCollator(
        _Processor(), _config(), conditional_mode=True
    )
    missing_request = _conditional_sample(0)
    del missing_request["request_code_switching"]
    with pytest.raises(ValueError, match="missing request fields"):
        collator([missing_request])

    wrong_view = _conditional_sample(0)
    wrong_view["request_code_switching"] = True
    with pytest.raises(ValueError, match="view_id disagrees"):
        collator([wrong_view])

    mismatched_text = _conditional_sample(0)
    mismatched_text["transcription"] = "different legacy alias"
    with pytest.raises(ValueError, match="disagrees with target_transcription"):
        collator([mismatched_text])

    invalid_application = _conditional_sample(2, content_has_code_switching=False)
    invalid_application["applied_code_switching"] = True
    with pytest.raises(ValueError, match="applied_code_switching=true"):
        collator([invalid_application])

    invalid_preprocessing = _conditional_sample(1)
    invalid_preprocessing.update(
        {"canvas_size": 8, "target_token_count": 999, "valid_view_mask": (True,) * 4}
    )
    with pytest.raises(ValueError, match="canvas_size disagrees"):
        collator([invalid_preprocessing])

    invalid_selected_view = _conditional_sample(1)
    invalid_selected_view["valid_view_mask"] = (True, False, True, True)
    with pytest.raises(ValueError, match="selected an invalid view_id"):
        collator([invalid_selected_view])


def test_device_corruption_does_not_mutate_clean_conditional_prefix():
    collator = MawJaisDiffusionCollator(
        _Processor(), _config(), conditional_mode=True
    )
    batch = collator([_conditional_sample(3)])
    prefix_before = batch["prefix_input_ids"].clone()
    corrupt_diffusion_batch(
        batch["clean_canvas_ids"],
        batch["canvas_valid_mask"],
        UniformNoiseProcess([5, 6, 7]),
        min_diffusion_time=1.0,
        time_generator=torch.Generator().manual_seed(1),
        corruption_generator=torch.Generator().manual_seed(2),
        token_noise_generator=torch.Generator().manual_seed(3),
    )
    assert torch.equal(batch["prefix_input_ids"], prefix_before)
