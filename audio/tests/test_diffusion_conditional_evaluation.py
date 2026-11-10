import random

import pytest

from audio.diffusion.conditional import CONDITIONAL_ASR_VIEWS
from audio.diffusion.conditional_evaluation import (
    edit_distance,
    latin_words,
    normalize_asr_text,
    score_conditional_predictions,
)


def _reference_edit_distance(reference, hypothesis):
    previous = list(range(len(reference) + 1))
    for index, item in enumerate(hypothesis, start=1):
        current = [index]
        for ref_index, ref_item in enumerate(reference, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[ref_index] + 1,
                    previous[ref_index - 1] + (item != ref_item),
                )
            )
        previous = current
    return previous[-1]


def _row(
    sample_key,
    view_id,
    reference,
    hypothesis,
    *,
    content_has_code_switching,
):
    view = CONDITIONAL_ASR_VIEWS[view_id]
    return {
        "sample_key": sample_key,
        "view_id": view_id,
        "view_name": view.name,
        "request_code_switching": view.request_code_switching,
        "request_diacritics": view.request_diacritics,
        "content_has_code_switching": content_has_code_switching,
        "reference": reference,
        "hypothesis": hypothesis,
        "country_code": "EG",
        "duration_sec": 1.0,
        "generation_seconds": 0.1,
        "nfe": 4,
        "inference_prompt_tag": view.prompt_tag,
    }


def _paired_rows():
    rows = [
        _row("cs", 0, "مرحبا هالو", "[EG] مرحبا هالو", content_has_code_switching=True),
        _row(
            "cs",
            1,
            "مَرْحَبًا هَالُو",
            "[EG] مَرْحَبًا هَالُو",
            content_has_code_switching=True,
        ),
        _row(
            "cs", 2, "مرحبا hello", "[EG] مرحبا hello", content_has_code_switching=True
        ),
        _row(
            "cs",
            3,
            "مَرْحَبًا hello",
            "[EG] مَرْحَبًا hello",
            content_has_code_switching=True,
        ),
        _row("arabic", 0, "مرحبا", "[EG] مرحبا", content_has_code_switching=False),
        _row(
            "arabic", 1, "مَرْحَبًا", "[EG] مَرْحَبًا", content_has_code_switching=False
        ),
        # Deliberate hallucination under a negative [Code-switch] request.
        _row(
            "arabic",
            2,
            "مرحبا",
            "[EG] مرحبا hello",
            content_has_code_switching=False,
        ),
        _row(
            "arabic", 3, "مَرْحَبًا", "[EG] مَرْحَبًا", content_has_code_switching=False
        ),
    ]
    return rows


def test_four_view_scorer_reports_asr_instruction_and_paired_metrics():
    report = score_conditional_predictions(_paired_rows())

    assert report["paired_samples"] == 2
    assert report["prediction_rows"] == 8
    assert report["content_code_switch_samples"] == 1
    assert set(report["views"]) == {view.name for view in CONDITIONAL_ASR_VIEWS}
    assert report["views"]["plain"]["cer"] == 0.0
    assert report["instruction_slices"]["diacritics_enabled"]["der"] == 0.0
    assert (
        report["instruction_slices"]["diacritics_disabled"][
            "diacritic_leakage_utterance_rate"
        ]
        == 0.0
    )
    assert (
        report["instruction_slices"]["code_switch_enabled_no_content"][
            "latin_hallucination_utterance_rate"
        ]
        == 0.5
    )
    assert report["aggregate"]["country_accuracy"] == 1.0
    assert report["aggregate"]["mean_nfe"] == 4.0
    assert report["aggregate"]["rtf"] == pytest.approx(0.1)
    assert report["content_slices"]["content_code_switch"]["utterances"] == 4
    assert (
        report["content_slices"]["content_without_code_switch"]["utterances"]
        == 4
    )
    assert (
        report["content_slices"]["content_code_switch_request_enabled"][
            "utterances"
        ]
        == 2
    )
    assert (
        report["content_slices"]["content_code_switch_request_disabled"][
            "utterances"
        ]
        == 2
    )
    assert report["paired_differences"]["diacritics"]["comparisons"] == 4
    assert report["paired_differences"]["code_switch"]["comparisons"] == 4
    assert report["paired_differences"]["code_switch"]["latin_rate_delta"] > 0


def test_normalization_keeps_script_identity_but_removes_arabic_marks():
    assert normalize_asr_text("[EG] مَرْحَبًا!", remove_diacritics=True) == "مرحبا"
    assert latin_words("[EG] مرحبا HÉLLO-world ثم café") == [
        "héllo-world",
        "café",
    ]


def test_four_view_scorer_rejects_missing_duplicate_or_wrong_prompt_views():
    rows = _paired_rows()
    with pytest.raises(ValueError, match="exactly four paired views"):
        score_conditional_predictions(rows[:-1])

    with pytest.raises(ValueError, match="duplicate prediction"):
        score_conditional_predictions(rows + [dict(rows[0])])

    wrong_prompt = [dict(row) for row in rows]
    wrong_prompt[2]["inference_prompt_tag"] = ""
    with pytest.raises(ValueError, match="used prompt"):
        score_conditional_predictions(wrong_prompt)

    wrong_requests = [dict(row) for row in rows]
    wrong_requests[0]["request_code_switching"] = True
    with pytest.raises(ValueError, match="view_id disagrees"):
        score_conditional_predictions(wrong_requests)


def test_negative_code_switch_rows_do_not_require_latin_references():
    report = score_conditional_predictions(_paired_rows())
    negative = report["instruction_slices"]["code_switch_enabled_no_content"]
    assert negative["reference_latin_words"] == 0
    assert negative["latin_span_wer"] is None
    assert negative["latin_word_recall"] is None
    assert negative["hypothesis_latin_words"] == 1


def test_bit_vector_edit_distance_matches_dynamic_programming():
    generator = random.Random(8142)
    alphabet = tuple("abc")
    for _ in range(500):
        reference = [generator.choice(alphabet) for _ in range(generator.randrange(40))]
        hypothesis = [generator.choice(alphabet) for _ in range(generator.randrange(40))]
        assert edit_distance(reference, hypothesis) == _reference_edit_distance(
            reference, hypothesis
        )


def test_parallel_scorer_is_bit_identical_to_single_process():
    assert score_conditional_predictions(
        _paired_rows(), num_workers=2
    ) == score_conditional_predictions(_paired_rows(), num_workers=1)
