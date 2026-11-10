"""Paired metrics for the four conditional MawJais ASR views.

The scorer is deliberately model-independent: generation jobs append one JSON
object per ``(sample_key, view_id)`` and this module validates that every audio
has the complete 2x2 factorial before aggregating metrics.  All reported error
rates are corpus-level ratios of additive edit counts, never means of per-row
rates.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import math
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from .conditional import CONDITIONAL_ASR_VIEWS, conditional_view_from_sample


CONDITIONAL_EVALUATION_SCHEMA_VERSION = 1
_LEADING_COUNTRY_TAG = re.compile(r"^\s*\[([^\]\r\n]+)\]\s*")
_ARABIC_DIACRITIC_RANGES = (
    (0x0610, 0x061A),
    (0x064B, 0x065F),
    (0x0670, 0x0670),
    (0x06D6, 0x06ED),
)
_TATWEEL = "\u0640"


def _is_arabic_diacritic(character: str) -> bool:
    value = ord(character)
    return any(lower <= value <= upper for lower, upper in _ARABIC_DIACRITIC_RANGES)


def strip_leading_country_tag(text: str) -> str:
    """Remove exactly one leading MawJais output-country tag."""

    return _LEADING_COUNTRY_TAG.sub("", str(text), count=1)


def remove_arabic_diacritics(text: str) -> str:
    """Remove Arabic combining marks and tatweel from text."""

    return "".join(
        character
        for character in str(text)
        if not _is_arabic_diacritic(character) and character != _TATWEEL
    )


def normalize_asr_text(text: str, *, remove_diacritics: bool) -> str:
    """Normalize text for corpus WER/CER while retaining script identity."""

    normalized = unicodedata.normalize("NFKC", strip_leading_country_tag(text))
    if remove_diacritics:
        normalized = remove_arabic_diacritics(normalized)
    normalized = normalized.casefold()
    normalized = "".join(
        " " if unicodedata.category(character)[0] in {"P", "S"} else character
        for character in normalized
    )
    return " ".join(normalized.split())


def edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    """Exact Levenshtein distance using Myers' bit-vector recurrence.

    Python's arbitrary-width integers evaluate a full dynamic-programming
    column with a handful of bit operations.  This is exact, supports both
    character and word sequences, and makes the 100-hour scorer practical.
    """

    if len(reference) > len(hypothesis):
        reference, hypothesis = hypothesis, reference
    width = len(reference)
    if width == 0:
        return len(hypothesis)
    equality_masks: dict[Any, int] = {}
    for index, item in enumerate(reference):
        equality_masks[item] = equality_masks.get(item, 0) | (1 << index)
    full_mask = (1 << width) - 1
    high_bit = 1 << (width - 1)
    positive_vertical = full_mask
    negative_vertical = 0
    distance = width
    for item in hypothesis:
        equality = equality_masks.get(item, 0)
        vertical_or_equality = equality | negative_vertical
        horizontal = (
            (((equality & positive_vertical) + positive_vertical)
             ^ positive_vertical)
            | equality
        )
        positive_horizontal = negative_vertical | ~(
            horizontal | positive_vertical
        )
        negative_horizontal = positive_vertical & horizontal
        if positive_horizontal & high_bit:
            distance += 1
        elif negative_horizontal & high_bit:
            distance -= 1
        positive_horizontal = ((positive_horizontal << 1) | 1) & full_mask
        negative_horizontal = (negative_horizontal << 1) & full_mask
        positive_vertical = (
            negative_horizontal | ~(vertical_or_equality | positive_horizontal)
        ) & full_mask
        negative_vertical = positive_horizontal & vertical_or_equality
    return distance


def _characters(text: str) -> list[str]:
    return list(text.replace(" ", ""))


def _diacritics(text: str) -> list[str]:
    return [
        character
        for character in strip_leading_country_tag(unicodedata.normalize("NFKC", text))
        if _is_arabic_diacritic(character) or character == _TATWEEL
    ]


def _is_latin(character: str) -> bool:
    return character.isalpha() and "LATIN" in unicodedata.name(character, "")


def latin_words(text: str) -> list[str]:
    """Extract Unicode Latin-script word spans without treating Arabic as Latin."""

    words: list[str] = []
    current: list[str] = []
    for character in unicodedata.normalize("NFKC", strip_leading_country_tag(text)):
        if _is_latin(character):
            current.append(character.casefold())
        elif character in {"'", "’", "-"} and current:
            current.append(character)
        else:
            if current:
                while current and current[-1] in {"'", "’", "-"}:
                    current.pop()
                if current:
                    words.append("".join(current))
                current = []
    if current:
        while current and current[-1] in {"'", "’", "-"}:
            current.pop()
        if current:
            words.append("".join(current))
    return words


def _latin_character_count(text: str) -> int:
    return sum(_is_latin(character) for character in strip_leading_country_tag(text))


def _lexical_character_count(text: str) -> int:
    return sum(character.isalnum() for character in strip_leading_country_tag(text))


def _arabic_base_count(text: str) -> int:
    count = 0
    for character in strip_leading_country_tag(text):
        value = ord(character)
        if (
            0x0600 <= value <= 0x06FF
            or 0x0750 <= value <= 0x077F
            or 0x08A0 <= value <= 0x08FF
        ) and not _is_arabic_diacritic(character) and character != _TATWEEL:
            count += 1
    return count


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    result = float(numerator) / float(denominator)
    if not math.isfinite(result):  # pragma: no cover - defensive guard
        raise ValueError("metric ratio is not finite")
    return result


def _row_reference(row: Mapping[str, Any]) -> str:
    reference = row.get("reference", row.get("target_transcription"))
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("every prediction row requires a non-empty reference")
    return reference


def _row_hypothesis(row: Mapping[str, Any]) -> str:
    hypothesis = row.get("hypothesis", row.get("transcript"))
    if not isinstance(hypothesis, str):
        raise ValueError("every prediction row requires a string hypothesis")
    return hypothesis


def _empty_totals() -> dict[str, int | float]:
    return {
        "utterances": 0,
        "word_errors": 0,
        "reference_words": 0,
        "character_errors": 0,
        "reference_characters": 0,
        "undiacritized_word_errors": 0,
        "undiacritized_reference_words": 0,
        "undiacritized_character_errors": 0,
        "undiacritized_reference_characters": 0,
        "diacritic_errors": 0,
        "reference_diacritic_marks": 0,
        "hypothesis_diacritic_marks": 0,
        "hypothesis_arabic_bases": 0,
        "utterances_with_diacritics": 0,
        "latin_word_errors": 0,
        "reference_latin_words": 0,
        "recalled_latin_words": 0,
        "hypothesis_latin_words": 0,
        "hypothesis_latin_characters": 0,
        "hypothesis_lexical_characters": 0,
        "utterances_with_latin": 0,
        "country_correct": 0,
        "audio_seconds": 0.0,
        "generation_seconds": 0.0,
        "nfe_total": 0,
        "nfe_observations": 0,
    }


def _score_row(row: Mapping[str, Any]) -> dict[str, int | float]:
    totals = _empty_totals()
    reference = _row_reference(row)
    hypothesis = _row_hypothesis(row)
    for remove, prefix in ((False, ""), (True, "undiacritized_")):
        ref = normalize_asr_text(reference, remove_diacritics=remove)
        hyp = normalize_asr_text(hypothesis, remove_diacritics=remove)
        ref_words, hyp_words = ref.split(), hyp.split()
        ref_characters, hyp_characters = _characters(ref), _characters(hyp)
        totals[prefix + "word_errors"] += edit_distance(ref_words, hyp_words)
        totals[prefix + "reference_words"] += len(ref_words)
        totals[prefix + "character_errors"] += edit_distance(
            ref_characters, hyp_characters
        )
        totals[prefix + "reference_characters"] += len(ref_characters)

    reference_marks = _diacritics(reference)
    hypothesis_marks = _diacritics(hypothesis)
    totals["diacritic_errors"] += edit_distance(reference_marks, hypothesis_marks)
    totals["reference_diacritic_marks"] += len(reference_marks)
    totals["hypothesis_diacritic_marks"] += len(hypothesis_marks)
    totals["hypothesis_arabic_bases"] += _arabic_base_count(hypothesis)
    totals["utterances_with_diacritics"] += int(bool(hypothesis_marks))

    reference_latin = latin_words(reference)
    hypothesis_latin = latin_words(hypothesis)
    totals["latin_word_errors"] += edit_distance(reference_latin, hypothesis_latin)
    totals["reference_latin_words"] += len(reference_latin)
    totals["hypothesis_latin_words"] += len(hypothesis_latin)
    totals["recalled_latin_words"] += sum(
        (Counter(reference_latin) & Counter(hypothesis_latin)).values()
    )
    latin_characters = _latin_character_count(hypothesis)
    totals["hypothesis_latin_characters"] += latin_characters
    totals["hypothesis_lexical_characters"] += _lexical_character_count(hypothesis)
    totals["utterances_with_latin"] += int(latin_characters > 0)

    country_code = row.get("country_code")
    country_match = _LEADING_COUNTRY_TAG.match(hypothesis)
    totals["country_correct"] += int(
        isinstance(country_code, str)
        and country_match is not None
        and country_match.group(1).strip() == country_code.strip()
    )
    duration = row.get("duration_sec", row.get("duration_seconds"))
    generation = row.get("generation_seconds")
    if not isinstance(duration, (int, float)) or duration < 0:
        raise ValueError("every prediction row requires non-negative duration_sec")
    if not isinstance(generation, (int, float)) or generation < 0:
        raise ValueError("every prediction row requires non-negative generation_seconds")
    totals["audio_seconds"] += float(duration)
    totals["generation_seconds"] += float(generation)
    nfe = row.get("nfe")
    if nfe is not None:
        if isinstance(nfe, bool) or not isinstance(nfe, int) or nfe <= 0:
            raise ValueError("nfe must be a positive integer when provided")
        totals["nfe_total"] += nfe
        totals["nfe_observations"] += 1
    totals["utterances"] += 1
    return totals


def _finalize_totals(totals: Mapping[str, int | float]) -> dict[str, Any]:

    utterances = int(totals["utterances"])
    output: dict[str, Any] = {
        **totals,
        "wer": _safe_ratio(totals["word_errors"], totals["reference_words"]),
        "cer": _safe_ratio(
            totals["character_errors"], totals["reference_characters"]
        ),
        "wer_without_diacritics": _safe_ratio(
            totals["undiacritized_word_errors"],
            totals["undiacritized_reference_words"],
        ),
        "cer_without_diacritics": _safe_ratio(
            totals["undiacritized_character_errors"],
            totals["undiacritized_reference_characters"],
        ),
        # DER is explicitly mark-level Levenshtein / reference mark count.
        "der": _safe_ratio(
            totals["diacritic_errors"], totals["reference_diacritic_marks"]
        ),
        "diacritic_mark_rate": _safe_ratio(
            totals["hypothesis_diacritic_marks"], totals["hypothesis_arabic_bases"]
        ),
        "diacritic_utterance_rate": _safe_ratio(
            totals["utterances_with_diacritics"], utterances
        ),
        "latin_span_wer": _safe_ratio(
            totals["latin_word_errors"], totals["reference_latin_words"]
        ),
        "latin_word_recall": _safe_ratio(
            totals["recalled_latin_words"], totals["reference_latin_words"]
        ),
        "latin_character_rate": _safe_ratio(
            totals["hypothesis_latin_characters"],
            totals["hypothesis_lexical_characters"],
        ),
        "latin_utterance_rate": _safe_ratio(totals["utterances_with_latin"], utterances),
        "country_accuracy": _safe_ratio(totals["country_correct"], utterances),
        "rtf": _safe_ratio(totals["generation_seconds"], totals["audio_seconds"]),
        "rtfx": _safe_ratio(totals["audio_seconds"], totals["generation_seconds"]),
        "mean_nfe": _safe_ratio(totals["nfe_total"], totals["nfe_observations"]),
    }
    # Explicit aliases make downstream gate dashboards self-documenting: these
    # are the raw, mark-sensitive rates, in contrast to the *_without_diacritics
    # lexical rates above.
    output["wer_with_diacritics"] = output["wer"]
    output["cer_with_diacritics"] = output["cer"]
    return output


def _score_precomputed(
    row_totals: Iterable[Mapping[str, int | float]],
) -> dict[str, Any]:
    totals = _empty_totals()
    for item in row_totals:
        for key in totals:
            totals[key] += item[key]
    return _finalize_totals(totals)


def _score_subset(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return _score_precomputed(_score_row(row) for row in rows)


def _prediction_rate(text: str, *, feature: str) -> float:
    if feature == "diacritics":
        return len(_diacritics(text)) / max(_arabic_base_count(text), 1)
    if feature == "latin":
        return _latin_character_count(text) / max(_lexical_character_count(text), 1)
    raise ValueError(f"unknown paired feature: {feature}")


def _paired_axis_metrics(
    grouped: Mapping[str, Mapping[int, Mapping[str, Any]]],
    pairs: Sequence[tuple[int, int]],
    *,
    feature: str,
) -> dict[str, Any]:
    comparisons = 0
    changed = 0
    rate_delta = 0.0
    lexical_errors = 0
    lexical_reference_characters = 0
    for views in grouped.values():
        for disabled_id, enabled_id in pairs:
            disabled = _row_hypothesis(views[disabled_id])
            enabled = _row_hypothesis(views[enabled_id])
            disabled_normalized = normalize_asr_text(
                disabled, remove_diacritics=False
            )
            enabled_normalized = normalize_asr_text(enabled, remove_diacritics=False)
            changed += int(disabled_normalized != enabled_normalized)
            rate_delta += _prediction_rate(enabled, feature=feature) - _prediction_rate(
                disabled, feature=feature
            )
            disabled_lexical = _characters(
                normalize_asr_text(disabled, remove_diacritics=True)
            )
            enabled_lexical = _characters(
                normalize_asr_text(enabled, remove_diacritics=True)
            )
            lexical_errors += edit_distance(disabled_lexical, enabled_lexical)
            lexical_reference_characters += len(disabled_lexical)
            comparisons += 1
    return {
        "comparisons": comparisons,
        "output_change_rate": _safe_ratio(changed, comparisons),
        f"{feature}_rate_delta": _safe_ratio(rate_delta, comparisons),
        "paired_lexical_cer_without_diacritics": _safe_ratio(
            lexical_errors, lexical_reference_characters
        ),
    }


def validate_paired_prediction_rows(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[int, Mapping[str, Any]]]:
    """Validate and group a complete four-view evaluation result."""

    grouped: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("every prediction row must be a mapping")
        sample_key = row.get("sample_key")
        if not isinstance(sample_key, str) or not sample_key:
            raise ValueError("every prediction row requires sample_key")
        view = conditional_view_from_sample(row)
        prompt_tag = row.get("inference_prompt_tag", row.get("prompt_tag"))
        if prompt_tag is not None and prompt_tag != view.prompt_tag:
            raise ValueError(
                f"sample {sample_key!r} view {view.view_id} used prompt {prompt_tag!r}; "
                f"expected {view.prompt_tag!r}"
            )
        if view.view_id in grouped[sample_key]:
            raise ValueError(
                f"duplicate prediction for sample {sample_key!r}, view {view.view_id}"
            )
        if not isinstance(row.get("content_has_code_switching"), bool):
            raise TypeError("content_has_code_switching must be bool in prediction rows")
        country_code = row.get("country_code")
        if not isinstance(country_code, str) or not country_code.strip():
            raise ValueError("every prediction row requires a non-empty country_code")
        _row_reference(row)
        _row_hypothesis(row)
        grouped[sample_key][view.view_id] = row

    if not grouped:
        raise ValueError("conditional evaluation contains no prediction rows")
    expected = {view.view_id for view in CONDITIONAL_ASR_VIEWS}
    for sample_key, views in grouped.items():
        missing = expected.difference(views)
        extra = set(views).difference(expected)
        if missing or extra:
            raise ValueError(
                f"sample {sample_key!r} does not have exactly four paired views; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        content_flags = {
            bool(row["content_has_code_switching"]) for row in views.values()
        }
        country_codes = {str(row["country_code"]) for row in views.values()}
        durations = {
            float(row.get("duration_sec", row.get("duration_seconds", -1)))
            for row in views.values()
        }
        if len(content_flags) != 1 or len(country_codes) != 1 or len(durations) != 1:
            raise ValueError(
                f"paired metadata disagrees between views for sample {sample_key!r}"
            )
    return dict(grouped)


def score_conditional_predictions(
    rows: Iterable[Mapping[str, Any]],
    *,
    num_workers: int = 1,
) -> dict[str, Any]:
    """Score a complete paired four-prompt prediction stream."""

    if isinstance(num_workers, bool) or int(num_workers) <= 0:
        raise ValueError("num_workers must be a positive integer")
    num_workers = int(num_workers)
    grouped = validate_paired_prediction_rows(rows)
    ordered = [
        grouped[sample_key][view.view_id]
        for sample_key in sorted(grouped)
        for view in CONDITIONAL_ASR_VIEWS
    ]
    if num_workers == 1:
        additive = [_score_row(row) for row in ordered]
    else:
        chunksize = max(1, len(ordered) // (num_workers * 8))
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            additive = list(executor.map(_score_row, ordered, chunksize=chunksize))
    additive_by_identity = {
        id(row): totals for row, totals in zip(ordered, additive, strict=True)
    }

    def score_rows(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return _score_precomputed(additive_by_identity[id(row)] for row in selected)

    by_view = {
        view.name: score_rows(
            [grouped[sample_key][view.view_id] for sample_key in sorted(grouped)]
        )
        for view in CONDITIONAL_ASR_VIEWS
    }
    diacritics_enabled = [row for row in ordered if bool(row["request_diacritics"])]
    diacritics_disabled = [row for row in ordered if not bool(row["request_diacritics"])]
    code_switch_enabled = [
        row for row in ordered if bool(row["request_code_switching"])
    ]
    code_switch_disabled = [
        row for row in ordered if not bool(row["request_code_switching"])
    ]
    negative_code_switch = [
        row
        for row in code_switch_enabled
        if not bool(row["content_has_code_switching"])
    ]
    content_code_switch = [
        row for row in ordered if bool(row["content_has_code_switching"])
    ]
    content_without_code_switch = [
        row for row in ordered if not bool(row["content_has_code_switching"])
    ]
    disabled_metrics = score_rows(diacritics_disabled)
    code_switch_disabled_metrics = score_rows(code_switch_disabled)
    negative_metrics = score_rows(negative_code_switch)
    instruction_metrics = {
        "diacritics_enabled": score_rows(diacritics_enabled),
        "diacritics_disabled": {
            **disabled_metrics,
            "diacritic_leakage_mark_rate": disabled_metrics["diacritic_mark_rate"],
            "diacritic_leakage_utterance_rate": disabled_metrics[
                "diacritic_utterance_rate"
            ],
        },
        "code_switch_enabled": score_rows(code_switch_enabled),
        "code_switch_disabled": {
            **code_switch_disabled_metrics,
            "latin_leakage_character_rate": code_switch_disabled_metrics[
                "latin_character_rate"
            ],
            "latin_leakage_utterance_rate": code_switch_disabled_metrics[
                "latin_utterance_rate"
            ],
        },
        "code_switch_enabled_no_content": {
            **negative_metrics,
            "latin_hallucination_character_rate": negative_metrics[
                "latin_character_rate"
            ],
            "latin_hallucination_utterance_rate": negative_metrics[
                "latin_utterance_rate"
            ],
        },
    }
    content_metrics = {
        "content_code_switch": score_rows(content_code_switch),
        "content_without_code_switch": score_rows(content_without_code_switch),
        "content_code_switch_request_enabled": score_rows(
            [row for row in content_code_switch if bool(row["request_code_switching"])]
        ),
        "content_code_switch_request_disabled": score_rows(
            [row for row in content_code_switch if not bool(row["request_code_switching"])]
        ),
    }
    return {
        "schema_version": CONDITIONAL_EVALUATION_SCHEMA_VERSION,
        "evaluation": "mawjais_diffusion_conditional_four_view_paired",
        "paired_samples": len(grouped),
        "prediction_rows": len(ordered),
        "content_code_switch_samples": sum(
            bool(next(iter(views.values()))["content_has_code_switching"])
            for views in grouped.values()
        ),
        "aggregate": score_rows(ordered),
        "views": by_view,
        "instruction_slices": instruction_metrics,
        "content_slices": content_metrics,
        "paired_differences": {
            "diacritics": _paired_axis_metrics(
                grouped,
                ((0, 1), (2, 3)),
                feature="diacritics",
            ),
            "code_switch": _paired_axis_metrics(
                grouped,
                ((0, 2), (1, 3)),
                feature="latin",
            ),
        },
    }


__all__ = [
    "CONDITIONAL_EVALUATION_SCHEMA_VERSION",
    "edit_distance",
    "latin_words",
    "normalize_asr_text",
    "remove_arabic_diacritics",
    "score_conditional_predictions",
    "strip_leading_country_tag",
    "validate_paired_prediction_rows",
]
