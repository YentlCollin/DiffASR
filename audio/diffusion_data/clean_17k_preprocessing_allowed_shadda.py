"""Principal preprocessing for raw clean-17k rows, with shadda and tatweel allowed.

NOTE: Under this preprocessing policy, shadda (U+0651) is not treated as a
removable diacritic. Existing shadda and tatweel (U+0640) are preserved even
when use_diacritics=False. Other Arabic diacritics are still removed in that
mode. This preprocessing preserves existing characters; it does not insert
shadda or tatweel, and it does not guarantee the model's generated output.
Linguistically, shadda remains a diacritic; tatweel is a graphical elongation.
When use_diacritics=True, the existing behavior is unchanged.

The main entry point is :func:`process_clean_17k_row`. It converts one row
from the raw clean-17k Parquet metadata into the metadata shape consumed by
the ASR indexed-audio loader, before model-specific token counting and split/
rank assignment.

For compatibility with the bounded legal-YTB snapshot already used by this
project, the function also accepts its prepared ``training.parquet`` rows.
Those rows store the transcription and country metadata inside
``target_json`` instead of the raw ``gemma.*`` / ``did.*`` columns.

Example::

    training_row = process_clean_17k_row(
        raw_row,
        use_code_switching=True,
        use_diacritics=False,
        source_root="/path/to/dataset/snapshot",
    )
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class TargetFormatError(ValueError):
    """Raised when a transcription cannot be converted without ambiguity."""


_CS_PAIR = re.compile(
    r"\[([^\[\]]+?):CS\]\s*\[((?:[A-Z]{2,3}|UNK)):\s*([^\[\]]*?)\]",
    flags=re.UNICODE,
)
_CS_SPAN = re.compile(r"\[([^\[\]]+?):CS\]", flags=re.UNICODE)
_LANGUAGE_SPAN = re.compile(r"\[(?:[A-Z]{2,3}|UNK):\s*[^\[\]]*?\]", flags=re.UNICODE)
_SQUARE_SPAN = re.compile(r"\[[^\[\]]*\]", flags=re.UNICODE)
_ANGLE_SPAN = re.compile(r"<[^<>]*>", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
# Shadda U+0651 and tatweel U+0640 are excluded from the removable signs.
_ARABIC_DIACRITICS = re.compile(
    "[\u0610-\u061a\u064b-\u0650\u0652-\u065f\u0670\u06d6-\u06ed]",
    flags=re.UNICODE,
)

_COUNTRY_CODES = frozenset(
    {
        "AE",
        "BH",
        "DZ",
        "EG",
        "IQ",
        "JO",
        "KW",
        "LB",
        "LY",
        "MA",
        "MR",
        "OM",
        "PS",
        "QA",
        "SA",
        "SD",
        "SY",
        "TN",
        "YE",
    }
)
_LANGUAGE_CODES = {
    "deu": "de",
    "eng": "en",
    "fra": "fr",
    "ita": "it",
    "por": "pt",
    "rus": "ru",
    "spa": "es",
    "tur": "tr",
    "ukr": "uk",
}
_ARABIC_LANGUAGE_CODES = frozenset({"ara", "arb"})

# MawJais uses this exact assistant target.  Some raw metadata uses the
# hyphenated spelling; accept it at the boundary and canonicalize it here.
_NO_TRANSCRIPT = "[IFM-NoTranscript]"
_NO_TRANSCRIPT_PAYLOADS = frozenset({"IFM-NoTranscript", "IFM-No-Transcript"})


def _field(row: Mapping[str, Any], name: str, default: Any = None) -> Any:
    """Read either a literal dotted column or equivalent nested mappings."""

    if name in row:
        return row[name]
    value: Any = row
    for part in name.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def _optional_bool(value: Any, *, name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no", ""}:
            return False
    raise ValueError(f"{name} must be boolean-like, got {value!r}")


def _json_object(value: Any, *, sample_key: str, name: str) -> Mapping[str, Any]:
    try:
        if isinstance(value, Mapping):
            decoded = value
        elif isinstance(value, (bytes, bytearray)):
            decoded = json.loads(value.decode("utf-8"))
        else:
            decoded = json.loads(str(value))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"Sample {sample_key} has invalid {name} JSON") from error
    if not isinstance(decoded, Mapping):
        raise ValueError(f"Sample {sample_key} {name} JSON is not an object")
    return decoded


def _prepared_target(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    payload = _field(row, "target_json")
    if payload is None:
        return None
    return _json_object(
        payload,
        sample_key=str(_field(row, "sample_key", "<unknown>")),
        name="target",
    )


def _uses_script(text: str, script: str) -> bool:
    return any(script in unicodedata.name(character, "") for character in text)


def _validate_annotation_delimiters(text: str) -> None:
    expected_close: str | None = None
    opening_position = -1
    pairs = {"[": "]", "<": ">"}
    for position, character in enumerate(text):
        if character in pairs:
            if expected_close is not None:
                raise TargetFormatError(f"Nested annotation delimiter at position {position}")
            expected_close = pairs[character]
            opening_position = position
        elif character in "]>":
            if character != expected_close:
                raise TargetFormatError(
                    f"Unmatched annotation delimiter {character!r} at position {position}"
                )
            expected_close = None
    if expected_close is not None:
        raise TargetFormatError(f"Unclosed annotation delimiter at position {opening_position}")


def _required_text(row: Mapping[str, Any], name: str) -> str:
    value = _field(row, name)
    if value is None or not str(value).strip():
        raise ValueError(f"Raw clean-17k row has an empty {name}")
    return str(value).strip()


def strip_arabic_diacritics(text: str) -> str:
    """Remove other Arabic diacritics; preserve shadda, tatweel and other text."""

    if not isinstance(text, str):
        raise TargetFormatError(f"Expected transcription text, got {type(text).__name__}")
    return _ARABIC_DIACRITICS.sub("", unicodedata.normalize("NFC", text))


def _arabic_cs_payload(match: re.Match[str]) -> str:
    payload = match.group(1).strip()
    if not _uses_script(payload, "ARABIC") or _uses_script(payload, "LATIN"):
        raise TargetFormatError(f"Non-Arabic :CS transliteration: {match.group(0)!r}")
    return payload


def _remove_non_target_annotations(text: str) -> str:
    """Remove events while retaining country and no-transcript control tags."""

    def replace_square_span(match: re.Match[str]) -> str:
        payload = match.group(0)[1:-1].strip()
        if payload in _NO_TRANSCRIPT_PAYLOADS:
            return _NO_TRANSCRIPT
        if payload in _COUNTRY_CODES:
            return f"[{payload}]"
        return " "

    return _ANGLE_SPAN.sub(" ", _SQUARE_SPAN.sub(replace_square_span, text))


def _validate_remaining_delimiters(text: str) -> None:
    for match in _SQUARE_SPAN.finditer(text):
        payload = match.group(0)[1:-1]
        if payload != _NO_TRANSCRIPT[1:-1] and payload not in _COUNTRY_CODES:
            raise TargetFormatError(f"Unexpected retained annotation: {match.group(0)!r}")
    without_valid_spans = _SQUARE_SPAN.sub("", text)
    if any(delimiter in without_valid_spans for delimiter in "[]<>"):
        raise TargetFormatError("Unbalanced bracket delimiter remains after training normalization")


def clean_diacritic_free_training_transcription(text: str) -> str:
    """Remove annotations and other diacritics; retain shadda, tatweel and Arabic CS text."""

    if not isinstance(text, str):
        raise TargetFormatError(f"Expected transcription text, got {type(text).__name__}")
    cleaned = unicodedata.normalize("NFC", text)
    _validate_annotation_delimiters(cleaned)
    cleaned = _CS_SPAN.sub(_arabic_cs_payload, cleaned)
    cleaned = _remove_non_target_annotations(cleaned)
    cleaned = strip_arabic_diacritics(cleaned)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    if not cleaned:
        raise TargetFormatError("Transcription is empty after training normalization")
    _validate_remaining_delimiters(cleaned)
    if _uses_script(_SQUARE_SPAN.sub("", cleaned), "LATIN"):
        raise TargetFormatError("Latin-script text remains after training normalization")
    if _ARABIC_DIACRITICS.search(cleaned) is not None:
        raise TargetFormatError("Arabic diacritic remains after training normalization")
    return cleaned


def clean_diacritized_training_transcription(text: str) -> str:
    """Remove annotations while retaining diacritics and Arabic CS speech."""

    if not isinstance(text, str):
        raise TargetFormatError(f"Expected transcription text, got {type(text).__name__}")
    cleaned = unicodedata.normalize("NFC", text)
    _validate_annotation_delimiters(cleaned)
    cleaned = _CS_SPAN.sub(_arabic_cs_payload, cleaned)
    cleaned = _remove_non_target_annotations(cleaned)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    if not cleaned:
        raise TargetFormatError("Transcription is empty after training normalization")
    _validate_remaining_delimiters(cleaned)
    if _uses_script(_SQUARE_SPAN.sub("", cleaned), "LATIN"):
        raise TargetFormatError("Latin-script text remains after training normalization")
    return cleaned


def clean_code_switch_training_transcription(text: str, *, keep_diacritics: bool) -> str:
    """Replace canonical code-switch pairs with their unbracketed target text."""

    if not isinstance(text, str):
        raise TargetFormatError(f"Expected transcription text, got {type(text).__name__}")
    cleaned = unicodedata.normalize("NFC", text)
    _validate_annotation_delimiters(cleaned)

    def latin_payload(match: re.Match[str]) -> str:
        payload = match.group(3).strip()
        if not payload:
            raise TargetFormatError(
                f"Code-switch pair has an empty Latin target: {match.group(0)!r}"
            )
        # A language target can legitimately be numeric (for example
        # ``[EN: 2011]``).  It has no Unicode LATIN character, but the paired
        # annotation still makes the desired target unambiguous.  Require
        # Latin text or at least one digit and never accept an Arabic target
        # here, since that would defeat the code-switch view.
        has_latin = _uses_script(payload, "LATIN")
        has_digit = any(character.isdigit() for character in payload)
        if _uses_script(payload, "ARABIC") or not (has_latin or has_digit):
            raise TargetFormatError(
                "Code-switch target is neither Latin-script nor numeric text: "
                f"{match.group(0)!r}"
            )
        return payload

    cleaned = _CS_PAIR.sub(latin_payload, cleaned)
    unpaired = _CS_SPAN.search(cleaned)
    if unpaired is not None:
        raise TargetFormatError(
            f"Code-switch annotation has no paired Latin target: {unpaired.group(0)!r}"
        )
    dangling = _LANGUAGE_SPAN.search(cleaned)
    if dangling is not None:
        raise TargetFormatError(
            f"Language annotation without Arabic :CS pair: {dangling.group(0)!r}"
        )
    cleaned = _remove_non_target_annotations(cleaned)
    if not keep_diacritics:
        cleaned = _ARABIC_DIACRITICS.sub("", cleaned)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    if not cleaned:
        raise TargetFormatError("Transcription is empty after training normalization")
    _validate_remaining_delimiters(cleaned)
    return cleaned


def process_training_transcription(
    text: str,
    *,
    use_code_switching: bool = False,
    use_diacritics: bool = False,
) -> str:
    """Produce one of four target views; False preserves shadda and tatweel, True all marks."""

    if use_code_switching:
        return clean_code_switch_training_transcription(
            text, keep_diacritics=use_diacritics
        )
    if use_diacritics:
        return clean_diacritized_training_transcription(text)
    return clean_diacritic_free_training_transcription(text)


def extract_raw_transcription(row: Mapping[str, Any]) -> str:
    """Extract the raw transcription from clean-17k or prepared proxy metadata."""

    sample_key = str(_field(row, "sample_key", "<unknown>"))
    payload = _field(row, "gemma.structured_output_json")
    name = "structured transcription"
    if payload is None:
        payload = _field(row, "target_json")
        name = "target"
    if payload is None:
        raise ValueError(
            f"Sample {sample_key} has neither gemma.structured_output_json nor target_json"
        )
    structured = _json_object(payload, sample_key=sample_key, name=name)
    transcription = structured.get("transcription")
    if not isinstance(transcription, str) or not transcription.strip():
        raise ValueError(f"Sample {sample_key} has an empty target transcription")
    return transcription.strip()


def _audio_tar_path(row: Mapping[str, Any], source_root: str | Path | None) -> str:
    raw_path = Path(_required_text(row, "audio_tar_path"))
    if raw_path.is_absolute() or source_root is None:
        return str(raw_path.resolve()) if raw_path.is_absolute() else str(raw_path)
    root = Path(source_root).resolve()
    resolved = (root / raw_path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Dataset path escapes source_root: {raw_path}")
    return str(resolved)


def _source_groups(sample_key: str) -> tuple[str, str]:
    parts = sample_key.split("__")
    if len(parts) < 4 or not parts[0].startswith("v") or len(parts[1]) != 2:
        raise ValueError(f"Cannot derive legal-YTB identity from sample key {sample_key!r}")
    return f"legal-ytb:video:{parts[2]}", f"legal-ytb:segment:{sample_key}"


def _labels(
    row: Mapping[str, Any],
    *,
    country_threshold: float,
    prepared_target: Mapping[str, Any] | None,
) -> dict[str, Any]:
    primary_language = str(
        _field(row, "gemma_meta.primary_language")
        or (None if prepared_target is None else prepared_target.get("primary_language"))
        or ""
    ).strip().lower()
    language_code = None
    if primary_language and primary_language not in _ARABIC_LANGUAGE_CODES | {"und"}:
        language_code = _LANGUAGE_CODES.get(primary_language)
        if language_code is None:
            raise ValueError(f"Unsupported non-Arabic primary language {primary_language!r}")

    score_value = _field(row, "did.top1_score")
    if score_value is None and prepared_target is not None:
        score_value = prepared_target.get("country_confidence")
    country_confidence = None if score_value is None else float(score_value)
    if country_confidence is not None and not math.isfinite(country_confidence):
        raise ValueError("Country confidence must be finite")

    country_value = _field(row, "did.top1_cc")
    if country_value is None and prepared_target is not None:
        country_value = prepared_target.get("country_code")
    country = None if country_value is None else str(country_value).strip().upper()
    country = country or None
    if country is not None and country not in _COUNTRY_CODES:
        raise ValueError(f"Clean row has unsupported country code {country!r}")

    msa_value = _field(row, "gemma_meta.msa")
    if msa_value is None and prepared_target is not None:
        msa_value = prepared_target.get("msa")
    msa = _optional_bool(msa_value, name="gemma_meta.msa")

    if language_code is not None:
        country_supervision = True
        msa = False
        msa_supervision = True
    elif prepared_target is not None and country is not None and score_value is None:
        # ``target_json.country_code`` in the prepared proxy metadata is the
        # authoritative label already used by the current MawJais trainer.
        country_supervision = True
        msa_supervision = msa is not None
    else:
        country_supervision = country is not None and country_confidence is not None
        country_supervision = country_supervision and country_confidence >= country_threshold
        msa_supervision = msa is not None
    return {
        "country_code": country,
        "dialect": country,
        "country_confidence": country_confidence,
        "country_supervision": country_supervision,
        "msa": msa,
        "msa_supervision": msa_supervision,
        "language_code": language_code,
    }


def process_clean_17k_row(
    row: Mapping[str, Any],
    *,
    use_code_switching: bool = False,
    use_diacritics: bool = False,
    source_root: str | Path | None = None,
    country_threshold: float = 0.90,
    max_audio_seconds: float | None = None,
) -> dict[str, Any]:
    """Convert one metadata row to pre-tokenization training metadata.

    Raw clean-17k metadata and the project's prepared legal-YTB
    ``training.parquet`` rows are both accepted. Code-switch markup is enabled
    only for rows advertised by ``richness.has_code_switch_tag`` or by the
    prepared ``target_json.has_code_switching`` field. On other rows,
    requesting it safely falls back to the equivalent non-code-switch target
    and leaves the prompt flag disabled. Diacritics can be selected
    independently.

    ``target_tokens``, CTC feasibility, split, rank, and epoch occurrence IDs
    are deliberately absent: those values depend on the selected model and
    experiment layout and must be added by the downstream preparation step.

    No duration limit is imposed by default. Set ``max_audio_seconds`` when a
    particular downstream model requires a hard preprocessing cap.
    """

    if not 0 <= country_threshold <= 1:
        raise ValueError("country_threshold must be between zero and one")
    if max_audio_seconds is not None and (
        not math.isfinite(max_audio_seconds) or max_audio_seconds <= 0
    ):
        raise ValueError("max_audio_seconds must be positive and finite when set")

    sample_key = _required_text(row, "sample_key")
    try:
        duration_sec = float(_field(row, "duration_sec"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Sample {sample_key} has an invalid duration") from error
    if not math.isfinite(duration_sec) or duration_sec <= 0:
        raise ValueError(f"Sample {sample_key} has non-positive or non-finite duration")
    if max_audio_seconds is not None and duration_sec > max_audio_seconds:
        raise ValueError(
            f"Sample {sample_key} is {duration_sec:.3f}s, exceeding the "
            f"{max_audio_seconds:.3f}s limit"
        )

    prepared_target = _prepared_target(row)
    raw_transcription = extract_raw_transcription(row)
    code_switch_value = _field(row, "richness.has_code_switch_tag")
    if code_switch_value is None and prepared_target is not None:
        code_switch_value = prepared_target.get("has_code_switching", False)
    has_code_switch_target = bool(
        _optional_bool(code_switch_value, name="richness.has_code_switch_tag")
    )
    selected_code_switching = bool(use_code_switching and has_code_switch_target)
    transcription = process_training_transcription(
        raw_transcription,
        use_code_switching=selected_code_switching,
        use_diacritics=use_diacritics,
    )
    no_transcript = _NO_TRANSCRIPT in transcription
    if no_transcript:
        residual = transcription.replace(_NO_TRANSCRIPT, "")
        residual = _SQUARE_SPAN.sub("", residual).strip()
        if residual:
            raise TargetFormatError(
                f"Sample {sample_key} combines {_NO_TRANSCRIPT} with spoken transcription text"
            )
    source_group, content_group = _source_groups(sample_key)

    wav_sha256 = _field(row, "wav_sha256")
    wav_sha256 = (
        None
        if wav_sha256 is None or not str(wav_sha256).strip()
        else str(wav_sha256)
    )
    return {
        "sample_key": sample_key,
        "source_sample_key": sample_key,
        "source_group": source_group,
        "content_group": content_group,
        "audio_tar_path": _audio_tar_path(row, source_root),
        "audio_member": _required_text(row, "audio_member"),
        "wav_sha256": wav_sha256,
        "duration_sec": duration_sec,
        "transcription": transcription,
        **_labels(
            row,
            country_threshold=country_threshold,
            prepared_target=prepared_target,
        ),
        "use_code_switching": selected_code_switching,
        "use_diacritics": bool(use_diacritics),
        "has_code_switching": selected_code_switching,
        "request_country_code": False,
        "request_msa_status": False,
        "no_transcript": no_transcript,
        "legacy_prompt_format": False,
        "family": "clean",
        "subcorpus": "clean-17k",
    }


__all__ = [
    "TargetFormatError",
    "clean_code_switch_training_transcription",
    "clean_diacritic_free_training_transcription",
    "clean_diacritized_training_transcription",
    "extract_raw_transcription",
    "process_clean_17k_row",
    "process_training_transcription",
    "strip_arabic_diacritics",
]
