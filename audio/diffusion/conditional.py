"""Canonical conditional-ASR view semantics shared by training and evaluation.

The request flags describe what the model is asked to emit.  They are kept
separate from ``content_has_code_switching``, which describes the recording,
and ``applied_code_switching``, which records whether the selected target
actually contains the code-switch transformation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ConditionalASRView:
    """One member of the fixed 2x2 conditional transcription factorial."""

    view_id: int
    name: str
    target_field: str
    request_code_switching: bool
    request_diacritics: bool
    prompt_tag: str


CONDITIONAL_ASR_VIEWS: tuple[ConditionalASRView, ...] = (
    ConditionalASRView(0, "plain", "target_plain", False, False, ""),
    ConditionalASRView(
        1,
        "diacritics",
        "target_diacritics",
        False,
        True,
        "[Diacritics]",
    ),
    ConditionalASRView(
        2,
        "code_switch",
        "target_code_switch",
        True,
        False,
        "[Code-switch]",
    ),
    ConditionalASRView(
        3,
        "code_switch_diacritics",
        "target_code_switch_diacritics",
        True,
        True,
        "[Code-switch] [Diacritics]",
    ),
)

CONDITIONAL_ASR_VIEW_BY_NAME = {
    view.name: view for view in CONDITIONAL_ASR_VIEWS
}


def conditional_asr_view(view_id: int) -> ConditionalASRView:
    """Return a view by ID, rejecting bools and out-of-range values."""

    if isinstance(view_id, bool):
        raise TypeError("view_id must be an integer in [0, 3], not bool")
    try:
        resolved = int(view_id)
    except (TypeError, ValueError) as error:
        raise TypeError("view_id must be an integer in [0, 3]") from error
    if resolved != view_id or not 0 <= resolved < len(CONDITIONAL_ASR_VIEWS):
        raise ValueError("view_id must be an integer in [0, 3]")
    return CONDITIONAL_ASR_VIEWS[resolved]


def conditional_prompt_tag(
    *, request_code_switching: bool, request_diacritics: bool
) -> str:
    """Build the exact system instruction in its canonical tag order."""

    if not isinstance(request_code_switching, bool):
        raise TypeError("request_code_switching must be bool")
    if not isinstance(request_diacritics, bool):
        raise TypeError("request_diacritics must be bool")
    tags: list[str] = []
    if request_code_switching:
        tags.append("[Code-switch]")
    if request_diacritics:
        tags.append("[Diacritics]")
    return " ".join(tags)


def view_id_from_requests(
    *, request_code_switching: bool, request_diacritics: bool
) -> int:
    """Encode the two request bits using the dataset's fixed view ordering."""

    # Calling the renderer centralizes strict bool validation.
    conditional_prompt_tag(
        request_code_switching=request_code_switching,
        request_diacritics=request_diacritics,
    )
    return (2 if request_code_switching else 0) + (1 if request_diacritics else 0)


def conditional_view_from_sample(sample: Mapping[str, Any]) -> ConditionalASRView:
    """Resolve and cross-check a selected conditional sample.

    The selected target text itself is intentionally not consulted.  Prompt
    construction therefore cannot leak whether the underlying audio contains
    code-switching.
    """

    missing = {
        "view_id",
        "request_code_switching",
        "request_diacritics",
    }.difference(sample)
    if missing:
        raise ValueError(
            f"conditional sample is missing request fields: {sorted(missing)}"
        )
    request_code_switching = sample["request_code_switching"]
    request_diacritics = sample["request_diacritics"]
    expected_id = view_id_from_requests(
        request_code_switching=request_code_switching,
        request_diacritics=request_diacritics,
    )
    view = conditional_asr_view(sample["view_id"])
    if view.view_id != expected_id:
        raise ValueError(
            "view_id disagrees with request_code_switching/request_diacritics: "
            f"view_id={view.view_id}, request view={expected_id}"
        )
    supplied_name = sample.get("view_name")
    if supplied_name is not None and supplied_name != view.name:
        raise ValueError(
            f"view_name {supplied_name!r} disagrees with view_id={view.view_id} "
            f"({view.name!r})"
        )
    return view


__all__ = [
    "CONDITIONAL_ASR_VIEWS",
    "CONDITIONAL_ASR_VIEW_BY_NAME",
    "ConditionalASRView",
    "conditional_asr_view",
    "conditional_prompt_tag",
    "conditional_view_from_sample",
    "view_id_from_requests",
]
