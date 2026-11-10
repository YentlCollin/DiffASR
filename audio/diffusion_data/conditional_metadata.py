"""Signed four-view metadata for conditional legal-YTB DiffASR training.

The preprocessing contract deliberately separates facts about the recording from
the instruction presented to the model:

``content_has_code_switching``
    Whether the source annotation says that the recording contains code switching.
``request_code_switching`` / ``request_diacritics``
    Which of the four model instructions was selected for this epoch.
``applied_code_switching``
    Whether the selected target actually uses the annotated Latin/numeric payload.

The base order always permutes all four IDs, including invalid views.  Selection then
filters that stable order to ``valid_view_ids`` and cycles over the survivors.  Thus a
sample with two feasible views remains available every epoch, while the same
``sample_key``, ``view_seed``, validity mask and epoch select the same view on every
rank and DataLoader worker.  Rows with no valid view are excluded before sharding.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .clean_17k_preprocessing_allowed_shadda import process_clean_17k_row

if TYPE_CHECKING:
    import pyarrow as pa


CONDITIONAL_METADATA_SCHEMA_VERSION = 1
CONDITIONAL_INDEX_SCHEMA_VERSION = 1

VIEW_NAMES = (
    "plain",
    "diacritics",
    "code_switch",
    "code_switch_diacritics",
)
VIEW_REQUESTS = (
    (False, False),
    (False, True),
    (True, False),
    (True, True),
)
TARGET_COLUMNS = tuple(f"target_{name}" for name in VIEW_NAMES)
VIEW_COUNT = len(VIEW_NAMES)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_NO_TRANSCRIPT = "[IFM-NoTranscript]"


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 of a regular file."""

    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"expected a regular file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    """Hash one JSON object after removing its self-referential content hash."""

    unsigned = dict(payload)
    unsigned.pop("content_sha256", None)
    encoded = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_view_permutation(sample_key: str, view_seed: int) -> tuple[int, ...]:
    """Return a cross-process stable permutation of all four view IDs.

    Sorting independent SHA-256 digests avoids dependence on Python's randomized
    hash seed and on implementation details of a PRNG.  View validity is
    intentionally not an argument.
    """

    if not isinstance(sample_key, str) or not sample_key:
        raise ValueError("sample_key must be a non-empty string")
    ranked: list[tuple[bytes, int]] = []
    for view_id in range(VIEW_COUNT):
        payload = f"conditional-view-v1\0{int(view_seed)}\0{sample_key}\0{view_id}".encode(
            "utf-8"
        )
        ranked.append((hashlib.sha256(payload).digest(), view_id))
    return tuple(view_id for _, view_id in sorted(ranked))


def select_view_id(
    sample_key: str,
    view_seed: int,
    epoch: int,
    valid_view_mask: Sequence[bool],
) -> int:
    """Select one valid view while retaining the stable four-view base order."""

    if int(epoch) < 0:
        raise ValueError("epoch must be non-negative")
    if len(valid_view_mask) != VIEW_COUNT:
        raise ValueError(f"valid_view_mask must contain {VIEW_COUNT} values")
    valid_ids = tuple(
        view_id
        for view_id in stable_view_permutation(sample_key, view_seed)
        if bool(valid_view_mask[view_id])
    )
    if not valid_ids:
        raise ValueError("cannot select a view from an all-invalid mask")
    return valid_ids[int(epoch) % len(valid_ids)]


def target_text_for_tokenizer(transcription: str, country_code: str | None) -> str:
    """Render the exact assistant answer whose tokens occupy the clean canvas."""

    if not isinstance(transcription, str) or not transcription.strip():
        raise ValueError("transcription must be a non-empty string")
    transcription = transcription.strip()
    if transcription == _NO_TRANSCRIPT:
        return transcription
    if not isinstance(country_code, str) or not country_code.strip():
        raise ValueError("country_code must be a non-empty string")
    return f"[{country_code.strip().upper()}] {transcription}"


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
    if hasattr(encoded, "detach"):
        encoded = encoded.detach().cpu().reshape(-1).tolist()
    elif hasattr(encoded, "reshape") and not isinstance(encoded, list):
        encoded = encoded.reshape(-1).tolist()
    elif encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("tokenizer returned an unexpected target batch")
        encoded = encoded[0]
    token_ids = [int(token_id) for token_id in encoded]
    if any(token_id < 0 for token_id in token_ids):
        raise ValueError("tokenizer returned a negative token ID")
    return token_ids


def conditional_metadata_schema() -> "pa.Schema":
    """Return the strict Parquet schema used by every per-TAR sidecar."""

    import pyarrow as pa

    bool4 = pa.list_(pa.bool_(), VIEW_COUNT)
    int4 = pa.list_(pa.int32(), VIEW_COUNT)
    error4 = pa.list_(pa.large_string(), VIEW_COUNT)
    return pa.schema(
        [
            pa.field("metadata_schema_version", pa.int16(), nullable=False),
            pa.field("sample_key", pa.large_string(), nullable=False),
            pa.field("audio_tar_path", pa.large_string(), nullable=False),
            pa.field("audio_member", pa.large_string(), nullable=False),
            pa.field("duration_sec", pa.float64(), nullable=False),
            pa.field("country_code", pa.string(), nullable=False),
            pa.field("content_has_code_switching", pa.bool_(), nullable=False),
            *(pa.field(name, pa.large_string()) for name in TARGET_COLUMNS),
            pa.field("valid_view_mask", bool4, nullable=False),
            pa.field("view_permutation", pa.list_(pa.int8(), VIEW_COUNT), nullable=False),
            pa.field("applied_code_switching_mask", bool4, nullable=False),
            pa.field("target_token_counts", int4, nullable=False),
            pa.field("canvas_sizes", int4, nullable=False),
            pa.field("view_errors", error4, nullable=False),
        ]
    )


def _prepared_target_json(row: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = row.get("target_json")
    try:
        if isinstance(raw, Mapping):
            target = raw
        elif isinstance(raw, (bytes, bytearray)):
            target = json.loads(raw.decode("utf-8"))
        else:
            target = json.loads(str(raw))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise ValueError("target_json is not valid JSON") from error
    if not isinstance(target, Mapping):
        raise ValueError("target_json must decode to an object")
    return target


def prepare_conditional_row(
    row: Mapping[str, Any],
    *,
    tokenizer: Any,
    canvas_sizes: Sequence[int],
    tokens_per_second: float,
    margin_tokens: float,
    view_seed: int = 0,
) -> dict[str, Any]:
    """Prepare all four targets and per-view feasibility for one metadata row.

    Transformation or token-overflow failures are recorded independently in
    ``view_errors``.  Structural failures shared by every view (bad identity,
    country, duration, or duration bucket overflow) raise ``ValueError`` so the
    calling preparation job can record a row-level exclusion explicitly.
    """

    sample_key = str(row.get("sample_key", "")).strip()
    if not sample_key:
        raise ValueError("row has an empty sample_key")
    audio_tar_path = str(row.get("audio_tar_path", "")).strip()
    audio_member = str(row.get("audio_member", "")).strip()
    if not audio_tar_path or not audio_member:
        raise ValueError(f"sample {sample_key} has incomplete audio metadata")
    try:
        duration_sec = float(row.get("duration_sec"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"sample {sample_key} has an invalid duration") from error
    if not math.isfinite(duration_sec) or duration_sec <= 0:
        raise ValueError(f"sample {sample_key} has a non-positive/non-finite duration")

    source_target = _prepared_target_json(row)
    country_value = source_target.get("country_code")
    if not isinstance(country_value, str) or not country_value.strip():
        raise ValueError(f"sample {sample_key} target_json has no country_code")
    # This field is deliberately taken from target_json, never from sample-key
    # geography or an auxiliary DID prediction.
    country_code = country_value.strip().upper()
    content_value = source_target.get("has_code_switching", False)
    if not isinstance(content_value, bool):
        raise ValueError(
            f"sample {sample_key} target_json.has_code_switching must be boolean"
        )
    content_has_code_switching = content_value

    sizes = tuple(int(size) for size in canvas_sizes)
    if not sizes or tuple(sorted(set(sizes))) != sizes or any(size <= 0 for size in sizes):
        raise ValueError("canvas_sizes must be sorted, unique and positive")
    if not math.isfinite(tokens_per_second) or tokens_per_second <= 0:
        raise ValueError("tokens_per_second must be positive and finite")
    if not math.isfinite(margin_tokens) or margin_tokens < 0:
        raise ValueError("margin_tokens must be non-negative and finite")
    estimated_tokens = int(math.ceil(tokens_per_second * duration_sec + margin_tokens))
    canvas_size = next((size for size in sizes if size >= estimated_tokens), None)
    if canvas_size is None:
        raise ValueError(
            f"sample {sample_key} duration estimate {estimated_tokens} exceeds "
            f"largest canvas {sizes[-1]}"
        )

    targets: list[str | None] = []
    valid: list[bool] = []
    applied_cs: list[bool] = []
    token_counts: list[int] = []
    selected_canvas_sizes: list[int] = []
    errors: list[str | None] = []
    for request_code_switching, request_diacritics in VIEW_REQUESTS:
        target: str | None = None
        token_count = -1
        error: str | None = None
        applied = False
        try:
            processed = process_clean_17k_row(
                row,
                use_code_switching=request_code_switching,
                use_diacritics=request_diacritics,
            )
            target = str(processed["transcription"])
            applied = bool(processed["use_code_switching"])
            # Keep content truth independent from whether a requested view was
            # applicable.  This also catches accidental preprocessing drift.
            if applied != bool(request_code_switching and content_has_code_switching):
                raise ValueError(
                    "clean preprocessing returned inconsistent code-switch application"
                )
            token_count = len(
                _token_ids(
                    tokenizer,
                    target_text_for_tokenizer(target, country_code),
                )
            )
            if token_count + 1 > canvas_size:
                error = (
                    f"target_overflow:requires={token_count + 1},"
                    f"canvas={canvas_size}"
                )
        except Exception as caught:  # per-view failure is signed into metadata
            if error is None:
                error = f"{type(caught).__name__}:{caught}"
        is_valid = error is None
        targets.append(target if is_valid else None)
        valid.append(is_valid)
        applied_cs.append(applied if is_valid else False)
        token_counts.append(token_count)
        selected_canvas_sizes.append(int(canvas_size))
        errors.append(error)

    return {
        "metadata_schema_version": CONDITIONAL_METADATA_SCHEMA_VERSION,
        "sample_key": sample_key,
        "audio_tar_path": audio_tar_path,
        "audio_member": audio_member,
        "duration_sec": duration_sec,
        "country_code": country_code,
        "content_has_code_switching": content_has_code_switching,
        **dict(zip(TARGET_COLUMNS, targets, strict=True)),
        "valid_view_mask": valid,
        "view_permutation": list(stable_view_permutation(sample_key, view_seed)),
        "applied_code_switching_mask": applied_cs,
        "target_token_counts": token_counts,
        "canvas_sizes": selected_canvas_sizes,
        "view_errors": errors,
    }


@dataclass(frozen=True, slots=True)
class ConditionalTarPartition:
    tar_path: str
    sidecar_path: str
    sidecar_sha256: str
    row_count: int
    any_valid_count: int
    per_view_valid_counts: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class ConditionalMetadataIndex:
    path: Path
    content_sha256: str
    repo_id: str
    revision: str
    source_metadata_sha256: str
    split_manifest_sha256: str
    tokenizer_sha256: str
    preprocessing_code_sha256: str
    total_train_rows: int
    any_valid_rows: int
    per_view_valid_counts: tuple[int, int, int, int]
    policy: dict[str, Any]
    partitions: tuple[ConditionalTarPartition, ...]
    raw: dict[str, Any]


def _require_sha256(value: Any, name: str) -> str:
    value = str(value)
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"conditional metadata index has invalid {name}")
    return value


def load_conditional_metadata_index(
    path: str | Path,
    *,
    validate_sidecar_hashes: bool = True,
) -> ConditionalMetadataIndex:
    """Authenticate a metadata index and all referenced sidecars."""

    resolved = Path(path).expanduser().resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"metadata index must be a regular file: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid conditional metadata index: {resolved}") from error
    if not isinstance(payload, dict):
        raise ValueError("conditional metadata index must be a JSON object")
    if payload.get("schema_version") != CONDITIONAL_INDEX_SCHEMA_VERSION:
        raise ValueError("unsupported conditional metadata index schema")
    content_hash = _require_sha256(payload.get("content_sha256"), "content_sha256")
    if canonical_json_sha256(payload) != content_hash:
        raise ValueError("conditional metadata index content hash mismatch")
    if payload.get("view_names") != list(VIEW_NAMES):
        raise ValueError("conditional metadata index view order mismatch")

    provenance = payload.get("provenance")
    totals = payload.get("totals")
    policy = payload.get("policy")
    raw_partitions = payload.get("partitions")
    if not isinstance(provenance, dict) or not isinstance(totals, dict):
        raise ValueError("conditional metadata index lacks provenance/totals")
    if not isinstance(policy, dict) or not isinstance(raw_partitions, list):
        raise ValueError("conditional metadata index lacks policy/partitions")

    partitions: list[ConditionalTarPartition] = []
    seen_tars: set[str] = set()
    seen_sidecars: set[str] = set()
    for raw in raw_partitions:
        if not isinstance(raw, dict):
            raise ValueError("conditional metadata partition must be an object")
        tar_path = str(raw.get("tar_path", ""))
        sidecar_path = str(raw.get("sidecar_path", ""))
        if not tar_path or Path(tar_path).is_absolute() or ".." in Path(tar_path).parts:
            raise ValueError(f"unsafe TAR path in metadata index: {tar_path!r}")
        if (
            not sidecar_path
            or Path(sidecar_path).is_absolute()
            or ".." in Path(sidecar_path).parts
        ):
            raise ValueError(f"unsafe sidecar path in metadata index: {sidecar_path!r}")
        if tar_path in seen_tars or sidecar_path in seen_sidecars:
            raise ValueError("duplicate TAR or sidecar in conditional metadata index")
        seen_tars.add(tar_path)
        seen_sidecars.add(sidecar_path)
        counts = tuple(int(value) for value in raw.get("per_view_valid_counts", ()))
        if len(counts) != VIEW_COUNT or any(value < 0 for value in counts):
            raise ValueError("invalid per-view partition counts")
        partition = ConditionalTarPartition(
            tar_path=tar_path,
            sidecar_path=sidecar_path,
            sidecar_sha256=_require_sha256(raw.get("sidecar_sha256"), "sidecar_sha256"),
            row_count=int(raw.get("row_count", -1)),
            any_valid_count=int(raw.get("any_valid_count", -1)),
            per_view_valid_counts=counts,  # type: ignore[arg-type]
        )
        if partition.row_count < 0 or not 0 <= partition.any_valid_count <= partition.row_count:
            raise ValueError("invalid conditional partition row counts")
        sidecar = resolved.parent / sidecar_path
        if not sidecar.is_file():
            raise FileNotFoundError(sidecar)
        if validate_sidecar_hashes and sha256_file(sidecar) != partition.sidecar_sha256:
            raise ValueError(f"conditional sidecar hash mismatch: {sidecar}")
        partitions.append(partition)

    total_rows = sum(item.row_count for item in partitions)
    any_valid = sum(item.any_valid_count for item in partitions)
    per_view = tuple(
        sum(item.per_view_valid_counts[index] for item in partitions)
        for index in range(VIEW_COUNT)
    )
    if int(totals.get("train_rows", -1)) != total_rows:
        raise ValueError("conditional metadata total train row count mismatch")
    if int(totals.get("any_valid_rows", -1)) != any_valid:
        raise ValueError("conditional metadata total valid row count mismatch")
    if tuple(totals.get("per_view_valid_counts", ())) != per_view:
        raise ValueError("conditional metadata total per-view counts mismatch")

    return ConditionalMetadataIndex(
        path=resolved,
        content_sha256=content_hash,
        repo_id=str(payload.get("repo_id", "")),
        revision=str(payload.get("revision", "")),
        source_metadata_sha256=_require_sha256(
            provenance.get("source_metadata_sha256"), "source_metadata_sha256"
        ),
        split_manifest_sha256=_require_sha256(
            provenance.get("split_manifest_sha256"), "split_manifest_sha256"
        ),
        tokenizer_sha256=_require_sha256(
            provenance.get("tokenizer_sha256"), "tokenizer_sha256"
        ),
        preprocessing_code_sha256=_require_sha256(
            provenance.get("preprocessing_code_sha256"),
            "preprocessing_code_sha256",
        ),
        total_train_rows=total_rows,
        any_valid_rows=any_valid,
        per_view_valid_counts=per_view,  # type: ignore[arg-type]
        policy=dict(policy),
        partitions=tuple(partitions),
        raw=dict(payload),
    )


__all__ = [
    "CONDITIONAL_INDEX_SCHEMA_VERSION",
    "CONDITIONAL_METADATA_SCHEMA_VERSION",
    "ConditionalMetadataIndex",
    "ConditionalTarPartition",
    "TARGET_COLUMNS",
    "VIEW_COUNT",
    "VIEW_NAMES",
    "VIEW_REQUESTS",
    "canonical_json_sha256",
    "conditional_metadata_schema",
    "load_conditional_metadata_index",
    "prepare_conditional_row",
    "select_view_id",
    "sha256_file",
    "stable_view_permutation",
    "target_text_for_tokenizer",
]
