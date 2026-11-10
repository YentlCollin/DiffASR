"""Signed sample-exclusion manifests for deterministic distributed streaming.

The manifest is intentionally small and contains only sample keys.  It is read
before TAR sample counts are computed, so distributed quotas describe examples
that can actually reach the collator.  This avoids the DDP deadlock/data-loss
failure mode where one rank discovers a target overflow after sharding.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


EXCLUSION_MANIFEST_SCHEMA_VERSION = 1


def canonical_sample_keys_sha256(sample_keys: Sequence[str]) -> str:
    """Hash a sorted, duplicate-free sequence using an unambiguous encoding."""

    ordered = sorted(str(key) for key in sample_keys)
    digest = hashlib.sha256()
    for key in ordered:
        encoded = key.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def exclusion_manifest_content_sha256(payload: Mapping[str, Any]) -> str:
    """Hash every manifest field except the self-referential content hash."""

    unsigned = dict(payload)
    unsigned.pop("content_sha256", None)
    encoded = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SampleExclusionManifest:
    path: Path
    dataset_revision: str
    subset_manifest_sha256: str
    sample_keys: frozenset[str]
    sample_keys_sha256: str
    content_sha256: str
    reason: str
    policy: dict[str, Any]

    @property
    def count(self) -> int:
        return len(self.sample_keys)


def load_sample_exclusion_manifest(
    path: str | Path,
    *,
    expected_dataset_revision: str,
    expected_subset_manifest_sha256: str,
    expected_count: int | None = None,
) -> SampleExclusionManifest:
    """Load and authenticate one exact exclusion set.

    The dataset revision and subset-manifest digest bind the keys to immutable
    metadata.  Both the key-list digest and whole-document digest are checked so
    editing either policy metadata or a key is always observable.
    """

    resolved = Path(path).expanduser().resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"sample exclusion manifest must be a regular file: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid sample exclusion manifest: {resolved}") from error
    required = {
        "schema_version",
        "reason",
        "dataset_revision",
        "subset_manifest_sha256",
        "policy",
        "excluded_sample_count",
        "excluded_sample_keys",
        "excluded_sample_keys_sha256",
        "content_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError(
            "sample exclusion manifest has an unexpected schema: "
            f"expected={sorted(required)}, actual={sorted(payload) if isinstance(payload, dict) else type(payload)}"
        )
    if payload["schema_version"] != EXCLUSION_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported sample exclusion manifest schema")
    if payload["dataset_revision"] != expected_dataset_revision:
        raise ValueError("sample exclusion manifest dataset revision mismatch")
    if payload["subset_manifest_sha256"] != expected_subset_manifest_sha256:
        raise ValueError("sample exclusion manifest subset hash mismatch")
    keys = payload["excluded_sample_keys"]
    if not isinstance(keys, list) or not all(isinstance(key, str) and key for key in keys):
        raise ValueError("excluded_sample_keys must be a list of non-empty strings")
    if keys != sorted(keys) or len(set(keys)) != len(keys):
        raise ValueError("excluded_sample_keys must be sorted and duplicate-free")
    if payload["excluded_sample_count"] != len(keys):
        raise ValueError("excluded_sample_count does not match excluded_sample_keys")
    if expected_count is not None and len(keys) != int(expected_count):
        raise ValueError(
            f"sample exclusion count mismatch: manifest={len(keys)}, expected={expected_count}"
        )
    keys_hash = canonical_sample_keys_sha256(keys)
    if payload["excluded_sample_keys_sha256"] != keys_hash:
        raise ValueError("excluded sample-key hash mismatch")
    content_hash = exclusion_manifest_content_sha256(payload)
    if payload["content_sha256"] != content_hash:
        raise ValueError("sample exclusion manifest content hash mismatch")
    if not isinstance(payload["reason"], str) or not payload["reason"]:
        raise ValueError("sample exclusion manifest reason must be non-empty")
    if not isinstance(payload["policy"], dict):
        raise ValueError("sample exclusion manifest policy must be an object")
    for value in (keys_hash, content_hash, payload["subset_manifest_sha256"]):
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("sample exclusion manifest contains an invalid SHA-256")
    return SampleExclusionManifest(
        path=resolved,
        dataset_revision=str(payload["dataset_revision"]),
        subset_manifest_sha256=str(payload["subset_manifest_sha256"]),
        sample_keys=frozenset(keys),
        sample_keys_sha256=keys_hash,
        content_sha256=content_hash,
        reason=str(payload["reason"]),
        policy=dict(payload["policy"]),
    )


__all__ = [
    "EXCLUSION_MANIFEST_SCHEMA_VERSION",
    "SampleExclusionManifest",
    "canonical_sample_keys_sha256",
    "exclusion_manifest_content_sha256",
    "load_sample_exclusion_manifest",
]
