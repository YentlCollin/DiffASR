"""Streaming data utilities with dependency-lazy public exports.

Inference-only environments do not necessarily install PyArrow. Importing this
package therefore must not import either streaming loader until its attribute is
actually requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULE = {
    "process_training_transcription": ".clean_17k_preprocessing_allowed_shadda",
    "process_clean_17k_row": ".clean_17k_preprocessing_allowed_shadda",
    "TargetFormatError": ".clean_17k_preprocessing_allowed_shadda",
    "CONDITIONAL_INDEX_SCHEMA_VERSION": ".conditional_metadata",
    "CONDITIONAL_METADATA_SCHEMA_VERSION": ".conditional_metadata",
    "TARGET_COLUMNS": ".conditional_metadata",
    "VIEW_NAMES": ".conditional_metadata",
    "VIEW_REQUESTS": ".conditional_metadata",
    "canonical_json_sha256": ".conditional_metadata",
    "conditional_metadata_schema": ".conditional_metadata",
    "load_conditional_metadata_index": ".conditional_metadata",
    "prepare_conditional_row": ".conditional_metadata",
    "select_view_id": ".conditional_metadata",
    "stable_view_permutation": ".conditional_metadata",
    "ConditionalDatasetCounters": ".conditional_tar_dataset",
    "ConditionalLocalTarDataset": ".conditional_tar_dataset",
    "EXCLUSION_MANIFEST_SCHEMA_VERSION": ".exclusions",
    "SampleExclusionManifest": ".exclusions",
    "canonical_sample_keys_sha256": ".exclusions",
    "exclusion_manifest_content_sha256": ".exclusions",
    "load_sample_exclusion_manifest": ".exclusions",
    "LocalTarParquetASRDataset": ".local_tar_dataset",
    "StreamingDatasetCounters": ".local_tar_dataset",
    "video_key_from_sample_key": ".local_tar_dataset",
}

__all__ = sorted(_EXPORT_MODULE)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
