"""MawJais prompt/canvas collation and device-side diffusion corruption."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch

from .canvas import (
    CanvasOverflowError,
    DurationBucketSelector,
    build_clean_canvas,
    pad_canvas_batch,
)
from .config import DiffusionConfig
from .conditional import conditional_view_from_sample
from .noise_process import DiscreteNoiseProcess, sample_diffusion_times


AUDIO_PLACEHOLDER = "<|ifm|audio|>"
NO_TRANSCRIPT_TARGET = "[IFM-NoTranscript]"


def build_mawjais_asr_prefix(system_tag: str = "") -> str:
    """Render the exact non-thinking R2 prefix used by validated inference."""

    return (
        "<|ifm|begin_of_text|><|ifm|im_start|>system\n"
        f"{system_tag}<|ifm|im_end|>"
        "<|ifm|im_start|>user\n"
        f"{AUDIO_PLACEHOLDER}<|ifm|im_end|>"
        "<|ifm|im_start|>assistant\n<ifm|think>\n</ifm|think>"
    )


def target_prompt_tag(
    target: Mapping[str, Any],
    *,
    request_diacritics: bool = True,
    request_code_switch_from_metadata: bool = True,
) -> str:
    tags = []
    if request_code_switch_from_metadata and bool(target.get("has_code_switching", False)):
        tags.append("[Code-switch]")
    if request_diacritics:
        tags.append("[Diacritics]")
    return " ".join(tags)


@dataclass(slots=True)
class DiffusionCollatorCounters:
    examples_seen: int = 0
    examples_emitted: int = 0
    duration_bucket_overflow: int = 0
    target_bucket_overflow: int = 0
    batches_fully_skipped: int = 0


@dataclass(frozen=True)
class NoisedCanvasBatch:
    noisy_canvas_ids: torch.Tensor
    clean_canvas_ids: torch.Tensor
    canvas_valid_mask: torch.Tensor
    corrupt_mask: torch.Tensor
    noise_level: torch.Tensor


class MawJaisDiffusionCollator:
    """Convert streamed ASR records into clean prefixes and duration buckets.

    Corruption is intentionally not performed here: DataLoader workers run on
    CPU, whereas the training step should create noise on the target device with
    explicit, checkpointable RNG streams.
    """

    def __init__(
        self,
        processor: Any,
        diffusion_config: DiffusionConfig,
        *,
        overflow_policy: Literal["raise", "skip"] = "raise",
        request_diacritics: bool = True,
        request_code_switch_from_metadata: bool = True,
        conditional_mode: bool = False,
    ) -> None:
        if overflow_policy not in ("raise", "skip"):
            raise ValueError("overflow_policy must be 'raise' or 'skip'")
        self.processor = processor
        self.diffusion_config = diffusion_config
        self.overflow_policy = overflow_policy
        self.request_diacritics = bool(request_diacritics)
        self.request_code_switch_from_metadata = bool(
            request_code_switch_from_metadata
        )
        self.conditional_mode = bool(conditional_mode)
        canvas = diffusion_config.canvas
        self.bucket_selector = DurationBucketSelector(
            canvas.sizes,
            tokens_per_second=canvas.duration_tokens_per_second,
            margin_tokens=canvas.duration_margin_tokens,
        )
        self.counters = DiffusionCollatorCounters()

    def _target_ids_and_text(
        self, sample: Mapping[str, Any]
    ) -> tuple[list[int], str, str]:
        if self.conditional_mode:
            transcription = sample.get("target_transcription")
            legacy_transcription = sample.get("transcription")
            if (
                legacy_transcription is not None
                and transcription is not None
                and legacy_transcription != transcription
            ):
                raise ValueError(
                    "conditional sample transcription disagrees with target_transcription"
                )
        else:
            transcription = sample.get("transcription")
        country_code = sample.get("country_code")
        if not isinstance(transcription, str) or not transcription.strip():
            field = "target_transcription" if self.conditional_mode else "transcription"
            raise ValueError(f"sample {field} must be a non-empty string")
        transcription = transcription.strip()
        if transcription == NO_TRANSCRIPT_TARGET:
            # R2 uses this sentinel as the complete assistant answer; adding a
            # country prefix would silently change its task semantics.
            target_text = NO_TRANSCRIPT_TARGET
        else:
            if not isinstance(country_code, str) or not country_code.strip():
                raise ValueError("sample country_code must be a non-empty string")
            target_text = f"[{country_code.strip()}] {transcription}"
        encoded = self.processor.tokenizer(
            target_text,
            add_special_tokens=False,
        )["input_ids"]
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.flatten().tolist()
        elif isinstance(encoded, np.ndarray):
            encoded = encoded.reshape(-1).tolist()
        elif encoded and isinstance(encoded[0], list):
            if len(encoded) != 1:
                raise ValueError("tokenizer returned an unexpected target batch")
            encoded = encoded[0]
        return [int(token_id) for token_id in encoded], target_text, transcription

    @staticmethod
    def _conditional_metadata(sample: Mapping[str, Any]) -> dict[str, Any]:
        """Validate request/content provenance without deriving one from another."""

        view = conditional_view_from_sample(sample)
        for field in ("content_has_code_switching", "applied_code_switching"):
            if field not in sample:
                raise ValueError(f"conditional sample is missing {field}")
            if not isinstance(sample[field], bool):
                raise TypeError(f"conditional sample {field} must be bool")
        content = sample["content_has_code_switching"]
        applied = sample["applied_code_switching"]
        if applied and not (content and view.request_code_switching):
            raise ValueError(
                "applied_code_switching=true requires both code-switched content "
                "and request_code_switching=true"
            )
        target = sample.get("target")
        if isinstance(target, Mapping):
            nested_country = target.get("country_code")
            if nested_country is not None and nested_country != sample.get("country_code"):
                raise ValueError(
                    "conditional sample country_code disagrees with target.country_code"
                )
            expected_nested = {
                "target_transcription": sample.get("target_transcription"),
                "request_code_switching": view.request_code_switching,
                "request_diacritics": view.request_diacritics,
                "content_has_code_switching": content,
                "has_code_switching": content,
                "applied_code_switching": applied,
                "view_id": view.view_id,
                "view_name": view.name,
            }
            for field, expected in expected_nested.items():
                if field in target and target[field] != expected:
                    raise ValueError(
                        f"conditional sample {field} disagrees with target.{field}"
                    )
        return {
            "view_id": view.view_id,
            "view_name": view.name,
            "prompt_tag": view.prompt_tag,
            "request_code_switching": view.request_code_switching,
            "request_diacritics": view.request_diacritics,
            "content_has_code_switching": content,
            "applied_code_switching": applied,
        }

    @staticmethod
    def _validate_precomputed_selection(
        sample: Mapping[str, Any],
        *,
        view_id: int,
        selected_canvas_size: int,
        actual_target_token_count: int,
    ) -> None:
        """Fail closed if signed preprocessing metadata drifts at runtime."""

        expected_canvas_size = sample.get("canvas_size")
        if expected_canvas_size is not None and (
            isinstance(expected_canvas_size, bool)
            or int(expected_canvas_size) != selected_canvas_size
        ):
            raise ValueError(
                "conditional sample canvas_size disagrees with duration-selected "
                f"bucket: metadata={expected_canvas_size!r}, runtime={selected_canvas_size}"
            )
        expected_token_count = sample.get("target_token_count")
        if expected_token_count is not None and (
            isinstance(expected_token_count, bool)
            or int(expected_token_count) != actual_target_token_count
        ):
            raise ValueError(
                "conditional sample target_token_count disagrees with runtime tokenizer: "
                f"metadata={expected_token_count!r}, runtime={actual_target_token_count}"
            )
        valid_view_mask = sample.get("valid_view_mask")
        if valid_view_mask is not None:
            if not isinstance(valid_view_mask, (list, tuple)) or len(valid_view_mask) != 4:
                raise ValueError("conditional sample valid_view_mask must have length four")
            if not all(isinstance(value, bool) for value in valid_view_mask):
                raise TypeError("conditional sample valid_view_mask entries must be bool")
            if not valid_view_mask[view_id]:
                raise ValueError("conditional sample selected an invalid view_id")

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("cannot collate an empty sample list")

        retained: list[Mapping[str, Any]] = []
        canvases = []
        canvas_sizes = []
        target_texts = []
        target_transcriptions = []
        prompt_tags = []
        conditional_metadata: list[dict[str, Any]] = []
        batch_duration_overflow = 0
        batch_target_overflow = 0
        for sample in samples:
            self.counters.examples_seen += 1
            duration = float(sample["duration_sec"])
            choice = self.bucket_selector.select(duration)
            if choice.overflow:
                self.counters.duration_bucket_overflow += 1
                batch_duration_overflow += 1
                if self.overflow_policy == "raise":
                    raise CanvasOverflowError(
                        f"duration estimate {choice.estimated_tokens} has no canvas bucket "
                        f"for sample {sample.get('sample_key')!r}"
                    )
                continue
            assert choice.canvas_size is not None
            metadata = (
                self._conditional_metadata(sample) if self.conditional_mode else None
            )
            target_ids, target_text, target_transcription = self._target_ids_and_text(
                sample
            )
            if metadata is not None:
                self._validate_precomputed_selection(
                    sample,
                    view_id=int(metadata["view_id"]),
                    selected_canvas_size=choice.canvas_size,
                    actual_target_token_count=len(target_ids),
                )
            try:
                canvas = build_clean_canvas(
                    target_ids,
                    choice.canvas_size,
                    tail_fill_mode=self.diffusion_config.canvas.tail_fill_mode,
                    turn_end_token_id=self.diffusion_config.canvas.turn_end_token_id,
                    pad_token_id=self.diffusion_config.canvas.pad_token_id,
                )
            except CanvasOverflowError:
                self.counters.target_bucket_overflow += 1
                batch_target_overflow += 1
                if self.overflow_policy == "raise":
                    raise CanvasOverflowError(
                        f"target does not fit duration-selected bucket {choice.canvas_size} "
                        f"for sample {sample.get('sample_key')!r}; it was not promoted or truncated"
                    )
                continue
            retained.append(sample)
            canvases.append(canvas)
            canvas_sizes.append(choice.canvas_size)
            target_texts.append(target_text)
            target_transcriptions.append(target_transcription)
            if metadata is not None:
                conditional_metadata.append(metadata)
                prompt_tags.append(metadata["prompt_tag"])
            else:
                prompt_tags.append(
                    target_prompt_tag(
                        sample.get("target", sample),
                        request_diacritics=self.request_diacritics,
                        request_code_switch_from_metadata=self.request_code_switch_from_metadata,
                    )
                )

        if not retained:
            if self.overflow_policy == "skip":
                self.counters.batches_fully_skipped += 1
                return {
                    "skip_batch": True,
                    "skipped_sample_keys": [
                        str(sample.get("sample_key", "")) for sample in samples
                    ],
                    "collator_metrics": {
                        "examples_seen": len(samples),
                        "examples_emitted": 0,
                        "duration_bucket_overflow": batch_duration_overflow,
                        "target_bucket_overflow": batch_target_overflow,
                        "batches_fully_skipped": 1,
                    },
                }
            raise CanvasOverflowError("every sample in this batch overflowed its duration bucket")
        prompts = [build_mawjais_asr_prefix(tag) for tag in prompt_tags]
        audio = [sample["audio"] for sample in retained]
        processor_output = self.processor(
            text=prompts,
            audio=audio,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        required = {
            "input_ids",
            "attention_mask",
            "audio_position_mask",
            "input_features",
            "feature_attention_mask",
        }
        missing = required.difference(processor_output)
        if missing:
            raise ValueError(f"processor output is missing diffusion fields: {sorted(missing)}")
        if processor_output["audio_position_mask"].dtype != torch.bool:
            raise TypeError("processor audio_position_mask must be torch.bool")

        clean_canvas_ids, canvas_valid_mask = pad_canvas_batch(
            canvases, pad_token_id=self.diffusion_config.canvas.pad_token_id
        )
        self.counters.examples_emitted += len(retained)
        batch = {
            "skip_batch": False,
            "collator_metrics": {
                "examples_seen": len(samples),
                "examples_emitted": len(retained),
                "duration_bucket_overflow": batch_duration_overflow,
                "target_bucket_overflow": batch_target_overflow,
                "batches_fully_skipped": 0,
            },
            "prefix_input_ids": processor_output["input_ids"],
            "prefix_attention_mask": processor_output["attention_mask"].to(torch.bool),
            "audio_position_mask": processor_output["audio_position_mask"],
            "input_features": processor_output["input_features"],
            "feature_attention_mask": processor_output["feature_attention_mask"],
            "clean_canvas_ids": clean_canvas_ids,
            "canvas_valid_mask": canvas_valid_mask,
            "canvas_sizes": torch.tensor(canvas_sizes, dtype=torch.long),
            "duration_seconds": torch.tensor(
                [float(sample["duration_sec"]) for sample in retained], dtype=torch.float32
            ),
            "sample_keys": [str(sample.get("sample_key", "")) for sample in retained],
            "target_texts": target_texts,
            "target_transcriptions": target_transcriptions,
            "prompt_tags": prompt_tags,
        }
        if self.conditional_mode:
            batch.update(
                {
                    "view_ids": torch.tensor(
                        [item["view_id"] for item in conditional_metadata],
                        dtype=torch.long,
                    ),
                    "view_names": [
                        str(item["view_name"]) for item in conditional_metadata
                    ],
                    "request_code_switching": torch.tensor(
                        [
                            item["request_code_switching"]
                            for item in conditional_metadata
                        ],
                        dtype=torch.bool,
                    ),
                    "request_diacritics": torch.tensor(
                        [item["request_diacritics"] for item in conditional_metadata],
                        dtype=torch.bool,
                    ),
                    "content_has_code_switching": torch.tensor(
                        [
                            item["content_has_code_switching"]
                            for item in conditional_metadata
                        ],
                        dtype=torch.bool,
                    ),
                    "applied_code_switching": torch.tensor(
                        [item["applied_code_switching"] for item in conditional_metadata],
                        dtype=torch.bool,
                    ),
                }
            )
        return batch


def corrupt_diffusion_batch(
    clean_canvas_ids: torch.Tensor,
    canvas_valid_mask: torch.Tensor,
    noise_process: DiscreteNoiseProcess,
    *,
    min_diffusion_time: float,
    full_noise_probability: float = 0.0,
    pad_token_id: int = 1,
    time_generator: torch.Generator | None = None,
    corruption_generator: torch.Generator | None = None,
    token_noise_generator: torch.Generator | None = None,
) -> NoisedCanvasBatch:
    """Create ``xt`` with independent RNG streams for time/mask/token prior."""

    if clean_canvas_ids.ndim != 2:
        raise ValueError("clean_canvas_ids must have shape [batch, canvas]")
    if canvas_valid_mask.shape != clean_canvas_ids.shape or canvas_valid_mask.dtype != torch.bool:
        raise ValueError("canvas_valid_mask must be boolean and match clean_canvas_ids")
    if not 0 < min_diffusion_time <= 1:
        raise ValueError("min_diffusion_time must be in (0, 1]")
    times = sample_diffusion_times(
        clean_canvas_ids.shape[0],
        device=clean_canvas_ids.device,
        generator=time_generator,
        full_noise_probability=full_noise_probability,
    ).clamp_min(min_diffusion_time)
    noisy, corrupt_mask = noise_process.q_sample(
        clean_canvas_ids,
        times,
        generator=corruption_generator,
        noise_generator=token_noise_generator,
    )
    corrupt_mask = corrupt_mask & canvas_valid_mask
    noisy = noisy.masked_fill(~canvas_valid_mask, int(pad_token_id))
    return NoisedCanvasBatch(
        noisy_canvas_ids=noisy,
        clean_canvas_ids=clean_canvas_ids,
        canvas_valid_mask=canvas_valid_mask,
        corrupt_mask=corrupt_mask,
        noise_level=times,
    )


__all__ = [
    "DiffusionCollatorCounters",
    "MawJaisDiffusionCollator",
    "NO_TRANSCRIPT_TARGET",
    "NoisedCanvasBatch",
    "build_mawjais_asr_prefix",
    "corrupt_diffusion_batch",
    "target_prompt_tag",
]
