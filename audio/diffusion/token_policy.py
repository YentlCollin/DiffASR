"""Reproducible MawJais output-token and Uniform-prior policies."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

import torch

from .config import TailFillMode


TURN_END_TOKEN_TEXT = "<|ifm|im_end|>"
AUDIO_TOKEN_TEXT = "<|ifm|audio|>"
# the user-supplied base model uses literal XML-like audio delimiters around the repeated
# ``<|ifm|audio|>`` placeholders. Resolve their IDs from the loaded
# tokenizer; do not infer their spelling from the config field names.
AUDIO_START_TOKEN_TEXT = "<audio>"
AUDIO_END_TOKEN_TEXT = "</audio>"


def _normalise_ids(ids: Iterable[int], *, vocab_size: int, name: str) -> tuple[int, ...]:
    values = tuple(sorted(set(int(token_id) for token_id in ids)))
    if values and (values[0] < 0 or values[-1] >= vocab_size):
        raise ValueError(f"{name} contains an ID outside [0, {vocab_size})")
    return values


def hash_token_ids(ids: Sequence[int]) -> str:
    """Hash an ordered ID table without depending on platform integer layout."""

    payload = "\n".join(str(int(token_id)) for token_id in ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class DiffusionTokenPolicy:
    """Exact token sets that must be saved next to every resolved run."""

    vocab_size: int
    base_vocab_size: int
    uniform_vocab_scope: Literal["full_vocab", "safe_vocab"]
    uniform_noise_ids: tuple[int, ...]
    training_forbidden_ids: tuple[int, ...]
    sampling_forbidden_ids: tuple[int, ...]
    allowed_added_output_ids: tuple[int, ...]

    @property
    def uniform_noise_ids_sha256(self) -> str:
        return hash_token_ids(self.uniform_noise_ids)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "base_vocab_size": self.base_vocab_size,
            "uniform_vocab_scope": self.uniform_vocab_scope,
            "uniform_noise_id_count": len(self.uniform_noise_ids),
            "uniform_noise_ids_sha256": self.uniform_noise_ids_sha256,
            "training_forbidden_ids": list(self.training_forbidden_ids),
            "sampling_forbidden_ids": list(self.sampling_forbidden_ids),
            "allowed_added_output_ids": list(self.allowed_added_output_ids),
        }

    def uniform_noise_tensor(self, *, device: torch.device | str | None = None) -> torch.Tensor:
        return torch.tensor(self.uniform_noise_ids, dtype=torch.long, device=device)


@dataclass(frozen=True)
class MawJaisTokenContract:
    """Tokenizer/processor/model IDs verified from the loaded the user-supplied base model snapshot."""

    mask_token_text: str
    mask_token_id: int
    turn_end_token_id: int
    pad_token_id: int
    audio_token_id: int
    audio_start_token_id: int
    audio_end_token_id: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "mask_token_text": self.mask_token_text,
            "mask_token_id": self.mask_token_id,
            "turn_end_token_text": TURN_END_TOKEN_TEXT,
            "turn_end_token_id": self.turn_end_token_id,
            "pad_token_id": self.pad_token_id,
            "audio_token_text": AUDIO_TOKEN_TEXT,
            "audio_token_id": self.audio_token_id,
            "audio_start_token_text": AUDIO_START_TOKEN_TEXT,
            "audio_start_token_id": self.audio_start_token_id,
            "audio_end_token_text": AUDIO_END_TOKEN_TEXT,
            "audio_end_token_id": self.audio_end_token_id,
        }


def _resolve_exact_token_id(tokenizer: Any, token_text: str) -> int:
    if not hasattr(tokenizer, "convert_tokens_to_ids"):
        raise TypeError("loaded tokenizer does not expose convert_tokens_to_ids()")
    token_id = tokenizer.convert_tokens_to_ids(token_text)
    if not isinstance(token_id, int):
        raise ValueError(f"tokenizer did not resolve {token_text!r} to one integer ID")
    return token_id


def validate_mawjais_token_contract(
    tokenizer: Any,
    *,
    processor: Any,
    model_config: Any,
    mask_token_text: str,
    mask_token_id: int,
    pad_token_id: int,
    turn_end_token_id: int,
    forbidden_output_token_ids: Sequence[int],
) -> MawJaisTokenContract:
    """Cross-check every diffusion-sensitive control token at runtime.

    IDs are resolved from the tokenizer and compared with both processor and
    model configuration. This prevents a config copied from another revision
    from silently corrupting audio injection or sequence termination.
    """

    if not isinstance(mask_token_text, str) or not mask_token_text:
        raise ValueError("diffusion.mask_token_text must be a non-empty string")
    resolved = {
        "mask": _resolve_exact_token_id(tokenizer, mask_token_text),
        "turn_end": _resolve_exact_token_id(tokenizer, TURN_END_TOKEN_TEXT),
        "audio": _resolve_exact_token_id(tokenizer, AUDIO_TOKEN_TEXT),
        "audio_start": _resolve_exact_token_id(tokenizer, AUDIO_START_TOKEN_TEXT),
        "audio_end": _resolve_exact_token_id(tokenizer, AUDIO_END_TOKEN_TEXT),
    }
    expected = {
        "mask": int(mask_token_id),
        "turn_end": int(turn_end_token_id),
    }
    for name, expected_id in expected.items():
        if resolved[name] != expected_id:
            token_text = mask_token_text if name == "mask" else TURN_END_TOKEN_TEXT
            raise ValueError(
                f"tokenizer token {token_text!r} resolves to {resolved[name]}, "
                f"but the configured {name} ID is {expected_id}"
            )

    tokenizer_pad_id = getattr(tokenizer, "pad_token_id", None)
    if not isinstance(tokenizer_pad_id, int) or tokenizer_pad_id != int(pad_token_id):
        raise ValueError(
            f"tokenizer.pad_token_id={tokenizer_pad_id!r} does not match configured "
            f"PAD ID {int(pad_token_id)}"
        )
    processor_audio_id = getattr(processor, "audio_token_id", None)
    if processor_audio_id != resolved["audio"]:
        raise ValueError(
            f"processor.audio_token_id={processor_audio_id!r} does not match tokenizer "
            f"{AUDIO_TOKEN_TEXT!r} ID {resolved['audio']}"
        )
    model_audio_fields = {
        "audio_token_id": "audio",
        "audio_start_token_id": "audio_start",
        "audio_end_token_id": "audio_end",
    }
    for field, name in model_audio_fields.items():
        model_id = getattr(model_config, field, None)
        if model_id != resolved[name]:
            raise ValueError(
                f"model config {field}={model_id!r} does not match tokenizer ID "
                f"{resolved[name]}"
            )

    distinct_ids = {
        resolved["mask"],
        resolved["turn_end"],
        int(pad_token_id),
        resolved["audio"],
        resolved["audio_start"],
        resolved["audio_end"],
    }
    if len(distinct_ids) != 6:
        raise ValueError(
            "MASK, TURN_END, PAD, audio placeholder, and audio delimiters must "
            "have distinct token IDs"
        )
    forbidden = {int(token_id) for token_id in forbidden_output_token_ids}
    required_forbidden = {
        resolved["mask"],
        resolved["audio"],
        resolved["audio_start"],
        resolved["audio_end"],
    }
    missing_forbidden = sorted(required_forbidden - forbidden)
    if missing_forbidden:
        raise ValueError(
            "diffusion output policy does not forbid MASK/audio control IDs: "
            f"{missing_forbidden}"
        )
    if resolved["turn_end"] in forbidden:
        raise ValueError("assistant TURN_END must remain an allowed diffusion output")

    return MawJaisTokenContract(
        mask_token_text=mask_token_text,
        mask_token_id=resolved["mask"],
        turn_end_token_id=resolved["turn_end"],
        pad_token_id=int(pad_token_id),
        audio_token_id=resolved["audio"],
        audio_start_token_id=resolved["audio_start"],
        audio_end_token_id=resolved["audio_end"],
    )


def build_diffusion_token_policy(
    tokenizer: Any,
    *,
    model_vocab_size: int,
    uniform_vocab_scope: Literal["full_vocab", "safe_vocab"],
    allowed_added_output_token_ids: Sequence[int],
    extra_forbidden_output_token_ids: Sequence[int],
    mask_token_id: int,
    tail_fill_mode: TailFillMode,
    pad_token_id: int,
    turn_end_token_id: int,
) -> DiffusionTokenPolicy:
    """Resolve control IDs from the loaded tokenizer, never from assumptions.

    ``safe_vocab`` contains ordinary base-vocabulary IDs plus explicitly
    allowed added output IDs. ``full_vocab`` deliberately retains every model
    row as the Uniform corruption prior; output masking remains independent.
    PAD is allowed only in ``eos_then_pad``, where it is a semantic target.
    """

    if uniform_vocab_scope not in ("full_vocab", "safe_vocab"):
        raise ValueError(f"unsupported uniform_vocab_scope: {uniform_vocab_scope!r}")
    if tail_fill_mode not in ("eos_repeat", "eos_then_pad"):
        raise ValueError(f"unsupported tail_fill_mode: {tail_fill_mode!r}")
    if model_vocab_size <= 0:
        raise ValueError("model_vocab_size must be positive")

    base_vocab_size = int(getattr(tokenizer, "vocab_size"))
    if not 0 < base_vocab_size <= model_vocab_size:
        raise ValueError("tokenizer.vocab_size is incompatible with the model vocabulary")
    if len(tokenizer) != model_vocab_size:
        raise ValueError(
            f"tokenizer length {len(tokenizer)} != model vocabulary {model_vocab_size}"
        )

    added_vocab: Mapping[str, int] = tokenizer.get_added_vocab()
    added_ids = _normalise_ids(
        added_vocab.values(), vocab_size=model_vocab_size, name="tokenizer added vocabulary"
    )
    allowed_added = _normalise_ids(
        allowed_added_output_token_ids,
        vocab_size=model_vocab_size,
        name="allowed_added_output_token_ids",
    )
    if turn_end_token_id not in allowed_added:
        raise ValueError("the MawJais assistant turn-end token must remain an allowed output")

    extra_forbidden = set(
        _normalise_ids(
            extra_forbidden_output_token_ids,
            vocab_size=model_vocab_size,
            name="extra_forbidden_output_token_ids",
        )
    )
    forbidden = (set(added_ids) - set(allowed_added)) | extra_forbidden | {int(mask_token_id)}
    forbidden.discard(int(turn_end_token_id))
    if tail_fill_mode == "eos_then_pad":
        forbidden.discard(int(pad_token_id))
    else:
        forbidden.add(int(pad_token_id))
    sampling_forbidden = _normalise_ids(
        forbidden, vocab_size=model_vocab_size, name="sampling forbidden IDs"
    )

    # The current objectives apply the same safety mask during training. A
    # batch-level assertion in objectives.py rejects any accidental clean
    # target/control-token collision rather than creating an infinite CE.
    training_forbidden = sampling_forbidden

    if uniform_vocab_scope == "full_vocab":
        uniform_noise_ids = tuple(range(model_vocab_size))
    else:
        safe = (set(range(base_vocab_size)) | set(allowed_added)) - set(sampling_forbidden)
        uniform_noise_ids = tuple(sorted(safe))
        if not uniform_noise_ids:
            raise ValueError("safe Uniform prior is empty")

    return DiffusionTokenPolicy(
        vocab_size=model_vocab_size,
        base_vocab_size=base_vocab_size,
        uniform_vocab_scope=uniform_vocab_scope,
        uniform_noise_ids=uniform_noise_ids,
        training_forbidden_ids=training_forbidden,
        sampling_forbidden_ids=sampling_forbidden,
        allowed_added_output_ids=allowed_added,
    )


__all__ = [
    "AUDIO_END_TOKEN_TEXT",
    "AUDIO_START_TOKEN_TEXT",
    "AUDIO_TOKEN_TEXT",
    "DiffusionTokenPolicy",
    "MawJaisTokenContract",
    "TURN_END_TOKEN_TEXT",
    "build_diffusion_token_policy",
    "hash_token_ids",
    "validate_mawjais_token_contract",
]
