"""PyTorch diffusion wrapper around the existing MawJais AR composite model.

The wrapper owns only diffusion-specific parameters. The original model remains
callable through ``autoregressive_model`` and its historical ``forward`` is not
replaced. This separation makes AR regression checks explicit and prevents the
standard causal-LM label shift from entering the denoising objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .alignment import DenoiserOutputAlignment, align_denoiser_logits
from .attention import (
    build_prefix_canvas_attention_mask,
    build_prefix_canvas_position_ids,
)
from .config import DiffusionConfig
from .noise_process import validate_diffusion_time
from .objectives import DenoisingLossOutput, diffusion_loss
from .time_embedding import TimeConditioningMLP


@dataclass(frozen=True)
class DiffusionForwardOutput:
    """Aligned denoiser output and optional decomposed training loss."""

    logits: torch.Tensor
    raw_logits: torch.Tensor
    loss: torch.Tensor | None
    loss_output: DenoisingLossOutput | None
    prefix_lengths: torch.Tensor
    canvas_lengths: torch.Tensor
    projected_audio_embeddings: torch.Tensor


def _validate_left_contiguous(mask: torch.Tensor, name: str) -> torch.Tensor:
    if mask.ndim != 2 or mask.dtype != torch.bool:
        raise ValueError(f"{name} must be a boolean [batch, sequence] tensor")
    lengths = mask.sum(dim=1)
    expected = torch.arange(mask.shape[1], device=mask.device)[None, :] < lengths[:, None]
    if not torch.equal(mask, expected):
        raise ValueError(f"{name} must be left-contiguous (True positions followed by padding)")
    return lengths


def pack_prefix_canvas_embeddings(
    prefix_embeddings: torch.Tensor,
    prefix_attention_mask: torch.Tensor,
    canvas_embeddings: torch.Tensor,
    canvas_valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack each row as ``[real prefix][allocated canvas][batch padding]``.

    Prefixes may have different lengths in their processor batch. Canvas tokens
    must never be placed after that prefix padding because doing so would create
    artificial RoPE gaps and visible padding keys.
    """

    if prefix_embeddings.ndim != 3 or canvas_embeddings.ndim != 3:
        raise ValueError("prefix_embeddings and canvas_embeddings must be rank-three")
    if prefix_embeddings.shape[0] != canvas_embeddings.shape[0]:
        raise ValueError("prefix and canvas batch sizes differ")
    if prefix_embeddings.shape[2] != canvas_embeddings.shape[2]:
        raise ValueError("prefix and canvas hidden sizes differ")
    if prefix_attention_mask.shape != prefix_embeddings.shape[:2]:
        raise ValueError("prefix_attention_mask does not match prefix embeddings")
    if canvas_valid_mask.shape != canvas_embeddings.shape[:2]:
        raise ValueError("canvas_valid_mask does not match canvas embeddings")
    if prefix_attention_mask.device != prefix_embeddings.device:
        raise ValueError("prefix_attention_mask and embeddings must share a device")
    if canvas_valid_mask.device != canvas_embeddings.device:
        raise ValueError("canvas_valid_mask and embeddings must share a device")

    prefix_lengths = prefix_attention_mask.to(torch.bool).sum(dim=1)
    if (prefix_lengths <= 0).any():
        raise ValueError("every example requires a non-empty prefix")
    canvas_lengths = _validate_left_contiguous(canvas_valid_mask, "canvas_valid_mask")
    if (canvas_lengths <= 0).any():
        raise ValueError("every example requires at least one allocated canvas position")

    total_lengths = prefix_lengths + canvas_lengths
    max_length = int(total_lengths.max().item())
    packed = prefix_embeddings.new_zeros(
        (prefix_embeddings.shape[0], max_length, prefix_embeddings.shape[2])
    )
    for row in range(prefix_embeddings.shape[0]):
        prefix = prefix_embeddings[row, prefix_attention_mask[row].to(torch.bool)]
        prefix_length = int(prefix_lengths[row].item())
        canvas_length = int(canvas_lengths[row].item())
        packed[row, :prefix_length] = prefix
        packed[row, prefix_length : prefix_length + canvas_length] = canvas_embeddings[
            row, :canvas_length
        ]
    return packed, prefix_lengths, canvas_lengths


class MawJaisDiffusionASRModel(nn.Module):
    """Add a denoising forward while preserving the loaded AR model intact."""

    def __init__(
        self,
        autoregressive_model: nn.Module,
        diffusion_config: DiffusionConfig,
        *,
        normal_text_token_count: int = 250_000,
        time_feature_dim: int = 128,
        time_embedding_seed: int = 3441,
    ) -> None:
        super().__init__()
        self.autoregressive_model = autoregressive_model
        self.diffusion_config = diffusion_config

        embeddings = autoregressive_model.get_input_embeddings().weight
        vocab_size, hidden_size = embeddings.shape
        if not 0 < normal_text_token_count <= vocab_size:
            raise ValueError("normal_text_token_count must fall inside the embedding table")
        if not 0 <= diffusion_config.mask_token_id < vocab_size:
            raise ValueError("mask_token_id falls outside the loaded MawJais vocabulary")

        # The reserved checkpoint row is nearly zero. A dedicated parameter avoids
        # unfreezing the 641M-parameter embedding table merely to train one input.
        with torch.no_grad():
            # Avoid materializing the whole 250000 x 2560 table in float32
            # (about 2.4 GiB) merely to compute one mean vector.
            mask_sum = torch.zeros(hidden_size, device=embeddings.device, dtype=torch.float32)
            chunk_size = 8192
            for start in range(0, normal_text_token_count, chunk_size):
                chunk = embeddings[
                    start : min(start + chunk_size, normal_text_token_count)
                ]
                mask_sum.add_(chunk.float().sum(dim=0))
            # Keep new trainable diffusion parameters in FP32.  AdamW otherwise
            # creates BF16 moments for this parameter when the base checkpoint is
            # loaded in BF16, which is needlessly fragile for a tiny new module.
            mask_initialization = mask_sum / normal_text_token_count
        self.diffusion_mask_embedding = nn.Parameter(mask_initialization)
        if time_embedding_seed < 0:
            raise ValueError("time_embedding_seed must be non-negative")
        if diffusion_config.use_time_conditioning:
            # Construct on CPU inside an isolated RNG scope.  Otherwise with_t
            # advances the global stream before PEFT creates LoRA matrices, so
            # the supposedly shared initialization differs across the 2x2x2
            # matrix even though the experiment seeds are identical.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(time_embedding_seed))
                time_mlp = TimeConditioningMLP(
                    hidden_size, feature_dim=time_feature_dim
                )
            self.time_mlp = time_mlp.to(device=embeddings.device)
        else:
            self.time_mlp = None

        if not hasattr(autoregressive_model, "freeze_audio_components"):
            raise TypeError("autoregressive_model does not expose freeze_audio_components()")
        autoregressive_model.freeze_audio_components()

    def train(self, mode: bool = True):
        super().train(mode)
        # The base model's own override keeps these modules in eval mode. Assert
        # here as well so a future refactor cannot silently weaken phase-1 policy.
        self.autoregressive_model.assert_audio_components_frozen()
        return self

    def encode_audio(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the frozen packed audio embeddings once, suitable for ISCT reuse."""

        self.autoregressive_model.freeze_audio_components()
        return self.autoregressive_model.model.get_audio_features(
            input_features,
            feature_attention_mask,
            no_grad=True,
        ).detach()

    def prepare_prefix_embeddings(
        self,
        *,
        prefix_input_ids: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        audio_position_mask: torch.Tensor,
        feature_attention_mask: torch.Tensor,
        input_features: torch.Tensor | None = None,
        projected_audio_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed the clean prefix and inject audio only through the explicit mask."""

        if prefix_input_ids.ndim != 2:
            raise ValueError("prefix_input_ids must have shape [batch, prefix]")
        if prefix_attention_mask.shape != prefix_input_ids.shape:
            raise ValueError("prefix_attention_mask must match prefix_input_ids")
        if audio_position_mask.shape != prefix_input_ids.shape:
            raise ValueError("audio_position_mask must match prefix_input_ids")
        if audio_position_mask.dtype != torch.bool:
            raise TypeError("audio_position_mask must have dtype torch.bool")
        prefix_valid = prefix_attention_mask.to(torch.bool)
        if (audio_position_mask & ~prefix_valid).any():
            raise ValueError("audio_position_mask marks a padded prefix position")
        if projected_audio_embeddings is None:
            if input_features is None:
                raise ValueError(
                    "provide input_features or reusable projected_audio_embeddings"
                )
            projected_audio_embeddings = self.encode_audio(
                input_features, feature_attention_mask
            )
        if projected_audio_embeddings.ndim != 2:
            raise ValueError("projected_audio_embeddings must be packed [frames, hidden]")

        prefix_embeddings = self.autoregressive_model.get_input_embeddings()(
            prefix_input_ids
        )
        expanded_mask = self.autoregressive_model.model._get_audio_placeholder_mask(
            prefix_input_ids,
            prefix_embeddings,
            projected_audio_embeddings.shape[0],
            audio_position_mask=audio_position_mask,
            feature_attention_mask=feature_attention_mask,
        )
        projected_audio_embeddings = projected_audio_embeddings.to(
            device=prefix_embeddings.device, dtype=prefix_embeddings.dtype
        ).detach()
        prefix_embeddings = prefix_embeddings.masked_scatter(
            expanded_mask, projected_audio_embeddings
        )
        return prefix_embeddings, projected_audio_embeddings

    def forward_diffusion(
        self,
        *,
        prefix_input_ids: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        audio_position_mask: torch.Tensor,
        feature_attention_mask: torch.Tensor,
        noisy_canvas_ids: torch.Tensor,
        canvas_valid_mask: torch.Tensor,
        noise_level: torch.Tensor,
        input_features: torch.Tensor | None = None,
        projected_audio_embeddings: torch.Tensor | None = None,
        clean_canvas_ids: torch.Tensor | None = None,
        corrupt_mask: torch.Tensor | None = None,
        denoiser_output_alignment: DenoiserOutputAlignment | None = None,
        forbidden_output_token_ids: Sequence[int] = (),
    ) -> DiffusionForwardOutput:
        """Run parallel denoising and, when targets are supplied, its explicit loss."""

        if noisy_canvas_ids.ndim != 2 or noisy_canvas_ids.dtype == torch.bool:
            raise ValueError("noisy_canvas_ids must be integer [batch, canvas]")
        if noisy_canvas_ids.is_floating_point() or noisy_canvas_ids.is_complex():
            raise TypeError("noisy_canvas_ids must contain integer token IDs")
        if canvas_valid_mask.shape != noisy_canvas_ids.shape:
            raise ValueError("canvas_valid_mask must match noisy_canvas_ids")
        if canvas_valid_mask.dtype != torch.bool:
            raise TypeError("canvas_valid_mask must have dtype torch.bool")
        validate_diffusion_time(noise_level, noisy_canvas_ids.shape[0], noisy_canvas_ids.device)
        if prefix_input_ids.device != noisy_canvas_ids.device:
            raise ValueError("prefix and canvas token IDs must share a device")

        prefix_embeddings, projected_audio_embeddings = self.prepare_prefix_embeddings(
            prefix_input_ids=prefix_input_ids,
            prefix_attention_mask=prefix_attention_mask,
            audio_position_mask=audio_position_mask,
            feature_attention_mask=feature_attention_mask,
            input_features=input_features,
            projected_audio_embeddings=projected_audio_embeddings,
        )
        canvas_embeddings = self.autoregressive_model.get_input_embeddings()(
            noisy_canvas_ids
        )
        if self.diffusion_config.noise_type == "mask":
            mask_positions = noisy_canvas_ids.eq(self.diffusion_config.mask_token_id)
            mask_embedding = self.diffusion_mask_embedding.to(
                device=canvas_embeddings.device,
                dtype=canvas_embeddings.dtype,
            )
            canvas_embeddings = torch.where(
                mask_positions[..., None],
                mask_embedding.view(1, 1, -1),
                canvas_embeddings,
            )
        else:
            # Keep the trainable-parameter set identical across the eight
            # matrix runs while satisfying DDP's static used-parameter graph.
            # Uniform noise has no semantic MASK input, so this dependency is
            # exactly zero and diffusion_mask_embedding receives a zero gradient.
            canvas_embeddings = canvas_embeddings + (
                self.diffusion_mask_embedding.sum().to(canvas_embeddings.dtype) * 0.0
            )
        if self.time_mlp is not None:
            time_embedding = self.time_mlp(noise_level.float()).to(canvas_embeddings.dtype)
            canvas_embeddings = canvas_embeddings + (
                time_embedding[:, None, :] * canvas_valid_mask[..., None]
            )

        packed_embeddings, prefix_lengths, canvas_lengths = pack_prefix_canvas_embeddings(
            prefix_embeddings,
            prefix_attention_mask,
            canvas_embeddings,
            canvas_valid_mask,
        )
        position_ids = build_prefix_canvas_position_ids(
            prefix_lengths,
            canvas_lengths,
            max_sequence_length=packed_embeddings.shape[1],
            device=packed_embeddings.device,
        )
        attention_mask = build_prefix_canvas_attention_mask(
            prefix_lengths,
            canvas_lengths,
            dtype=packed_embeddings.dtype,
            max_sequence_length=packed_embeddings.shape[1],
            device=packed_embeddings.device,
        )
        outputs = self.autoregressive_model.model.language_model(
            input_ids=None,
            inputs_embeds=packed_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            attention_mode="diffusion",
        )
        hidden = outputs.last_hidden_state
        max_canvas = noisy_canvas_ids.shape[1]
        boundary_hidden = hidden.new_empty((hidden.shape[0], 1, hidden.shape[2]))
        canvas_hidden = hidden.new_zeros((hidden.shape[0], max_canvas, hidden.shape[2]))
        for row in range(hidden.shape[0]):
            prefix_length = int(prefix_lengths[row].item())
            canvas_length = int(canvas_lengths[row].item())
            boundary_hidden[row, 0] = hidden[row, prefix_length - 1]
            canvas_hidden[row, :canvas_length] = hidden[
                row, prefix_length : prefix_length + canvas_length
            ]

        raw_logits = self.autoregressive_model.lm_head(
            torch.cat((boundary_hidden, canvas_hidden), dim=1)
        )
        alignment = (
            self.diffusion_config.denoiser_output_alignment
            if denoiser_output_alignment is None
            else denoiser_output_alignment
        )
        logits = align_denoiser_logits(raw_logits, alignment)
        if logits.shape[:2] != noisy_canvas_ids.shape:
            raise AssertionError("aligned logits lost the canvas coordinate contract")

        loss_output = None
        if clean_canvas_ids is not None:
            if corrupt_mask is None:
                raise ValueError("corrupt_mask is required with clean_canvas_ids")
            if clean_canvas_ids.shape != noisy_canvas_ids.shape:
                raise ValueError("clean_canvas_ids must match noisy_canvas_ids")
            if self.diffusion_config.resolved_objective.name == "mdm_absorbing":
                expected_corruption = (
                    noisy_canvas_ids.eq(self.diffusion_config.mask_token_id)
                    & canvas_valid_mask
                )
                if not torch.equal(corrupt_mask & canvas_valid_mask, expected_corruption):
                    raise ValueError(
                        "mdm_absorbing requires corrupt_mask to match exactly the "
                        "allocated noisy-canvas MASK positions"
                    )
            loss_output = diffusion_loss(
                logits,
                clean_canvas_ids,
                corrupt_mask,
                objective=self.diffusion_config.resolved_objective,
                t=noise_level,
                mask_token_id=self.diffusion_config.mask_token_id,
                canvas_valid_mask=canvas_valid_mask,
                tail_fill_mode=self.diffusion_config.canvas.tail_fill_mode,
                eos_tail_loss_weight=self.diffusion_config.canvas.eos_tail_loss_weight,
                pad_loss_weight=self.diffusion_config.canvas.pad_loss_weight,
                turn_end_token_id=self.diffusion_config.canvas.turn_end_token_id,
                pad_token_id=self.diffusion_config.canvas.pad_token_id,
                forbidden_output_token_ids=forbidden_output_token_ids,
                min_diffusion_time=self.diffusion_config.min_diffusion_time,
            )
        elif corrupt_mask is not None:
            raise ValueError("corrupt_mask was supplied without clean_canvas_ids")

        return DiffusionForwardOutput(
            logits=logits,
            raw_logits=raw_logits,
            loss=None if loss_output is None else loss_output.loss,
            loss_output=loss_output,
            prefix_lengths=prefix_lengths,
            canvas_lengths=canvas_lengths,
            projected_audio_embeddings=projected_audio_embeddings,
        )

    def forward(self, **kwargs) -> DiffusionForwardOutput:
        """Alias for trainers; AR calls remain on ``autoregressive_model``."""

        return self.forward_diffusion(**kwargs)


__all__ = [
    "DiffusionForwardOutput",
    "MawJaisDiffusionASRModel",
    "pack_prefix_canvas_embeddings",
]
