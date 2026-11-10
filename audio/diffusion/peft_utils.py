"""PEFT compatibility helpers for the MawJais composite model contract."""

from __future__ import annotations

from torch import nn


class PeftCompositeProxy(nn.Module):
    """Expose ``XllmAuTForConditionalGeneration`` through a ``PeftModel``.

    PEFT's public ``.model`` points at its tuner wrapper, while MawJais
    diffusion needs ``autoregressive_model.model`` to remain the audio/text
    composite. The proxy delegates composite access to ``get_base_model()`` and
    retains the PEFT owner for adapter parameters and ``save_pretrained``.
    LoRA-injected projections remain shared underlying module objects.
    """

    def __init__(self, peft_model: nn.Module) -> None:
        super().__init__()
        if not hasattr(peft_model, "get_base_model"):
            raise TypeError("peft_model must expose get_base_model()")
        self.peft_model = peft_model

    def get_base_model(self) -> nn.Module:
        return self.peft_model.get_base_model()

    @property
    def model(self) -> nn.Module:
        return self.get_base_model().model

    @property
    def lm_head(self) -> nn.Module:
        return self.get_base_model().lm_head

    @property
    def config(self):
        return self.get_base_model().config

    def get_input_embeddings(self):
        return self.get_base_model().get_input_embeddings()

    def get_output_embeddings(self):
        return self.get_base_model().get_output_embeddings()

    def freeze_audio_components(self):
        self.get_base_model().freeze_audio_components()
        return self

    def assert_audio_components_frozen(self):
        return self.get_base_model().assert_audio_components_frozen()

    def gradient_checkpointing_enable(self, *args, **kwargs):
        return self.get_base_model().gradient_checkpointing_enable(*args, **kwargs)

    def gradient_checkpointing_disable(self):
        return self.get_base_model().gradient_checkpointing_disable()

    def save_pretrained(self, *args, **kwargs):
        return self.peft_model.save_pretrained(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.peft_model(*args, **kwargs)


__all__ = ["PeftCompositeProxy"]
