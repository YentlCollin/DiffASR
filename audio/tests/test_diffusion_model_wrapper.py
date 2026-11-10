from types import SimpleNamespace

import pytest
import torch
from torch import nn

from audio.diffusion.config import CanvasConfig, DiffusionConfig
from audio.diffusion.model import MawJaisDiffusionASRModel, pack_prefix_canvas_embeddings


class _FakeLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_call = None

    def forward(self, **kwargs):
        self.last_call = kwargs
        # Identity makes the readout coordinates directly inspectable.
        return SimpleNamespace(last_hidden_state=kwargs["inputs_embeds"])


class _FakeInnerModel(nn.Module):
    def __init__(self, vocab_size=16, hidden_size=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.language_model = _FakeLanguageModel()
        self.audio_tower = nn.Linear(2, 2)
        self.embed_audio = nn.Linear(2, hidden_size)

    def get_input_embeddings(self):
        return self.embedding

    def get_audio_features(self, input_features, feature_attention_mask, *, no_grad=False):
        del feature_attention_mask
        output = input_features.reshape(-1, input_features.shape[-1])[:, :4]
        return output.detach() if no_grad else output

    def _get_audio_placeholder_mask(
        self,
        input_ids,
        inputs_embeds,
        n_audio_frames,
        *,
        audio_position_mask,
        feature_attention_mask,
    ):
        del input_ids, feature_attention_mask
        assert int(audio_position_mask.sum()) == n_audio_frames
        return audio_position_mask[..., None].expand_as(inputs_embeds)


class _FakeARModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _FakeInnerModel()
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.frozen = False

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def freeze_audio_components(self):
        self.model.audio_tower.requires_grad_(False).eval()
        self.model.embed_audio.requires_grad_(False).eval()
        self.frozen = True
        return self

    def assert_audio_components_frozen(self):
        assert self.frozen
        assert not self.model.audio_tower.training
        assert not self.model.embed_audio.training

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.model.audio_tower.eval()
            self.model.embed_audio.eval()
        return self


def _wrapper(*, noise_type="mask", with_time=False, alignment="same_position"):
    ar = _FakeARModel()
    config = DiffusionConfig(
        noise_type=noise_type,
        use_time_conditioning=with_time,
        denoiser_output_alignment=alignment,
        mask_token_id=9,
        canvas=CanvasConfig(
            sizes=(2, 3),
            duration_tokens_per_second=1.0,
            duration_margin_tokens=0.0,
            turn_end_token_id=7,
            pad_token_id=1,
        ),
    )
    return MawJaisDiffusionASRModel(ar, config, normal_text_token_count=8), ar


def test_pack_removes_prefix_padding_before_canvas():
    prefix = torch.arange(2 * 4 * 2, dtype=torch.float32).reshape(2, 4, 2)
    prefix_mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    canvas = 100 + torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    canvas_valid = torch.tensor([[True, True, False], [True, True, True]])
    packed, prefix_lengths, canvas_lengths = pack_prefix_canvas_embeddings(
        prefix, prefix_mask, canvas, canvas_valid
    )
    assert prefix_lengths.tolist() == [2, 4]
    assert canvas_lengths.tolist() == [2, 3]
    assert torch.equal(packed[0, 2:4], canvas[0, :2])
    assert torch.equal(packed[1, 4:7], canvas[1])


def test_forward_uses_explicit_audio_mask_and_central_alignment():
    wrapper, ar = _wrapper(alignment="same_position")
    prefix_ids = torch.tensor([[2, 5, 1], [2, 5, 6]])
    prefix_valid = torch.tensor([[True, True, False], [True, True, True]])
    audio_positions = torch.tensor([[False, True, False], [False, True, False]])
    noisy = torch.tensor([[9, 5, 1], [9, 5, 6]])
    canvas_valid = torch.tensor([[True, True, False], [True, True, True]])
    projected = torch.tensor([[21.0, 22.0, 23.0, 24.0], [31.0, 32.0, 33.0, 34.0]])

    output = wrapper.forward_diffusion(
        prefix_input_ids=prefix_ids,
        prefix_attention_mask=prefix_valid,
        audio_position_mask=audio_positions,
        feature_attention_mask=torch.ones(2, 1, dtype=torch.long),
        projected_audio_embeddings=projected,
        noisy_canvas_ids=noisy,
        canvas_valid_mask=canvas_valid,
        noise_level=torch.tensor([1.0, 0.5]),
    )

    call = ar.model.language_model.last_call
    assert call["attention_mode"] == "diffusion"
    assert call["use_cache"] is False
    assert call["attention_mask"].ndim == 4
    # In row 0 the audio embedding is prefix position 1 and canvas starts right
    # after it, not after the processor padding at original position 2.
    assert torch.equal(call["inputs_embeds"][0, 1], projected[0])
    assert torch.equal(call["inputs_embeds"][0, 2], wrapper.diffusion_mask_embedding)
    assert output.raw_logits.shape == (2, 4, 16)
    assert torch.equal(output.logits, output.raw_logits[:, 1:])

    previous = wrapper.forward_diffusion(
        prefix_input_ids=prefix_ids,
        prefix_attention_mask=prefix_valid,
        audio_position_mask=audio_positions,
        feature_attention_mask=torch.ones(2, 1, dtype=torch.long),
        projected_audio_embeddings=projected,
        noisy_canvas_ids=noisy,
        canvas_valid_mask=canvas_valid,
        noise_level=torch.tensor([1.0, 0.5]),
        denoiser_output_alignment="previous_position",
    )
    assert torch.equal(previous.logits, previous.raw_logits[:, :-1])


def test_uniform_audio_token_in_canvas_is_an_ordinary_token():
    wrapper, ar = _wrapper(noise_type="uniform")
    # ID 5 stands in for the audio placeholder. Only the explicit prefix mask
    # receives projected audio; the identical canvas ID remains its text embedding.
    output = wrapper.forward_diffusion(
        prefix_input_ids=torch.tensor([[2, 5]]),
        prefix_attention_mask=torch.ones(1, 2, dtype=torch.bool),
        audio_position_mask=torch.tensor([[False, True]]),
        feature_attention_mask=torch.ones(1, 1, dtype=torch.long),
        projected_audio_embeddings=torch.tensor([[20.0, 21.0, 22.0, 23.0]]),
        noisy_canvas_ids=torch.tensor([[5, 3]]),
        canvas_valid_mask=torch.ones(1, 2, dtype=torch.bool),
        noise_level=torch.tensor([1.0]),
    )
    del output
    packed = ar.model.language_model.last_call["inputs_embeds"]
    assert torch.equal(packed[0, 1], torch.tensor([20.0, 21.0, 22.0, 23.0]))
    assert torch.equal(packed[0, 2], ar.get_input_embeddings().weight[5])


def test_uniform_keeps_mask_parameter_in_ddp_graph_with_exact_zero_gradient():
    wrapper, _ = _wrapper(noise_type="uniform")
    output = wrapper.forward_diffusion(
        prefix_input_ids=torch.tensor([[2, 5]]),
        prefix_attention_mask=torch.ones(1, 2, dtype=torch.bool),
        audio_position_mask=torch.tensor([[False, True]]),
        feature_attention_mask=torch.ones(1, 1, dtype=torch.long),
        projected_audio_embeddings=torch.ones(1, 4),
        noisy_canvas_ids=torch.tensor([[5, 3]]),
        clean_canvas_ids=torch.tensor([[2, 7]]),
        corrupt_mask=torch.ones(1, 2, dtype=torch.bool),
        canvas_valid_mask=torch.ones(1, 2, dtype=torch.bool),
        noise_level=torch.tensor([1.0]),
    )
    assert output.loss is not None
    output.loss.backward()
    assert wrapper.diffusion_mask_embedding.grad is not None
    assert torch.count_nonzero(wrapper.diffusion_mask_embedding.grad) == 0


def test_loss_is_unshifted_and_audio_stays_frozen_after_train():
    wrapper, _ = _wrapper(noise_type="mask")
    wrapper.train()
    clean = torch.tensor([[2, 7]])
    noisy = torch.tensor([[9, 9]])
    output = wrapper.forward_diffusion(
        prefix_input_ids=torch.tensor([[2, 5]]),
        prefix_attention_mask=torch.ones(1, 2, dtype=torch.bool),
        audio_position_mask=torch.tensor([[False, True]]),
        feature_attention_mask=torch.ones(1, 1, dtype=torch.long),
        projected_audio_embeddings=torch.ones(1, 4),
        noisy_canvas_ids=noisy,
        clean_canvas_ids=clean,
        corrupt_mask=torch.ones_like(noisy, dtype=torch.bool),
        canvas_valid_mask=torch.ones_like(noisy, dtype=torch.bool),
        noise_level=torch.tensor([1.0]),
    )
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert wrapper.diffusion_mask_embedding.grad is not None
    assert all(parameter.grad is None for parameter in wrapper.autoregressive_model.model.audio_tower.parameters())


def test_mdm_forward_rejects_mask_and_corruption_disagreement():
    wrapper, _ = _wrapper(noise_type="mask")
    with pytest.raises(ValueError, match="MASK positions"):
        wrapper.forward_diffusion(
            prefix_input_ids=torch.tensor([[2, 5]]),
            prefix_attention_mask=torch.ones(1, 2, dtype=torch.bool),
            audio_position_mask=torch.tensor([[False, True]]),
            feature_attention_mask=torch.ones(1, 1, dtype=torch.long),
            projected_audio_embeddings=torch.ones(1, 4),
            noisy_canvas_ids=torch.tensor([[9, 3]]),
            clean_canvas_ids=torch.tensor([[2, 7]]),
            corrupt_mask=torch.tensor([[False, False]]),
            canvas_valid_mask=torch.ones(1, 2, dtype=torch.bool),
            noise_level=torch.tensor([0.5]),
        )


def test_pack_rejects_gapped_canvas_mask():
    with pytest.raises(ValueError, match="left-contiguous"):
        pack_prefix_canvas_embeddings(
            torch.zeros(1, 2, 4),
            torch.ones(1, 2, dtype=torch.bool),
            torch.zeros(1, 3, 4),
            torch.tensor([[True, False, True]]),
        )


def test_time_conditioning_changes_only_canvas_queries():
    wrapper, ar = _wrapper(noise_type="mask", with_time=True)
    assert wrapper.time_mlp is not None
    with torch.no_grad():
        wrapper.time_mlp.out_proj.weight.fill_(0.1)
    common = dict(
        prefix_input_ids=torch.tensor([[2, 5]]),
        prefix_attention_mask=torch.ones(1, 2, dtype=torch.bool),
        audio_position_mask=torch.tensor([[False, True]]),
        feature_attention_mask=torch.ones(1, 1, dtype=torch.long),
        projected_audio_embeddings=torch.ones(1, 4),
        noisy_canvas_ids=torch.tensor([[9, 3]]),
        canvas_valid_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    wrapper.forward_diffusion(**common, noise_level=torch.tensor([0.1]))
    low_t = ar.model.language_model.last_call["inputs_embeds"].clone()
    wrapper.forward_diffusion(**common, noise_level=torch.tensor([0.9]))
    high_t = ar.model.language_model.last_call["inputs_embeds"].clone()
    assert torch.equal(low_t[:, :2], high_t[:, :2])
    assert not torch.equal(low_t[:, 2:], high_t[:, 2:])


def test_new_diffusion_parameters_stay_fp32_with_bf16_base():
    ar = _FakeARModel().to(dtype=torch.bfloat16)
    config = DiffusionConfig(
        noise_type="mask",
        use_time_conditioning=True,
        mask_token_id=9,
        canvas=CanvasConfig(
            sizes=(2,),
            duration_tokens_per_second=1.0,
            duration_margin_tokens=0.0,
            turn_end_token_id=7,
            pad_token_id=1,
        ),
    )
    wrapper = MawJaisDiffusionASRModel(ar, config, normal_text_token_count=8)
    assert wrapper.diffusion_mask_embedding.dtype == torch.float32
    assert wrapper.time_mlp is not None
    assert {parameter.dtype for parameter in wrapper.time_mlp.parameters()} == {
        torch.float32
    }
    output = wrapper.forward_diffusion(
        prefix_input_ids=torch.tensor([[2, 5]]),
        prefix_attention_mask=torch.ones(1, 2, dtype=torch.bool),
        audio_position_mask=torch.tensor([[False, True]]),
        feature_attention_mask=torch.ones(1, 1, dtype=torch.long),
        projected_audio_embeddings=torch.ones(1, 4, dtype=torch.bfloat16),
        noisy_canvas_ids=torch.tensor([[9, 3]]),
        canvas_valid_mask=torch.ones(1, 2, dtype=torch.bool),
        noise_level=torch.tensor([0.5]),
    )
    assert output.logits.dtype == torch.bfloat16
    new_parameters = [wrapper.diffusion_mask_embedding, *wrapper.time_mlp.parameters()]
    optimizer = torch.optim.AdamW(new_parameters, lr=1.0e-3)
    optimizer.zero_grad(set_to_none=True)
    sum(parameter.sum() for parameter in new_parameters).backward()
    optimizer.step()
    for parameter in new_parameters:
        state = optimizer.state[parameter]
        assert state["exp_avg"].dtype == torch.float32
        assert state["exp_avg_sq"].dtype == torch.float32


def test_optional_time_module_does_not_advance_global_rng():
    without_time_ar = _FakeARModel()
    with_time_ar = _FakeARModel()
    torch.manual_seed(9876)
    MawJaisDiffusionASRModel(
        without_time_ar,
        DiffusionConfig(noise_type="mask", use_time_conditioning=False, mask_token_id=9),
        normal_text_token_count=8,
    )
    after_without_time = torch.rand(5)
    torch.manual_seed(9876)
    MawJaisDiffusionASRModel(
        with_time_ar,
        DiffusionConfig(noise_type="mask", use_time_conditioning=True, mask_token_id=9),
        normal_text_token_count=8,
        time_embedding_seed=123,
    )
    after_with_time = torch.rand(5)
    assert torch.equal(after_without_time, after_with_time)


def test_audio_parameters_remain_bit_identical_across_optimizer_steps():
    wrapper, ar = _wrapper(noise_type="mask")
    wrapper.train()
    before = {
        name: parameter.detach().clone()
        for name, parameter in ar.named_parameters()
        if "audio_tower" in name or "embed_audio" in name
    }
    optimizer = torch.optim.SGD(
        [parameter for parameter in wrapper.parameters() if parameter.requires_grad],
        lr=0.01,
    )
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = wrapper.forward_diffusion(
            prefix_input_ids=torch.tensor([[2, 5]]),
            prefix_attention_mask=torch.ones(1, 2, dtype=torch.bool),
            audio_position_mask=torch.tensor([[False, True]]),
            feature_attention_mask=torch.ones(1, 1, dtype=torch.long),
            projected_audio_embeddings=torch.ones(1, 4),
            noisy_canvas_ids=torch.tensor([[9, 9]]),
            clean_canvas_ids=torch.tensor([[2, 7]]),
            corrupt_mask=torch.ones(1, 2, dtype=torch.bool),
            canvas_valid_mask=torch.ones(1, 2, dtype=torch.bool),
            noise_level=torch.tensor([1.0]),
        )
        assert output.loss is not None
        output.loss.backward()
        optimizer.step()
        ar.assert_audio_components_frozen()

    after = dict(ar.named_parameters())
    assert before
    for name, value in before.items():
        assert torch.equal(value, after[name])
