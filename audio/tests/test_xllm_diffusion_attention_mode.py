import pytest
import torch

from audio.diffusion.attention import (
    build_prefix_canvas_attention_mask,
    build_prefix_canvas_position_ids,
)
from huggingface.xllm.configuration_xllm import XllmConfig
from huggingface.xllm.modeling_xllm import XllmModel


def _tiny_model():
    config = XllmConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        query_key_norm=False,
        layernorm_num_groups=1,
        use_cache=False,
        pad_token_id=1,
    )
    config._attn_implementation = "eager"
    return XllmModel(config).eval()


def test_explicit_causal_mode_preserves_default_path_exactly():
    torch.manual_seed(5)
    model = _tiny_model()
    input_ids = torch.tensor([[2, 3, 4, 5]])
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        default = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        explicit = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            attention_mode="causal",
        ).last_hidden_state
    assert torch.equal(default, explicit)


def test_diffusion_mode_consumes_explicit_4d_prefix_canvas_mask():
    torch.manual_seed(7)
    model = _tiny_model()
    input_ids = torch.tensor([[2, 3, 4, 5]])  # prefix length 2, canvas length 2
    mask = build_prefix_canvas_attention_mask([2], [2], dtype=torch.float32)
    positions = build_prefix_canvas_position_ids([2], [2])
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=mask,
            position_ids=positions,
            use_cache=False,
            attention_mode="diffusion",
        ).last_hidden_state
    assert output.shape == (1, 4, 16)
    assert torch.isfinite(output).all()


def test_diffusion_mode_rejects_implicit_mask_and_cache():
    model = _tiny_model()
    input_ids = torch.tensor([[2, 3, 4]])
    with pytest.raises(ValueError, match="explicit 4D"):
        model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            attention_mode="diffusion",
        )

    mask = build_prefix_canvas_attention_mask([1], [2], dtype=torch.float32)
    with pytest.raises(ValueError, match="use_cache=False"):
        model(
            input_ids=input_ids,
            attention_mask=mask,
            use_cache=True,
            attention_mode="diffusion",
        )


def test_diffusion_prefix_is_invariant_to_canvas_content():
    torch.manual_seed(11)
    model = _tiny_model()
    mask = build_prefix_canvas_attention_mask([2], [2], dtype=torch.float32)
    positions = build_prefix_canvas_position_ids([2], [2])
    first = torch.tensor([[2, 3, 4, 5]])
    second = torch.tensor([[2, 3, 4, 7]])
    with torch.no_grad():
        output_first = model(
            input_ids=first,
            attention_mask=mask,
            position_ids=positions,
            use_cache=False,
            attention_mode="diffusion",
        ).last_hidden_state
        output_second = model(
            input_ids=second,
            attention_mask=mask,
            position_ids=positions,
            use_cache=False,
            attention_mode="diffusion",
        ).last_hidden_state
    assert torch.equal(output_first[:, :2], output_second[:, :2])
    # The first canvas token is identical, but its state changes because it can
    # attend bidirectionally to the modified later canvas position.
    assert not torch.equal(output_first[:, 2], output_second[:, 2])


def test_gradient_checkpointing_runs_each_decoder_layer_and_backpropagates():
    torch.manual_seed(13)
    model = _tiny_model().train()
    model.gradient_checkpointing_enable()
    calls = [0 for _ in model.layers]
    for index, layer in enumerate(model.layers):
        original = layer._gradient_checkpointing_func

        def counted(*args, _index=index, _original=original, **kwargs):
            calls[_index] += 1
            return _original(*args, **kwargs)

        layer._gradient_checkpointing_func = counted

    mask = build_prefix_canvas_attention_mask([2], [2], dtype=torch.float32)
    positions = build_prefix_canvas_position_ids([2], [2])
    output = model(
        input_ids=torch.tensor([[2, 3, 4, 5]]),
        attention_mask=mask,
        position_ids=positions,
        use_cache=False,
        attention_mode="diffusion",
    ).last_hidden_state
    output.square().mean().backward()

    assert calls == [1, 1]
    assert model.embed_tokens.weight.grad is not None
    assert torch.isfinite(model.embed_tokens.weight.grad).all()
