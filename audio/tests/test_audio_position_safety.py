"""Small CPU tests for explicit audio positions and the phase-1 audio freeze.

These tests deliberately avoid constructing the 4B language model or the base audio tower.
"""

from pathlib import Path
from types import ModuleType, SimpleNamespace
import importlib.util
import sys

import numpy as np
import pytest
import torch
from torch import nn
from transformers.configuration_utils import PreTrainedConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "audio" / "model"


def _load_processor_module():
    spec = importlib.util.spec_from_file_location(
        "xllm_aut_processor_safety_test", MODEL_DIR / "processing_xllm_aut.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_model_module(monkeypatch):
    # modeling_xllm_aut.py supports a flat trust_remote_code layout. Supply tiny stand-ins for its
    # two local siblings so importing the glue does not instantiate or even import the 4B backbone.
    backbone_module = ModuleType("modeling_xllm")

    class TinyXllmModel(nn.Module):
        def __init__(self, _config=None):
            super().__init__()

    backbone_module.XllmModel = TinyXllmModel
    config_module = ModuleType("configuration_xllm_aut")

    class TinyConfig(PreTrainedConfig):
        model_type = "xllm_aut_audio_safety_test"

    config_module.XllmAuTConfig = TinyConfig
    monkeypatch.setitem(sys.modules, "modeling_xllm", backbone_module)
    monkeypatch.setitem(sys.modules, "configuration_xllm_aut", config_module)

    spec = importlib.util.spec_from_file_location(
        "xllm_aut_model_safety_test", MODEL_DIR / "modeling_xllm_aut.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _TinyTokenizer:
    audio_token_id = 99
    pad_token_id = 0

    def convert_tokens_to_ids(self, token):
        return self.audio_token_id if token == "<|ifm|audio|>" else 7

    def __call__(self, *, text, return_tensors=None, **_kwargs):
        encoded = []
        audio_token = "<|ifm|audio|>"
        for value in text:
            ids = []
            cursor = 0
            while cursor < len(value):
                if value.startswith(audio_token, cursor):
                    ids.append(self.audio_token_id)
                    cursor += len(audio_token)
                else:
                    ids.append(7)
                    cursor += 1
            encoded.append(ids)
        max_length = max(map(len, encoded))
        input_ids = [row + [self.pad_token_id] * (max_length - len(row)) for row in encoded]
        attention_mask = [[1] * len(row) + [0] * (max_length - len(row)) for row in encoded]
        if return_tensors == "pt":
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _TinyFeatureExtractor:
    sampling_rate = 16_000

    def __call__(self, audio, *, return_tensors=None, **_kwargs):
        lengths = [len(clip) for clip in audio]
        max_length = max(lengths)
        features = np.zeros((len(audio), 2, max_length), dtype=np.float32)
        mask = np.zeros((len(audio), max_length), dtype=np.int64)
        for index, length in enumerate(lengths):
            mask[index, :length] = 1
        if return_tensors == "pt":
            features = torch.from_numpy(features)
            mask = torch.from_numpy(mask)
        return {"input_features": features, "attention_mask": mask}


def _tiny_processor(processor_module):
    processor = object.__new__(processor_module.XllmAuTProcessor)
    processor.audio_token = "<|ifm|audio|>"
    processor.boa_token = "<audio>"
    processor.eoa_token = "</audio>"
    processor.audio_token_id = 99
    processor.feature_extractor = _TinyFeatureExtractor()
    processor.tokenizer = _TinyTokenizer()
    return processor


def _bare_model(model_module):
    model = object.__new__(model_module.XllmAuTModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(audio_token_id=99)
    return model


def test_processor_returns_exact_padded_bool_audio_mask():
    module = _load_processor_module()
    processor = _tiny_processor(module)
    result = processor(
        text=["before <|ifm|audio|> after", "<|ifm|audio|>"],
        # Mel lengths 1 and 101 expand to 1 and 14 AuT output tokens respectively.
        audio=[np.zeros(1, dtype=np.float32), np.zeros(101, dtype=np.float32)],
        return_tensors="pt",
        padding=True,
    )

    mask = result["audio_position_mask"]
    assert mask.dtype == torch.bool
    assert mask.shape == result["input_ids"].shape
    assert mask.sum(dim=1).tolist() == [1, 14]
    assert torch.equal(mask, result["input_ids"].eq(processor.audio_token_id))
    assert not mask[~result["attention_mask"].bool()].any()


def test_processor_without_audio_returns_all_false_mask():
    module = _load_processor_module()
    processor = _tiny_processor(module)
    result = processor(text=["plain text", "short"], return_tensors="pt", padding=True)
    assert result["audio_position_mask"].dtype == torch.bool
    assert result["audio_position_mask"].shape == result["input_ids"].shape
    assert not result["audio_position_mask"].any()


def test_explicit_mask_is_the_only_injection_source(monkeypatch):
    module = _load_model_module(monkeypatch)
    model = _bare_model(module)
    embeddings = torch.zeros(2, 4, 3)
    input_ids = torch.tensor(
        [
            [99, 5, 99, 5],  # second 99 simulates a noisy canvas token and must be ignored
            [5, 99, 5, 5],
        ]
    )
    explicit = torch.tensor(
        [
            [True, False, False, False],
            [False, True, False, False],
        ]
    )
    feature_mask = torch.ones(2, 1, dtype=torch.long)  # one output frame per example

    expanded = model._get_audio_placeholder_mask(
        input_ids,
        embeddings,
        n_audio_frames=2,
        audio_position_mask=explicit,
        feature_attention_mask=feature_mask,
    )
    assert torch.equal(expanded[..., 0], explicit)
    assert not expanded[0, 2].any()


def test_explicit_mask_validates_dtype_shape_and_each_example(monkeypatch):
    module = _load_model_module(monkeypatch)
    model = _bare_model(module)
    embeddings = torch.zeros(2, 4, 3)
    input_ids = torch.full((2, 4), 5)
    feature_mask = torch.ones(2, 1, dtype=torch.long)

    with pytest.raises(TypeError, match="torch.bool"):
        model._get_audio_placeholder_mask(
            input_ids,
            embeddings,
            2,
            audio_position_mask=torch.zeros(2, 4, dtype=torch.long),
        )
    with pytest.raises(ValueError, match=r"shape \[batch, sequence\]"):
        model._get_audio_placeholder_mask(
            input_ids,
            embeddings,
            2,
            audio_position_mask=torch.zeros(2, 3, dtype=torch.bool),
        )

    # The global total is still two, but both positions are in example 1. Per-example validation
    # must reject this instead of silently scattering example 0's frame into example 1.
    displaced = torch.tensor(
        [
            [False, False, False, False],
            [True, True, False, False],
        ]
    )
    with pytest.raises(ValueError, match="audio spans|example"):
        model._get_audio_placeholder_mask(
            input_ids,
            embeddings,
            2,
            audio_position_mask=displaced,
            feature_attention_mask=feature_mask,
        )


def test_legacy_ar_id_fallback_remains_available(monkeypatch):
    module = _load_model_module(monkeypatch)
    model = _bare_model(module)
    embeddings = torch.zeros(1, 4, 2)
    input_ids = torch.tensor([[5, 99, 99, 5]])
    expanded = model._get_audio_placeholder_mask(input_ids, embeddings, n_audio_frames=2)
    assert expanded[..., 0].tolist() == [[False, True, True, False]]


def test_frozen_audio_modules_stay_eval_after_train(monkeypatch):
    module = _load_model_module(monkeypatch)
    model = _bare_model(module)
    model.language_model = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
    model.audio_tower = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
    model.embed_audio = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5))
    model._audio_components_frozen = False

    # A late freeze must also discard stale gradients left by an earlier AR/debug step.
    next(model.audio_tower.parameters()).grad = torch.ones_like(
        next(model.audio_tower.parameters())
    )

    model.freeze_audio_components()
    model.train()

    assert model.language_model.training
    assert not model.audio_tower.training
    assert not model.embed_audio.training
    assert all(not parameter.requires_grad for parameter in model.audio_tower.parameters())
    assert all(not parameter.requires_grad for parameter in model.embed_audio.parameters())
    assert all(parameter.grad is None for parameter in model.audio_tower.parameters())
    assert all(parameter.grad is None for parameter in model.embed_audio.parameters())
    model.assert_audio_components_frozen()


def test_audio_no_grad_path_returns_detached_embeddings(monkeypatch):
    module = _load_model_module(monkeypatch)
    model = _bare_model(module)

    class TinyTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(()))

        @property
        def dtype(self):
            return self.scale.dtype

        def forward(self, packed_features, feature_lens=None):
            return SimpleNamespace(last_hidden_state=packed_features.T * self.scale)

    model.audio_tower = TinyTower()
    model.embed_audio = nn.Linear(2, 2, bias=False)
    input_features = torch.ones(1, 2, 3, requires_grad=True)
    feature_mask = torch.ones(1, 3, dtype=torch.long)

    regular = model.get_audio_features(input_features, feature_mask)
    detached = model.get_audio_features(input_features, feature_mask, no_grad=True)
    assert regular.requires_grad
    assert not detached.requires_grad
    assert detached.grad_fn is None


def test_diffusion_forward_requires_explicit_audio_mask(monkeypatch):
    module = _load_model_module(monkeypatch)
    model = _bare_model(module)
    model.language_model = nn.Identity()
    model.get_input_embeddings = lambda: nn.Embedding(128, 2)

    with pytest.raises(ValueError, match="audio_position_mask is required"):
        model.forward(
            input_ids=torch.tensor([[99, 7]]),
            input_features=torch.ones(1, 2, 1),
            feature_attention_mask=torch.ones(1, 1, dtype=torch.long),
            diffusion_mode=True,
        )
