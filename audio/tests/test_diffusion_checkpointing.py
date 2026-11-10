from types import SimpleNamespace

import pytest
import torch
from torch import nn

from audio.diffusion.checkpointing import (
    load_diffusion_component_state,
    read_diffusion_checkpoint_manifest,
    save_diffusion_component_state,
    write_diffusion_checkpoint_manifest,
)


class TinyDiffusionWrapper(nn.Module):
    def __init__(self, with_time=True):
        super().__init__()
        self.diffusion_mask_embedding = nn.Parameter(torch.arange(4, dtype=torch.float32))
        self.time_mlp = nn.Linear(2, 4) if with_time else None


def test_diffusion_component_state_round_trip(tmp_path):
    source = TinyDiffusionWrapper(with_time=True)
    expected_mask = source.diffusion_mask_embedding.detach().clone()
    expected_time = {name: value.detach().clone() for name, value in source.time_mlp.state_dict().items()}
    path = save_diffusion_component_state(source, tmp_path / "diffusion_state.pt")

    with torch.no_grad():
        source.diffusion_mask_embedding.zero_()
        for parameter in source.time_mlp.parameters():
            parameter.zero_()
    payload = load_diffusion_component_state(source, path)
    assert payload["schema_version"] == 1
    assert torch.equal(source.diffusion_mask_embedding, expected_mask)
    for name, value in source.time_mlp.state_dict().items():
        assert torch.equal(value, expected_time[name])


def test_diffusion_component_state_rejects_time_architecture_mismatch(tmp_path):
    path = save_diffusion_component_state(
        TinyDiffusionWrapper(with_time=True), tmp_path / "with_time.pt"
    )
    with pytest.raises(ValueError, match="contains time_mlp"):
        load_diffusion_component_state(TinyDiffusionWrapper(with_time=False), path)


def test_checkpoint_manifest_contract(tmp_path):
    path = write_diffusion_checkpoint_manifest(
        tmp_path,
        {
            "strategy": "lora",
            "base_checkpoint": "org/model",
            "revision": "revision",
            "update": 12,
            "diffusion_state": "diffusion_state.pt",
            "resolved_config": "resolved_config.yaml",
            "adapter_dir": "adapter",
        },
    )
    payload = read_diffusion_checkpoint_manifest(path)
    assert payload["schema_version"] == 1
    assert payload["strategy"] == "lora"
    assert payload["adapter_dir"] == "adapter"

    with pytest.raises(ValueError, match="full_model_dir"):
        write_diffusion_checkpoint_manifest(
            tmp_path / "bad",
            {
                "strategy": "full",
                "base_checkpoint": "org/model",
                "revision": "revision",
                "update": 1,
                "diffusion_state": "state.pt",
                "resolved_config": "resolved.yaml",
            },
        )
