from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import threading
import time

import pytest
import torch
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "examples" / "export_fsdp_diffusion_checkpoint_for_inference.py"


def _load_exporter():
    spec = importlib.util.spec_from_file_location("fsdp_diffusion_checkpoint_export", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_checkpoint(tmp_path: Path, *, update: int = 100, trained_hash: str = "a" * 64) -> Path:
    checkpoint = tmp_path / f"checkpoint-{update:06d}"
    dcp_dir = checkpoint / "full_model" / "pytorch_model_fsdp_0"
    dcp_dir.mkdir(parents=True)
    (dcp_dir / ".metadata").write_bytes(b"metadata")
    (dcp_dir / "__0_0.distcp").write_bytes(b"shard")
    manifest = {
        "schema_version": 1,
        "strategy": "full",
        "checkpoint_format": "accelerate_fsdp_sharded_state_dict",
        "full_model_dir": "full_model",
        "update": update,
        "artifact_sha256": {"trained_model": trained_hash},
    }
    (checkpoint / "diffusion_checkpoint.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return checkpoint


def _wrapper_state(*, model_prefix: bool = False) -> dict[str, torch.Tensor]:
    prefix = "model." if model_prefix else ""
    return {
        prefix + "diffusion_mask_embedding": torch.zeros(2),
        prefix + "autoregressive_model.lm_head.weight": torch.ones(2, 2),
        prefix
        + "autoregressive_model.model.language_model.embed_tokens.weight": torch.full(
            (2, 2), 2.0
        ),
        prefix
        + "autoregressive_model.model.language_model.layers.0.self_attn.q_proj.weight": torch.full(
            (2, 2), 3.0
        ),
        prefix + "autoregressive_model.model.audio_tower.probe": torch.full((1,), 4.0),
        prefix + "autoregressive_model.model.embed_audio.probe": torch.full((1,), 5.0),
    }


def _successful_merge(counter: list[int] | None = None, *, delay: float = 0.0):
    def merge(source, destination, *, safe_serialization, remove_checkpoint_dir):
        assert Path(source).name == "pytorch_model_fsdp_0"
        assert safe_serialization is True
        assert remove_checkpoint_dir is False
        if counter is not None:
            counter.append(1)
        if delay:
            time.sleep(delay)
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        save_file(_wrapper_state(), destination / "model.safetensors")

    return merge


def test_export_writes_verified_weights_and_source_sidecar(tmp_path):
    exporter = _load_exporter()
    checkpoint = _make_checkpoint(tmp_path, update=100, trained_hash="b" * 64)
    output = tmp_path / "export"

    payload = exporter.export_fsdp_diffusion_checkpoint(
        checkpoint, output, merge_function=_successful_merge()
    )

    weights = output / "model.safetensors"
    sidecar = output / "fsdp_export_manifest.json"
    assert weights.is_file()
    assert sidecar.is_file()
    assert json.loads(sidecar.read_text(encoding="utf-8")) == payload
    assert payload["checkpoint"] == str(checkpoint.resolve())
    assert payload["checkpoint_update"] == 100
    assert payload["source_trained_model_sha256"] == "b" * 64
    assert payload["source_dcp_dir"] == str(
        (checkpoint / "full_model" / "pytorch_model_fsdp_0").resolve()
    )
    assert payload["weights_file"] == "model.safetensors"
    assert payload["weights_sha256"] == exporter._sha256_file(weights)
    assert payload["key_count"] == len(_wrapper_state())


def test_export_accepts_accelerate_outer_model_prefix(tmp_path):
    exporter = _load_exporter()
    checkpoint = _make_checkpoint(tmp_path)

    def merge(_source, destination, **_kwargs):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        save_file(_wrapper_state(model_prefix=True), destination / "model.safetensors")

    result = exporter.export_fsdp_diffusion_checkpoint(
        checkpoint, tmp_path / "export", merge_function=merge
    )
    assert result["key_count"] == len(_wrapper_state())


def test_valid_export_is_reused_without_second_merge(tmp_path):
    exporter = _load_exporter()
    checkpoint = _make_checkpoint(tmp_path)
    output = tmp_path / "export"
    first = exporter.export_fsdp_diffusion_checkpoint(
        checkpoint, output, merge_function=_successful_merge()
    )

    def forbidden_merge(*_args, **_kwargs):
        raise AssertionError("valid export should have been reused")

    second = exporter.export_fsdp_diffusion_checkpoint(
        checkpoint, output, merge_function=forbidden_merge
    )
    assert second == first


def test_concurrent_callers_merge_only_once(tmp_path):
    exporter = _load_exporter()
    checkpoint = _make_checkpoint(tmp_path)
    output = tmp_path / "export"
    counter: list[int] = []
    merge = _successful_merge(counter, delay=0.1)
    barrier = threading.Barrier(2)
    results = []
    errors = []

    def run():
        try:
            barrier.wait()
            results.append(
                exporter.export_fsdp_diffusion_checkpoint(
                    checkpoint, output, merge_function=merge
                )
            )
        except BaseException as exc:  # pragma: no cover - diagnostic capture
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not errors
    assert len(results) == 2
    assert results[0] == results[1]
    assert len(counter) == 1


def test_export_for_another_checkpoint_is_refused(tmp_path):
    exporter = _load_exporter()
    first = _make_checkpoint(tmp_path / "first", update=100, trained_hash="a" * 64)
    second = _make_checkpoint(tmp_path / "second", update=200, trained_hash="c" * 64)
    output = tmp_path / "export"
    exporter.export_fsdp_diffusion_checkpoint(
        first, output, merge_function=_successful_merge()
    )

    with pytest.raises(ValueError, match="another checkpoint"):
        exporter.export_fsdp_diffusion_checkpoint(
            second, output, merge_function=_successful_merge()
        )


def test_failed_merge_never_publishes_completion_files(tmp_path):
    exporter = _load_exporter()
    checkpoint = _make_checkpoint(tmp_path)
    output = tmp_path / "export"

    def failing_merge(_source, destination, **_kwargs):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        save_file(_wrapper_state(), destination / "model.safetensors")
        raise RuntimeError("synthetic merge failure")

    with pytest.raises(RuntimeError, match="synthetic merge failure"):
        exporter.export_fsdp_diffusion_checkpoint(
            checkpoint, output, merge_function=failing_merge
        )
    assert not (output / "model.safetensors").exists()
    assert not (output / "fsdp_export_manifest.json").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("strategy", "lora", "strategy='full'"),
        ("checkpoint_format", "huggingface", "checkpoint_format"),
        ("update", "100", "invalid update"),
    ],
)
def test_invalid_checkpoint_manifest_is_rejected(tmp_path, field, value, message):
    exporter = _load_exporter()
    checkpoint = _make_checkpoint(tmp_path)
    manifest_path = checkpoint / "diffusion_checkpoint.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        exporter.export_fsdp_diffusion_checkpoint(
            checkpoint, tmp_path / "export", merge_function=_successful_merge()
        )


def test_missing_wrapper_keys_are_rejected_without_publication(tmp_path):
    exporter = _load_exporter()
    checkpoint = _make_checkpoint(tmp_path)
    output = tmp_path / "export"

    def incomplete_merge(_source, destination, **_kwargs):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        save_file({"unrelated.weight": torch.ones(1)}, destination / "model.safetensors")

    with pytest.raises(ValueError, match="expected DiffASR wrapper state"):
        exporter.export_fsdp_diffusion_checkpoint(
            checkpoint, output, merge_function=incomplete_merge
        )
    assert not (output / "model.safetensors").exists()
    assert not (output / "fsdp_export_manifest.json").exists()
