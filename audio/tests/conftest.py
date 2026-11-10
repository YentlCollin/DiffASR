"""Tests requiring the withheld upstream implementation are explicitly skipped."""
from pathlib import Path
import pytest

UPSTREAM_TESTS = {
    "test_audio_position_safety.py",
    "test_xllm_diffusion_attention_mode.py",
    "test_diffusion_repository_loader.py",
}

def pytest_ignore_collect(collection_path, config):
    return (collection_path.name in UPSTREAM_TESTS and
            not (Path(__file__).resolve().parents[2] / "audio/model/modeling_xllm_aut.py").is_file())
