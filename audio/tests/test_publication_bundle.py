"""Portable configuration, preprocessing and CLI integration checks (no GPU)."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from audio.diffusion.config import ConfidenceSamplerConfig, DiffusionConfig
from audio.diffusion.experiment_config import load_experiment_config, validate_experiment_config
from audio.diffusion_data import process_training_transcription

ROOT = Path(__file__).resolve().parents[2]


def load_example(filename):
    name = f"bundle_{filename}"
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / f"{filename}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("config", sorted((ROOT / "configs/diffusion").glob("*.yaml")))
def test_published_configs_resolve_without_external_files(config):
    resolved = load_experiment_config(config)
    assert resolved["model"]["freeze_audio_tower"]
    assert resolved["model"]["freeze_audio_projector"]
    assert not resolved["diffusion"]["use_iterative_self_correction"]


def test_explicit_epoch_plan_still_rejects_wrong_update_total():
    config = deepcopy(load_experiment_config(ROOT / "configs/diffusion/base.yaml"))
    config["training"]["expected_epoch_updates"] = [1, 2]
    with pytest.raises(ValueError, match="sum"):
        validate_experiment_config(config)


@pytest.mark.parametrize("code_switch", [False, True])
def test_public_preprocessing_preserves_shadda_and_tatweel(code_switch):
    original = "السُّؤَالُ مِنْكَـ"
    plain = process_training_transcription(original, use_diacritics=False,
                                           use_code_switching=code_switch)
    vocalized = process_training_transcription(original, use_diacritics=True,
                                               use_code_switching=code_switch)
    assert plain == "السّؤال منكـ"
    assert "ُ" in vocalized and "ّ" in vocalized and "ـ" in vocalized


@pytest.mark.parametrize("sampler,mode", [("confidence_schedule", "argmax"),
                                       ("argmax_udm", "argmax"),
                                       ("udm_marginalization", "categorical")])
def test_inference_and_evaluation_select_compatible_sampler_settings(sampler, mode):
    inference = load_example("infer_mawjais_diffusion")
    evaluation = load_example("evaluate_mawjais_diffusion_conditional")
    resolved = load_experiment_config(ROOT / "configs/diffusion/base.yaml")
    args = SimpleNamespace(sampler=sampler, nfe=4, prediction_mode=None,
                           temperature_start=None, temperature_end=None,
                           entropy_budget_gamma=None, position_bias_lambda=None,
                           adaptive_stopping=False)
    controls = inference._resolve_sampling_controls(resolved, args)
    inference._validate_sampling_semantics(DiffusionConfig(noise_type="uniform"), controls)
    cfg = ConfidenceSamplerConfig(**{k: v for k, v in controls.items() if k != "adaptive_stopping"})
    assert cfg.sampler == sampler and cfg.prediction_mode == mode
    eval_cfg = evaluation._sampling_config(inference, resolved,
                                                DiffusionConfig(noise_type="uniform"), args)
    assert eval_cfg.sampler == sampler and eval_cfg.prediction_mode == mode
    assert eval_cfg.max_nfe == cfg.max_nfe == 4
    configured = deepcopy(resolved)
    configured["sampling"].update(sampler=sampler, prediction_mode=mode,
                                   temperature_start=cfg.temperature_start,
                                   temperature_end=cfg.temperature_end)
    assert validate_experiment_config(configured)["sampling"]["sampler"] == sampler
    parsed = inference.build_parser().parse_args([
        "--checkpoint", "checkpoint-001000", "--wav", "example.wav", "--sampler", sampler])
    assert parsed.sampler == sampler
    parsed_eval = evaluation.build_parser().parse_args([
        "run", "--checkpoint", "checkpoint-001000", "--manifest", "eval.jsonl",
        "--dataset-root", "data", "--output-jsonl", "out.jsonl", "--sampler", sampler])
    assert parsed_eval.sampler == sampler


def test_udm_rejects_mask_checkpoint_before_loading_weights():
    inference = load_example("infer_mawjais_diffusion")
    with pytest.raises(ValueError, match="uniform"):
        inference._validate_sampling_semantics(DiffusionConfig(noise_type="mask"), {
            "sampler": "argmax_udm", "max_nfe": 4, "prediction_mode": "argmax",
            "temperature_start": 1.0, "temperature_end": 1.0,
        })


def test_node_launcher_preserves_paths_and_resume_arguments(tmp_path):
    command = ["bash", str(ROOT / "examples/run_training_node.sh"),
               str(tmp_path / "config with spaces.yaml"), str(tmp_path / "output run"),
               str(tmp_path / "parity.json"), "--resume-from-checkpoint", "latest",
               "--stop-after-updates", "100"]
    env = dict(os.environ, MAWJAIS_REPOSITORY_DIR=str(ROOT), GPUS_PER_NODE="8",
               SLURM_NNODES="2", SLURM_NODEID="1", MASTER_ADDR="node-a",
               MASTER_PORT="29512", TRAINING_DRY_RUN="1", CONTAINER_IMAGE="")
    result = subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    tokens = shlex.split(result.stdout)
    assert "--node-rank=1" in tokens and "--nnodes=2" in tokens
    assert "--nproc-per-node=8" in tokens
    assert str(tmp_path / "output run") in tokens
    assert tokens[-4:] == ["--resume-from-checkpoint", "latest", "--stop-after-updates", "100"]


def test_submission_dry_run_never_submits_or_creates_output(tmp_path):
    config = tmp_path / "config.yaml"
    parity = tmp_path / "parity.json"
    config.write_text("{}")
    parity.write_text("{}")
    output = tmp_path / "future run"
    logs = tmp_path / "logs"
    env = dict(os.environ, TRAIN_NODES="2", GPUS_PER_NODE="4", TRAIN_LOG_DIR=str(logs))
    result = subprocess.run(["bash", str(ROOT / "examples/submit_training.sh"),
                             "--dry-run", str(config), str(output), str(parity)],
                            env=env, capture_output=True, text=True, check=True)
    tokens = shlex.split(result.stdout)
    assert tokens[0] == "sbatch" and "--nodes=2" in tokens
    assert "--gres=gpu:4" in tokens
    assert str(output) in tokens
    assert not output.exists() and not logs.exists()
