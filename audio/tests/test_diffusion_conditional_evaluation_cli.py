import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch
from torch import nn

from audio.diffusion.conditional import CONDITIONAL_ASR_VIEWS
from audio.diffusion.config import ConfidenceSamplerConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPOSITORY_ROOT / "examples" / "evaluate_mawjais_diffusion_conditional.py"


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _manifest_row(sample_key, *, valid=(True, True, True, True), content=True):
    return {
        "sample_key": sample_key,
        "audio_tar_path": "audio/train.tar",
        "audio_member": f"{sample_key}.wav",
        "duration_sec": 1.0,
        "country_code": "EG",
        "content_has_code_switching": content,
        "valid_view_mask": valid,
        "target_plain": "plain",
        "target_diacritics": "vocalized",
        "target_code_switch": "hello",
        "target_code_switch_diacritics": "vocalized hello",
        "canvas_sizes": [4, 4, 4, 4],
    }


def _prediction(sample_key, view_id):
    view = CONDITIONAL_ASR_VIEWS[view_id]
    return {
        "sample_key": sample_key,
        "view_id": view_id,
        "view_name": view.name,
        "request_code_switching": view.request_code_switching,
        "request_diacritics": view.request_diacritics,
        "content_has_code_switching": True,
        "reference": "hello" if view.request_code_switching else "هالو",
        "hypothesis": "[EG] hello" if view.request_code_switching else "[EG] هالو",
        "country_code": "EG",
        "duration_sec": 1.0,
        "generation_seconds": 0.1,
        "inference_prompt_tag": view.prompt_tag,
        "checkpoint": "/checkpoint",
        "checkpoint_update": 100,
        "manifest_sha256": "manifest-hash",
        "sampler": "confidence_schedule",
        "nfe": 4,
        "evaluation_seed": 3421,
    }


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_cli_exposes_resumable_shards_and_exact_merge():
    module = _load_script("conditional_eval_cli_parser_test")
    help_text = module.build_parser().format_help()
    assert "run" in help_text
    assert "score" in help_text
    run_help = module.build_parser()._subparsers._group_actions[0].choices[
        "run"
    ].format_help()
    for option in (
        "--checkpoint",
        "--fsdp-merged-weights",
        "--manifest",
        "--output-jsonl",
        "--limit",
        "--num-shards",
        "--shard-index",
        "--shard-mode",
        "--nfe",
        "--dry-run",
    ):
        assert option in run_help


def test_production_parquet_requires_and_verifies_signed_companion(tmp_path):
    module = _load_script("conditional_eval_signature_test")
    parquet = tmp_path / "validation-metadata.parquet"
    parquet.write_bytes(b"signed-parquet-payload")
    with pytest.raises(FileNotFoundError, match="signed validation manifest"):
        module.validate_manifest_signature(parquet)

    payload = {
        "schema_version": 1,
        "metadata_path": parquet.name,
        "metadata_sha256": module._sha256_file(parquet),
        "rows": 12,
    }
    payload["content_sha256"] = module.hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    companion = parquet.with_suffix(".manifest.json")
    companion.write_text(json.dumps(payload), encoding="utf-8")
    signature = module.validate_manifest_signature(parquet)
    assert signature["content_sha256"] == payload["content_sha256"]

    parquet.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        module.validate_manifest_signature(parquet)


def test_global_limit_precedes_sample_level_sharding_and_skips_incompatible():
    module = _load_script("conditional_eval_sharding_test")
    rows = [_manifest_row(f"sample-{index:02d}") for index in range(10)]
    rows.append(_manifest_row("invalid", valid=(True, False, True, True)))
    shards = []
    for index in range(3):
        selected, counts = module.select_compatible_rows(
            rows, limit=7, num_shards=3, shard_index=index
        )
        shards.append({row["sample_key"] for row in selected})
        assert counts["compatible_rows"] == 10
        assert counts["incompatible_rows_skipped"] == 1
        assert counts["globally_selected_rows"] == 7
    assert not (shards[0] & shards[1] or shards[0] & shards[2] or shards[1] & shards[2])
    assert set.union(*shards) == {f"sample-{index:02d}" for index in range(7)}


def test_tar_balanced_sharding_never_splits_an_archive():
    module = _load_script("conditional_eval_tar_sharding_test")
    rows = []
    for tar_index, count in enumerate((7, 5, 3, 2)):
        for sample_index in range(count):
            row = _manifest_row(f"tar-{tar_index}-sample-{sample_index}")
            row["audio_tar_path"] = f"audio/tar-{tar_index}.tar"
            row["duration_sec"] = float(sample_index + 1)
            rows.append(row)
    ownership = {}
    selected_keys = set()
    for shard_index in range(3):
        selected, counts = module.select_compatible_rows(
            rows,
            limit=None,
            num_shards=3,
            shard_index=shard_index,
            shard_mode="tar_balanced",
        )
        assert counts["shard_mode"] == "tar_balanced"
        selected_keys.update(str(row["sample_key"]) for row in selected)
        for row in selected:
            tar_name = str(row["audio_tar_path"])
            assert ownership.setdefault(tar_name, shard_index) == shard_index
    assert selected_keys == {str(row["sample_key"]) for row in rows}


def test_merge_deduplicates_identical_rows_scores_exactly_and_rejects_conflicts(
    tmp_path,
):
    module = _load_script("conditional_eval_merge_test")
    first = [_prediction("a", view.view_id) for view in CONDITIONAL_ASR_VIEWS]
    second = [_prediction("b", view.view_id) for view in CONDITIONAL_ASR_VIEWS]
    shard_a = tmp_path / "a.jsonl"
    shard_b = tmp_path / "b.jsonl"
    duplicate = tmp_path / "duplicate.jsonl"
    _write_jsonl(shard_a, first)
    _write_jsonl(shard_b, second)
    _write_jsonl(duplicate, [first[0]])

    merged = module.merge_prediction_rows([shard_a, shard_b, duplicate])
    assert len(merged) == 8
    report = module.score_conditional_predictions(merged)
    assert report["paired_samples"] == 2
    assert report["aggregate"]["cer"] == 0.0

    merged_path = tmp_path / "checkpoint-000100" / "all.jsonl"
    summary_path = tmp_path / "checkpoint-000100" / "metrics.json"
    scored = module.run_scoring(
        SimpleNamespace(
            predictions=[shard_a, shard_b],
            merged_jsonl=merged_path,
            summary_json=summary_path,
            artifacts_dir=None,
            manifest=None,
            score_workers=1,
        )
    )
    artifact_dir = summary_path.parent / "artifacts"
    assert scored["artifacts"]["directory"] == str(artifact_dir.resolve())
    assert merged_path.is_file()
    assert summary_path.is_file()
    assert (artifact_dir / "conditional_metrics.csv").is_file()
    svg = (artifact_dir / "conditional_metrics.svg").read_text(encoding="utf-8")
    assert "Évaluation DiffASR conditionnelle" in svg
    artifact_manifest = json.loads(
        (artifact_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert set(artifact_manifest["files"]) == {
        "merged_jsonl",
        "metrics_csv",
        "metrics_svg",
        "summary_json",
    }

    conflicting = dict(first[0])
    conflicting["hypothesis"] = "[EG] conflict"
    _write_jsonl(duplicate, [conflicting])
    with pytest.raises(ValueError, match="conflicting duplicate"):
        module.merge_prediction_rows([shard_a, duplicate])


class _Tokenizer:
    def decode(self, ids, **kwargs):
        del kwargs
        return "[EG] decoded-" + "-".join(str(value) for value in ids)


class _Processor:
    tokenizer = _Tokenizer()

    def __init__(self):
        self.prompts = []

    def __call__(self, *, text, audio, **kwargs):
        del audio, kwargs
        self.prompts.extend(text)
        return {
            "input_ids": torch.tensor([[2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[True, True]]),
            "audio_position_mask": torch.tensor([[False, True]]),
            "input_features": torch.zeros(1, 2, 3),
            "feature_attention_mask": torch.ones(1, 3, dtype=torch.long),
        }


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.encode_calls = 0

    def encode_audio(self, features, mask):
        del features, mask
        self.encode_calls += 1
        return torch.zeros(1, 2)


def test_four_view_generation_reuses_audio_projection_and_resets_paired_seed(monkeypatch):
    module = _load_script("conditional_eval_generation_test")
    seeds = []

    def fake_sample(denoiser, process, shape, **kwargs):
        del denoiser, process
        seeds.append(kwargs["generator"].initial_seed())
        canvas = torch.tensor([[7, 14, 14, 14]], dtype=torch.long)
        assert shape == (1, 4)
        return SimpleNamespace(canvas=canvas, nfe=4)

    monkeypatch.setattr(module, "sample_diffusion", fake_sample)
    model = _Model()
    processor = _Processor()
    diffusion = SimpleNamespace(
        denoiser_output_alignment="same_position",
        canvas=SimpleNamespace(
            sizes=(4,),
            duration_tokens_per_second=1.0,
            duration_margin_tokens=1.0,
            turn_end_token_id=14,
            pad_token_id=1,
        ),
    )
    sampler = ConfidenceSamplerConfig(
        sampler="confidence_schedule", max_nfe=4, prediction_mode="argmax"
    )
    rows = module.generate_four_views(
        row=_manifest_row("paired"),
        waveform=np.zeros(160, dtype=np.float32),
        model=model,
        processor=processor,
        diffusion=diffusion,
        noise_process=object(),
        sampler_config=sampler,
        forbidden_output_token_ids=(),
        layout=SimpleNamespace(root=Path("/checkpoint"), metadata={"update": 100}),
        manifest_sha256="hash",
        evaluation_seed=123,
    )

    assert len(rows) == 4
    assert model.encode_calls == 1
    assert len(set(seeds)) == 1
    assert [row["inference_prompt_tag"] for row in rows] == [
        view.prompt_tag for view in CONDITIONAL_ASR_VIEWS
    ]
    assert [row["reference"] for row in rows] == [
        "plain",
        "vocalized",
        "hello",
        "vocalized hello",
    ]
