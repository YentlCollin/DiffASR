from __future__ import annotations

import io
import json
import tarfile
import wave
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from audio.diffusion_data import ConditionalLocalTarDataset
from examples.prepare_legal_ytb_conditional_metadata import (
    finalize_tasks,
    run_task,
)


RAW_CS = "مَرْحَبًا فِي [يُوتْيُوبْ:CS] [EN: YouTube] الْيَوْمَ."


class CharacterTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in text]}


def _wav_bytes(frequency: float) -> bytes:
    sample_rate = 8_000
    time = np.arange(sample_rate // 50, dtype=np.float32) / sample_rate
    samples = (0.2 * np.sin(2 * np.pi * frequency * time) * 32767).astype(np.int16)
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())
    return output.getvalue()


def _write_tar(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def _source_row(
    key: str,
    tar_path: str,
    transcription: str,
    *,
    country_code: str = "DZ",
    has_code_switching: bool = True,
) -> dict[str, object]:
    return {
        "sample_key": key,
        "audio_tar_path": tar_path,
        "audio_member": f"{key}.wav",
        "duration_sec": 1.0,
        "target_json": json.dumps(
            {
                "transcription": transcription,
                "country_code": country_code,
                "has_code_switching": has_code_switching,
            },
            ensure_ascii=False,
        ),
    }


def _build_source(tmp_path: Path) -> tuple[Path, Path, Path, list[dict[str, object]]]:
    root = tmp_path / "source"
    rows = [
        _source_row("v3__QA__video-a__0-0", "data/audio/a.tar", RAW_CS),
        _source_row(
            "v3__QA__video-a__1-0",
            "data/audio/a.tar",
            "مَرْحَبًا بِكُمْ",
            has_code_switching=False,
        ),
        # Only plain/diacritics are valid, exercising filtered view cycling.
        _source_row(
            "v3__QA__video-b__0-0",
            "data/audio/b.tar",
            "هَذَا [يُوتْيُوبْ:CS]",
        ),
        # Held out before sidecar/shard assignment.
        _source_row("v3__QA__heldout__0-0", "data/audio/b.tar", RAW_CS),
        # All views invalid and therefore present only in exclusions.
        _source_row("v3__QA__video-c__0-0", "data/audio/c.tar", "هَذَا [مكسور"),
    ]
    by_tar: dict[str, dict[str, bytes]] = {}
    for index, row in enumerate(rows):
        by_tar.setdefault(str(row["audio_tar_path"]), {})[str(row["audio_member"])] = (
            _wav_bytes(200 + index * 10)
        )
    for relative, members in by_tar.items():
        _write_tar(root / relative, members)
    metadata = root / "data/training.parquet"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), metadata, row_group_size=2)
    split = tmp_path / "train-excluded.jsonl"
    split.write_text('{"video_key":"v3__QA__heldout","split":"validation"}\n')
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    (tokenizer_path / "tokenizer.json").write_text('{"test":true}\n')
    return root, metadata, split, rows


def _prepare(tmp_path: Path, *, max_source_rows: int | None = None) -> tuple[Path, Path]:
    root, metadata, split, _ = _build_source(tmp_path)
    tokenizer_path = tmp_path / "tokenizer"
    output = tmp_path / "conditional"
    run_task(
        source_metadata=metadata,
        split_manifest=split,
        output_dir=output,
        tokenizer_path=tokenizer_path,
        tokenizer=CharacterTokenizer(),
        repo_id="test/legal-ytb",
        revision="fixed-revision",
        canvas_sizes=[128],
        tokens_per_second=1.0,
        margin_tokens=0.0,
        view_seed=77,
        task_count=1,
        task_index=0,
        parquet_batch_size=2,
        sidecar_row_group_size=1,
        max_source_rows=max_source_rows,
    )
    finalize_tasks(
        output_dir=output,
        task_count=1,
        expected_train_rows=None if max_source_rows is not None else 4,
    )
    return root, output


def test_preparation_filters_split_before_index_and_records_all_invalid(tmp_path: Path) -> None:
    _, output = _prepare(tmp_path)
    index = json.loads((output / "metadata-index.json").read_text())
    assert index["totals"]["assigned_train_rows_including_exclusions"] == 4
    assert index["totals"]["train_rows"] == 3
    assert index["totals"]["all_invalid_rows_excluded"] == 1
    assert index["totals"]["any_valid_rows"] == 3
    assert all(partition["tar_path"] != "data/audio/c.tar" for partition in index["partitions"])
    sidecar_keys = set()
    for partition in index["partitions"]:
        table = pq.read_table(output / partition["sidecar_path"])
        sidecar_keys.update(table.column("sample_key").to_pylist())
    assert "v3__QA__heldout__0-0" not in sidecar_keys
    assert "v3__QA__video-c__0-0" not in sidecar_keys


def test_loader_streams_without_extraction_and_cycles_valid_views(tmp_path: Path) -> None:
    root, output = _prepare(tmp_path)
    files_before = {path.relative_to(root) for path in root.rglob("*") if path.is_file()}
    dataset = ConditionalLocalTarDataset(
        local_root=root,
        metadata_index_path=output / "metadata-index.json",
        view_seed=77,
        seed=12,
        shuffle_buffer_size=1,
        target_sample_rate=16_000,
    )
    by_epoch: list[dict[str, dict[str, object]]] = []
    for epoch in range(4):
        dataset.set_epoch(epoch)
        samples = {sample["sample_key"]: sample for sample in dataset}
        assert len(samples) == 3
        assert all(sample["country_code"] == "DZ" for sample in samples.values())
        assert all(sample["target_transcription"] == sample["transcription"] for sample in samples.values())
        assert all(sample["sampling_rate"] == 16_000 for sample in samples.values())
        by_epoch.append(samples)

    partial = "v3__QA__video-b__0-0"
    assert {epoch[partial]["view_id"] for epoch in by_epoch} == {0, 1}
    assert all(epoch[partial]["view_id"] in (0, 1) for epoch in by_epoch)
    full = "v3__QA__video-a__0-0"
    assert {epoch[full]["view_id"] for epoch in by_epoch} == {0, 1, 2, 3}
    for epoch in by_epoch:
        sample = epoch[full]
        assert sample["target"]["request_code_switching"] == sample["request_code_switching"]
        assert sample["target"]["applied_code_switching"] == sample["applied_code_switching"]
        if sample["request_code_switching"]:
            assert sample["applied_code_switching"] is True

    restored = ConditionalLocalTarDataset(
        local_root=root,
        metadata_index_path=output / "metadata-index.json",
        view_seed=77,
        seed=12,
        shuffle_buffer_size=1,
    )
    restored.load_state_dict(dataset.state_dict())
    assert [item["view_id"] for item in restored] == [item["view_id"] for item in dataset]
    files_after = {path.relative_to(root) for path in root.rglob("*") if path.is_file()}
    assert files_after == files_before


def test_dry_run_max_source_rows_is_exact(tmp_path: Path) -> None:
    _, output = _prepare(tmp_path, max_source_rows=3)
    task = json.loads(next((output / "tasks").glob("*.json")).read_text())
    assert task["dry_run"] is True
    assert task["statistics"]["source_rows_scanned"] == 3
    assert task["statistics"]["assigned_train_rows"] == 3
    index = json.loads((output / "metadata-index.json").read_text())
    assert index["totals"]["assigned_train_rows_including_exclusions"] == 3


def test_distributed_counts_use_prevalidated_rows(tmp_path: Path) -> None:
    root, output = _prepare(tmp_path)
    datasets = [
        ConditionalLocalTarDataset(
            local_root=root,
            metadata_index_path=output / "metadata-index.json",
            view_seed=77,
            rank=rank,
            world_size=2,
            shuffle_buffer_size=1,
        )
        for rank in range(2)
    ]
    loads = datasets[0].samples_per_rank()
    assert loads is not None
    assert loads == datasets[1].samples_per_rank()
    observed = [{sample["sample_key"] for sample in dataset} for dataset in datasets]
    assert len(observed[0]) == len(observed[1]) == min(loads)
    assert observed[0].isdisjoint(observed[1])
