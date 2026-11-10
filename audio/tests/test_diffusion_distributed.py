from __future__ import annotations

import ast
from copy import deepcopy
import importlib.util
import json
import multiprocessing
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn

from audio.diffusion.distributed import (
    collective_unexpected_overflow,
    gather_rank_objects,
    reduce_additive_values,
    reduce_max_value,
    resolve_distributed_training_layout,
    step_raw_scheduler_for_synchronized_update,
    validate_rank_states,
)
from audio.diffusion_data.exclusions import load_sample_exclusion_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "examples" / "train_mawjais_diffusion.py"
SPEC = importlib.util.spec_from_file_location(
    "mawjais_diffusion_distributed_trainer_test", SCRIPT_PATH
)
trainer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = trainer
SPEC.loader.exec_module(trainer)


def _gloo_worker(rank: int, world_size: int, rendezvous: str, result_dir: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        reduced = reduce_additive_values(
            {"tokens": rank + 1, "updates": 1},
            device=torch.device("cpu"),
            world_size=world_size,
        )
        overflow = collective_unexpected_overflow(
            rank == 1,
            device=torch.device("cpu"),
            world_size=world_size,
        )
        maximum = reduce_max_value(
            rank + 0.5,
            device=torch.device("cpu"),
            world_size=world_size,
        )
        generator_state = torch.Generator(device="cpu").manual_seed(1000 + rank).get_state()
        gathered_rich = gather_rank_objects(
            {
                "rank": rank,
                "cursor": f"cursor-{rank}",
                "rng_state": generator_state,
                "nested_tuple": ("rank", rank, (rank + 1, rank + 2)),
            },
            world_size=world_size,
        )
        rich_roundtrip_exact = all(
            item["rank"] == source_rank
            and item["cursor"] == f"cursor-{source_rank}"
            and item["nested_tuple"]
            == ("rank", source_rank, (source_rank + 1, source_rank + 2))
            and item["rng_state"].device.type == "cpu"
            and torch.equal(
                item["rng_state"],
                torch.Generator(device="cpu")
                .manual_seed(1000 + source_rank)
                .get_state(),
            )
            for source_rank, item in enumerate(gathered_rich)
        )
        gathered = [
            {"rank": item["rank"], "cursor": item["cursor"]}
            for item in gathered_rich
        ]
        Path(result_dir, f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "reduced": reduced,
                    "overflow": overflow,
                    "maximum": maximum,
                    "gathered": gathered,
                    "rich_roundtrip_exact": rich_roundtrip_exact,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("world_size", "accumulation"),
    ((1, 16), (2, 8), (4, 4), (8, 2)),
)
def test_global_batch_16_is_invariant(world_size: int, accumulation: int) -> None:
    layout = resolve_distributed_training_layout(
        {
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "global_batch_size": 16,
            "supported_world_sizes": [1, 2, 4, 8],
        },
        world_size=world_size,
    )
    assert layout.gradient_accumulation_steps == accumulation
    assert world_size * layout.per_device_batch_size * accumulation == 16


def test_global_batch_16_uses_one_microbatch_on_sixteen_ranks() -> None:
    layout = resolve_distributed_training_layout(
        {
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "global_batch_size": 16,
            "supported_world_sizes": [1, 2, 4, 8, 16],
        },
        world_size=16,
    )
    assert layout.gradient_accumulation_steps == 1
    assert layout.global_batch_size == 16


@pytest.mark.parametrize(
    ("world_size", "accumulation"),
    ((32, 4), (64, 2), (128, 1)),
)
def test_full_corpus_global_batch_128_scales_to_sixteen_nodes(
    world_size: int, accumulation: int
) -> None:
    layout = resolve_distributed_training_layout(
        {
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "global_batch_size": 128,
            "supported_world_sizes": [32, 64, 128],
        },
        world_size=world_size,
    )
    assert layout.gradient_accumulation_steps == accumulation
    assert world_size * layout.per_device_batch_size * accumulation == 128


def test_global_batch_rejects_unsupported_or_indivisible_layout() -> None:
    base = {
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "global_batch_size": 16,
        "supported_world_sizes": [1, 2, 4, 8],
    }
    with pytest.raises(ValueError, match="not supported"):
        resolve_distributed_training_layout(base, world_size=3)
    with pytest.raises(ValueError, match="divisible"):
        resolve_distributed_training_layout(
            {**base, "per_device_batch_size": 3}, world_size=2
        )


@pytest.mark.parametrize(
    ("world_size", "accumulation"),
    ((1, 16), (2, 8), (4, 4), (8, 2)),
)
def test_raw_scheduler_steps_once_per_successful_global_update_for_every_topology(
    world_size: int,
    accumulation: int,
) -> None:
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.12)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: max(0.0, 1.0 - update / 6.0)
    )
    initial_last_epoch = scheduler.last_epoch
    scheduler_steps = 0
    boundary_lrs: list[float] = []
    attempted_updates = 0

    for microbatch in range(5 * accumulation):
        sync_gradients = (microbatch + 1) % accumulation == 0
        if sync_gradients:
            attempted_updates += 1
        optimizer_step_was_skipped = sync_gradients and attempted_updates == 3
        if sync_gradients and not optimizer_step_was_skipped:
            optimizer.step()
        stepped = step_raw_scheduler_for_synchronized_update(
            scheduler,
            sync_gradients=sync_gradients,
            optimizer_step_was_skipped=optimizer_step_was_skipped,
        )
        scheduler_steps += int(stepped)
        if sync_gradients:
            boundary_lrs.append(scheduler.get_last_lr()[0])

    assert world_size * accumulation == 16
    assert scheduler_steps == 4
    assert scheduler.last_epoch - initial_last_epoch == 4
    assert boundary_lrs == pytest.approx([0.10, 0.08, 0.08, 0.06, 0.04])


def test_continuation_cosine_warms_and_decays_over_only_the_new_50k_updates():
    multiplier, warmup = trainer.build_learning_rate_multiplier(
        {
            "max_updates": 100000,
            "scheduler_type": "continuation_cosine",
            "continuation_start_update": 50000,
            "scheduler_warmup_updates": 1000,
            "scheduler_min_lr_ratio": 0.1,
        }
    )
    assert warmup == 1000
    assert multiplier(0) == pytest.approx(0.1009)
    assert multiplier(999) == pytest.approx(1.0)
    assert multiplier(1000) == pytest.approx(1.0)
    assert multiplier(25500) == pytest.approx(0.55)
    assert multiplier(50000) == pytest.approx(0.1)


def test_fresh_cosine_has_explicit_625_update_warmup_and_two_epoch_horizon():
    multiplier, warmup = trainer.build_learning_rate_multiplier(
        {
            "max_updates": 58862,
            "scheduler_type": "cosine",
            "scheduler_warmup_updates": 625,
            "scheduler_min_lr_ratio": 0.1,
            "scheduler_horizon_updates": 58862,
        }
    )
    assert warmup == 625
    assert multiplier(0) == pytest.approx(1 / 625)
    assert multiplier(624) == pytest.approx(1.0)
    assert multiplier(625) == pytest.approx(1.0)
    assert multiplier(58862) == pytest.approx(0.1)
    assert multiplier(70000) == pytest.approx(0.1)


def test_raw_scheduler_state_dict_resume_preserves_exact_trajectory() -> None:
    def build():
        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=0.2)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda update: max(0.0, 1.0 - update / 8.0)
        )
        return optimizer, scheduler

    optimizer, scheduler = build()
    for _ in range(3):
        optimizer.step()
        assert step_raw_scheduler_for_synchronized_update(
            scheduler,
            sync_gradients=True,
            optimizer_step_was_skipped=False,
        )
    saved_optimizer = deepcopy(optimizer.state_dict())
    saved_scheduler = deepcopy(scheduler.state_dict())

    for _ in range(2):
        optimizer.step()
        step_raw_scheduler_for_synchronized_update(
            scheduler,
            sync_gradients=True,
            optimizer_step_was_skipped=False,
        )
    expected_scheduler = deepcopy(scheduler.state_dict())
    expected_lr = scheduler.get_last_lr()

    resumed_optimizer, resumed_scheduler = build()
    resumed_optimizer.load_state_dict(saved_optimizer)
    resumed_scheduler.load_state_dict(saved_scheduler)
    assert resumed_scheduler.state_dict() == saved_scheduler
    for _ in range(2):
        resumed_optimizer.step()
        step_raw_scheduler_for_synchronized_update(
            resumed_scheduler,
            sync_gradients=True,
            optimizer_step_was_skipped=False,
        )

    assert resumed_scheduler.state_dict() == expected_scheduler
    assert resumed_scheduler.get_last_lr() == expected_lr


def test_diffusion_trainer_keeps_scheduler_out_of_accelerator_prepare() -> None:
    for filename in (
        "train_mawjais_diffusion.py",
    ):
        source = (REPO_ROOT / "examples" / filename).read_text(encoding="utf-8")
        tree = ast.parse(source)
        prepare_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "prepare"
        ]
        assert prepare_calls, filename
        assert all(len(node.args) == 2 for node in prepare_calls), filename
        direct_scheduler_steps = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "step"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "scheduler"
        ]
        assert direct_scheduler_steps == [], filename


def test_rank_object_roundtrip_preserves_nested_tuples_and_rng_tensors() -> None:
    rng_state = torch.Generator(device="cpu").manual_seed(1234).get_state()
    payload = {
        "python_rng_shape": (3, (1, 2, 3), None),
        "rng_state": rng_state,
    }
    [restored] = gather_rank_objects(payload, world_size=1)
    assert restored["python_rng_shape"] == payload["python_rng_shape"]
    assert restored["rng_state"].device.type == "cpu"
    assert torch.equal(restored["rng_state"], rng_state)
    assert restored is not payload


def test_rank_object_collective_receives_only_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    def fake_all_gather_object(outputs: list[object], local_payload: object) -> None:
        observed["wire_type"] = type(local_payload)
        assert type(local_payload) is bytes
        outputs[:] = [local_payload, local_payload]

    monkeypatch.setattr(dist, "all_gather_object", fake_all_gather_object)
    rng_state = torch.Generator(device="cpu").manual_seed(5678).get_state()
    gathered = gather_rank_objects(
        {"rng": rng_state, "nested": ("a", (1, 2))},
        world_size=2,
    )
    assert observed["wire_type"] is bytes
    assert len(gathered) == 2
    assert all(torch.equal(item["rng"], rng_state) for item in gathered)
    assert all(item["nested"] == ("a", (1, 2)) for item in gathered)


def test_rank_object_collective_rejects_non_bytes_from_any_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    def fake_all_gather_object(outputs: list[object], local_payload: object) -> None:
        outputs[:] = [local_payload, bytearray(b"not-an-accepted-wire-payload")]

    monkeypatch.setattr(dist, "all_gather_object", fake_all_gather_object)
    with pytest.raises(RuntimeError, match="non-bytes payload for rank 1"):
        gather_rank_objects({"rank": 0}, world_size=2)


def test_rank_state_validation_rejects_wrong_types_and_rank_order() -> None:
    with pytest.raises(ValueError, match="must be mappings"):
        validate_rank_states([{"rank": 0, "world_size": 2}, None], world_size=2)
    with pytest.raises(ValueError, match="invalid rank values"):
        validate_rank_states(
            [
                {"rank": 0, "world_size": 2},
                {"rank": "1", "world_size": 2},
            ],
            world_size=2,
        )
    with pytest.raises(ValueError, match="ordered and complete"):
        validate_rank_states(
            [
                {"rank": 1, "world_size": 2},
                {"rank": 0, "world_size": 2},
            ],
            world_size=2,
        )


def test_cpu_gloo_reduces_metrics_guards_overflow_and_gathers_rank_state(
    tmp_path: Path,
) -> None:
    world_size = 2
    rendezvous = str(tmp_path / "gloo-rendezvous")
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_gloo_worker,
            args=(rank, world_size, rendezvous, str(tmp_path)),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    for rank in range(world_size):
        payload = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert payload["reduced"] == {"tokens": 3.0, "updates": 2.0}
        assert payload["overflow"] is True
        assert payload["maximum"] == 1.5
        assert payload["rich_roundtrip_exact"] is True
        assert payload["gathered"] == [
            {"cursor": "cursor-0", "rank": 0},
            {"cursor": "cursor-1", "rank": 1},
        ]


def test_distributed_checkpoint_restores_the_selected_rank_only(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    identities = {
        "training_config_sha256": "1" * 64,
        "dataset_identity_sha256": "2" * 64,
        "token_policy_sha256": "3" * 64,
        "trainable_parameters_sha256": "4" * 64,
        "base_snapshot_sha256": "5" * 64,
        "source_code_sha256": "6" * 64,
    }
    seeds = {
        "time_seed": 11,
        "corruption_seed": 12,
        "uniform_prior_seed": 13,
        "sampler_seed": 14,
        "dataloader_seed": 15,
    }
    rngs = [
        trainer.TrainingRNGStreams.create(
            seeds, device=torch.device("cpu"), rank=rank
        )
        for rank in range(2)
    ]
    rank_states = [
        trainer.build_local_rank_checkpoint_state(
            rank=rank,
            world_size=2,
            epoch=3,
            raw_batches_consumed_in_epoch=10 + rank,
            usable_batches_consumed_in_epoch=9 + rank,
            raw_batch_cursor_sha256=str(rank + 1) * 64,
            epoch_dataloader_generator_state=rngs[rank].dataloader.get_state(),
            rng=rngs[rank],
            overflow_counters={name: rank for name in trainer.COLLATOR_METRIC_NAMES},
        )
        for rank in range(2)
    ]
    state_path = tmp_path / "training_state.pt"
    trainer._save_training_loop_state(
        state_path,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_scaler=None,
        rng=rngs[0],
        update=7,
        epoch=3,
        raw_batches_consumed_in_epoch=10,
        usable_batches_consumed_in_epoch=9,
        raw_batch_cursor_sha256="1" * 64,
        epoch_dataloader_generator_state=rngs[0].dataloader.get_state(),
        overflow_counters={name: 0 for name in trainer.COLLATOR_METRIC_NAMES},
        last_reported_loss=0.5,
        elapsed_seconds=2.0,
        identities=identities,
        rank_states=rank_states,
    )
    checkpoint = SimpleNamespace(training_state=state_path, update=7)
    rank_one = trainer._load_training_loop_state(
        checkpoint,
        identities=identities,
        rank=1,
        world_size=2,
    )
    assert rank_one["rank"] == 1
    assert rank_one["raw_batches_consumed_in_epoch"] == 11
    assert rank_one["usable_batches_consumed_in_epoch"] == 10
    assert rank_one["training_rng"]["seeds"]["time"] == 12
    with pytest.raises(ValueError, match="world_size"):
        trainer._load_training_loop_state(
            checkpoint,
            identities=identities,
            rank=0,
            world_size=1,
        )
