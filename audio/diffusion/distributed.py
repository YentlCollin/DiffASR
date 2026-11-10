"""Small, testable distributed-training contracts for MawJais diffusion."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist


# Keep one explicit topology allow-list.  The full-corpus campaign scales one
# data-parallel process per MI210 from the existing single-node gates through
# 16 nodes x 8 GPUs.  Rejecting every other value catches partial torchrun
# launches before any rank opens the 4B checkpoint.
SUPPORTED_WORLD_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)


@dataclass(frozen=True, slots=True)
class DistributedTrainingLayout:
    world_size: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    global_batch_size: int


def step_raw_scheduler_for_synchronized_update(
    scheduler: Any,
    *,
    sync_gradients: bool,
    optimizer_step_was_skipped: bool,
) -> bool:
    """Advance an unprepared scheduler exactly once per successful update.

    Accelerate's scheduler wrapper normally advances the wrapped scheduler by
    ``world_size`` when batches are not split. MawJais already keeps its global
    batch invariant through rank-aware gradient accumulation, so that behavior
    would make the learning-rate trajectory depend on DDP topology. The
    trainers therefore keep the native PyTorch scheduler out of
    ``Accelerator.prepare`` and call this helper after ``optimizer.step``.

    Returns ``True`` exactly when ``scheduler.step`` was called.
    """

    if not isinstance(sync_gradients, bool):
        raise TypeError("sync_gradients must be a bool")
    if not isinstance(optimizer_step_was_skipped, bool):
        raise TypeError("optimizer_step_was_skipped must be a bool")
    scheduler_type = type(scheduler)
    if (
        scheduler_type.__name__ == "AcceleratedScheduler"
        or scheduler_type.__module__.startswith("accelerate.scheduler")
    ):
        raise TypeError(
            "scheduler must remain a raw PyTorch scheduler and must not be "
            "passed to Accelerator.prepare"
        )
    if not sync_gradients or optimizer_step_was_skipped:
        return False
    scheduler.step()
    return True


def environment_world_size() -> int:
    """Read torchrun's world size without initializing a process group."""

    raw = os.environ.get("WORLD_SIZE", "1")
    try:
        world_size = int(raw)
    except ValueError as error:
        raise ValueError(f"WORLD_SIZE must be an integer, got {raw!r}") from error
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError(
            f"unsupported WORLD_SIZE={world_size}; expected one of {SUPPORTED_WORLD_SIZES}"
        )
    return world_size


def resolve_distributed_training_layout(
    training_config: Mapping[str, Any],
    *,
    world_size: int,
) -> DistributedTrainingLayout:
    """Derive accumulation while holding the semantic global batch fixed."""

    raw_configured = training_config.get(
        "supported_world_sizes", SUPPORTED_WORLD_SIZES
    )
    if not isinstance(raw_configured, (list, tuple)) or not raw_configured:
        raise ValueError("supported_world_sizes must be a non-empty list or tuple")
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in raw_configured
    ):
        raise ValueError("supported_world_sizes must contain only integers")
    configured = tuple(raw_configured)
    if configured != tuple(sorted(set(configured))):
        raise ValueError(
            "supported_world_sizes must be strictly increasing without duplicates"
        )
    unsupported = tuple(
        value for value in configured if value not in SUPPORTED_WORLD_SIZES
    )
    if unsupported:
        raise ValueError(
            "supported_world_sizes contains unsupported topologies "
            f"{unsupported}; allowed values are {SUPPORTED_WORLD_SIZES}"
        )
    if world_size not in configured:
        raise ValueError(f"world_size={world_size} is not supported")
    per_device = int(training_config["per_device_batch_size"])
    global_batch = int(training_config.get(
        "global_batch_size",
        per_device * int(training_config["gradient_accumulation_steps"]),
    ))
    micro_global = world_size * per_device
    if per_device <= 0 or global_batch <= 0:
        raise ValueError("per-device and global batch sizes must be positive")
    if global_batch % micro_global:
        raise ValueError(
            "global batch must be divisible by world_size * per_device_batch_size: "
            f"{global_batch} % ({world_size} * {per_device}) != 0"
        )
    accumulation = global_batch // micro_global
    if accumulation <= 0:
        raise ValueError("resolved gradient accumulation must be positive")
    return DistributedTrainingLayout(
        world_size=world_size,
        per_device_batch_size=per_device,
        gradient_accumulation_steps=accumulation,
        global_batch_size=global_batch,
    )


def _serialize_rank_object(local_object: Any) -> bytes:
    """Serialize tensor-rich state before entering an object collective.

    ``all_gather_object`` pickles its input internally.  Passing tensors through
    that implicit pickle path is not reliable with every PyTorch/ROCm release
    (notably PyTorch 2.11 can reconstruct an ``UntypedStorage`` incorrectly).
    ``torch.save`` owns tensor serialization here; the collective only ever
    sees an ordinary byte string.
    """

    buffer = io.BytesIO()
    torch.save(local_object, buffer)
    return buffer.getvalue()


def _deserialize_rank_object(payload: bytes, *, source_rank: int) -> Any:
    if type(payload) is not bytes:
        raise RuntimeError(
            "distributed rank-state gather returned a non-bytes payload for "
            f"rank {source_rank}: {type(payload).__name__}"
        )
    try:
        return torch.load(
            io.BytesIO(payload),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise RuntimeError(
            f"could not deserialize checkpoint state gathered from rank {source_rank}"
        ) from error


def gather_rank_objects(local_object: Any, *, world_size: int) -> list[Any]:
    """Gather tensor-rich checkpoint objects through a bytes-only wire format."""

    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
        raise ValueError(f"world_size must be a positive integer, got {world_size!r}")

    local_payload = _serialize_rank_object(local_object)

    if world_size == 1:
        gathered_payloads: list[Any] = [local_payload]
    else:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "distributed rank-state gathering requires an initialized process group"
            )
        process_group_world_size = dist.get_world_size()
        if process_group_world_size != world_size:
            raise RuntimeError(
                "process-group world size mismatch: "
                f"{process_group_world_size} != {world_size}"
            )
        gathered_payloads = [None] * world_size
        dist.all_gather_object(gathered_payloads, local_payload)

    if len(gathered_payloads) != world_size:
        raise RuntimeError(
            "distributed rank-state payload count mismatch: "
            f"{len(gathered_payloads)} != {world_size}"
        )
    return [
        _deserialize_rank_object(payload, source_rank=rank)
        for rank, payload in enumerate(gathered_payloads)
    ]


def validate_rank_states(rank_states: Sequence[Mapping[str, Any]], *, world_size: int) -> None:
    """Reject missing, duplicated or out-of-order rank checkpoint state."""

    if len(rank_states) != world_size:
        raise ValueError(
            f"rank checkpoint state count mismatch: {len(rank_states)} != {world_size}"
        )
    if any(not isinstance(state, Mapping) for state in rank_states):
        types = [type(state).__name__ for state in rank_states]
        raise ValueError(f"rank checkpoint states must be mappings, got {types}")
    ranks = [state.get("rank") for state in rank_states]
    if any(not isinstance(rank, int) or isinstance(rank, bool) for rank in ranks):
        raise ValueError(f"rank checkpoint states contain invalid rank values: {ranks}")
    if ranks != list(range(world_size)):
        raise ValueError(f"rank checkpoint states are not ordered and complete: {ranks}")
    saved_world_sizes = [state.get("world_size") for state in rank_states]
    if any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value != world_size
        for value in saved_world_sizes
    ):
        raise ValueError("rank checkpoint states disagree on world_size")


def collective_unexpected_overflow(
    local_overflow: bool,
    *,
    device: torch.device,
    world_size: int,
) -> bool:
    """Return the global overflow decision; every rank participates each batch."""

    flag = torch.tensor(int(local_overflow), device=device, dtype=torch.int64)
    if world_size > 1:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("collective overflow guard requires an initialized process group")
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def reduce_additive_values(
    values: Mapping[str, torch.Tensor | float | int],
    *,
    device: torch.device,
    world_size: int,
) -> dict[str, float]:
    """Sum scalar additive metrics across ranks with one ordered collective."""

    names = sorted(values)
    tensor = torch.stack(
        [
            torch.as_tensor(values[name], device=device)
            .detach()
            .to(dtype=torch.float64)
            .reshape(())
            for name in names
        ]
    )
    if world_size > 1:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("metric reduction requires an initialized process group")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {name: float(value) for name, value in zip(names, tensor.cpu().tolist(), strict=True)}


def reduce_max_value(
    value: float | int | torch.Tensor,
    *,
    device: torch.device,
    world_size: int,
) -> float:
    """Return the largest scalar value observed by any rank."""

    tensor = torch.as_tensor(value, device=device, dtype=torch.float64).reshape(())
    if world_size > 1:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("maximum reduction requires an initialized process group")
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.cpu())


__all__ = [
    "DistributedTrainingLayout",
    "SUPPORTED_WORLD_SIZES",
    "collective_unexpected_overflow",
    "environment_world_size",
    "gather_rank_objects",
    "reduce_additive_values",
    "reduce_max_value",
    "resolve_distributed_training_layout",
    "validate_rank_states",
]
