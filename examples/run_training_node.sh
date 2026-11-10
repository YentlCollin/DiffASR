#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 3 ]] || { echo "Expected CONFIG OUTPUT_DIR PARITY_JSON [trainer arguments...]" >&2; exit 2; }
: "${MAWJAIS_REPOSITORY_DIR:?Missing repository directory}"
: "${GPUS_PER_NODE:?Missing GPU count}"
config_path="$1"
output_dir="$2"
parity_path="$3"
shift 3
cd -- "$MAWJAIS_REPOSITORY_DIR"
export PYTHONPATH="$MAWJAIS_REPOSITORY_DIR${PYTHONPATH:+:$PYTHONPATH}"
command=("${PYTHON_BIN:-python}" -m torch.distributed.run
    --nnodes="${SLURM_NNODES:-1}" --nproc-per-node="$GPUS_PER_NODE"
    --node-rank="${SLURM_NODEID:-0}"
    --master-addr="${MASTER_ADDR:-127.0.0.1}" --master-port="${MASTER_PORT:-29500}"
    examples/train_mawjais_diffusion.py --train
    --config "$config_path" --output-dir "$output_dir" --parity-artifact "$parity_path" "$@")
if [[ -n "${CONTAINER_IMAGE:-}" ]]; then
    [[ -f "$CONTAINER_IMAGE" ]] || { echo "Container image not found" >&2; exit 2; }
    # Optional existing ROCm image; additional data/cache mounts use APPTAINER_BIND.
    command=(apptainer exec --rocm --bind "$MAWJAIS_REPOSITORY_DIR:$MAWJAIS_REPOSITORY_DIR"
        "$CONTAINER_IMAGE" "${command[@]}")
fi
if [[ "${TRAINING_DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
exec "${command[@]}"
