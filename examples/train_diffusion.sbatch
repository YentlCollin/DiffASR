#!/usr/bin/env bash
# Launched by submit_training.sh, which exports the real source directory.
# Slurm runs a spooled copy, so BASH_SOURCE cannot locate the repository here.
set -euo pipefail
[[ $# -ge 3 ]] || { echo "Expected CONFIG OUTPUT_DIR PARITY_JSON [trainer arguments...]" >&2; exit 2; }
: "${MAWJAIS_REPOSITORY_DIR:?Use examples/submit_training.sh to submit this job}"
: "${SLURM_JOB_ID:?This script runs inside a Slurm allocation}"
: "${SLURM_JOB_NODELIST:?Missing Slurm node list}"
: "${SLURM_NNODES:?Missing Slurm node count}"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
exec srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
    --kill-on-bad-exit=1 bash "$MAWJAIS_REPOSITORY_DIR/examples/run_training_node.sh" "$@"
