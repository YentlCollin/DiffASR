#!/usr/bin/env bash
# Submit one run. No checkpoint, dataset, account or partition is hard-coded.
set -euo pipefail

dry_run=0
if [[ "${1:-}" == --dry-run ]]; then
    dry_run=1
    shift
fi
if (( $# < 3 )); then
    echo "Usage: bash $0 [--dry-run] CONFIG OUTPUT_DIR PARITY_JSON [trainer arguments...]" >&2
    exit 2
fi
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export MAWJAIS_REPOSITORY_DIR="$repo_dir"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
nodes="${TRAIN_NODES:-1}"
for value in "$nodes" "$GPUS_PER_NODE"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Node and GPU counts must be positive integers" >&2; exit 2; }
done
# Resolve paths without requiring GNU realpath (also supports macOS dry-runs).
resolve_path() {
    "${PYTHON_BIN:-python3}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[2]).resolve(strict=sys.argv[1] == "existing"))' "$1" "$2"
}
config_path="$(resolve_path existing "$1")"
output_dir="$(resolve_path future "$2")"
parity_path="$(resolve_path existing "$3")"
shift 3
[[ -f "$config_path" && -f "$parity_path" ]] || { echo "Config and parity artifact must be files" >&2; exit 2; }
log_dir="${TRAIN_LOG_DIR:-$repo_dir/logs}"
submit=(sbatch --export=ALL --job-name=diffasr
    --nodes="$nodes" --ntasks-per-node=1
    --gres="${TRAIN_GRES:-gpu:$GPUS_PER_NODE}"
    --cpus-per-task="${TRAIN_CPUS_PER_NODE:-32}"
    --time="${TRAIN_TIME:-1-00:00:00}"
    --output="$log_dir/train-%j.out")
[[ -z "${TRAIN_PARTITION:-}" ]] || submit+=(--partition="$TRAIN_PARTITION")
[[ -z "${TRAIN_ACCOUNT:-}" ]] || submit+=(--account="$TRAIN_ACCOUNT")
[[ -z "${TRAIN_QOS:-}" ]] || submit+=(--qos="$TRAIN_QOS")
submit+=("$repo_dir/examples/train_diffusion.sbatch" "$config_path" "$output_dir" "$parity_path" "$@")
if (( dry_run )); then
    printf '%q ' "${submit[@]}"
    printf '\n'
    exit 0
fi
command -v sbatch >/dev/null || { echo "sbatch is not available" >&2; exit 1; }
mkdir -p -- "$log_dir"
exec "${submit[@]}"
