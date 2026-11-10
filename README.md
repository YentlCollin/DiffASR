# DiffASR — Discrete Diffusion for Speech Recognition

**Diffusion research and implementation: Yentl Collin · IFM / MBZUAI**

Research code for adapting an autoregressive speech recognizer into a discrete
diffusion denoiser. The work was developed around MawJais / Jais-ASR: the
pretrained audio encoder and projector are frozen while the text decoder is
adapted to denoise transcript tokens in parallel. This repository presents the
diffusion contribution; it is not an official release of Jais-ASR.

## Availability and authorization

**Some code is currently omitted because redistribution authorization has not
yet been granted.** In particular, the underlying model/processor implementation
and its modified integration files are not included. Model weights, tokenizer
assets, datasets, internal model/dataset repository identifiers, private revision
references, and cluster access configuration are also excluded.

The diffusion modules, training/inference entry points, data-processing utilities,
example configurations and applicable CPU tests are included. End-to-end training
and transcription require separately authorized base-model files and assets;
the public checkout alone cannot run the complete recognizer. See
[the integration guide](docs/INTEGRATION.md) for the required interfaces.

This is a research code release, with no released checkpoints or benchmark claims.
No new redistribution license for third-party model code is granted here.

## Code overview

- `audio/diffusion/`: corruption, denoising objectives, prefix/canvas attention,
  time conditioning, samplers, LoRA support, checkpoints and training state.
- `audio/diffusion_data/`: local audio-TAR loading and conditional transcript preparation.
- `examples/`: training, inference, evaluation, parity checks and generic Slurm launchers.
- `configs/diffusion/`: example ablations; replace placeholder paths and revisions.
- `audio/tests/`: CPU tests, including toy models that exercise the diffusion wrapper.

For a guided code review, start with:

- [Files added for diffusion](added_files.md): module inventory and responsibilities.
- [Changes to the AR model](change_code.md): function-level integration changes,
  attention visibility and audio-freezing safeguards.

## Architecture

The original audio tower and audio projector remain frozen. The text decoder
sees a clean audio/prompt prefix and a noisy transcript canvas. The prefix
remains causal and cannot attend to the canvas; canvas positions attend to the
prefix and to one another. The denoiser predicts clean tokens in parallel,
using same-position or previous-position logits. Optional time conditioning
is applied to the canvas. The original causal forward remains available.

The canvas size is selected from audio duration: the examples use
`ceil(9 × duration_seconds + 100)` followed by bucketing into
`64, 128, 256, 384, 512`. References never determine the inference canvas size.
Mask and uniform corruption are supported; UDM requires uniform corruption.

## Environment

The training entry point retains the production **ROCm/MI210 requirement**;
it does not enable NVIDIA or CPU training. Use your existing ROCm PyTorch
environment/container. The production stack used Python 3.12, PyTorch
2.11 / ROCm 7.14, Transformers 5.14, Accelerate 1.14 and PEFT 0.19.1.
PyTorch is deliberately not installed by the requirements file, to avoid
replacing a working GPU-specific build. vLLM is not needed for this workflow.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Public-source checks use a separate CPU-test environment; see
[validation notes](docs/VALIDATION.md). They are not a new GPU training run.

## Prepare data

Run commands from this directory. Supply an existing model/tokenizer snapshot
and locally accessible audio TARs; the preparation commands do not extract
or download audio. Obtain model/dataset access separately if needed.

1. Create a video-disjoint split. `validation-manifest.jsonl` is your existing
   held-out validation manifest with video/sample identifiers.

```bash
python examples/prepare_legal_ytb_train_test_split.py \
  --parquet /path/to/legal-ytb/data/training.parquet \
  --validation-manifest /path/to/validation-manifest.jsonl \
  --output-dir data/split
```

2. Build the four conditional targets and signed metadata index in a **new**
   directory. A single-task preparation finalizes the index automatically.

```bash
python examples/prepare_legal_ytb_conditional_metadata.py \
  --source-metadata /path/to/legal-ytb/data/training.parquet \
  --split-manifest data/split/train-excluded-video-groups.jsonl \
  --tokenizer-path /path/to/mawjais-user-provided-revision \
  --output-dir data/conditional

python examples/prepare_legal_ytb_conditional_metadata.py \
  --source-metadata /path/to/legal-ytb/data/training.parquet \
  --split-manifest data/split/train-excluded-video-groups.jsonl \
  --tokenizer-path /path/to/mawjais-user-provided-revision \
  --output-dir data/conditional --finalize \
  --validation-manifest /path/to/validation-manifest.jsonl
```

For parallel preprocessing, run task indices `0..N-1` with `--task-count N`
and `--task-index I`, then finalize with the same `--task-count N`.
`--expected-train-rows` optionally enforces a known exact count.

The current preprocessing **preserves existing shadda and tatweel even when
diacritics are not requested**. Historical datasets are not rewritten.
See [the preprocessing policy](docs/PREPROCESSING.md).

## Configure and launch training

Edit `configs/diffusion/base.yaml` (or create an override using `extends`):
set `model.checkpoint`, `data.local_root`, `data.conditional_metadata_index`,
the update budget and the scheduler horizon. Relative data paths resolve from
the repository root. Keep the original model revision recorded in the config.

| Config | Adaptation | Alignment | Time conditioning |
|---|---|---|---|
| `lora_no_t_same.yaml` | LoRA | same position | off |
| `lora_no_t_shifted.yaml` | LoRA | previous position | off |
| `lora_with_t_same.yaml` | LoRA | same position | on |
| `full.yaml` | Full text decoder + lm_head | same position | off |
| `full_no_t_shifted.yaml` | Full | previous position | off |
| `full_with_t_same.yaml` | Full | same position | on |
| `mask_no_t_same.yaml` | LoRA, mask corruption | same position | off |

The example global batch is 128; per-rank accumulation is derived from the
world size and microbatch. Full fine-tuning uses FSDP hybrid sharding, BF16
and gradient checkpointing, with at least eight GPUs in these configs.
The LoRA/full starting LRs are `3e-5` / `3e-6`, with a cosine schedule.
These are starting configurations, not a claim of optimal LR for new data.
The example 58,838-update budget is **not automatically two epochs after a
preprocessing change**. An optional `expected_epoch_updates` list can bind
a recalculated epoch plan to signed data quotas.

Validate configuration without loading weights:

```bash
python examples/train_mawjais_diffusion.py \
  --config configs/diffusion/lora_no_t_same.yaml \
  --validate-only --skip-data-validation --output-dir outputs/config-check
```

The trainer retains the original AR parity gate. Generate a passing artifact
using an audio/reference pair with the exact intended AR output, including its
country tag. This checks the local PyTorch model path before adapting it.

```bash
python examples/validate_mawjais_ar_parity.py \
  --checkpoint /path/to/mawjais-user-provided-revision --revision user-provided-revision \
  --audio /path/to/parity.wav --reference-file /path/to/parity-reference.txt \
  --prompt-tag '[Diacritics]' --attention-backend sdpa \
  --write-artifact outputs/ar-parity.json
```

Submit one Slurm job; configure the account/partition for your cluster:

```bash
TRAIN_NODES=1 GPUS_PER_NODE=8 TRAIN_PARTITION=your-partition \
TRAIN_ACCOUNT=your-account \
bash examples/submit_training.sh \
  configs/diffusion/lora_no_t_same.yaml outputs/lora-no-t-same outputs/ar-parity.json
```

Add `--dry-run` immediately after `submit_training.sh` to print the submission
without sending it. Other options: `TRAIN_QOS`, `TRAIN_GRES` (e.g.
`gpu:mi210:8`), `TRAIN_CPUS_PER_NODE`, `TRAIN_TIME`, `TRAIN_LOG_DIR`,
`PYTHON_BIN`, and optional `CONTAINER_IMAGE` for an existing Apptainer ROCm image.
The environment and all data/model paths must be accessible on every node;
use `APPTAINER_BIND` for any additional container mounts. Access groups and
cluster-specific network settings remain the caller's environment settings.

Inside an existing single-node allocation, the equivalent direct launch is:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  examples/train_mawjais_diffusion.py --train \
  --config configs/diffusion/lora_no_t_same.yaml \
  --output-dir outputs/lora-no-t-same --parity-artifact outputs/ar-parity.json
```

For a resume, append `--resume-from-checkpoint latest` to the same command
and output directory. `--stop-after-updates N` provides a bounded run without
changing the scheduler horizon. A continuation into a new output uses
`--continue-from-checkpoint PATH` and explicit continuation config fields;
it is distinct from an exact resume. Optimizer, scheduler, RNG, data cursor,
token policy and source identity checks are retained. This cleaned source
tree has a different source identity from historical runs; it is not a
drop-in replacement for their active training processes.

Training metrics are emitted as JSON lines to stdout (captured by the Slurm
log), and checkpoints/summary files are written to the output directory.
The external W&B log bridge and campaign watchers are not bundled.

## Inference: select a sampler

```bash
python examples/infer_mawjais_diffusion.py \
  --checkpoint outputs/lora-no-t-same/checkpoint-001000 \
  --base-checkpoint /path/to/mawjais-user-provided-revision \
  --wav /path/to/example.wav --prompt-tag '[Diacritics]' \
  --sampler argmax_udm --nfe 32 --attention-backend sdpa
```

| `--sampler` | Meaning |
|---|---|
| `confidence_schedule` | Confidence-based decoding; mask or uniform checkpoint. |
| `argmax_udm` | One-hot clean prediction followed by categorical UDM bridge transitions; uniform checkpoint. |
| `udm_marginalization` | Marginalized categorical UDM transitions at temperature 1; uniform checkpoint. |

Switching from a saved Confidence config to UDM selects compatible defaults.
Argmax UDM uses argmax clean predictions, while marginalized UDM uses
categorical predictions; both use temperature 1 in these examples. Invalid
explicit combinations are rejected. Their transitions need not be deterministic;
use `--seed` for reproducibility. The existing lower-level sampler variants
remain in `audio/diffusion/sampler.py`, without separate campaign scripts.

Full FSDP checkpoints first require an inference export:

```bash
python examples/export_fsdp_diffusion_checkpoint_for_inference.py \
  --checkpoint outputs/full/checkpoint-001000 --output-dir outputs/full-export
```

Then pass `--fsdp-merged-weights outputs/full-export/model.safetensors`
alongside the original checkpoint to inference/evaluation. The export preserves
the sharded training checkpoint and verifies the merged weights' provenance.

## Conditional evaluation

```bash
python examples/evaluate_mawjais_diffusion_conditional.py run \
  --checkpoint outputs/lora-no-t-same/checkpoint-001000 \
  --base-checkpoint /path/to/mawjais-user-provided-revision \
  --manifest data/conditional/validation-metadata.parquet \
  --dataset-root /path/to/legal-ytb \
  --sampler argmax_udm --nfe 32 --output-jsonl outputs/eval/predictions.jsonl

python examples/evaluate_mawjais_diffusion_conditional.py score \
  --predictions outputs/eval/predictions.jsonl \
  --manifest data/conditional/validation-metadata.parquet \
  --summary-json outputs/eval/metrics.json
```

Generation can be partitioned with `--num-shards`, `--shard-index` and
`--shard-mode tar_balanced`; each audio retains its four views. Use a distinct
output for each checkpoint/sampler/NFE. Scores include CER/WER with and without
diacritics, conditional slices, country accuracy and generation timing.
The historical `der` compares extracted mark sequences, **not marks aligned
to their base letters**; do not interpret it as a fully aligned diacritization
accuracy. No silence/no-speech quality claim follows from speech-only data.
