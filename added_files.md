# Files added for DiffASR

This document presents the diffusion files added to the original AR repository:
what they do and where to look to understand the diffusion conversion,
preprocessing, training, and inference.

**Public-release scope:** the base-model implementation and modified upstream
files are withheld pending redistribution authorization. The inventory describes
the diffusion contribution; full-model execution requires those separately
authorized dependencies. See [change_code.md](change_code.md) and
[docs/INTEGRATION.md](docs/INTEGRATION.md).

## 1. `audio/diffusion/`: the AR → diffusion conversion

This directory is an addition to the original model. It does not contain a
second Transformer implementation: it organizes how the model is used as a
denoiser.

In practice, the collator prepares the prefix and a noisy target; the wrapper
assembles their embeddings and calls the decoder with diffusion attention;
the objective compares predictions with the clean text. At inference, the
sampler replaces that training loop with repeated canvas updates.

### The model, canvas, noise, and loss

| Added file | What I put in it / what to review |
|---|---|
| [model.py](audio/diffusion/model.py) | The `MawJaisDiffusionASRModel` wrapper. It freezes the audio path, prepares the prefix with `prepare_prefix_embeddings()`, supports reuse of encoded audio, and assembles the prefix/canvas in `forward_diffusion()`. It calls the decoder with `attention_mode="diffusion"`, retrieves aligned logits, and computes the denoising loss. |
| [attention.py](audio/diffusion/attention.py) | The `build_prefix_canvas_allowed_mask()`, `build_prefix_canvas_attention_mask()`, and `build_prefix_canvas_position_ids()` functions. They build causal-prefix / bidirectional-canvas visibility, handle padding, and provide RoPE positions. |
| [alignment.py](audio/diffusion/alignment.py) | Logit selection for the `same_position` and `previous_position` variants: prediction from the same position or the preceding one. This alignment is explicit, without an additional implicit AR shift. |
| [canvas.py](audio/diffusion/canvas.py) | Generation length derived from audio duration, bucket selection, target construction with TURN_END/EOS and padding, overflow detection, and output reading up to the first TURN_END. |
| [collator.py](audio/diffusion/collator.py) | Assembly of examples into batches: prompts, audio prefixes, variable lengths, conditional targets, canvases, and corruption. It connects the prepared data to the wrapper inputs. |
| [noise_process.py](audio/diffusion/noise_process.py) | Uniform and Mask corruption, the prior token distribution, and diffusion-time sampling. Token noise is distinct from the attention mask. |
| [objectives.py](audio/diffusion/objectives.py) | Reconstruction of clean text `x0` through cross-entropy. Options determine supervised positions and weights: by default, all positions for Uniform and corrupted positions for Mask, with EOS/PAD-tail handling. |
| [time_embedding.py](audio/diffusion/time_embedding.py) | Sinusoidal time features and the optional small MLP that conditions canvas embeddings in the `with_t` variants. |
| [conditional.py](audio/diffusion/conditional.py) | The contract for the four instructions: `plain`, `code_switch`, `diacritics`, and `code_switch_diacritics`, and their mapping to preprocessing flags. |
| [token_policy.py](audio/diffusion/token_policy.py) | Rules for MASK, TURN_END, PAD, and audio IDs, the prior vocabulary, forbidden output tokens, and policy fingerprints. This prevents mixing checkpoints with incompatible token rules. |

Canvas-selection example using the published settings: for 10 seconds of
audio, `ceil(9 × 10 + 100) = 190`, so the next available bucket is 256 among
`64, 128, 256, 384, 512`. Reference transcript length does not determine the
canvas at inference. This duration heuristic can be replaced in future experiments. During preparation, an overly long target is flagged;
it does not silently increase the bucket size.

### Sampling and scores

| Added file | What I put in it / what to review |
|---|---|
| [sampler.py](audio/diffusion/sampler.py) | Canvas-update algorithms, including Confidence, Argmax UDM, and marginalized UDM, plus the shared `sample_diffusion` entry point. |
| [metrics.py](audio/diffusion/metrics.py) | Denoising measurements during training and aggregation through additive counters, producing consistent ratios across batches/ranks. |
| [conditional_evaluation.py](audio/diffusion/conditional_evaluation.py) | Strict and diacritic-stripped WER/CER, breakdowns by instruction and content, country tags, paired comparisons, and timing. Note: the historical `der` measures distance between extracted sequences of marks, not a diacritic-to-letter alignment. |

A single inference entry point lets you choose the sampler:

```text
--sampler confidence_schedule --nfe 64
--sampler argmax_udm --nfe 64
--sampler udm_marginalization --nfe 64
```

These are options to pass to the script, not three separate programs.
UDM requires a Uniform noise configuration. In YAML, UDM temperatures must
be 1, with `prediction_mode: argmax` for Argmax UDM or
`prediction_mode: categorical` for marginalized UDM. The inference CLI selects
compatible defaults when switching sampler families.

### Configuration, loading, and resuming

| Added file | What I put in it / what to review |
|---|---|
| [config.py](audio/diffusion/config.py) | Dataclasses and validation for noise, alignment, token, objective, and sampling choices. |
| [experiment_config.py](audio/diffusion/experiment_config.py) | YAML loading and inheritance, training parameters, global batch size, and consistency checks. For publication, UDM selection is exposed in YAML, and an update budget no longer enforces the old exact two-epoch plan. An explicitly declared epoch plan is still validated. |
| [loading.py](audio/diffusion/loading.py) | Loader contract for separately authorized local base-model files. It raises an explicit error when those withheld dependencies are absent, rather than loading remote executable model code. |
| [peft_utils.py](audio/diffusion/peft_utils.py) | Adaptation of the PEFT interface to the audio/text composite for LoRA training. |
| [distributed.py](audio/diffusion/distributed.py) | DDP layouts, per-rank accumulation, metric reductions, collective overflow checks, and scheduler synchronization. |
| [checkpointing.py](audio/diffusion/checkpointing.py) | Saving/restoring diffusion-specific components and the checkpoint manifest. |
| [training_resume.py](audio/diffusion/training_resume.py) | Resume checks: code and data identity, hashes, training state, RNG, and position within the data. |
| [training_fork.py](audio/diffusion/training_fork.py) | The distinction between exactly resuming a run and continuing in a new run, with explicit policies for retaining or resetting states. |
| [isct.py](audio/diffusion/isct.py) | Legacy iterative self-correction support for Mask, retained for imports and compatibility. It is disabled in every published configuration. Additional research branches are not included. |
| [__init__.py](audio/diffusion/__init__.py) | Public package exports, including the `sample_diffusion` dispatcher. |

## 2. `audio/diffusion_data/`: preprocessing and data loading

This directory groups the targets and loaders used to train the denoiser.
It contains no audio, corpus Parquet files, or private examples.

| Added file | What I put in it / what to review |
|---|---|
| [clean_17k_preprocessing_allowed_shadda.py](audio/diffusion_data/clean_17k_preprocessing_allowed_shadda.py) | The new 17k preprocessing: annotation cleanup, Arabic/code-switch/diacritics targets, country tags, and missing-transcription cases. The main entry points are `process_training_transcription()` and `process_clean_17k_row()`. This version preserves existing shadda and tatweel even with `use_diacritics=False`. |
| [conditional_metadata.py](audio/diffusion_data/conditional_metadata.py) | Construction of the four conditional targets, their tokenized lengths and validity masks, reproducible per-epoch view permutation, and metadata signatures. |
| [conditional_tar_dataset.py](audio/diffusion_data/conditional_tar_dataset.py) | Reading prepared metadata and audio TARs without full extraction, with distribution across ranks/workers, signature checks, balancing, and quotas. |
| [local_tar_dataset.py](audio/diffusion_data/local_tar_dataset.py) | Parsing/audio utilities and the original local loader from diffusion development. It remains used by the conditional path and inference on a dataset sample. "Original" does not mean it existed in the AR baseline commit. |
| [exclusions.py](audio/diffusion_data/exclusions.py) | Reading and checking signed exclusion manifests, particularly to separate training and evaluation data. The actual manifests are not published. |
| [__init__.py](audio/diffusion_data/__init__.py) | Lazy package exports. The public preprocessor points to the `allowed_shadda` version. |

### The exact shadda rule

With `use_diacritics=False`, I preserve existing shadda `U+0651` and tatweel
`U+0640` and remove the other configured Arabic marks. With
`use_diacritics=True`, diacritics are retained. The code does not add any
shadda or tatweel that was absent from the original text.

Linguistically, shadda remains a diacritic: this is a **training-target
convention**. The historical `*_without_diacritics` scorer also removes
shadda and tatweel from references and hypotheses. The preprocessing and
scoring policies are therefore not identical.
[docs/PREPROCESSING.md](docs/PREPROCESSING.md) explains this distinction.

This new code does not retroactively transform old Parquet files. To apply
the policy, metadata must be prepared in a new destination, and epoch quotas
must be recalculated.

## 3. `examples/`: preparing, training, launching, and evaluating

All files below are additions relative to the AR baseline commit. Some are
copies of the research code; others are publication adaptations. This does
not change their "added" status.

| Added file | Purpose and origin of the published version |
|---|---|
| [prepare_legal_ytb_train_test_split.py](examples/prepare_legal_ytb_train_test_split.py) | Prepares a deterministic video-level split, excluding validation. Derived from the workspace preparation script; the three default private paths were replaced with required arguments. |
| [prepare_legal_ytb_conditional_metadata.py](examples/prepare_legal_ytb_conditional_metadata.py) | Prepares the four targets and signed index, with partitioning, resuming, and finalization. Publication adaptation: the assertion on the exact historical row count becomes optional. |
| [validate_mawjais_ar_parity.py](examples/validate_mawjais_ar_parity.py) | Checks the PyTorch AR path and produces the parity artifact required by the trainer before training. Included with private identifiers replaced by placeholders. |
| [train_mawjais_diffusion.py](examples/train_mawjais_diffusion.py) | The conditional trainer: AR loading, LoRA/full selection, audio freezing, Accelerate/DDP/FSDP, Adam, scheduler, loss, JSON metrics, and checkpoints. Includes the `materialize_adam_state_for_fsdp_load` fix for restoring Adam moments under FSDP. |
| [submit_training.sh](examples/submit_training.sh) | New generic publication launcher: explicit arguments, Slurm resources through environment variables, and `--dry-run`. No personal account or partition is embedded. |
| [train_diffusion.sbatch](examples/train_diffusion.sbatch) | New generic Slurm script: one task per node, master address, launching through `srun`, and collective shutdown on failure. |
| [run_training_node.sh](examples/run_training_node.sh) | New per-node launcher: `torchrun`, argument forwarding, an optional ROCm container, and local dry-run support. |
| [infer_mawjais_diffusion.py](examples/infer_mawjais_diffusion.py) | Inference on an audio file or local sample, checkpoint and token checks, and a duration-derived canvas. Adapted to use `sample_diffusion`, offer Confidence/UDM, and select compatible defaults. |
| [evaluate_mawjais_diffusion_conditional.py](examples/evaluate_mawjais_diffusion_conditional.py) | Four-view evaluation with generation, sharding, resuming, merging, and scoring. Marginalized UDM was added to the CLI choices during preparation; Confidence and Argmax UDM were already exposed. |
| [export_fsdp_diffusion_checkpoint_for_inference.py](examples/export_fsdp_diffusion_checkpoint_for_inference.py) | Verified export of a full FSDP checkpoint to safetensors, with locking, provenance, and atomic publication of the export. It writes a separate export without modifying the training checkpoint. Included with private identifiers replaced by placeholders. |

The three shell scripts are portable versions prepared for this directory,
**not the scripts that launched the historical runs**. Their dry-runs are
tested, but they must be configured for the target cluster.

The trainer retains its production ROCm/MI210 requirement; these launchers
do not automatically make it CUDA/NVIDIA- or CPU-compatible. Account,
partition, environment, network, and data-access settings must be supplied
by the user. Full commands are in the [README](README.md).

## 4. `configs/diffusion/`: the selected experiments

| Added file | Configuration |
|---|---|
| [base.yaml](configs/diffusion/base.yaml) | Shared base: Uniform, conditional data, editable paths, LoRA rank 128 / alpha 256, global batch size 128, LR `3e-5`, cosine scheduler, and gradient checkpointing enabled. The `58838`-update budget is an example, not a guarantee of two epochs on new data. |
| [lora_no_t_same.yaml](configs/diffusion/lora_no_t_same.yaml) | Uniform LoRA, no time conditioning, same-position alignment. |
| [lora_no_t_shifted.yaml](configs/diffusion/lora_no_t_shifted.yaml) | Uniform LoRA, no time conditioning, previous-position alignment. |
| [lora_with_t_same.yaml](configs/diffusion/lora_with_t_same.yaml) | Uniform LoRA, with time conditioning, same position. |
| [full.yaml](configs/diffusion/full.yaml) | Full fine-tuning of the text decoder and `lm_head`, with audio still frozen; no time conditioning, same position, LR `3e-6`, and FSDP hybrid sharding. Declared distributed world sizes start at 8 GPUs. |
| [full_no_t_shifted.yaml](configs/diffusion/full_no_t_shifted.yaml) | Full variant without time conditioning, previous position. |
| [full_with_t_same.yaml](configs/diffusion/full_with_t_same.yaml) | Full variant with time conditioning, same position. |
| [mask_no_t_same.yaml](configs/diffusion/mask_no_t_same.yaml) | Mask-corruption example, no time conditioning, same position, and a trainable MASK embedding. |

These YAML files let you start from the AR model and choose a clearly defined
variant. The many old resume YAML files, LR sweeps, and job-specific topologies
are not copied. Effective batch size remains
`microbatch per GPU × number of GPUs × accumulation`; updates per epoch
therefore depend on the data and this batch size.

## 5. The model import bridge

[_mawjais_diffusion_repository/__init__.py](_mawjais_diffusion_repository/__init__.py)
is an added package that connects imports between `audio/model` and
`huggingface/xllm`. In particular, it lets DataLoader workers started with
`spawn`/`forkserver` locate the local code. It is not an additional model copy.

## 6. `audio/tests/`: the added tests

The tests use small models, tensors, and synthetic fixtures. They check code
behavior, not the ASR quality of the actual model weights.

| Added file | What it checks |
|---|---|
| [test_audio_position_safety.py](audio/tests/test_audio_position_safety.py) | Explicit audio positions, lengths, and injection safety. |
| [test_xllm_diffusion_attention_mode.py](audio/tests/test_xllm_diffusion_attention_mode.py) | Preservation of the causal path, the diffusion mask, cache rejection, and gradients. |
| [test_diffusion_alignment_attention.py](audio/tests/test_diffusion_alignment_attention.py) | Same/previous alignment and prefix/canvas visibility. |
| [test_diffusion_canvas_objectives.py](audio/tests/test_diffusion_canvas_objectives.py) | Buckets, targets, EOS/PAD, and objectives. |
| [test_diffusion_collator.py](audio/tests/test_diffusion_collator.py) | Batch preparation and corruption. |
| [test_diffusion_noise.py](audio/tests/test_diffusion_noise.py) | Noise processes and time. |
| [test_diffusion_sampler_time.py](audio/tests/test_diffusion_sampler_time.py) | Sampler transitions, schedules, and invariants, including UDM tests on small tensors. |
| [test_diffusion_token_policy.py](audio/tests/test_diffusion_token_policy.py) | Vocabularies, special tokens, and output restrictions. |
| [test_diffusion_model_wrapper.py](audio/tests/test_diffusion_model_wrapper.py) | The wrapper, audio/projector freezing, and loss. |
| [test_diffusion_isct.py](audio/tests/test_diffusion_isct.py) | The retained optional Mask support. |
| [test_diffusion_checkpointing.py](audio/tests/test_diffusion_checkpointing.py) | Diffusion state and the checkpoint manifest. |
| [test_diffusion_training_resume.py](audio/tests/test_diffusion_training_resume.py) | Resuming and data/state identity. |
| [test_diffusion_training_fork.py](audio/tests/test_diffusion_training_fork.py) | Continuation and Adam/scheduler/RNG policies. |
| [test_diffusion_distributed.py](audio/tests/test_diffusion_distributed.py) | Reductions, the synchronized scheduler, collective checks, and a two-process CPU Gloo test. |
| [test_diffusion_repository_loader.py](audio/tests/test_diffusion_repository_loader.py) | Loading with local code and importing in a fresh worker. |
| [forkserver_repository_probe.py](audio/tests/forkserver_repository_probe.py) | The importable helper function used by the worker test; it is not a training experiment. |
| [test_diffusion_metrics.py](audio/tests/test_diffusion_metrics.py) | Denoising metric counters and ratios. |
| [test_conditional_metadata.py](audio/tests/test_conditional_metadata.py) | Conditional targets, flags, and indexes. |
| [test_conditional_tar_dataset.py](audio/tests/test_conditional_tar_dataset.py) | TAR reading, signatures, and quotas. |
| [test_diffusion_conditional_evaluation.py](audio/tests/test_diffusion_conditional_evaluation.py) | Conditional scores and paired comparisons. |
| [test_diffusion_conditional_evaluation_cli.py](audio/tests/test_diffusion_conditional_evaluation_cli.py) | The CLI, resuming/merging, shared audio encoding, and per-view seeds. Its mock was connected to the dispatcher actually used by the published version. |
| [test_fsdp_diffusion_checkpoint_export.py](audio/tests/test_fsdp_diffusion_checkpoint_export.py) | Verified export, resuming, concurrency, and the absence of partially published exports. |
| [test_publication_bundle.py](audio/tests/test_publication_bundle.py) | Checks created for publication: standalone configurations, shadda preprocessing, CLI/YAML sampler selection, and launcher dry-runs. |

Assertions tied to the cluster's old Slurm scripts were replaced with checks
for the new launchers. Checks depending on an actual historical manifest,
old unpublished wrappers, or the out-of-scope AR control trainer are not
included.

The public release passed **189 CPU tests**, without downloading model weights
or launching GPU training. Three modules requiring withheld upstream files are
excluded from collection while those files are absent (audio position safety,
upstream attention-mode integration and repository-loader integration). Test limitations and the environment
are detailed in [docs/VALIDATION.md](docs/VALIDATION.md).

## 7. Documentation, dependencies, and publication verification

These files document the implementation, dependencies and source integrity.

| Added file | Role |
|---|---|
| [change_code.md](change_code.md) | A function-by-function explanation of the three modified upstream files, with public interface examples and an explicit availability notice. |
| [added_files.md](added_files.md) | This inventory of added files. |
| [docs/PREPROCESSING.md](docs/PREPROCESSING.md) | Target policy, the four views, and the distinction between preprocessing and scoring. |
| [docs/VALIDATION.md](docs/VALIDATION.md) | Checks performed, their results, and their limitations. |
| [docs/SOURCE_MANIFEST.json](docs/SOURCE_MANIFEST.json) | The public file inventory and source checksums; private provenance is omitted. |
| [tools/verify_bundle.py](tools/verify_bundle.py) | Checks hashes and reports missing, modified, or unlisted files without depending on the research copies. |
| [requirements.txt](requirements.txt) | Workflow dependencies. PyTorch is not installed automatically, to avoid replacing the GPU-specific build. |
| [requirements-dev.txt](requirements-dev.txt) | Additional test dependencies. |
| [pytest.ini](pytest.ini) | Test discovery and local import paths. |

Two supporting files **already existed in the original repository** and were
adapted for this directory: [README.md](README.md) for commands and
[.gitignore](.gitignore) for exclusions. They are therefore not counted as
added files or as changes to the model's Python code.

Two existing model configuration dependencies,
`audio/model/configuration_xllm_aut.py` and
`huggingface/xllm/configuration_xllm.py`, are also withheld pending authorization.
They are base-model dependencies, not new diffusion modules.

## 8. What is not included

This selection does not aim to copy the entire research workspace. It
excludes weights, adapters, Adam states, audio, Parquet files, actual example
manifests, Slurm/W&B logs, keys, containers, evaluation results, and frozen
code snapshots. vLLM/serving changes, ms-swift, the AR control trainer, and
additional research branches are not copied.

The directory contains the selected diffusion workflow and its dependencies,
not every historical experiment. Its resume signatures remain strict: do not
replace the code of active jobs with this publication copy.

The public repository has an independent history. The original private
repositories and their complete model integration remain separate.

To verify the selection from its root:

```bash
python tools/verify_bundle.py
```
