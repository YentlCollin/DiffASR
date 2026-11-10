# Base-model integration and intentionally omitted code

This work adapts a MawJais / Jais-ASR autoregressive recognizer into a discrete
diffusion model. The base recognizer is a dependency, not a model released here.

## Diffusion integration contract

1. Encode audio with a frozen audio tower and frozen projector, and inject its
   embeddings at explicitly identified audio positions in the clean prefix.
2. Build a noisy transcript canvas after that prefix. The prefix keeps causal
   attention and cannot attend to the canvas; canvas positions see the prefix
   and one another. See `audio/diffusion/attention.py`.
3. Let the language decoder accept this explicit attention mask, disable mutable
   autoregressive KV caching in diffusion mode, and preserve its original causal
   forward for parity checks.
4. Apply optional time conditioning to canvas embeddings and predict clean tokens
   using the alignment selected in the configuration. See `model.py`,
   `objectives.py`, `noise_process.py` and `sampler.py` under `audio/diffusion/`.
5. The processor must provide an explicit audio-position mask before corruption.
   The model must validate audio-feature and placeholder lengths and keep the
   frozen audio path in evaluation mode during decoder fine-tuning.

## Files withheld pending authorization

- `audio/model/configuration_xllm_aut.py`
- `audio/model/processing_xllm_aut.py`
- `audio/model/modeling_xllm_aut.py`
- `huggingface/xllm/configuration_xllm.py`
- `huggingface/xllm/modeling_xllm.py`

The full upstream model files and a patch containing upstream code are omitted.
The public loader deliberately raises a descriptive error when those files are
missing; it never substitutes remotely downloaded executable model code.
Authorized users need compatible, locally supplied implementations of the above
interfaces, not simply an arbitrary public ASR checkpoint.

Weights, tokenizer assets, training/evaluation corpora, private revisions and
production infrastructure are not distributed. Paths and provenance values in
examples are placeholders. Token IDs and text formatting in the diffusion code
are interface assumptions and must be checked against the authorized tokenizer.

## Validation scope

The public CPU suite uses synthetic inputs and toy models. Three test modules
that directly import the withheld implementation are excluded from collection
when those files are absent: audio position safety, upstream attention-mode
integration and repository-loader integration. They remain in the tree for
users with authorized dependencies. This does not validate full-model parity,
GPU training, real checkpoint export, data access or recognition quality.
