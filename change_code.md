# The three AR model files I modified

This document explains my integration changes for adapting MawJais / Jais-ASR
from an autoregressive recognizer into a discrete diffusion denoiser.
The added diffusion modules are described in [added_files.md](added_files.md).

**Availability:** the three upstream files discussed below are currently omitted
pending redistribution authorization, together with the base-model configuration
files. Their original source excerpts and full patch are also withheld. This
guide describes the changes and interfaces without redistributing the underlying
implementation. See [docs/INTEGRATION.md](docs/INTEGRATION.md).

## Overall approach

I reuse the pretrained audio encoder, audio projector, text decoder and prediction
head. The audio path stays frozen; the text decoder is adapted through LoRA or
full fine-tuning to reconstruct a noisy transcript canvas in parallel.

The clean audio/prompt prefix remains causal and cannot attend to the canvas.
Canvas positions attend to the prefix and one another. Token corruption and
attention visibility are separate operations. Optional time conditioning is
applied to canvas embeddings. The original autoregressive path remains available
for parity checks when the authorized base implementation is supplied.

## 1. `audio/model/modeling_xllm_aut.py`: freezing and placing audio

This upstream composite assembles the audio encoder, projector and text decoder.
My changes make the frozen audio path safe to reuse during diffusion training.

| Function or method | Integration change and purpose |
|---|---|
| `_aut_output_lengths()` | Match the processor's output-length convention so encoded audio can be checked against the expected placeholder count. The upstream encoder-specific formula is not reproduced here. |
| `_true_run_lengths()` | Measure contiguous runs of audio positions to validate clip boundaries and ordering. |
| `XllmAuTModel.__init__()` | Track whether the explicit diffusion audio-freeze policy has been enabled. |
| `freeze_audio_components()` | Disable gradients for the audio encoder and projector and keep both in evaluation mode. |
| `assert_audio_components_frozen()` | Check the freeze contract before decoder adaptation. |
| `train()` | Preserve evaluation mode for the frozen audio components when the outer model enters training mode. |
| `get_audio_features(..., no_grad=False)` | Add an optional no-gradient path and detached audio embeddings; retain the original computation for ordinary AR calls. The diffusion wrapper can reuse those embeddings across denoising steps. |
| `_get_audio_placeholder_mask()` | Accept an explicit `audio_position_mask`, validate its shape/type and use it as the source of injection positions. Validate counts per example and per clip as well as across the entire batch. |
| `XllmAuTModel.forward()` | Accept `audio_position_mask` and `diffusion_mode`; require explicit positions in diffusion mode, freeze/encode the audio path without gradients and validate injection before supplying embeddings to the decoder. |
| `XllmAuTForConditionalGeneration` | Expose the freeze/check methods and forward the optional arguments to the composite model. The original prediction head and AR loss remain separate from the diffusion objective. |

### Why explicit positions matter

A randomly corrupted transcript token must never become an audio injection
position merely because its ID happens to equal the tokenizer's audio token.
The processor records positions from the clean prefix before corruption:

```text
                       clean prefix             | noisy canvas
Audio positions:       [False, True, True, False] | [False, False]
```

Only the two marked prefix positions receive audio vectors. The model does not
combine the explicit mask with a fresh ID comparison on the noisy canvas.

A global count alone is insufficient: if two examples need 10 and 20 vectors,
incorrect counts of 12 and 18 still sum to 30. Per-example and per-clip checks
prevent that cross-example assignment error.

`diffusion_mode=True` controls the audio path. The decoder's `attention_mode`
selects diffusion visibility; the two switches have different responsibilities.

## 2. `audio/model/processing_xllm_aut.py`: recording positions before corruption

The existing processor prepares audio features, the prompt and its tokenization.
I add `audio_position_mask` immediately after tokenizing the clean prefix,
before the diffusion collator adds the noisy transcript canvas.

The output supports PyTorch tensors, NumPy arrays and Python lists. Without
audio, the mask is entirely false. Existing processor outputs remain available;
the new field accompanies them rather than replacing the feature mask.

| Field | Shape | Purpose |
|---|---|---|
| `feature_attention_mask` | `[clips, mel_frames]` | Identify real audio-feature frames rather than padding. |
| `audio_position_mask` | `[batch, prefix_length]` | Identify embedding positions where audio vectors belong. |
| Diffusion `attention_mask` | `[batch, 1, sequence_length, sequence_length]` | Control which prefix/canvas positions can attend to one another. |

Computing positions from clean IDs in the processor is safe because the canvas
does not exist yet. Reconstructing those positions from corrupted IDs later
would lose that guarantee.

## 3. `huggingface/xllm/modeling_xllm.py`: accepting diffusion attention

My changes let the decoder accept an explicit prefix/canvas mask. They reuse
the existing Transformer layers, projections, MLPs and positional embeddings.

### `_create_attention_mask_compat()`

This helper inspects the Transformers mask builder's accepted parameters and
forwards the supported embedding/cache arguments. It preserves compatibility
of the original causal path across API versions; it is not a diffusion sampler.

### `XllmModel.forward(..., attention_mode="causal")`

The default remains causal and uses the Transformers causal or sliding-window
mask builder. The explicit `"diffusion"` mode instead requires:

- `use_cache=False` and no mutable `past_key_values`;
- a floating-point additive mask with exact shape `[B, 1, S, S]`;
- conversion of that mask to the embeddings' device and numerical type.

Missing, boolean, 2D or incorrectly sized masks are rejected. The validated mask
is forwarded through the decoder's existing attention-mask interface.

### Prefix/canvas visibility

The public [attention.py](audio/diffusion/attention.py) constructs this mask.
For two prefix positions and three canvas positions, without padding:

```text
                  key position
                P0 P1 C0 C1 C2
query      P0    1  0  0  0  0
position   P1    1  1  0  0  0
           C0    1  1  1  1  1
           C1    1  1  1  1  1
           C2    1  1  1  1  1
```

`1` means allowed. The actual additive tensor uses zero for allowed positions
and a very negative floating-point value for forbidden positions. Padding is
handled explicitly, including safe self-visibility for ignored padding rows.

### Why mutable AR caches are rejected

Canvas tokens change between denoising steps and attend to one another.
Reusing their old keys and values would introduce stale context. Reusing frozen
audio embeddings is a separate operation and remains supported.

### The diffusion call in the public wrapper

The following call is present in the published
[model.py](audio/diffusion/model.py); it does not reproduce the withheld decoder:

```python
outputs = self.autoregressive_model.model.language_model(
    input_ids=None,
    inputs_embeds=packed_embeddings,
    attention_mask=attention_mask,
    position_ids=position_ids,
    use_cache=False,
    attention_mode="diffusion",
)
```

The wrapper constructs the prefix/canvas embeddings, optional time conditioning,
positions and mask. Noise, denoising objectives, same/previous-position alignment
and iterative sampling live in the added diffusion modules.

## Validation and remaining dependencies

The public wrapper is tested with toy models in
[test_diffusion_model_wrapper.py](audio/tests/test_diffusion_model_wrapper.py).
Tests that directly import the withheld composite, processor or decoder remain
in the tree but are excluded from collection while those dependencies are
absent. Full-model parity and end-to-end recognition require separately
authorized model files and weights. See [docs/VALIDATION.md](docs/VALIDATION.md).
