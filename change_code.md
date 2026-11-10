# The three AR model files I modified

This document explains the changes to the three Python files already present
in the MawJais AR model: what existed before, what I added, and why. The new
diffusion, data, and launch modules are described separately in
[added_files.md](added_files.md).

## Availability of the files discussed here

**The changes described below were made in existing MawJais / Jais-ASR model
files. Those files are not currently pushed to this public repository because
I do not yet have authorization to redistribute their complete implementation.**
Their absence from the public tree does not mean that the changes were not
implemented. This document explains my integration work function by function,
with excerpts of my additions and illustrative examples. The complete upstream
files, their full patch, encoder-specific internals and model weights remain
omitted pending authorization.

The paths below identify the files in the research implementation; they are
shown as text rather than links when the files are not available publicly.
The Python and diff blocks are selected excerpts, not complete runnable files.
Full-model execution requires the separately authorized base-model dependencies.
See [the integration guide](docs/INTEGRATION.md).

## The overall approach

I start from the architecture and initial weights of autoregressive MawJais:
the audio encoder, audio projector, text decoder, and prediction head.
I do not replace its Transformer with a new diffusion network.

I adapt the text decoder to reconstruct a noisy transcript, called the
**canvas**, in parallel. The audio encoder and projector remain frozen;
the text model is adapted through LoRA or full fine-tuning.

Attention becomes bidirectional **within the canvas**: each position can
attend to the other canvas positions and the audio/prompt prefix. The prefix
remains causal and cannot attend to the canvas. This is therefore not a
fully open attention mask over the entire sequence.

The attention mask must also be distinguished from the noise applied to the
text: Uniform uses random tokens; Mask uses masked positions. The denoiser
learns to recover the clean tokens. At inference, Confidence or UDM updates
the canvas repeatedly. NFE counts denoiser calls, not generated words. The
`with_t` variant adds a small time-conditioning module; the Mask variant has
a dedicated MASK embedding. These additions live in `audio/diffusion/`.

| Modified existing file | What I change |
|---|---|
| `audio/model/modeling_xllm_aut.py` (not published yet) | Persistent freezing of the audio path, computation without gradients, audio injection based on an explicit mask, and length validation. |
| `audio/model/processing_xllm_aut.py` (not published yet) | Creation of that explicit mask before the noisy canvas is added. |
| `huggingface/xllm/modeling_xllm.py` (not published yet) | Selection between causal and diffusion attention, using a supplied 4D mask and no mutable KV cache in diffusion mode. |

## 1. `audio/model/modeling_xllm_aut.py`: securing the audio path

This file already assembles the audio encoder, projector, and text decoder.
Before my changes, it encoded the audio and replaced the audio-token embeddings
with the resulting vectors. I keep this architecture and operation, adding
the safeguards needed to use them in diffusion.

### 1.1. New function: `_aut_output_lengths()`

I add the processor-compatible audio-length calculation to the model so it
can validate the number of audio vectors before injection. The encoder-specific
formula and numerical examples are omitted here pending redistribution
authorization. The important interface requirement is that the model and
processor agree on the number of vectors produced for each clip.

This function checks that count **per clip**. It does not choose the size
of the text-generation canvas: that calculation is in
[audio/diffusion/canvas.py](audio/diffusion/canvas.py).

### 1.2. New function: `_true_run_lengths()`

```python
def _true_run_lengths(mask: torch.Tensor) -> list[tuple[int, int]]:
    """Return `(batch_index, run_length)` for contiguous True runs, in sequence order."""
    runs = []
    for batch_index, row in enumerate(mask):
        positions = row.nonzero(as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        run_start = 0
        boundaries = (positions[1:] != positions[:-1] + 1).nonzero(as_tuple=False).flatten()
        for boundary in boundaries.tolist():
            runs.append((batch_index, boundary + 1 - run_start))
            run_start = boundary + 1
        runs.append((batch_index, positions.numel() - run_start))
    return runs
```

It identifies consecutive blocks of audio positions. Consider this example:

```text
Mask for example 0 : [False, True, True, False, True, True, True, False]
Result             : [(0, 2), (0, 3)]
```

There are two audio blocks, of lengths 2 and 3. This lets me validate multiple
clips in the same prompt, in the order in which their vectors will be injected.

### 1.3. `XllmAuTModel.__init__()`: tracking the audio freeze

The three modules are constructed exactly as before. I only add this state:

```python
self._audio_components_frozen = False
```

Constructing the model therefore does not immediately enforce the diffusion
audio freeze. It is explicitly enabled by `freeze_audio_components()` or
when processing audio with `diffusion_mode=True`.

### 1.4. New method: `freeze_audio_components()`

```python
def freeze_audio_components(self):
    """Freeze the complete audio path and keep it in eval mode across future `.train()` calls."""
    for module in (self.audio_tower, self.embed_audio):
        module.requires_grad_(False)
        for parameter in module.parameters():
            parameter.grad = None
        module.eval()
    self._audio_components_frozen = True
    return self
```

I freeze **the audio encoder and projector**, not the text decoder.

- `requires_grad_(False)` prevents gradients from being computed for their parameters.
- `parameter.grad = None` clears any gradient that may already be present.
- `eval()` disables their training-specific behavior, such as dropout.
  `eval()` alone would not freeze their parameters.
- The boolean records that this freeze must be maintained afterward.

The goal is to preserve the audio representation learned by the AR model
while adapting its decoder to denoising.

### 1.5. New method: `assert_audio_components_frozen()`

```python
def assert_audio_components_frozen(self):
    """Fail fast if a diffusion trainer accidentally made an audio parameter trainable."""
    for module_name, module in (
        ("model.audio_tower", self.audio_tower),
        ("model.embed_audio", self.embed_audio),
    ):
        if module.training:
            raise AssertionError(f"{module_name} must remain in eval mode during diffusion training")
        for parameter_name, parameter in module.named_parameters():
            if parameter.requires_grad:
                raise AssertionError(f"{module_name}.{parameter_name} is unexpectedly trainable")
            if parameter.grad is not None:
                raise AssertionError(f"{module_name}.{parameter_name} unexpectedly has a gradient")
```

This method does not freeze anything: it **checks** the freeze. It fails
immediately if an audio module switches back to training mode, if a parameter
becomes trainable again, or if an audio gradient appears. The diffusion wrapper
uses this check to catch regressions in the trainer.

### 1.6. New override: `train()`

```python
def train(self, mode: bool = True):
    super().train(mode)
    if self._audio_components_frozen:
        self.audio_tower.eval()
        self.embed_audio.eval()
    return self
```

A global call to `model.train()` normally switches every submodule to training
mode, including those with frozen parameters. I let the decoder switch to
training mode, then put the audio path back in evaluation mode if its freeze
has been enabled.

For example, after `model.freeze_audio_components()` followed by
`model.train()`, the audio path remains in `eval`, but the decoder can be
trained. Switching back to `diffusion_mode=False` does not automatically
unfreeze the audio path.

### 1.7. `get_audio_features()`: adding computation without gradients

Previously, this method directly encoded and projected the audio. I add an
optional argument without changing the feature computation:

```diff
- def get_audio_features(self, input_features, feature_attention_mask=None):
+ def get_audio_features(self, input_features, feature_attention_mask=None, *, no_grad=False):
```

The existing audio encoding and projection are grouped into an inner
`encode_and_project()` helper. The upstream feature computation and numerical
model dimensions are omitted here pending authorization; my change is to make
its gradient behavior explicit.

The new branch is:

```python
if no_grad:
    with torch.no_grad():
        return encode_and_project().detach()
return encode_and_project()
```

In diffusion, I do not need to retain a gradient graph for the frozen audio
path. This avoids that memory cost; the decoder can still compute its own
gradients using the resulting audio embeddings. The method does not create
a cache itself: the wrapper can reuse the embeddings it returns.

### 1.8. `_get_audio_placeholder_mask()`: keeping noise separate from audio

This method already existed. It identified where to replace text embeddings
with audio vectors. When `input_ids` were provided, the original detection was:

```python
special_audio_mask = input_ids == self.config.audio_token_id
```

The issue in diffusion is this: **if a token in the noisy canvas has the audio
ID**, its value must not make it an audio-injection position. The mask supplied
by the processor therefore becomes the source of truth.

I add two keyword-only arguments:

```python
def _get_audio_placeholder_mask(
    self,
    input_ids,
    inputs_embeds,
    n_audio_frames,
    *,
    audio_position_mask=None,
    feature_attention_mask=None,
):
```

When `audio_position_mask` is provided, I check that it is a boolean tensor
of shape `[batch, sequence]`, then use **only that mask**:

```python
if audio_position_mask is not None:
    if not isinstance(audio_position_mask, torch.Tensor):
        raise TypeError("audio_position_mask must be a torch.Tensor")
    if audio_position_mask.dtype != torch.bool:
        raise TypeError(
            f"audio_position_mask must have dtype torch.bool, got {audio_position_mask.dtype}"
        )
    if tuple(audio_position_mask.shape) != tuple(inputs_embeds.shape[:2]):
        raise ValueError(
            "audio_position_mask must have shape [batch, sequence] matching inputs_embeds; "
            f"got {tuple(audio_position_mask.shape)} and {tuple(inputs_embeds.shape[:2])}"
        )
    # This is the only source of injection positions when supplied. In particular, never OR
    # it with `input_ids == audio_token_id`: a noisy diffusion canvas may contain that id.
    special_audio_mask = audio_position_mask.to(device=inputs_embeds.device)
```

I do not combine this mask with another ID check. The old ID-based detection
branch, or embedding-based detection when `input_ids` is absent, remains
available for legacy AR calls without an explicit mask.

Illustrative example using audio ID `250062`:

```text
                         clean prefix             | noisy canvas
IDs                 : [42, 250062, 250062, 17]     | [250062, 81]
Audio positions     : [ F,      T,      T,  F]     | [     F,  F]
```

Only the two prefix positions receive audio vectors. The `250062` in the
canvas remains a noisy token, never a third audio vector. This example
illustrates an interface safeguard; it does not imply that every prior
configuration allows this ID. In the current wrapper, injection happens on
the prefix before it is assembled with the canvas.

I also add checks **per example and per clip**, in addition to the existing
global check. The expected count is calculated from valid mel frames:

```python
expected_per_clip = _aut_output_lengths(
    feature_attention_mask.sum(dim=1).to(dtype=torch.long)
)
```

The code checks, in order:

1. That the total number of audio positions equals the number of encoded vectors.
2. That `feature_attention_mask` is 2D and the calculated per-clip lengths
   add up to the same total.
3. For one clip per example, that each example has exactly the right number
   of positions, not just the batch as a whole.
4. Otherwise, that every contiguous block identified by `_true_run_lengths()`
   matches the right clip, with the right length and in the right order.

Why is the total alone insufficient? Suppose A produces 10 vectors and B
produces 20. A mask with 12 positions for A and 18 for B would still total
30, but would assign vectors to the wrong example. The new checks reject
this case.

Finally, the mask is expanded across the embedding dimension for injection:

```python
return special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
```

It goes from `[B, S]` to `[B, S, H]` for `masked_scatter`.

### 1.9. `XllmAuTModel.forward()`: enabling these safeguards

I add two arguments, both optional to preserve AR calls:

```python
audio_position_mask: Optional[torch.BoolTensor] = None,
diffusion_mode: bool = False,
```

In the branch that encodes audio during prefill, I add:

```python
if diffusion_mode:
    if audio_position_mask is None:
        raise ValueError(
            "audio_position_mask is required when diffusion_mode=True; "
            "the legacy input_ids == audio_token_id fallback is unsafe for a noisy canvas."
        )
    self.freeze_audio_components()
audio_embeds = self.get_audio_features(
    input_features,
    feature_attention_mask,
    no_grad=diffusion_mode,
)
special_audio_mask = self._get_audio_placeholder_mask(
    input_ids,
    inputs_embeds,
    audio_embeds.shape[0],
    audio_position_mask=audio_position_mask,
    feature_attention_mask=feature_attention_mask,
)
audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
if diffusion_mode:
    audio_embeds = audio_embeds.detach()
inputs_embeds = inputs_embeds.masked_scatter(special_audio_mask, audio_embeds)
```

In practical terms: require an explicit mask for this injection in diffusion
mode, freeze the audio path, encode without gradients, validate the positions,
then replace their embeddings.

**Prefill** detection already existed: it avoids re-encoding the audio for
every new token during AR generation with a KV cache. `masked_scatter` also
already existed; I make its input mask safer.

Passing `attention_mask` and `**kwargs` to the decoder also already existed.
**`diffusion_mode=True` does not enable bidirectional attention by itself**:
here, it controls the audio path. Attention selection is explained in the
third section.

### 1.10. `XllmAuTForConditionalGeneration`: exposing and forwarding the additions

This class wraps the composite model and its prediction head. I expose the
two freeze/check methods by delegating to the internal model:

```python
def freeze_audio_components(self):
    """Opt into the strict phase-1 diffusion freeze without changing default AR behavior."""
    self.model.freeze_audio_components()
    return self

def assert_audio_components_frozen(self):
    self.model.assert_audio_components_frozen()
```

Its `forward()` also receives the two new arguments and forwards them in
the call to `self.model(...)`:

```python
audio_position_mask=audio_position_mask,
diffusion_mode=diffusion_mode,
```

I do not change the construction of `lm_head`, the logits computation, or
this class's AR loss. The denoising loss is defined separately in
[audio/diffusion/objectives.py](audio/diffusion/objectives.py).

## 2. `audio/model/processing_xllm_aut.py`: recording audio positions

### 2.1. What already existed

`XllmAuTProcessor.__call__()` received the prompt text and audio. It computed
mel features, determined the expected number of audio vectors, replaced each
clip placeholder with the correct number of audio tokens, then tokenized
the text.

Audio extraction without truncation, the `_aut_output_lengths()` formula,
the audio delimiters, the tokenizer, and the chat template are not new changes
in this file.

### 2.2. What I add to `__call__()`

I add `import torch`, then build an `audio_position_mask` field immediately
after tokenization, **before the diffusion code adds a noisy canvas**:

```python
input_ids = text_inputs["input_ids"]
if audio is None:
    if isinstance(input_ids, torch.Tensor):
        audio_position_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    elif isinstance(input_ids, np.ndarray):
        audio_position_mask = np.zeros_like(input_ids, dtype=np.bool_)
    else:
        audio_position_mask = [[False] * len(row) for row in input_ids]
elif isinstance(input_ids, torch.Tensor):
    audio_position_mask = input_ids.eq(self.audio_token_id)
elif isinstance(input_ids, np.ndarray):
    audio_position_mask = input_ids == self.audio_token_id
else:
    audio_position_mask = [
        [token_id == self.audio_token_id for token_id in row] for row in input_ids
    ]
```

The result is `True` at audio positions and `False` elsewhere. The three
branches support PyTorch tensors, NumPy arrays, and Python lists. Without
audio, the mask is entirely false, even if the text contains the special
ID: there are no audio vectors to inject.

Yes, I still compare IDs at this stage. The difference is **when the mask is
computed**: it is built from the clean prompt prepared by the processor,
then retained. The model does not reconstruct it from a noisy sequence.

### 2.3. New processor output

Before:

```python
return BatchFeature(data={**text_inputs, **audio_inputs}, tensor_type=return_tensors)
```

After:

```python
return BatchFeature(
    data={
        **text_inputs,
        **audio_inputs,
        "audio_position_mask": audio_position_mask,
    },
    tensor_type=return_tensors,
)
```

The caller therefore receives the same information as before, plus the
explicit map of audio positions. For the illustrative prefix
`[42, 250062, 250062, 17]`, the mask is `[False, True, True, False]`.

### 2.4. The three masks that must not be confused

| Field | Shape in the relevant path | The question it answers |
|---|---|---|
| `feature_attention_mask` | `[number_of_clips, mel_frames]` | Which mel frames are real rather than padding? |
| `audio_position_mask` | `[batch, prefix_length]` in the processor's PyTorch output | Where should audio vectors be injected into the embedding sequence? |
| `attention_mask` | A 2D text mask in the processor output; a new 4D mask `[B, 1, S, S]` built by the wrapper for the diffusion decoder | Which sequence positions can each position attend to? |

Adding `audio_position_mask` therefore does not change visibility between
tokens. It describes **placement**, not the diffusion attention mask.

## 3. `huggingface/xllm/modeling_xllm.py`: accepting diffusion attention

This file contains the text decoder. My functional changes concern a
compatibility helper and `XllmModel.forward()`. The Q/K/V attention projections,
Transformer layers, their MLPs, and RoPE positional embeddings are not replaced.

### 3.1. New function: `_create_attention_mask_compat()`

I introduce a compatibility helper that passes only the arguments accepted
by the selected Transformers mask-building function:

```python
def _create_attention_mask_compat(
    mask_function,
    *,
    config,
    attention_mask,
    cache_position,
    past_key_values,
    position_ids,
    inputs_embeds,
):
    """Call Transformers masking helpers across the 4.5x/5.x API split.

    Pass only parameters advertised by the selected helper, preserving the
    causal path across supported Transformers API versions.
    """

    parameters = _inspect.signature(mask_function).parameters
    values = {
        "config": config,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
        "input_embeds": inputs_embeds,
        "inputs_embeds": inputs_embeds,
    }
    return mask_function(**{name: values[name] for name in parameters if name in values})
```

For example, if the helper expects `inputs_embeds` but not `cache_position`,
I pass the former but not the latter. This preserves compatibility of the
causal path across the environments used. This helper does not create
bidirectional attention and is not a new diffusion algorithm.

### 3.2. `XllmModel.forward()`: new `attention_mode` parameter

I add `Literal` to the `typing` imports, then add this argument:

```python
attention_mode: Literal["causal", "diffusion"] = "causal",
```

Unless explicitly requested otherwise, the mode remains causal. Previously,
this `forward()` went through causal-mask construction; now it selects a branch.

### 3.3. The `causal` branch: preserving AR behavior

```python
if attention_mode == "causal":
    # Keep the historical path byte-for-byte equivalent by delegating to
    # Transformers' causal-mask builder exactly as before.
    mask_function = (
        create_causal_mask
        if self.config.sliding_window is None
        else create_sliding_window_causal_mask
    )
    causal_mask = _create_attention_mask_compat(
        mask_function,
        config=self.config,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
        inputs_embeds=inputs_embeds,
    )
```

The code continues to use the Transformers causal-mask builder, or its
sliding-window variant depending on the configuration. I do not replace AR
behavior with diffusion for every call.

### 3.4. The `diffusion` branch: using the supplied 4D mask

```python
elif attention_mode == "diffusion":
    if use_cache or past_key_values is not None:
        raise ValueError(
            "attention_mode='diffusion' requires use_cache=False and no mutable KV cache"
        )
    if attention_mask is None or attention_mask.ndim != 4:
        raise ValueError(
            "attention_mode='diffusion' requires an explicit 4D additive attention mask"
        )
    expected_shape = (
        inputs_embeds.shape[0],
        1,
        inputs_embeds.shape[1],
        inputs_embeds.shape[1],
    )
    if tuple(attention_mask.shape) != expected_shape:
        raise ValueError(
            f"diffusion attention mask must have shape {expected_shape}, "
            f"got {tuple(attention_mask.shape)}"
        )
    if not attention_mask.is_floating_point():
        raise TypeError("diffusion attention mask must be additive and floating point")
    causal_mask = attention_mask.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
else:
    raise ValueError(f"unsupported attention_mode: {attention_mode!r}")
```

I enforce three requirements: no mutable KV cache, a mask with the exact shape
`[batch, 1, sequence_length, sequence_length]`, and floating-point values.
An absent, 2D, boolean, or incorrectly sized mask is rejected.

The final two axes represent the querying positions and the positions they
attend to, respectively. The `1` allows the mask to be shared across attention
heads. It is converted to the same device and numerical type as the embeddings.

The variable name `causal_mask` is retained, but its contents are no longer
causal in this branch. The layers then receive this mask through the existing
`attention_mask=causal_mask` argument.

### 3.5. Example: which positions can attend to which?

The matrix is built in the **new** file
[audio/diffusion/attention.py](audio/diffusion/attention.py), then accepted
here. For two prefix positions `P0, P1` and three canvas positions
`C0, C1, C2`, visibility without padding is as follows
(`1` = allowed, `0` = forbidden):

```text
                  key position
                P0 P1 C0 C1 C2
query      P0    1  0  0  0  0
position   P1    1  1  0  0  0
           C0    1  1  1  1  1
           C1    1  1  1  1  1
           C2    1  1  1  1  1
```

Thus, `C0` can attend to `C2`, whereas causal attention would forbid this.
The prefix never attends to the canvas. Real positions do not attend to
padding; a padding row attends only to itself to avoid an entirely masked
row, and its output must be ignored.

The tensor actually passed is **additive**, not this 0/1 matrix: it contains
`0.0` where attention is allowed and `torch.finfo(dtype).min` where it is
forbidden. This very negative value suppresses the forbidden position's
contribution in the attention softmax.

### 3.6. Why reject the AR KV cache?

During AR generation, previous tokens remain fixed; their keys and values
can be reused while a new token is added. In diffusion, the canvas changes
between steps, and its positions attend to one another. Naively reusing old
keys/values would provide stale context.

This implementation therefore requires `use_cache=False` and no
`past_key_values`. This does not mean that specialized prefix optimization
would be impossible; it simply is not this cache path. Reusing audio
embeddings is a separate operation, compatible with the absence of a KV cache.

### 3.7. The actual diffusion call

In [audio/diffusion/model.py](audio/diffusion/model.py), the wrapper prepares
the audio-prefix embeddings, adds the noisy canvas and, optionally, time
conditioning. It builds the positions and the 4D mask, then calls the text
decoder directly:

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

**This call** selects diffusion attention. The wrapper also uses the composite
model's methods for freezing, encoding, and placing audio; it does not
necessarily go through both of the composite's `forward()` methods at each NFE.

The loss, same/shifted alignment, noise, and sampler remain in the new modules.
I did not add a new gradient-checkpointing implementation in this diff: the
decoder infrastructure already existed, and its compatibility is covered
by the tests.

## What these three changes do not do

They do not replace the encoder, projector, or Transformer layers; nor do they
implement all of diffusion on their own. They make the AR model usable as a
denoiser: frozen and explicitly placed audio, a controlled visibility mask,
and a preserved causal path.

For the rest, [added_files.md](added_files.md) shows where I added the wrapper,
noise, loss, sampling, data preprocessing, and training/launch scripts.
The tests specifically targeting these three files are
[test_audio_position_safety.py](audio/tests/test_audio_position_safety.py),
[test_xllm_diffusion_attention_mode.py](audio/tests/test_xllm_diffusion_attention_mode.py),
and [test_diffusion_model_wrapper.py](audio/tests/test_diffusion_model_wrapper.py).


## Public-checkout validation

The public release passed **189 CPU tests**. The tests that directly require
the withheld processor, composite model or decoder are excluded from collection
while those files are absent. The public diffusion-wrapper tests use toy models.
This validation does not establish full-model parity or recognition quality;
see [docs/VALIDATION.md](docs/VALIDATION.md).
