# Architecture: Nemotron-3.5-ASR-Streaming-Multilingual 0.6B

End-to-end model graph and how language conditioning actually works.
Derived from direct inspection of `model_weights.ckpt` + `model_config.yaml`
plus runtime introspection of the live NeMo model class
(`EncDecRNNTBPEModelWithPrompt.forward`, `conformer_stream_step`,
`_apply_prompt_to_encoded`, `set_inference_prompt`).

## Top-level modules (only these — verified from checkpoint)

| Module | Tensors | Params | Role |
|--------|---------|--------|------|
| `encoder` | 636 | 609.1 M | FastConformer Cache-Aware (24 layers, d_model=1024, 8 heads) |
| `decoder.prediction` | 9 | 14.9 M | RNNT prediction network (2× LSTM @ 640) + token embedding (13088 × 640) |
| `joint` | 6 | 9.5 M | enc/pred projections (→ 640) + output Linear (→ 13088) |
| `prompt_kernel` | 4 | 4.5 M | 2-layer MLP `Linear(1152, 2048) → ReLU → Linear(2048, 1024)` |
| `preprocessor` | 2 | 0.03 M | 128-bin mel filterbank, 16 kHz |
| **Total** | **657** | **638.0 M** | |

**There are no other modules.** Specifically: no language-ID classifier, no separate detection head, no auxiliary CTC weights (config mentions `aux_ctc` but the checkpoint ships no aux weights — `aux_ctc.decoder.vocabulary: []` in config confirms it's not active at inference).

## Forward graph (single chunk, streaming)

The prompt is applied **after the full encoder runs**, not between the
front-end and the conformer body. The 24× conformer is
language-agnostic; only `prompt_kernel` is conditioned.

```
audio [B, 16000·t]                                 ──┐
                                                     │
preprocessor (mel filterbank)                        │
   └─► mel [B, 128, T]                               │  language-agnostic
                                                     │  acoustic frontend
encoder.pre_encode (3× strided conv, factor 8)       │
   └─► features [B, T/8, 1024]                       │
                                                     │
+ cache_channel [1, 24, 70, 1024]                    │  language-agnostic
+ cache_time    [1, 24, 1024, 8]                     │  acoustic body
+ cache_len     [1]                                  │
                                                     │
24× FastConformer layers (cache-aware self-attn      │
                          + causal conv + 2× FFN)    │
   └─► encoded [B, 1024, T/8]   (B, D, T order)      │
   └─► updated caches                              ──┤
                                                     │
                                                   ──┤  ← LANGUAGE PROMPT
prompt_id (int [B], == _inference_prompt_index)      │     injects HERE,
   └─► one_hot [B, 128]                              │     after the encoder
   └─► broadcast over T/8 ──► [B, T/8, 128]          │
                                                     │
transpose encoded → [B, T/8, 1024]                   │
concat([encoded, one_hot], dim=-1) → [B, T/8, 1152]  │
   └─► prompt_kernel (1152 → ReLU → 2048 → 1024)     │
   └─► conditioned [B, T/8, 1024]                    │
transpose back → [B, 1024, T/8]                    ──┤
                                                     │
decoder.prediction (autoregressive)                  │
   └─► dec_out [B, U, 640]                           │  RNNT prediction
                                                     │  (token-level, vocab-aware)
joint (Linear(enc→640) ⊕ Linear(dec→640) → ReLU      │
       → Linear(640 → 13088))                        │
   └─► logits [B, T/8, U, 13088]                     │
                                                     │
greedy argmax (with blank=13087)                   ──┘
   └─► token sequence
```

## What the prompt actually does

`prompt_kernel` is **not** a lookup table. It's a small MLP whose first
layer takes `concat(encoded_1024, one_hot_128)` — note encoded is in
positions `[0:1024]` and the prompt one-hot is in positions
`[1024:1152]`. This ordering matters when wiring up the CoreML graph;
it must match the trained weight matrix orientation.

`initialize_prompt_feature` in the model class spells out the
construction:

```python
proj_in_size  = self.num_prompts + self._cfg.model_defaults.enc_hidden  # 128 + 1024
proj_out_size = self._cfg.model_defaults.enc_hidden                     # 1024
self.prompt_kernel = nn.Sequential(
    nn.Linear(proj_in_size,  proj_out_size * 2),  # 1152 → 2048
    nn.ReLU(),
    nn.Linear(proj_out_size * 2, proj_out_size),  # 2048 → 1024
)
```

Each language id (0..127) effectively activates a different subset of
the 2048 hidden units. The MLP learns a **language-conditional
post-encoder adapter**.

| `target_lang` argument | Resolved prompt_id | What the model does |
|------------------------|--------------------|---------------------|
| `"en-US"` (or `"en"`)  | 0  | Biases features toward English phonotactics; emits `<en-US>` as first token, then English |
| `"es-ES"`              | 2  | Biases toward European Spanish; emits `<es-ES>` first, then Spanish |
| `"zh-CN"` (or `"zh-ZH"`) | 4  | Mandarin Simplified; emits `<zh-CN>` first, then Chinese characters |
| `"zh-TW"`              | 5  | Mandarin Traditional (NOTE: vocab has no `<zh-TW>` tag — model emits `<zh-CN>` since that's the closest vocab entry, but encoder is conditioned for Traditional script) |
| `"auto"`               | 101 | Generic prompt trained on all languages; model emits whatever lang-tag it detects acoustically |

## How the prompt is fed at inference (two NeMo paths, one CoreML path)

NeMo exposes the prompt via different mechanisms depending on whether
you call the high-level `forward` or the streaming `conformer_stream_step`:

| API | How the prompt enters |
|-----|-----------------------|
| `model.forward(input_signal, ..., prompt_indices=tensor([id]))` | Pass an int tensor `[B]` directly as the `prompt_indices` kwarg. Full-utterance offline path. |
| `model.conformer_stream_step(processed_signal, cache_*, ...)` | **No prompt kwarg.** Caller must call `model.set_inference_prompt("en-US")` **once before the streaming loop** — this sets `model._inference_prompt_index: int`. Each chunk's `_apply_prompt_to_encoded(encoded)` reads that attribute and builds the one-hot internally. |
| CoreML encoder mlpackage | Takes `prompt_id: int32[1]` as a normal input every chunk. The graph builds the one-hot, concats with the encoder output, and runs `prompt_kernel` — all internal. Equivalent to the streaming path but stateless across chunks. |

`prompt_dictionary` lives at **`model.cfg.model_defaults.prompt_dictionary`**
(also mirrored under `cfg.train_ds`, `cfg.validation_ds`, `cfg.test_ds`,
but `model_defaults` is canonical). The setter:

```python
def set_inference_prompt(self, target_lang: str):
    prompt_dict = self.cfg.model_defaults.get("prompt_dictionary", {})
    if target_lang not in prompt_dict:
        raise ValueError(...)
    self._inference_prompt_index = prompt_dict[target_lang]
```

In the converted CoreML graph we deliberately bypass this state-based
API: the encoder mlpackage takes `prompt_id` as a per-chunk int input
and behaves identically whether the surrounding code calls
`set_inference_prompt` or not. Swift/Python only needs to know the
prompt-id mapping (carried in `metadata.json["prompt_dictionary"]`).

## How "auto" mode detects language (the only mechanism)

There is **no explicit language detector**. The "detection" is entirely implicit:

1. During training, examples were fed with `target_lang = "auto"` (prompt id 101) paired with the same `<xx-XX>` + transcript labels as their per-language counterparts.
2. With prompt 101, the model learned that the joint must predict the correct `<xx-XX>` as the first non-blank token directly from acoustics.
3. At inference with `prompt_id=101`, the encoder produces "generic" conditioned features; the RNNT decoder then emits the leading `<xx-XX>` it determines best matches the audio.
4. Subsequent tokens are conditioned (autoregressively) on that emitted lang tag.

**Practical consequence:** in `auto` mode the first emitted non-blank token IS the language detection result. You can read it off without any extra head. Empirical verification (from `decoder.prediction.embed.weight` analysis):

- Lang-tag tokens have normal embedding magnitudes (mean norm 21.66 vs 21.96 for content tokens) → they're treated as regular vocabulary, not special
- Lang-tag intra-group cosine is ~0.014 (essentially orthogonal) → they don't cluster as a separate "indicator" group; each lives near its own language's content tokens

This is consistent with: "the model just learned to emit them first."

## Recommended Swift API

Always pass a `prompt_id` (the encoder requires it as an input). Treat detection as a side-effect of the first emitted token.

```swift
public enum NemotronTargetLang {
    case forced(String)   // e.g., "en-US", "zh-CN" — biases the model
    case auto             // prompt_id = 101; model picks language itself

    /// Resolves to an integer prompt id using metadata.json's prompt_dictionary
    func promptId(in dict: [String: Int]) -> Int32 {
        switch self {
        case .forced(let lang): return Int32(dict[lang] ?? dict["auto"] ?? 101)
        case .auto:             return Int32(dict["auto"] ?? 101)
        }
    }
}

public struct NemotronTranscription {
    public let detectedLang: String?   // populated from leading <xx-XX> token
    public let text: String            // lang-tag stripped
}
```

Output handling:

1. Run the RNNT decoding loop normally
2. Inspect the **first emitted non-blank token id**. If it's in `metadata.json["lang_tag_token_ids"]`, record it as `detectedLang` (look up the piece text via tokenizer) and strip it from the output.
3. Filter any further lang-tag tokens from `text` (they shouldn't occur mid-utterance in a well-behaved decode, but strip defensively).

## Why not have separate models per language?

The 105-entry `prompt_dictionary` would require 105 separate exports if we baked the prompt at conversion time. The current design (CoreML graph builds the one-hot from an int32 `prompt_id` input) keeps it to a single set of `.mlpackage` files (~600 MB encoder int8 + ~30 MB decoder + ~10 MB joint).

## Why use `auto` over a forced language?

| Choice | Pros | Cons |
|--------|------|------|
| **Forced** (`en-US`, etc.) | Higher accuracy when the user actually speaks that language (encoder is biased correctly); no first-token "detection error" risk | Wrong choice → degraded transcription, may still emit a different `<xx-XX>` if acoustics strongly mismatch |
| **`auto`** (101) | No need to ask the user; handles code-switching gracefully (at utterance boundaries) | Slightly lower per-language accuracy; depends on training coverage of `auto`-prompted examples |

For VoiceLink: `auto` is the right default. Provide a UI override to force a specific language if the user wants.

## Streaming considerations

- The prompt is fed **every chunk** in the CoreML graph (not just the first). The conformer body itself doesn't consume the prompt, but `prompt_kernel` runs on every chunk's encoder output, so the int32 input must be supplied each call. NeMo's streaming API sidesteps this by reading from a stashed `self._inference_prompt_index` per chunk; the CoreML graph keeps the same effective behavior but makes the prompt explicit at the I/O boundary.
- Switching `prompt_id` mid-utterance is theoretically possible but untested; safest to fix at session start.
- The leading `<xx-XX>` will appear in the first chunk that produces a non-blank emission. Subsequent chunks should not re-emit it (the autoregressive decoder state knows it was already emitted).

## Verification commands

```bash
# Confirm no LID head
python3 -c "
import torch
sd = torch.load('model_weights.ckpt', map_location='cpu', mmap=True, weights_only=True)
heads = set(k.split('.')[0] for k in sd)
print('Top-level namespaces:', sorted(heads))
"
# Expected: ['decoder', 'encoder', 'joint', 'preprocessor', 'prompt_kernel']

# Confirm prompt_kernel shape
python3 -c "
import torch
sd = torch.load('model_weights.ckpt', map_location='cpu', mmap=True, weights_only=True)
for k in sd:
    if 'prompt' in k: print(k, tuple(sd[k].shape))
"
# Expected:
#   prompt_kernel.0.weight (2048, 1152)
#   prompt_kernel.0.bias   (2048,)
#   prompt_kernel.2.weight (1024, 2048)
#   prompt_kernel.2.bias   (1024,)
```
