# Debugging Methodology

Techniques used to diagnose issues during Qwen3-TTS CoreML conversion and Swift integration. These patterns are reusable for future model conversions.

---

## 1. Step-by-Step Token Comparison (CoreML vs PyTorch)

**When to use:** Output sounds wrong or tokens diverge from reference.

**Technique:** Run both CoreML and PyTorch decoders side-by-side, printing the CB0 token at each step. The first step where they diverge pinpoints where the bug is.

```
Step  0: CoreML=1995  PyTorch=1995  ✓
Step  1: CoreML=215   PyTorch=215   ✓
Step  2: CoreML=1181  PyTorch=212   ✗ ← divergence starts here
```

**What it revealed:**
- KV cache corruption (garbage padding entries caused divergence at step 2)
- Wrong speaker embedding (token 0 was 1221 instead of 1995)
- Greedy vs sampled decoding (identical first token but different sequences due to sampling)

**Key insight:** If step 0 matches but step 1+ diverges, the issue is in the decode loop (KV cache, position IDs, embedding sum). If step 0 itself is wrong, the issue is in prefill (wrong inputs, wrong embedding).

---

## 2. EOS Logit Rank Tracking

**When to use:** Model never stops generating (always hits max frames).

**Technique:** At each decode step, sort the logits and log where the EOS token ranks:

```python
ranks = np.argsort(logits)[::-1]
eos_rank = np.where(ranks == EOS_TOKEN_ID)[0][0]
print(f"Step {step}: EOS rank {eos_rank}, logit {logits[EOS_TOKEN_ID]:.3f}")
```

**What it revealed:**
- EOS at index 2049 (wrong) was always masked out → rank never improved
- EOS at index 2150 (correct) rose from rank 847 → rank 2-4 after speech ends
- Even at rank 2, greedy argmax never selects EOS → proves sampling is required

---

## 3. First-Token Diagnostic

**When to use:** Quick sanity check that the pipeline is wired correctly.

**Technique:** The first CB0 token is deterministic for a given input (prefill has no autoregressive feedback). Compare against a known-good reference:

| Condition | First CB0 Token |
|-----------|----------------|
| Correct pipeline | 1995 |
| Wrong speaker embedding | 1221 |
| Wrong role_ids (system instead of assistant) | 1724 |
| Missing codec embedding sum | varies |

**What it revealed:** Each misconfiguration produces a characteristic first token. If the first token is wrong, the issue is in the prefill inputs (not the decode loop).

---

## 4. Greedy-vs-Greedy Cross-Validation

**When to use:** Verify that a CoreML model matches PyTorch numerically, independent of sampling randomness.

**Technique:** Run both CoreML and PyTorch with `argmax` (greedy, no sampling) on the same input. Compare outputs token-for-token. Since greedy is deterministic, any difference is a conversion bug.

```python
# PyTorch
pt_token = torch.argmax(pt_logits).item()
# CoreML
cm_token = int(np.argmax(cm_logits))
assert pt_token == cm_token, f"Step {step}: PT={pt_token} CM={cm_token}"
```

**What it revealed:**
- Confirmed that the V10 decode model matches PyTorch exactly for the first ~15 steps
- Divergence after step 15 is expected (float32 rounding compounds through autoregression)
- Both greedy pipelines produce identical stuck/repetitive patterns, proving the issue is algorithmic (not a conversion bug)

---

## 5. Hidden State Comparison

**When to use:** Token outputs match at step N but diverge at step N+1 — need to find where internal values differ.

**Technique:** Extract and compare intermediate tensors: `past_hidden`, logits, and KV cache values between CoreML and PyTorch.

```python
pt_hidden = pt_output.last_hidden_state[0, -1, :10].tolist()
cm_hidden = cm_output['past_hidden'][0, 0, :10].tolist()
diff = max(abs(a-b) for a,b in zip(pt_hidden, cm_hidden))
print(f"Step {step}: max hidden diff = {diff:.6f}")
```

**What it revealed:**
- KV cache not trimmed: hidden states diverged immediately because the model attended to padding
- Embedding sum missing: hidden states were close but systematically offset because CB1-15 embeddings weren't included

---

## 6. Audio RMS Energy Analysis

**When to use:** Output audio sounds wrong, silent, or has unexpected artifacts.

**Technique:** Compute RMS energy per second (or per 20ms window) to visualize the audio's energy profile:

```python
for sec in range(int(len(samples) / sr)):
    chunk = samples[sec*sr : (sec+1)*sr]
    rms = np.sqrt(np.mean(chunk**2))
    print(f"  {sec}-{sec+1}s: RMS={rms:.4f}")
```

**What it revealed:**
- Leading silence: 1.5-2.0s of near-zero RMS before speech
- Trailing blip: tiny spike (RMS=0.0156) at ~9s in the padding region
- Completely silent audio (RMS=0.0000): greedy code predictor or greedy Chinese CB0
- Format string bug: `:.1f` rounds 0.035 to "0.0" — use `:.4f`

---

## 7. Spectral Cosine Similarity

**When to use:** Two audio outputs sound roughly correct but you need to quantify how similar they are.

**Technique:** Compute mel spectrograms and calculate cosine similarity:

```python
import librosa
mel1 = librosa.feature.melspectrogram(y=audio1, sr=24000)
mel2 = librosa.feature.melspectrogram(y=audio2, sr=24000)
# Align lengths, flatten, cosine similarity
cos_sim = np.dot(m1, m2) / (np.linalg.norm(m1) * np.linalg.norm(m2))
```

**Baselines established:**
- Same model, same seed: ~0.92
- Same model, different seed: ~0.12 (completely different waveforms!)
- CoreML vs PyTorch (different seeds): 0.73-0.92
- Garbled audio vs reference: < 0.3

**Key insight:** For stochastic TTS, spectral similarity between different runs is meaningless. Use ASR transcription accuracy instead.

---

## 8. Whisper ASR Verification

**When to use:** Final validation that generated speech says the right words.

**Technique:** Run Whisper on the output WAV and compare the transcription against the input text:

```python
import whisper
model = whisper.load_model("base")
result = model.transcribe("output.wav")
print(f"Expected: '{input_text}'")
print(f"Got:      '{result['text']}'")
```

**What it revealed:**
- Garbled audio transcribed as Chinese/Japanese gibberish → wrong model version pairing
- Chinese "世界" (shijie) sometimes transcribed as "事件" (shijian) → phonetic ambiguity, not a bug
- All three pipelines (Swift, CoreML Python, PyTorch) transcribed correctly → conversion is valid

---

## 9. Config.json Inspection

**When to use:** Constants or special token IDs seem wrong.

**Technique:** Always check the model's `config.json` and `generation_config.json` from HuggingFace for ground-truth values:

```json
{
  "codec_eos_token_id": 2150,
  "codec_bos_id": 2149,
  "do_sample": true,
  "temperature": 0.9,
  "top_k": 50
}
```

**What it revealed:**
- EOS/BOS constants were wrong (2049/2048 vs 2150/2149)
- `do_sample: true` is the default for a reason — greedy fails
- `temperature: 0.9, top_k: 50` are the intended sampling parameters

---

## 10. Source Code Reading (PyTorch Reference)

**When to use:** Behavior is unclear from config alone — need to understand what the official code actually does.

**Technique:** Read the HuggingFace model's `modeling_*.py` source, particularly the `generate()` method, `forward()`, and any comments:

**Key findings from source reading:**
- The V3 decode source has an explicit comment: *"do_sample=True is required for the subtalker to work correctly!"*
- The prefill code has: *"IMPORTANT: Only positions 0 to actual_len-1 are valid! Caller must slice kv_cache"*
- The embedding sum is done in `TracableDecodeV3.forward()`, not in a separate step
- The code predictor uses `use_cache=False` (recomputes from scratch each step in the official code, but we added KV caching for efficiency)

---

## 11. Explore Scripts for Architecture Discovery

**When to use:** Starting a new model conversion — need to understand the architecture before writing conversion code.

**Technique:** Write lightweight scripts that load the model and print its structure, shapes, and configs without running inference:

```python
for name, child in model.named_children():
    params = sum(p.numel() for p in child.parameters())
    print(f"  {name}: {type(child).__name__} ({params/1e6:.1f}M params)")
```

**What the explore scripts revealed:**
- LM has 28 transformer layers, 16 attention heads, 8 KV heads, head_dim=128
- Code predictor has 5 layers, 16 heads, 8 KV heads, head_dim=128, hidden=1024
- Audio decoder: 16 codebooks × T frames → audio at 1920 samples/frame
- Separate embedding tables for CB0 (in LM) vs CB1-15 (in code predictor)

---

## General Debugging Principles

1. **Bisect the pipeline.** If end-to-end output is wrong, test each stage independently: prefill → decode → code predictor → audio decoder. The first stage that diverges from reference contains the bug.

2. **Use deterministic mode first.** Fix greedy decoding to work before adding sampling. Greedy is reproducible and makes A/B comparisons trivial.

3. **Compare against PyTorch, not expectations.** Don't assume what the output "should" be. Run the PyTorch reference with identical inputs and compare numerically.

4. **Log the first divergence.** In autoregressive models, errors compound. The first step where output differs is the only one worth investigating — everything after is downstream noise.

5. **Check the simple things first.** Wrong constants, wrong file paths, wrong model versions, and missing inputs account for most bugs. Architecture misunderstandings are rarer.
