# Shape bounds reference

All RangeDim bounds in `scripts/convert-coreml.py` derive from a single
`--max-frames` (default 2000), via `compute_shape_bounds`:

```python
{
    'max_T_enc':       512,
    'max_T_a':         max_frames,            # 2000
    'max_T2':          max_frames * 2,        # 4000
    'max_ns0_T':       max_frames * 20,       # 40000
    'max_ns1_T':       max_frames * 120 + 1,  # 240001
    'max_x_pre_T':     max_frames * 120 + 1,  # 240001
    'max_audio_samples': max_frames * 600,    # 1.2 M  ≈ 50 s @ 24 kHz
}
```

## Per-stage I/O

| Stage | Input → Output | Notable shapes |
|---|---|---|
| Albert     | `(input_ids[1, T_enc], attention_mask[1, T_enc])` → `bert_dur[1, T_enc, 768]` | T_enc ≤ 512 |
| PostAlbert | `(bert_dur[1, T_enc, 768], input_ids[1, T_enc], style_s[1, 128], speed[1], attention_mask[1, T_enc])` → `(duration[1, T_enc], d[1, 256, T_enc], t_en[1, 512, T_enc])` | static T_enc |
| Alignment  | `(pred_dur[1, T_enc] int32, d[1, 256, T_enc], t_en[1, 512, T_enc])` → `(en[1, 512, T_a], asr[1, 512, T_a])` | T_a = sum(pred_dur), T_a ≤ 2000 |
| Prosody    | `(en[1, 512, T_a], style_s[1, 128])` → `(F0[1, T2], N[1, T2])` | T2 = T_a * 2 |
| Noise      | `(F0_curve[1, T2], style_timbre[1, 128])` → `(x_source_0[1, C0, T_ns0], x_source_1[1, C1, T_ns1])` | T_ns0 = T_a * 20, T_ns1 ≈ T_a * 120 |
| Vocoder    | `(asr[1, 512, T_a], F0_curve[1, T2], N_pred[1, T2], x_source_0, x_source_1, style_timbre[1, 128])` → `(anchor[discard], x_pre[1, 128, T_pre])` | T_pre ≈ T_a * 120 |
| Tail       | `(x_pre[1, 128, T_pre])` → `(audio[1, N_samples])` | N ≈ T_a * 600 (24 kHz @ 600 samples/frame) |

## Capacities at default `--max-frames=2000`

* **T_enc cap**: 512 IPA characters (incl. BOS/EOS). Practically ~80–100 words.
* **T_a cap**: 2000 frames ≈ 24 seconds of speech.
* **Audio cap**: 1,200,000 samples = 50 s @ 24 kHz.

If a sentence's `T_a` (computable after PostAlbert) exceeds `max_T_a`, the
benchmark/inference scripts must skip or chunk it. `scripts/convert-coreml.py` and
`scripts/benchmark.py` both probe `T_a` after PostAlbert and abort that sentence
gracefully if it exceeds `max_T_a`.

## Voice pack indexing

```python
row = max(min(len(phonemes) - 1, voice_pack.shape[0] - 1), 0)  # 0..509
ref_s        = voice_pack[row]              # shape [1, 256]
style_timbre = ref_s[:, :128]               # → Noise + Vocoder
style_s      = ref_s[:, 128:]               # → PostAlbert + Prosody
```

The voice pack itself is **independent of T_enc** — it's a fixed lookup
table generated offline from Kokoro's voice training. It only varies along
its first axis (utterance length bucket).

## Tail audio length formula

`audio_samples ≈ T_a × hop_length × 2 × ratio_factor`

For default Kokoro config (n_fft=1024, hop=300):
* `T_pre ≈ T_a × 120` (vocoder upsamples T_a→T_pre by ~120×)
* `audio_samples ≈ T_pre × 5` (iSTFT hop=5 samples per pre-frame? — verify
  empirically; the practical relation observed is `audio_samples ≈ T_a × 600`)
