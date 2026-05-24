# C2: Cache INT8 compression — implementation sketch

## Goal

Reduce cache_channel/cache_time Swift→CoreML marshalling per chunk by
quantizing them to INT8 in-flight instead of FP16.

- cache_channel: FP16 [1,24,56,1024] = ~5.5 MB per call → INT8 = ~2.75 MB
- cache_time:    FP16 [1,24,1024,8]  = ~400 KB         → INT8 = ~200 KB

Expected save: ~3-5% RTFx from marshalling reduction.

## Why deferred

- Cache contains attention state (Q/K/V projections from previous chunks).
- INT8 quant noise propagates through attention softmax across chunks → can amplify.
- No existing parity-test infrastructure for inter-chunk cache drift.
- Implementation requires modifying the encoder's I/O interface (4 changes:
  PyTorch wrapper, traced inputs, traced outputs, Swift cache mgmt).
- Half-day engineering for expected +3-5% RTFx with high WER risk.
- Diminishing returns: B1 fusion already landed +5.5%, cache shape reduction
  already at 60.1 RTFx ceiling. C2 would push to maybe 63-65 RTFx (n=100).

## Implementation steps

1. **Encoder wrapper** (`multilingual_components.py`):
   - Add new inputs: `cache_channel_int8`, `cache_channel_scale`,
     `cache_time_int8`, `cache_time_scale`
   - At forward start: `cache_channel_fp16 = cache_channel_int8.to(fp16) * cache_channel_scale`
   - At forward end:
     `scale = cache_channel_n.abs().amax() / 127`
     `cache_channel_int8_n = (cache_channel_n / scale).round().to(int8)`
   - Return INT8 + scale instead of FP16

2. **Convert script**: thread through new I/O types.

3. **Swift `StreamingNemotronMultilingualAsrManager`**:
   - Cache storage: keep INT8 instead of FP16
   - On encoder call: send INT8 + scale to encoder
   - On encoder return: store INT8 + scale

4. **Parity test**: full LS test-clean WER comparison vs FP16 cache baseline.

## Alternative: per-channel scales

Instead of per-tensor scale, use per-channel (along dim=24-layer or
dim=1024-channel). Reduces quant noise. More complex marshalling.

## Risk profile

- If WER survives: +3-5% RTFx win, real
- If WER degrades by even 0.5pp: not worth shipping (we have higher-WER
  options already at higher RTFx, e.g. cache-14)
- If WER catastrophically breaks: hours of debugging to figure out which
  step introduced the corruption
