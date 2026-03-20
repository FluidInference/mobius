# VoxCPM 1.5 CoreML Conversion

Convert [VoxCPM 1.5](https://huggingface.co/openbmb/VoxCPM1.5) (800M params, 44.1kHz, EN/ZH) to CoreML.

## Source

- **Paper**: [arxiv.org/abs/2509.24650](https://arxiv.org/abs/2509.24650)
- **Code**: [github.com/OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)
- **Weights**: [huggingface.co/openbmb/VoxCPM1.5](https://huggingface.co/openbmb/VoxCPM1.5)
- **License**: Apache 2.0

## Architecture

Tokenizer-free diffusion autoregressive TTS built on MiniCPM-4 (0.5B backbone).

| Component | Layers | Hidden | Params (approx) |
|-----------|--------|--------|-----------------|
| base_lm (MiniCPM4) | 24 | 1024 | ~450M |
| residual_lm | 8 | 1024 | ~80M |
| feat_encoder (LocEnc) | 8 | 1024 | ~80M |
| feat_decoder (LocDiT) | 8 | 1024 | ~80M |
| AudioVAE | encoder [2,3,6,7,7] / decoder [7,7,6,3,2] | 64/2048 | ~130M |

**Total**: ~800M parameters, 44.1kHz output, 6.25 Hz token rate (patch_size=4)

## CoreML Model Split

| Model | Purpose | Input | Output | Calls |
|-------|---------|-------|--------|-------|
| `audio_vae_encoder` | Encode prompt audio to latents | `[1, 1, 220500]` (5s fixed) | `[1, 64, T]` | 1 |
| `audio_vae_decoder` | Decode latents to 44.1kHz audio | `[1, 64, T]` (flexible) | `[1, 1, T*1764]` | 1 |
| `feat_encoder` | Encode latent patches to LM embeddings | `[1, 1, 4, 64]` | `[1, 1, 1024]` | 1 + N |
| `base_lm_step` | Single AR step (24-layer LM + FSQ + stop) | embed + pos + 48 KV caches | lm_hidden + fsq + stop + 48 caches | prefill + N |
| `residual_lm_step` | Single AR step (8-layer residual LM) | embed + pos + 16 KV caches | res_hidden + 16 caches | prefill + N |
| `locdit_step` | Single Euler diffusion step | x, mu, t, cond, dt (batch=2 for CFG) | velocity `[2, 64, 4]` | N * 10 |

Constants exported to `constants/` (~299 MB):
- `embed_tokens.npy` — [73448, 1024] text embedding table (287 MB)
- `enc_to_lm_proj_{w,b}.npy` — encoder-to-LM projection
- `lm_to_dit_proj_{w,b}.npy` — LM-to-DiT projection
- `res_to_dit_proj_{w,b}.npy` — residual-to-DiT projection
- `config.json` — generation config

## Conversion Validation

All models use **Float16** precision (`compute_precision=ct.precision.FLOAT16`, `compute_units=CPU_AND_GPU`).

| Model | Correlation (F16) | Notes |
|-------|-------------------|-------|
| audio_vae_encoder | 0.999989 | Fixed 5s input, Snake activations patched |
| audio_vae_decoder | 0.999999 | Flexible latent length via RangeDim |
| feat_encoder | 1.000000 | 8-layer non-causal transformer |
| base_lm_step | 0.999998 | 24 layers, GQA patched, scatter-based KV cache |
| residual_lm_step | 1.000000 | 8 layers, same GQA/cache pattern |
| locdit_step | 0.999999 | Flow matching estimator, cond_len=4 |

## Generation Pipeline

```
1. Encode prompt audio: latent = audio_vae_encoder(pad_to_5s(prompt))
2. Reshape into patches: [1, 64, T] -> [1, n_patches, 4, 64]
3. Encode patches: feat_emb = feat_encoder(each patch)
4. Project: feat_lm = enc_to_lm_proj(feat_emb)
5. Embed text: text_emb = embed_tokens[ids + audio_start_token] * scale_emb
6. Combine: [text_emb, feat_lm] -> [1, seq_len, 1024]
7. Prefill: step through all tokens via base_lm_step + residual_lm_step
8. Loop (autoregressive):
   a. dit_hidden = lm_to_dit_proj(lm_hidden) + res_to_dit_proj(res_hidden)
   b. noise = randn(1, 64, 4)
   c. For t in 10 Euler steps:
        vel = locdit_step(noise, dit_hidden, prefix_cond, t)  # batch=2 for CFG
        noise += vel * dt
   d. pred_feat = noise (after all steps)
   e. If stop_head predicts stop and step > min_len: break
   f. prefix_cond = pred_feat
   g. next_emb = enc_to_lm_proj(feat_encoder(pred_feat))
   h. lm_hidden, fsq, stop = base_lm_step(next_emb, pos, caches)
   i. res_hidden = residual_lm_step(fsq + next_emb, pos, caches)
9. Decode: audio = audio_vae_decoder(concat(all pred_feats))
```

## Setup

```bash
cd mobius/models/tts/voxcpm-1.5/coreml
uv sync
```

## Conversion

```bash
# 1. Verify PyTorch inference
uv run python verify_pytorch.py

# 2. Export constants (embeddings, projections, config)
uv run python export_constants.py

# 3. Convert each component
uv run python convert_audio_vae_encoder.py
uv run python convert_audio_vae_decoder.py
uv run python convert_feat_encoder.py
uv run python convert_base_lm_step.py
uv run python convert_residual_lm_step.py
uv run python convert_locdit_step.py

# 4. Validate end-to-end
uv run python generate_coreml.py --text "Hello world" --prompt prompt.wav --output output.wav
```

## Key Conversion Challenges

See [TRIALS.md](TRIALS.md) for detailed conversion log.

1. **GQA attention**: 16 query heads, 2 KV heads — SDPA `enable_gqa=True` doesn't trace to CoreML. Fixed by manually expanding KV heads with `repeat_interleave`.
2. **Snake activations**: `@torch.jit.script` + shape tuple indexing breaks CoreML. Replaced with simple module.
3. **In-place KV cache**: CoreML doesn't support in-place ops. Replaced with functional `scatter`.
4. **MuP scaling**: `scale_depth / sqrt(num_hidden_layers)` in residual connections.
5. **AudioVAE data-dependent assertions**: Baked into trace, prevents flexible encoder shapes.
6. **LM step validation methodology**: Comparing against different model loads gave false failures (see Trials).
