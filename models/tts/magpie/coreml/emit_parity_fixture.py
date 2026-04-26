"""Emit intermediate-tensor fixtures for cross-implementation parity testing.

Runs the Magpie CoreML pipeline for a fixed (text, speaker, language, seed)
and dumps intermediate tensors so the Swift port (or any other
implementation) can replay each stage and diff against this ground truth.

Two output modes:

- ``--mode full`` (default): runs the full pipeline and saves an ``.npz`` with
  text tokens, encoder output, post-prefill KV caches, per-step decoder
  hidden states, per-step sampled codes, the final ``(8, N)`` codes matrix,
  and the decoded PCM.
- ``--mode tokenizer``: tokenizes only and saves a ``.json`` mapping
  ``{text, speaker, language, token_ids}`` — cheap to diff against the Swift
  ``MagpieTokenizer`` output without requiring CoreML at all.

Example:

    python emit_parity_fixture.py "Hello world." \\
        --speaker 0 --language en --seed 42 \\
        --output fixture_en_s0.npz

    python emit_parity_fixture.py "Hello world." \\
        --speaker 0 --language en --mode tokenizer \\
        --output fixture_en_s0_tokens.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any

import coremltools as ct
import numpy as np
import soundfile as sf

# Re-use everything from the main script so we never drift from the reference.
from generate_coreml import (  # noqa: E402
    BUILD_DIR,
    DECODER_CACHE_K_OUT_KEYS,
    DECODER_CACHE_V_OUT_KEYS,
    DECODER_HIDDEN_KEY,
    DECODER_POSITION_KEYS,
    _tokenize_text,
    embed_audio_codes,
    load_audio_embeddings,
    load_constants,
    load_local_transformer,
    load_speaker_embedding,
    local_transformer_sample,
)


def _make_caches(n_layers: int, max_seq_len: int, n_heads: int, d_head: int):
    c, p = {}, {}
    for i in range(n_layers):
        c[f"cache_k{i}"] = np.zeros(
            (1, max_seq_len, n_heads, d_head), dtype=np.float32
        )
        c[f"cache_v{i}"] = np.zeros(
            (1, max_seq_len, n_heads, d_head), dtype=np.float32
        )
        p[f"position{i}"] = np.array([0.0], dtype=np.float32)
    return c, p


def emit_tokenizer_fixture(
    text: str,
    speaker: int,
    language: str,
    output_path: str,
) -> None:
    constants = load_constants()
    token_ids = _tokenize_text(text, language, constants).tolist()
    fixture = {
        "text": text,
        "speakerIndex": speaker,
        "languageCode": language,
        "expectedTokenIds": token_ids,
    }
    with open(output_path, "w") as f:
        json.dump(fixture, f, indent=2, ensure_ascii=False)
    print(f"Wrote tokenizer fixture → {output_path}  ({len(token_ids)} tokens)")


def emit_full_fixture(
    text: str,
    speaker: int,
    language: str,
    output_path: str,
    temperature: float,
    topk: int,
    max_steps: int,
    seed: int,
    use_cfg: bool,
    cfg_scale: float,
) -> None:
    np.random.seed(seed)
    constants = load_constants()

    num_codebooks = constants["num_audio_codebooks"]
    audio_bos_id = constants["special_tokens"]["audio_bos_id"]
    audio_eos_id = constants["special_tokens"]["audio_eos_id"]
    sample_rate = constants["output_sample_rate"]
    d_model = constants["decoder"]["d_model"]
    n_layers = constants["decoder"]["n_layers"]
    sa_n_heads = constants["decoder"]["sa_n_heads"]
    d_head = d_model // sa_n_heads
    max_text_len = 256
    max_seq_len = 512
    min_frames = constants["inference"].get("min_generated_frames", 4)

    # --- 1. Tokenize ---
    text_tokens = _tokenize_text(text, language, constants)
    T_text = int(len(text_tokens))
    text_tokens_padded = np.zeros(max_text_len, dtype=np.int32)
    text_tokens_padded[:T_text] = text_tokens
    text_mask = np.zeros(max_text_len, dtype=np.float32)
    text_mask[:T_text] = 1.0

    # --- 2. Load models ---
    text_encoder = ct.models.MLModel(
        os.path.join(BUILD_DIR, "text_encoder.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )
    decoder_step = ct.models.MLModel(
        os.path.join(BUILD_DIR, "decoder_step.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )
    nanocodec = ct.models.MLModel(
        os.path.join(BUILD_DIR, "nanocodec_decoder.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )

    # --- 3. Encode text ---
    enc_out = text_encoder.predict({
        "text_tokens": text_tokens_padded[np.newaxis, :],
        "text_mask": text_mask[np.newaxis, :],
    })
    encoder_output = np.asarray(enc_out["encoder_output"], dtype=np.float32)

    if use_cfg:
        uncond_encoder_output = np.zeros_like(encoder_output)
        uncond_text_mask = np.zeros_like(text_mask)
        uncond_text_mask[0] = 1.0

    # --- 4. Load embeddings + LT weights ---
    speaker_emb = load_speaker_embedding(speaker)
    T_ctx = int(speaker_emb.shape[0])
    audio_emb_tables = load_audio_embeddings(constants)
    lt_weights = load_local_transformer()

    caches, positions = _make_caches(n_layers, max_seq_len, sa_n_heads, d_head)
    if use_cfg:
        u_caches, u_positions = _make_caches(n_layers, max_seq_len, sa_n_heads, d_head)

    def _run_step(audio_embed, enc_out_np, mask_np, cache_dict, pos_dict):
        inputs: dict[str, Any] = {
            "audio_embed": audio_embed.astype(np.float32),
            "encoder_output": enc_out_np.astype(np.float32),
            "encoder_mask": mask_np[np.newaxis, :].astype(np.float32),
        }
        inputs.update(cache_dict)
        inputs.update(pos_dict)
        out = decoder_step.predict(inputs)
        for i in range(n_layers):
            cache_dict[f"cache_k{i}"] = out[DECODER_CACHE_K_OUT_KEYS[i]]
            cache_dict[f"cache_v{i}"] = out[DECODER_CACHE_V_OUT_KEYS[i]]
            pos_dict[f"position{i}"] = out[DECODER_POSITION_KEYS[i]]
        return np.asarray(out[DECODER_HIDDEN_KEY], dtype=np.float32)

    # --- 5. Prefill ---
    uncond_ctx = np.zeros((1, 1, d_model), dtype=np.float32)
    for t in range(T_ctx):
        ctx = speaker_emb[np.newaxis, np.newaxis, t, :]
        _run_step(ctx, encoder_output, text_mask, caches, positions)
        if use_cfg:
            _run_step(uncond_ctx, uncond_encoder_output, uncond_text_mask,
                      u_caches, u_positions)

    # Snapshot KV caches after prefill (deep-copied so later rotation doesn't
    # mutate the fixture).
    prefill_caches = {k: v.copy() for k, v in caches.items()}
    prefill_positions = {k: v.copy() for k, v in positions.items()}

    # --- 6. AR loop ---
    current_codes = np.full(num_codebooks, audio_bos_id, dtype=np.int32)
    per_step_hidden: list[np.ndarray] = []
    per_step_codes: list[np.ndarray] = []

    gen_start = time.time()
    for step in range(max_steps):
        audio_embed = embed_audio_codes(current_codes, audio_emb_tables, num_codebooks)
        cond_hidden = _run_step(audio_embed, encoder_output, text_mask, caches, positions)

        if use_cfg:
            uncond_hidden = _run_step(
                audio_embed, uncond_encoder_output, uncond_text_mask,
                u_caches, u_positions,
            )
            uncond_dec_hidden = uncond_hidden[0, 0]
        else:
            uncond_dec_hidden = None

        decoder_hidden = cond_hidden[0, 0]
        per_step_hidden.append(decoder_hidden.copy())

        forbid_eos = step < min_frames
        next_codes = local_transformer_sample(
            decoder_hidden, lt_weights, audio_emb_tables,
            num_codebooks, temperature, topk, forbid_eos,
            uncond_decoder_hidden=uncond_dec_hidden,
            cfg_scale=cfg_scale if use_cfg else 1.0,
        )

        is_eos = bool(np.any(next_codes == audio_eos_id))
        if is_eos and step >= min_frames:
            per_step_codes.append(next_codes.copy())
            break
        per_step_codes.append(next_codes.copy())
        current_codes = next_codes

    gen_time = time.time() - gen_start

    predicted_codes_full = np.stack(per_step_codes, axis=1)  # (8, N)

    # --- 7. NanoCodec decode ---
    max_frames = 256
    T_total = min(predicted_codes_full.shape[1], max_frames)
    padded = np.zeros((num_codebooks, max_frames), dtype=np.int32)
    padded[:, :T_total] = predicted_codes_full[:, :T_total]
    codec_out = nanocodec.predict({
        "tokens": padded[np.newaxis, :, :].astype(np.int32),
    })
    audio = np.asarray(codec_out["audio"], dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.flatten()
    expected_samples = T_total * constants["codec_samples_per_frame"]
    audio = audio[:expected_samples]
    peak = float(np.abs(audio).max())
    if peak > 0:
        audio = audio / peak * 0.9

    # --- 8. Pack fixture ---
    # NPZ contains tensors only. The Swift `NpyReader` rejects 0-D shapes,
    # `<U…` string dtypes, and bool dtype, so all scalar/string metadata is
    # written to a sidecar JSON next to the .npz.
    fixture: dict[str, Any] = {
        # Stage 1: tokenizer
        "textTokens": text_tokens.astype(np.int32),
        "textTokensPadded": text_tokens_padded.astype(np.int32),
        "textMask": text_mask.astype(np.float32),
        # Stage 2: text encoder
        "encoderOutput": encoder_output.astype(np.float32),
        # Stage 3: post-prefill caches (rank-4 split-K/V)
        **{f"prefillCacheK{i}": prefill_caches[f"cache_k{i}"].astype(np.float32)
           for i in range(n_layers)},
        **{f"prefillCacheV{i}": prefill_caches[f"cache_v{i}"].astype(np.float32)
           for i in range(n_layers)},
        **{f"prefillPosition{i}": prefill_positions[f"position{i}"].astype(np.float32)
           for i in range(n_layers)},
        # Stage 4: per-step AR trace
        "perStepDecoderHidden": np.stack(per_step_hidden, axis=0).astype(np.float32),
        "perStepCodes": np.stack(per_step_codes, axis=0).astype(np.int32),
        "predictedCodes": predicted_codes_full.astype(np.int32),
        # Stage 5: audio
        "audioPcm": audio.astype(np.float32),
    }

    np.savez_compressed(output_path, **fixture)

    # Sidecar metadata JSON (next to the .npz with the same basename).
    meta = {
        "text": text,
        "speakerIndex": int(speaker),
        "languageCode": language,
        "seed": int(seed),
        "useCfg": bool(use_cfg),
        "cfgScale": float(cfg_scale),
        "temperature": float(temperature),
        "topk": int(topk),
        "sampleRate": int(sample_rate),
        "minFrames": int(min_frames),
        "audioSamples": int(len(audio)),
        "genTimeSeconds": float(gen_time),
        "frames": int(predicted_codes_full.shape[1]),
        "tokens": int(T_text),
    }
    meta_path = os.path.splitext(output_path)[0] + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  metadata sidecar → {meta_path}")

    duration = len(audio) / sample_rate if sample_rate > 0 else 0.0
    rtf = gen_time / duration if duration > 0 else math.inf
    print(f"Wrote full fixture → {output_path}")
    print(f"  tokens={T_text}  frames={predicted_codes_full.shape[1]}  "
          f"duration={duration:.2f}s  rtf={rtf:.2f}x")

    wav_path = os.path.splitext(output_path)[0] + ".wav"
    sf.write(wav_path, audio, sample_rate)
    print(f"  reference audio → {wav_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit Magpie TTS parity fixtures for cross-impl testing.",
    )
    parser.add_argument("text", type=str, help="Text to synthesize")
    parser.add_argument("--mode", choices=["full", "tokenizer"], default="full",
                        help="'full' dumps .npz of all intermediates; "
                             "'tokenizer' dumps a small .json of token ids")
    parser.add_argument("--speaker", type=int, default=0)
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path (.npz for full, .json for tokenizer)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--topk", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--no-cfg", action="store_true")
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    args = parser.parse_args()

    if args.mode == "tokenizer":
        emit_tokenizer_fixture(
            text=args.text,
            speaker=args.speaker,
            language=args.language,
            output_path=args.output,
        )
    else:
        emit_full_fixture(
            text=args.text,
            speaker=args.speaker,
            language=args.language,
            output_path=args.output,
            temperature=args.temperature,
            topk=args.topk,
            max_steps=args.max_steps,
            seed=args.seed,
            use_cfg=not args.no_cfg,
            cfg_scale=args.cfg_scale,
        )


if __name__ == "__main__":
    main()
