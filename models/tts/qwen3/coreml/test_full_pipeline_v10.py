#!/usr/bin/env python3
"""
Full end-to-end pipeline test: LM Prefill → V10 Decode + KV Code Predictor → Audio Decoder

Tests using CoreML models for all components.
Compares generated codebooks against reference (test_codes_v9.npy).
Generates WAV audio and optionally evaluates with ASR (Whisper).
"""

import torch
import numpy as np
import coremltools as ct
import time
import os
import sys


def load_pytorch_model():
    """Load the PyTorch model for reference code predictor."""
    from qwen_tts import Qwen3TTSModel
    print("Loading Qwen3-TTS model...")
    model = Qwen3TTSModel.from_pretrained(
        "./model_0.6b", device_map="cpu", torch_dtype=torch.float32
    )
    return model


def run_code_predictor_pytorch(cp, cp_embeddings, codec_embedding, past_hidden, cb0_token):
    """Run code predictor using original PyTorch (reference)."""
    cb0_token_t = torch.tensor([[cb0_token]])
    with torch.no_grad():
        cb0_embed = codec_embedding(cb0_token_t)
        hidden = torch.cat([past_hidden, cb0_embed], dim=1)
        tokens = []
        for i in range(15):
            outputs = cp.model(inputs_embeds=hidden, use_cache=False)
            hs = outputs.last_hidden_state
            logits = cp.lm_head[i](hs[:, -1:, :])
            token = torch.argmax(logits, dim=-1)
            tokens.append(token.item())
            embed = cp_embeddings[i](token)
            hidden = torch.cat([hidden, embed], dim=1)
    return tokens


def sample_top_k(logits_np, temperature=0.9, top_k=50):
    """Sample from logits with temperature and top-k filtering."""
    logits = logits_np.copy().flatten().astype(np.float64)
    logits = logits / temperature
    if top_k > 0 and top_k < len(logits):
        top_k_idx = np.argsort(logits)[-top_k:]
        mask = np.full_like(logits, -1e9)
        mask[top_k_idx] = logits[top_k_idx]
        logits = mask
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    return int(np.random.choice(len(probs), p=probs))


def run_code_predictor_coreml(cp_prefill_ml, cp_decode_ml, cp_embeds_np, past_hidden_np, cb0_token):
    """Run code predictor using CoreML models with sampling (required for correct audio)."""
    # CP Prefill
    pf_out = cp_prefill_ml.predict({
        "past_hidden": past_hidden_np.astype(np.float32),
        "cb0_token": np.array([[cb0_token]], dtype=np.int32),
    })
    kv = pf_out["kv_cache"]
    logits = pf_out["all_logits"]

    # CB1 from logits[0] - MUST use sampling, not greedy
    cb1 = sample_top_k(logits[0], temperature=0.9, top_k=50)
    tokens = [cb1]

    # CP Decode steps
    for step in range(1, 15):
        embed = cp_embeds_np[step - 1][tokens[-1]].reshape(1, 1, 1024)
        dc_out = cp_decode_ml.predict({
            "input_embed": embed.astype(np.float32),
            "kv_cache": kv.astype(np.float32),
            "position": np.array([step + 1], dtype=np.int32),
        })
        kv = dc_out["new_kv_cache"]
        token = sample_top_k(dc_out["all_logits"][step], temperature=0.9, top_k=50)
        tokens.append(token)

    return tokens


def main():
    np.random.seed(42)

    print("=" * 60)
    print("Full Pipeline: Prefill → V10 Decode + KV CP → Audio Decoder")
    print("=" * 60)

    # Load CoreML models
    print("\n1. Loading CoreML models...")
    t0 = time.time()

    prefill_ml = ct.models.MLModel("qwen3_tts_lm_prefill_v9.mlpackage")
    decode_ml = ct.models.MLModel("qwen3_tts_lm_decode_v10.mlpackage")
    cp_prefill_ml = ct.models.MLModel("qwen3_tts_cp_prefill.mlpackage")
    cp_decode_ml = ct.models.MLModel("qwen3_tts_cp_decode.mlpackage")
    audio_decoder_ml = ct.models.MLModel("qwen3_tts_decoder_10s.mlpackage")

    cp_embeds_np = np.load("cp_embeddings.npy")  # [15, 2048, 1024]

    # Load embeddings
    tts_bos = np.load("tts_bos_embed.npy").reshape(1, 1, 1024).astype(np.float32)
    tts_pad = np.load("tts_pad_embed.npy").reshape(1, 1, 1024).astype(np.float32)
    tts_eos = np.load("tts_eos_embed.npy").reshape(1, 1, 1024).astype(np.float32)
    speaker = np.load("speaker_embedding_official.npy").reshape(1, 1024).astype(np.float32)

    print(f"   Loaded in {time.time() - t0:.2f}s")

    # Also load PyTorch model for reference code predictor comparison
    model = load_pytorch_model()
    talker = model.model.talker
    cp = talker.code_predictor
    codec_embedding = talker.model.codec_embedding
    cp_embeddings = cp.get_input_embeddings()

    # Load reference codebooks
    ref_codes = np.load("test_codes_v9.npy")  # [125, 16]
    print(f"\n   Reference codebooks: {ref_codes.shape}")

    # Text tokens for "Hello world, this is a test of the text-to-speech system."
    text_tokens = np.load("test_text_tokens.npy") if os.path.exists("test_text_tokens.npy") else None

    if text_tokens is None:
        # Use known token IDs from the test
        # These are the token IDs for the test text used in previous runs
        print("   No test_text_tokens.npy found, using default test tokens")
        # We'll use placeholder - the real test needs actual tokens
        # For now, let's test with the prefill model using known inputs
        text_tokens = np.array([9906, 1917, 11, 419, 374, 264, 1273, 315, 279, 1467, 4669, 12, 38384, 1894, 13])

    print(f"   Text tokens: {len(text_tokens)}")

    # 2. Run Prefill
    print("\n2. Running prefill...")
    t0 = time.time()

    # Build prefill inputs
    role_ids = np.array([[151644, 77091, 198]], dtype=np.int32)  # <|im_start|>assistant\n
    max_text_len = 128
    text_ids = np.zeros((1, max_text_len), dtype=np.int32)
    for i, t in enumerate(text_tokens[:max_text_len]):
        text_ids[0, i] = int(t)
    text_length = np.array([min(len(text_tokens), max_text_len)], dtype=np.int32)

    prefill_out = prefill_ml.predict({
        "role_ids": role_ids,
        "text_ids": text_ids,
        "text_length": text_length,
        "tts_bos_embed": tts_bos,
        "tts_pad_embed": tts_pad,
        "tts_eos_embed": tts_eos,
        "speaker_embed": speaker,
    })

    prefill_logits = prefill_out["logits"]
    kv_cache_raw = prefill_out["kv_cache"]
    past_hidden = prefill_out["past_hidden"]

    # CRITICAL: Trim KV cache to actual length (only valid positions)
    # Prefill pads to max_text_len+11 but only text_len+11 positions are valid
    actual_len = min(len(text_tokens), max_text_len) + 11
    kv_cache = kv_cache_raw[:, :, :, :actual_len, :]

    prefill_time = time.time() - t0
    print(f"   Prefill: {prefill_time:.2f}s")
    print(f"   Logits shape: {prefill_logits.shape}")
    print(f"   KV cache shape (trimmed): {kv_cache.shape} (from {kv_cache_raw.shape})")
    print(f"   Past hidden shape: {past_hidden.shape}")

    # Sample first CB0 token
    # Apply suppression: only allow tokens 0-2047 and EOS
    EOS_TOKEN = 2150  # codec_eos_token_id from model config
    masked_logits = prefill_logits.copy().flatten()
    for i in range(2048, len(masked_logits)):
        if i != EOS_TOKEN:
            masked_logits[i] = -1e9
    first_cb0 = int(np.argmax(masked_logits))
    print(f"   First CB0: {first_cb0} (ref: {ref_codes[0, 0]})")

    # 3. Run decode loop
    print("\n3. Running decode loop...")
    t0 = time.time()

    MAX_TOKENS = 125
    all_codebooks = []
    current_kv = kv_cache
    current_past_hidden_np = past_hidden
    current_cb0 = first_cb0
    position = actual_len  # role(3) + think(6) + text + bos + speaker = text_len + 11

    for step in range(MAX_TOKENS):
        # Step A: Run code predictor (CoreML) to get CB1-15
        cb1_15 = run_code_predictor_coreml(
            cp_prefill_ml, cp_decode_ml, cp_embeds_np,
            current_past_hidden_np, current_cb0
        )

        # Also run PyTorch code predictor for comparison (first few steps)
        if step < 3:
            past_hidden_torch = torch.from_numpy(current_past_hidden_np)
            cb1_15_ref = run_code_predictor_pytorch(
                cp, cp_embeddings, codec_embedding, past_hidden_torch, current_cb0
            )
            match = cb1_15 == cb1_15_ref
            print(f"   Step {step}: CB0={current_cb0}, CB1-3={cb1_15[:3]}, ref CB1-3={cb1_15_ref[:3]}, match={match}")

        # Build full frame
        frame = [current_cb0] + cb1_15
        all_codebooks.append(frame)

        # Step B: Run V10 LM decode
        v10_out = decode_ml.predict({
            "cb0_id": np.array([[current_cb0]], dtype=np.int32),
            "cb1_15_ids": np.array([cb1_15], dtype=np.int32),
            "trailing_text_embed": tts_pad,
            "kv_cache": current_kv.astype(np.float32),
            "position": np.array([position], dtype=np.int32),
        })

        new_logits = v10_out["logits"].flatten()
        current_kv = v10_out["new_kv_cache"]
        current_past_hidden_np = v10_out["past_hidden"]

        # Sample next CB0
        masked = new_logits.copy()
        for i in range(2048, len(masked)):
            if i != EOS_TOKEN:
                masked[i] = -1e9
        next_cb0 = int(np.argmax(masked))

        position += 1

        # Check EOS
        if next_cb0 == EOS_TOKEN:
            print(f"   EOS at step {step + 1}")
            break

        current_cb0 = next_cb0

    decode_time = time.time() - t0
    print(f"   Decode: {decode_time:.2f}s for {len(all_codebooks)} frames")
    print(f"   Speed: {decode_time / len(all_codebooks) * 1000:.0f}ms/frame")

    # Compare with reference
    all_codes_np = np.array(all_codebooks)
    print(f"\n   Generated codebooks shape: {all_codes_np.shape}")
    if ref_codes.shape[0] == all_codes_np.shape[0]:
        cb0_match = np.sum(all_codes_np[:, 0] == ref_codes[:, 0])
        total_match = np.sum(all_codes_np == ref_codes)
        print(f"   CB0 match: {cb0_match}/{len(ref_codes)}")
        print(f"   Total match: {total_match}/{ref_codes.size}")
    else:
        print(f"   Different lengths: generated={len(all_codebooks)}, ref={len(ref_codes)}")
        min_len = min(len(all_codebooks), len(ref_codes))
        if min_len > 0:
            cb0_match = np.sum(all_codes_np[:min_len, 0] == ref_codes[:min_len, 0])
            print(f"   CB0 match (first {min_len}): {cb0_match}/{min_len}")

    # 4. Run audio decoder
    print("\n4. Running audio decoder...")
    t0 = time.time()

    # Build codes tensor [1, 16, 125]
    fixed_len = 125
    codes = np.zeros((1, 16, fixed_len), dtype=np.int32)
    for t in range(min(len(all_codebooks), fixed_len)):
        frame = all_codebooks[t]
        for cb in range(min(len(frame), 16)):
            codes[0, cb, t] = frame[cb]

    audio_out = audio_decoder_ml.predict({"codes": codes})
    audio_samples = audio_out["audio"].flatten()

    decoder_time = time.time() - t0
    print(f"   Audio decoder: {decoder_time:.2f}s")
    print(f"   Audio samples: {len(audio_samples)}")
    audio_rms = np.sqrt(np.mean(audio_samples ** 2))
    print(f"   Audio RMS: {audio_rms:.4f}")
    print(f"   Audio range: [{audio_samples.min():.4f}, {audio_samples.max():.4f}]")

    # Also decode reference codes for comparison
    ref_codes_tensor = np.zeros((1, 16, fixed_len), dtype=np.int32)
    for t in range(min(len(ref_codes), fixed_len)):
        for cb in range(16):
            ref_codes_tensor[0, cb, t] = int(ref_codes[t, cb])
    ref_audio_out = audio_decoder_ml.predict({"codes": ref_codes_tensor})
    ref_audio = ref_audio_out["audio"].flatten()
    ref_rms = np.sqrt(np.mean(ref_audio ** 2))
    print(f"   Reference RMS: {ref_rms:.4f}")

    # 5. Save WAV
    print("\n5. Saving WAV files...")
    try:
        import soundfile as sf
        sf.write("test_v10_pipeline.wav", audio_samples, 24000)
        sf.write("test_v10_reference.wav", ref_audio, 24000)
        print("   Saved test_v10_pipeline.wav and test_v10_reference.wav")
    except ImportError:
        # Manual WAV writing
        import struct
        def write_wav(filename, samples, sr=24000):
            n = len(samples)
            with open(filename, "wb") as f:
                f.write(b"RIFF")
                f.write(struct.pack("<I", 36 + n * 2))
                f.write(b"WAVE")
                f.write(b"fmt ")
                f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
                f.write(b"data")
                f.write(struct.pack("<I", n * 2))
                for s in samples:
                    val = max(-1.0, min(1.0, s / 32768.0 if abs(s) > 1 else s))
                    f.write(struct.pack("<h", int(val * 32767)))
        write_wav("test_v10_pipeline.wav", audio_samples)
        write_wav("test_v10_reference.wav", ref_audio)
        print("   Saved WAV files (manual)")

    # 6. ASR evaluation
    print("\n6. ASR evaluation...")
    try:
        import whisper
        whisper_model = whisper.load_model("base")

        result = whisper_model.transcribe("test_v10_pipeline.wav")
        gen_text = result["text"].strip()
        print(f"   Generated: '{gen_text}'")

        ref_result = whisper_model.transcribe("test_v10_reference.wav")
        ref_text = ref_result["text"].strip()
        print(f"   Reference: '{ref_text}'")

        # Simple WER
        expected = "Hello world, this is a test of the text to speech system."
        gen_words = gen_text.lower().split()
        expected_words = expected.lower().split()
        if len(expected_words) > 0:
            errors = sum(1 for a, b in zip(gen_words, expected_words) if a != b)
            errors += abs(len(gen_words) - len(expected_words))
            wer = errors / len(expected_words)
            print(f"   WER: {wer:.1%}")
    except ImportError:
        print("   Whisper not installed, skipping ASR evaluation")
        print("   Install with: pip install openai-whisper")

    # Summary
    total_time = prefill_time + decode_time + decoder_time
    audio_duration = len(audio_samples) / 24000
    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Prefill:       {prefill_time:.2f}s")
    print(f"  Decode:        {decode_time:.2f}s ({len(all_codebooks)} frames)")
    print(f"  Audio decoder: {decoder_time:.2f}s")
    print(f"  Total:         {total_time:.2f}s")
    print(f"  Audio:         {audio_duration:.2f}s")
    print(f"  RTFx:          {audio_duration / total_time:.2f}x")
    print(f"  Audio RMS:     {np.sqrt(np.mean(audio_samples ** 2)):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
