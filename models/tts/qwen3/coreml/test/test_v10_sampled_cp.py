#!/usr/bin/env python3
"""
Test V10 pipeline with sampled CB1-15 (temperature=0.9, top_k=50).
The code predictor REQUIRES sampling - greedy produces silent audio.
"""

import torch
import numpy as np
import coremltools as ct
import time
import os


def sample_top_k(logits_np, temperature=0.9, top_k=50):
    """Sample from logits with temperature and top-k filtering."""
    logits = logits_np.copy().flatten().astype(np.float64)
    logits = logits / temperature
    # Top-k: zero out everything below top-k
    if top_k > 0 and top_k < len(logits):
        top_k_idx = np.argsort(logits)[-top_k:]
        mask = np.full_like(logits, -1e9)
        mask[top_k_idx] = logits[top_k_idx]
        logits = mask
    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    return int(np.random.choice(len(probs), p=probs))


def sample_cb0(logits_np, temperature=0.9, top_k=50, eos_token=2150):
    """Sample CB0 with suppression of non-codec tokens."""
    logits = logits_np.copy().flatten()
    for i in range(2048, len(logits)):
        if i != eos_token:
            logits[i] = -1e9
    return sample_top_k(logits, temperature=temperature, top_k=top_k)


def greedy_cb0(logits_np, eos_token=2150):
    """Greedy CB0 with suppression."""
    logits = logits_np.copy().flatten()
    for i in range(2048, len(logits)):
        if i != eos_token:
            logits[i] = -1e9
    return int(np.argmax(logits))


def run_code_predictor_sampled(cp_prefill_ml, cp_decode_ml, cp_embeds_np, past_hidden_np, cb0_token,
                                temperature=0.9, top_k=50):
    """Run code predictor with sampling (like the original model)."""
    pf_out = cp_prefill_ml.predict({
        "past_hidden": past_hidden_np.astype(np.float32),
        "cb0_token": np.array([[cb0_token]], dtype=np.int32),
    })
    kv = pf_out["kv_cache"]
    all_logits = pf_out["all_logits"]  # [15, 1, 2048]

    # CB1: sample from logits[0]
    cb1 = sample_top_k(all_logits[0], temperature=temperature, top_k=top_k)
    tokens = [cb1]

    # CB2-CB15: decode steps
    for step in range(1, 15):
        embed = cp_embeds_np[step - 1][tokens[-1]].reshape(1, 1, 1024)
        dc_out = cp_decode_ml.predict({
            "input_embed": embed.astype(np.float32),
            "kv_cache": kv.astype(np.float32),
            "position": np.array([step + 1], dtype=np.int32),
        })
        kv = dc_out["new_kv_cache"]
        token = sample_top_k(dc_out["all_logits"][step], temperature=temperature, top_k=top_k)
        tokens.append(token)

    return tokens


def run_code_predictor_greedy(cp_prefill_ml, cp_decode_ml, cp_embeds_np, past_hidden_np, cb0_token):
    """Run code predictor with greedy decoding."""
    pf_out = cp_prefill_ml.predict({
        "past_hidden": past_hidden_np.astype(np.float32),
        "cb0_token": np.array([[cb0_token]], dtype=np.int32),
    })
    kv = pf_out["kv_cache"]
    all_logits = pf_out["all_logits"]

    cb1 = int(np.argmax(all_logits[0], axis=-1).item())
    tokens = [cb1]

    for step in range(1, 15):
        embed = cp_embeds_np[step - 1][tokens[-1]].reshape(1, 1, 1024)
        dc_out = cp_decode_ml.predict({
            "input_embed": embed.astype(np.float32),
            "kv_cache": kv.astype(np.float32),
            "position": np.array([step + 1], dtype=np.int32),
        })
        kv = dc_out["new_kv_cache"]
        token = int(np.argmax(dc_out["all_logits"][step], axis=-1).item())
        tokens.append(token)

    return tokens


def run_pipeline(name, decode_ml, cp_prefill_ml, cp_decode_ml, cp_embeds_np,
                 audio_decoder_ml, prefill_logits, kv_cache, past_hidden, tts_pad,
                 actual_len, sample_cb1_15=True, sample_cb0_flag=False, seed=42):
    """Run full pipeline with specified sampling config."""
    np.random.seed(seed)
    EOS_TOKEN = 2150

    # First CB0
    if sample_cb0_flag:
        first_cb0 = sample_cb0(prefill_logits)
    else:
        first_cb0 = greedy_cb0(prefill_logits)

    current_kv = kv_cache.copy()
    current_ph = past_hidden.copy()
    current_cb0 = first_cb0
    position = actual_len
    all_codebooks = []

    t0 = time.time()
    for step in range(125):
        if sample_cb1_15:
            cb1_15 = run_code_predictor_sampled(
                cp_prefill_ml, cp_decode_ml, cp_embeds_np,
                current_ph, current_cb0
            )
        else:
            cb1_15 = run_code_predictor_greedy(
                cp_prefill_ml, cp_decode_ml, cp_embeds_np,
                current_ph, current_cb0
            )

        frame = [current_cb0] + cb1_15
        all_codebooks.append(frame)

        v10_out = decode_ml.predict({
            "cb0_id": np.array([[current_cb0]], dtype=np.int32),
            "cb1_15_ids": np.array([cb1_15], dtype=np.int32),
            "trailing_text_embed": tts_pad,
            "kv_cache": current_kv.astype(np.float32),
            "position": np.array([position], dtype=np.int32),
        })
        new_logits = v10_out["logits"].flatten()
        current_kv = v10_out["new_kv_cache"]
        current_ph = v10_out["past_hidden"]

        if sample_cb0_flag:
            next_cb0 = sample_cb0(new_logits)
        else:
            next_cb0 = greedy_cb0(new_logits)
        position += 1

        if next_cb0 == EOS_TOKEN:
            print(f"   EOS at step {step + 1}")
            break
        current_cb0 = next_cb0

    decode_time = time.time() - t0
    print(f"   Decode: {decode_time:.1f}s for {len(all_codebooks)} frames ({decode_time/len(all_codebooks)*1000:.0f}ms/frame)")
    print(f"   CB0 first 20: {[f[0] for f in all_codebooks[:20]]}")

    # Decode to audio
    fixed_len = 125
    codes = np.zeros((1, 16, fixed_len), dtype=np.int32)
    for t in range(min(len(all_codebooks), fixed_len)):
        frame = all_codebooks[t]
        for cb in range(min(len(frame), 16)):
            codes[0, cb, t] = frame[cb]

    audio_out = audio_decoder_ml.predict({"codes": codes})
    audio = audio_out["audio"].flatten()
    rms = np.sqrt(np.mean(audio ** 2))
    print(f"   Audio RMS: {rms:.4f}")
    print(f"   Audio range: [{audio.min():.4f}, {audio.max():.4f}]")

    # Save WAV
    wav_name = f"test_v10_{name}.wav"
    try:
        import soundfile as sf
        sf.write(wav_name, audio, 24000)
        print(f"   Saved: {wav_name}")
    except ImportError:
        pass

    return wav_name, audio, all_codebooks


def main():
    print("=" * 60)
    print("V10 Pipeline: Sampled vs Greedy Code Predictor")
    print("=" * 60)

    # Load models
    print("\n1. Loading CoreML models...")
    t0 = time.time()
    prefill_ml = ct.models.MLModel("qwen3_tts_lm_prefill_v9.mlpackage")
    decode_ml = ct.models.MLModel("qwen3_tts_lm_decode_v10.mlpackage")
    cp_prefill_ml = ct.models.MLModel("qwen3_tts_cp_prefill.mlpackage")
    cp_decode_ml = ct.models.MLModel("qwen3_tts_cp_decode.mlpackage")
    audio_decoder_ml = ct.models.MLModel("qwen3_tts_decoder_10s.mlpackage")
    cp_embeds_np = np.load("cp_embeddings.npy")
    tts_bos = np.load("tts_bos_embed.npy").reshape(1, 1, 1024).astype(np.float32)
    tts_pad = np.load("tts_pad_embed.npy").reshape(1, 1, 1024).astype(np.float32)
    tts_eos = np.load("tts_eos_embed.npy").reshape(1, 1, 1024).astype(np.float32)
    speaker = np.load("speaker_embedding_official.npy").reshape(1, 1024).astype(np.float32)
    print(f"   Loaded in {time.time() - t0:.1f}s")

    # Load text tokens
    text_tokens = np.load("test_text_tokens.npy") if os.path.exists("test_text_tokens.npy") else None
    if text_tokens is None:
        text_tokens = np.array([9707, 1879, 11, 419, 374, 264, 1273, 315, 279, 1467, 4686, 1331, 39586, 1849, 13])
    print(f"   Text tokens: {len(text_tokens)}")

    # Run prefill
    print("\n2. Running CoreML prefill...")
    max_text_len = 128
    role_ids = np.array([[151644, 77091, 198]], dtype=np.int32)
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
    actual_len = min(len(text_tokens), max_text_len) + 11
    kv_cache = kv_cache_raw[:, :, :, :actual_len, :]
    print(f"   KV cache: {kv_cache.shape}")
    print(f"   First CB0 (greedy): {greedy_cb0(prefill_logits)}")

    # Load reference
    ref_codes = np.load("test_codes_v9.npy")

    # =========================================================
    # Test A: Greedy CB0 + Sampled CB1-15 (MAIN TEST)
    # =========================================================
    print("\n" + "=" * 60)
    print("Test A: Greedy CB0 + Sampled CB1-15 (temp=0.9, top_k=50)")
    print("=" * 60)
    wav_a, audio_a, codes_a = run_pipeline(
        "greedy_cb0_sampled_cp", decode_ml, cp_prefill_ml, cp_decode_ml, cp_embeds_np,
        audio_decoder_ml, prefill_logits, kv_cache, past_hidden, tts_pad, actual_len,
        sample_cb1_15=True, sample_cb0_flag=False, seed=42
    )

    # =========================================================
    # Test B: Sampled CB0 + Sampled CB1-15 (full sampling)
    # =========================================================
    print("\n" + "=" * 60)
    print("Test B: Sampled CB0 + Sampled CB1-15 (both temp=0.9, top_k=50)")
    print("=" * 60)
    wav_b, audio_b, codes_b = run_pipeline(
        "sampled_both", decode_ml, cp_prefill_ml, cp_decode_ml, cp_embeds_np,
        audio_decoder_ml, prefill_logits, kv_cache, past_hidden, tts_pad, actual_len,
        sample_cb1_15=True, sample_cb0_flag=True, seed=42
    )

    # =========================================================
    # Test C: Greedy CB0 + Greedy CB1-15 (baseline - expected to fail)
    # =========================================================
    print("\n" + "=" * 60)
    print("Test C: Greedy CB0 + Greedy CB1-15 (baseline)")
    print("=" * 60)
    wav_c, audio_c, codes_c = run_pipeline(
        "greedy_both", decode_ml, cp_prefill_ml, cp_decode_ml, cp_embeds_np,
        audio_decoder_ml, prefill_logits, kv_cache, past_hidden, tts_pad, actual_len,
        sample_cb1_15=False, sample_cb0_flag=False, seed=42
    )

    # =========================================================
    # ASR Evaluation
    # =========================================================
    print("\n" + "=" * 60)
    print("ASR Evaluation (Whisper)")
    print("=" * 60)
    try:
        import whisper
        whisper_model = whisper.load_model("base")

        for label, wav in [("Test A (greedy CB0 + sampled CP)", wav_a),
                           ("Test B (sampled both)", wav_b),
                           ("Test C (greedy both)", wav_c)]:
            result = whisper_model.transcribe(wav)
            text = result["text"].strip()
            print(f"   {label}: '{text}'")

        # Also reference
        ref_codes_tensor = np.zeros((1, 16, 125), dtype=np.int32)
        for t in range(min(len(ref_codes), 125)):
            for cb in range(16):
                ref_codes_tensor[0, cb, t] = int(ref_codes[t, cb])
        ref_audio_out = audio_decoder_ml.predict({"codes": ref_codes_tensor})
        ref_audio = ref_audio_out["audio"].flatten()
        try:
            import soundfile as sf
            sf.write("test_v10_reference_check.wav", ref_audio, 24000)
            result = whisper_model.transcribe("test_v10_reference_check.wav")
            print(f"   Reference codes: '{result['text'].strip()}'")
        except ImportError:
            pass

    except ImportError:
        print("   Whisper not available, skipping ASR")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    rms_a = np.sqrt(np.mean(audio_a ** 2))
    rms_b = np.sqrt(np.mean(audio_b ** 2))
    rms_c = np.sqrt(np.mean(audio_c ** 2))
    print(f"  Test A (greedy CB0 + sampled CP): RMS={rms_a:.4f}")
    print(f"  Test B (sampled both):            RMS={rms_b:.4f}")
    print(f"  Test C (greedy both):             RMS={rms_c:.4f}")


if __name__ == "__main__":
    main()
