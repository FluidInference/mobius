#!/usr/bin/env python3
"""
Diagnostic test: Compare CoreML V10 decode vs PyTorch V10 decode step by step.
Also test with temperature+top_k sampling vs greedy.
"""

import torch
import numpy as np
import coremltools as ct
import time
import os
import sys


def load_pytorch_model():
    from qwen_tts import Qwen3TTSModel
    print("Loading Qwen3-TTS model...")
    model = Qwen3TTSModel.from_pretrained(
        "./model_0.6b", device_map="cpu", torch_dtype=torch.float32
    )
    return model


def run_code_predictor_coreml(cp_prefill_ml, cp_decode_ml, cp_embeds_np, past_hidden_np, cb0_token):
    pf_out = cp_prefill_ml.predict({
        "past_hidden": past_hidden_np.astype(np.float32),
        "cb0_token": np.array([[cb0_token]], dtype=np.int32),
    })
    kv = pf_out["kv_cache"]
    logits = pf_out["all_logits"]
    cb1 = int(np.argmax(logits[0], axis=-1).item())
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


def sample_top_k(logits_np, temperature=0.9, top_k=50, suppress_above=2048, eos_token=2150):
    """Sample from logits with temperature and top-k."""
    logits = logits_np.copy().flatten().astype(np.float64)
    # Suppress invalid tokens
    for i in range(suppress_above, len(logits)):
        if i != eos_token:
            logits[i] = -1e9
    # Apply temperature
    logits = logits / temperature
    # Top-k filtering
    top_k_idx = np.argsort(logits)[-top_k:]
    mask = np.full_like(logits, -1e9)
    mask[top_k_idx] = logits[top_k_idx]
    # Softmax
    exp_logits = np.exp(mask - np.max(mask))
    probs = exp_logits / np.sum(exp_logits)
    # Sample
    return int(np.random.choice(len(probs), p=probs))


def greedy_sample(logits_np, suppress_above=2048, eos_token=2150):
    """Greedy argmax with suppression."""
    logits = logits_np.copy().flatten()
    for i in range(suppress_above, len(logits)):
        if i != eos_token:
            logits[i] = -1e9
    return int(np.argmax(logits))


def main():
    print("=" * 60)
    print("V10 Diagnostic: CoreML vs PyTorch + Greedy vs Sampling")
    print("=" * 60)

    # Load CoreML models
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

    # Load PyTorch model
    model = load_pytorch_model()
    talker = model.model.talker

    # Create PyTorch V10 wrapper for comparison
    from convert_lm_decode_v10 import TracableDecodeV10
    pytorch_v10 = TracableDecodeV10(talker)
    pytorch_v10.eval()

    # Load text tokens
    text_tokens = np.load("test_text_tokens.npy") if os.path.exists("test_text_tokens.npy") else None
    if text_tokens is None:
        text_tokens = np.array([9707, 1879, 11, 419, 374, 264, 1273, 315, 279, 1467, 4686, 1331, 39586, 1849, 13])
    print(f"\n   Text tokens ({len(text_tokens)}): {text_tokens.tolist()}")

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
    print(f"   KV cache trimmed: {kv_cache.shape}")

    EOS_TOKEN = 2150
    first_cb0 = greedy_sample(prefill_logits)
    print(f"   First CB0: {first_cb0}")

    # =========================================================
    # Test A: CoreML V10 greedy - first 20 steps
    # =========================================================
    print("\n" + "=" * 60)
    print("Test A: CoreML V10 Greedy - First 20 CB0 tokens")
    print("=" * 60)

    current_kv_cml = kv_cache.copy()
    current_ph_cml = past_hidden.copy()
    current_cb0 = first_cb0
    position = actual_len
    cml_cb0_tokens = [first_cb0]

    for step in range(20):
        cb1_15 = run_code_predictor_coreml(
            cp_prefill_ml, cp_decode_ml, cp_embeds_np,
            current_ph_cml, current_cb0
        )
        v10_out = decode_ml.predict({
            "cb0_id": np.array([[current_cb0]], dtype=np.int32),
            "cb1_15_ids": np.array([cb1_15], dtype=np.int32),
            "trailing_text_embed": tts_pad,
            "kv_cache": current_kv_cml.astype(np.float32),
            "position": np.array([position], dtype=np.int32),
        })
        new_logits = v10_out["logits"].flatten()
        current_kv_cml = v10_out["new_kv_cache"]
        current_ph_cml = v10_out["past_hidden"]
        next_cb0 = greedy_sample(new_logits)
        cml_cb0_tokens.append(next_cb0)
        position += 1
        if next_cb0 == EOS_TOKEN:
            print(f"   EOS at step {step + 1}")
            break
        current_cb0 = next_cb0

    print(f"   CB0 tokens: {cml_cb0_tokens}")

    # =========================================================
    # Test B: PyTorch V10 greedy - first 20 steps
    # =========================================================
    print("\n" + "=" * 60)
    print("Test B: PyTorch V10 Greedy - First 20 CB0 tokens")
    print("=" * 60)

    kv_torch = torch.from_numpy(kv_cache).float()
    ph_torch = torch.from_numpy(past_hidden).float()
    current_cb0 = first_cb0
    position = actual_len
    pt_cb0_tokens = [first_cb0]
    tts_pad_torch = torch.from_numpy(tts_pad).float()

    for step in range(20):
        cb1_15 = run_code_predictor_coreml(
            cp_prefill_ml, cp_decode_ml, cp_embeds_np,
            ph_torch.numpy(), current_cb0
        )
        with torch.no_grad():
            logits_pt, kv_torch, ph_torch = pytorch_v10(
                torch.tensor([[current_cb0]]),
                torch.tensor([cb1_15]),
                tts_pad_torch,
                kv_torch,
                torch.tensor([position]),
            )
        next_cb0_pt = greedy_sample(logits_pt.numpy())
        pt_cb0_tokens.append(next_cb0_pt)
        position += 1
        if next_cb0_pt == EOS_TOKEN:
            print(f"   EOS at step {step + 1}")
            break
        current_cb0 = next_cb0_pt

    print(f"   CB0 tokens: {pt_cb0_tokens}")

    # Compare
    print("\n   Step-by-step comparison (CML vs PT):")
    for i in range(min(len(cml_cb0_tokens), len(pt_cb0_tokens))):
        match = "✓" if cml_cb0_tokens[i] == pt_cb0_tokens[i] else "✗"
        print(f"   Step {i}: CML={cml_cb0_tokens[i]:5d}  PT={pt_cb0_tokens[i]:5d}  {match}")

    # =========================================================
    # Test C: CoreML V10 with sampling - full 125 frames
    # =========================================================
    print("\n" + "=" * 60)
    print("Test C: CoreML V10 with Sampling (temp=0.9, top_k=50)")
    print("=" * 60)

    np.random.seed(42)  # For reproducibility
    current_kv = kv_cache.copy()
    current_ph = past_hidden.copy()
    current_cb0 = first_cb0
    position = actual_len
    all_codebooks = []

    t0 = time.time()
    for step in range(125):
        cb1_15 = run_code_predictor_coreml(
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

        next_cb0 = sample_top_k(new_logits, temperature=0.9, top_k=50)
        position += 1

        if next_cb0 == EOS_TOKEN:
            print(f"   EOS at step {step + 1}")
            break
        current_cb0 = next_cb0

    decode_time = time.time() - t0
    print(f"   Decode: {decode_time:.1f}s for {len(all_codebooks)} frames")
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

    # Save
    try:
        import soundfile as sf
        sf.write("test_v10_sampled.wav", audio, 24000)
        print("   Saved: test_v10_sampled.wav")
    except ImportError:
        pass

    # ASR
    try:
        import whisper
        whisper_model = whisper.load_model("base")
        result = whisper_model.transcribe("test_v10_sampled.wav")
        print(f"   ASR: '{result['text'].strip()}'")
    except ImportError:
        print("   Whisper not available")

    # =========================================================
    # Test D: PyTorch V10 greedy - full 125 frames
    # =========================================================
    print("\n" + "=" * 60)
    print("Test D: PyTorch V10 Greedy - Full 125 frames")
    print("=" * 60)

    kv_torch = torch.from_numpy(kv_cache).float()
    ph_torch = torch.from_numpy(past_hidden).float()
    current_cb0 = first_cb0
    position = actual_len
    all_codebooks_pt = []

    t0 = time.time()
    for step in range(125):
        cb1_15 = run_code_predictor_coreml(
            cp_prefill_ml, cp_decode_ml, cp_embeds_np,
            ph_torch.numpy(), current_cb0
        )
        frame = [current_cb0] + cb1_15
        all_codebooks_pt.append(frame)

        with torch.no_grad():
            logits_pt, kv_torch, ph_torch = pytorch_v10(
                torch.tensor([[current_cb0]]),
                torch.tensor([cb1_15]),
                tts_pad_torch,
                kv_torch,
                torch.tensor([position]),
            )
        next_cb0 = greedy_sample(logits_pt.numpy())
        position += 1

        if next_cb0 == EOS_TOKEN:
            print(f"   EOS at step {step + 1}")
            break
        current_cb0 = next_cb0

    decode_time = time.time() - t0
    print(f"   Decode: {decode_time:.1f}s for {len(all_codebooks_pt)} frames")
    print(f"   CB0 first 20: {[f[0] for f in all_codebooks_pt[:20]]}")

    # Decode to audio
    codes_pt = np.zeros((1, 16, fixed_len), dtype=np.int32)
    for t in range(min(len(all_codebooks_pt), fixed_len)):
        frame = all_codebooks_pt[t]
        for cb in range(min(len(frame), 16)):
            codes_pt[0, cb, t] = frame[cb]

    audio_out_pt = audio_decoder_ml.predict({"codes": codes_pt})
    audio_pt = audio_out_pt["audio"].flatten()
    rms_pt = np.sqrt(np.mean(audio_pt ** 2))
    print(f"   Audio RMS: {rms_pt:.4f}")
    print(f"   Audio range: [{audio_pt.min():.4f}, {audio_pt.max():.4f}]")

    try:
        import soundfile as sf
        sf.write("test_v10_pytorch_greedy.wav", audio_pt, 24000)
        print("   Saved: test_v10_pytorch_greedy.wav")
    except ImportError:
        pass

    try:
        import whisper
        result = whisper_model.transcribe("test_v10_pytorch_greedy.wav")
        print(f"   ASR: '{result['text'].strip()}'")
    except ImportError:
        pass

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
