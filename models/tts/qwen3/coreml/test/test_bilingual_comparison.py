#!/usr/bin/env python3
"""
Bilingual (English + Chinese) comparison test:
  Phase 1: Generate PyTorch reference audio for both languages
  Phase 2: Generate CoreML (Python) audio for both languages
  Phase 3: Spectral similarity comparison + ASR evaluation

Saves token IDs for Swift pipeline testing.
"""

import torch
import numpy as np
import coremltools as ct
import time
import os
import struct
import warnings
warnings.filterwarnings('ignore')

SAMPLE_RATE = 24000
MAX_CODEC_TOKENS = 125

# ─── Test sentences ────────────────────────────────────────────────────
ENGLISH_TEXT = "Hello world, this is a test of the text to speech system."
CHINESE_TEXT = "你好世界，这是一个文字转语音系统的测试。"

OUTPUT_DIR = "bilingual_test_outputs"

def write_wav(filename, samples, sr=24000):
    """Write float32 samples to 16-bit PCM WAV."""
    samples = np.clip(samples, -1.0, 1.0)
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
            f.write(struct.pack("<h", int(s * 32767)))


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


def compute_mel_spectrogram(audio, sr=24000, n_fft=1024, hop_length=256, n_mels=80):
    """Compute log mel spectrogram using numpy (no librosa dependency)."""
    # STFT
    window = np.hanning(n_fft)
    frames = []
    for i in range(0, len(audio) - n_fft, hop_length):
        frame = audio[i:i + n_fft] * window
        spectrum = np.fft.rfft(frame)
        frames.append(np.abs(spectrum) ** 2)
    power_spec = np.array(frames).T  # [freq_bins, time]

    # Mel filterbank
    fmin, fmax = 0.0, sr / 2.0
    mel_min = 2595 * np.log10(1 + fmin / 700)
    mel_max = 2595 * np.log10(1 + fmax / 700)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_left, f_center, f_right = bins[m - 1], bins[m], bins[m + 1]
        for k in range(f_left, f_center):
            if f_center > f_left:
                fbank[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right > f_center:
                fbank[m - 1, k] = (f_right - k) / (f_right - f_center)

    mel_spec = fbank @ power_spec
    log_mel = np.log(mel_spec + 1e-10)
    return log_mel


def spectral_similarity(audio_a, audio_b, sr=24000):
    """Compute spectral similarity metrics between two audio signals."""
    # Pad shorter to same length
    max_len = max(len(audio_a), len(audio_b))
    a = np.pad(audio_a, (0, max_len - len(audio_a)))
    b = np.pad(audio_b, (0, max_len - len(audio_b)))

    mel_a = compute_mel_spectrogram(a, sr)
    mel_b = compute_mel_spectrogram(b, sr)

    # Align time dimensions
    min_t = min(mel_a.shape[1], mel_b.shape[1])
    mel_a = mel_a[:, :min_t]
    mel_b = mel_b[:, :min_t]

    # 1. Frame-wise cosine similarity (averaged over time)
    cos_sims = []
    for t in range(min_t):
        a_frame = mel_a[:, t]
        b_frame = mel_b[:, t]
        norm_a = np.linalg.norm(a_frame)
        norm_b = np.linalg.norm(b_frame)
        if norm_a > 0 and norm_b > 0:
            cos_sims.append(np.dot(a_frame, b_frame) / (norm_a * norm_b))
    avg_cosine = np.mean(cos_sims) if cos_sims else 0.0

    # 2. Mel Cepstral Distortion (MCD) - lower is better
    # Approximate: use DCT of log mel as MFCC
    from numpy.fft import fft as np_fft
    n_mfcc = 13
    mcds = []
    for t in range(min_t):
        # Poor man's DCT via real FFT (approximation)
        mfcc_a = np.fft.rfft(mel_a[:, t])[:n_mfcc].real
        mfcc_b = np.fft.rfft(mel_b[:, t])[:n_mfcc].real
        diff = mfcc_a - mfcc_b
        mcd = np.sqrt(2 * np.sum(diff[1:] ** 2))  # skip c0
        mcds.append(mcd)
    avg_mcd = np.mean(mcds) if mcds else float('inf')

    # 3. Overall correlation
    flat_a = mel_a.flatten()
    flat_b = mel_b.flatten()
    if np.std(flat_a) > 0 and np.std(flat_b) > 0:
        correlation = np.corrcoef(flat_a, flat_b)[0, 1]
    else:
        correlation = 0.0

    # 4. RMS ratio
    rms_a = np.sqrt(np.mean(audio_a ** 2))
    rms_b = np.sqrt(np.mean(audio_b ** 2))
    rms_ratio = min(rms_a, rms_b) / max(rms_a, rms_b) if max(rms_a, rms_b) > 0 else 0.0

    return {
        'cosine_similarity': avg_cosine,
        'mel_cepstral_distortion': avg_mcd,
        'correlation': correlation,
        'rms_ratio': rms_ratio,
        'rms_a': rms_a,
        'rms_b': rms_b,
    }


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: PyTorch reference
# ═══════════════════════════════════════════════════════════════════════

def generate_pytorch_reference(tts_model, tokenizer_model, text, language, speaker_embed_np, label):
    """Generate reference audio using official PyTorch pipeline."""
    print(f"\n  [{label}] Text: '{text}'")

    input_text = tts_model._build_assistant_text(text)
    full_input_ids = tts_model._tokenize_texts([input_text])[0]
    text_token_ids = full_input_ids.tolist()
    print(f"  [{label}] Full input tokens: {len(text_token_ids)}")

    # Also get just the text tokens (without role prefix etc)
    processor = tts_model.processor
    inputs = processor(text=text, return_tensors="pt")
    pure_text_ids = inputs.input_ids[0].tolist()
    print(f"  [{label}] Pure text tokens ({len(pure_text_ids)}): {pure_text_ids}")

    voice_clone_prompt = {
        'ref_spk_embedding': [torch.from_numpy(speaker_embed_np.squeeze(0).copy())],
        'x_vector_only_mode': [True],
        'icl_mode': [False],
        'ref_code': None,
    }

    t0 = time.time()
    with torch.no_grad():
        result = tts_model.model.generate(
            input_ids=[full_input_ids],
            languages=[language],
            voice_clone_prompt=voice_clone_prompt,
            non_streaming_mode=True,
            max_new_tokens=MAX_CODEC_TOKENS,
            do_sample=True,       # model default: sampling required for Chinese
            temperature=0.9,
            top_k=50,
            subtalker_dosample=True,  # sampled CB1-15 (required!)
            subtalker_temperature=0.9,
            subtalker_top_k=50,
        )

    gen_time = time.time() - t0
    codes = result[0][0]  # [num_tokens, 16]
    print(f"  [{label}] Generated {codes.shape[0]} frames in {gen_time:.2f}s")

    # Decode to audio
    with torch.no_grad():
        codes_for_decoder = codes.T.unsqueeze(0)  # [1, 16, num_tokens]
        audio = tokenizer_model.model.decoder(codes_for_decoder)
    audio_np = audio[0, 0].numpy()

    rms = np.sqrt(np.mean(audio_np ** 2))
    print(f"  [{label}] Audio: {len(audio_np)} samples, {len(audio_np)/SAMPLE_RATE:.2f}s, RMS={rms:.4f}")

    return audio_np, codes.numpy(), pure_text_ids


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: CoreML pipeline
# ═══════════════════════════════════════════════════════════════════════

def run_code_predictor_coreml(cp_prefill_ml, cp_decode_ml, cp_embeds_np, past_hidden_np, cb0_token):
    """Run code predictor using CoreML with sampling."""
    pf_out = cp_prefill_ml.predict({
        "past_hidden": past_hidden_np.astype(np.float32),
        "cb0_token": np.array([[cb0_token]], dtype=np.int32),
    })
    kv = pf_out["kv_cache"]
    logits = pf_out["all_logits"]

    cb1 = sample_top_k(logits[0], temperature=0.9, top_k=50)
    tokens = [cb1]

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


def generate_coreml_audio(text_tokens, label, models, embeddings):
    """Generate audio using CoreML V10 pipeline."""
    prefill_ml, decode_ml, cp_prefill_ml, cp_decode_ml, audio_decoder_ml = models
    tts_bos, tts_pad, tts_eos, speaker, cp_embeds_np = embeddings

    print(f"\n  [{label}] Text tokens ({len(text_tokens)}): {text_tokens}")

    # Prefill
    role_ids = np.array([[151644, 77091, 198]], dtype=np.int32)
    max_text_len = 128
    text_ids = np.zeros((1, max_text_len), dtype=np.int32)
    for i, t in enumerate(text_tokens[:max_text_len]):
        text_ids[0, i] = int(t)
    text_length = np.array([min(len(text_tokens), max_text_len)], dtype=np.int32)

    t0 = time.time()
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
    prefill_time = time.time() - t0

    # First CB0
    EOS_TOKEN = 2150  # codec_eos_token_id from model config
    masked_logits = prefill_logits.copy().flatten()
    for i in range(2048, len(masked_logits)):
        if i != EOS_TOKEN:
            masked_logits[i] = -1e9
    first_cb0 = int(np.argmax(masked_logits))
    print(f"  [{label}] Prefill: {prefill_time:.2f}s, first CB0={first_cb0}")

    # Decode loop
    t0 = time.time()
    all_codebooks = []
    current_kv = kv_cache
    current_past_hidden = past_hidden
    current_cb0 = first_cb0
    position = actual_len

    for step in range(MAX_CODEC_TOKENS):
        cb1_15 = run_code_predictor_coreml(
            cp_prefill_ml, cp_decode_ml, cp_embeds_np,
            current_past_hidden, current_cb0
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
        current_past_hidden = v10_out["past_hidden"]

        masked = new_logits.copy()
        for i in range(2048, len(masked)):
            if i != EOS_TOKEN:
                masked[i] = -1e9
        next_cb0 = int(np.argmax(masked))
        position += 1

        if next_cb0 == EOS_TOKEN:
            print(f"  [{label}] EOS at step {step + 1}")
            break
        current_cb0 = next_cb0

    decode_time = time.time() - t0
    print(f"  [{label}] Decode: {decode_time:.2f}s for {len(all_codebooks)} frames")

    # Audio decoder
    t0 = time.time()
    codes = np.zeros((1, 16, 125), dtype=np.int32)
    # Fill unused frames with pad token (2050)
    codes[:] = 2050
    for t_idx in range(min(len(all_codebooks), 125)):
        frame = all_codebooks[t_idx]
        for cb in range(min(len(frame), 16)):
            codes[0, cb, t_idx] = frame[cb]

    audio_out = audio_decoder_ml.predict({"codes": codes})
    audio_samples = audio_out["audio"].flatten()
    decoder_time = time.time() - t0

    rms = np.sqrt(np.mean(audio_samples ** 2))
    print(f"  [{label}] Audio: {len(audio_samples)} samples, {len(audio_samples)/SAMPLE_RATE:.2f}s, RMS={rms:.4f}")
    print(f"  [{label}] Total: prefill={prefill_time:.2f}s, decode={decode_time:.2f}s, audio_dec={decoder_time:.2f}s")

    return audio_samples, np.array(all_codebooks)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    np.random.seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("Bilingual Comparison: PyTorch vs CoreML (English + Chinese)")
    print("=" * 70)

    # ── Load models ──────────────────────────────────────────────────
    print("\n1. Loading models...")
    t0 = time.time()

    # PyTorch
    from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
    tts_model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
    tokenizer_model = Qwen3TTSTokenizer.from_pretrained("./tokenizer_12hz", device_map="cpu")

    # CoreML
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

    models = (prefill_ml, decode_ml, cp_prefill_ml, cp_decode_ml, audio_decoder_ml)
    embeddings = (tts_bos, tts_pad, tts_eos, speaker, cp_embeds_np)

    print(f"   All models loaded in {time.time() - t0:.1f}s")

    # Speaker embed for PyTorch
    speaker_embed_np = np.load("speaker_embedding_official.npy").reshape(1, 1024)

    # ── Phase 1: PyTorch reference ───────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 1: PyTorch Reference Generation")
    print("=" * 70)

    # English
    np.random.seed(42)
    torch.manual_seed(42)
    en_pytorch_audio, en_pytorch_codes, en_text_tokens = generate_pytorch_reference(
        tts_model, tokenizer_model, ENGLISH_TEXT, "english", speaker_embed_np, "EN-PyTorch"
    )
    en_pytorch_path = os.path.join(OUTPUT_DIR, "en_pytorch_reference.wav")
    write_wav(en_pytorch_path, en_pytorch_audio)
    print(f"  Saved: {en_pytorch_path}")

    # Save English token IDs for Swift
    np.save(os.path.join(OUTPUT_DIR, "en_text_tokens.npy"), np.array(en_text_tokens))
    print(f"  English token IDs: {en_text_tokens}")

    # Chinese
    np.random.seed(42)
    torch.manual_seed(42)
    zh_pytorch_audio, zh_pytorch_codes, zh_text_tokens = generate_pytorch_reference(
        tts_model, tokenizer_model, CHINESE_TEXT, "chinese", speaker_embed_np, "ZH-PyTorch"
    )
    zh_pytorch_path = os.path.join(OUTPUT_DIR, "zh_pytorch_reference.wav")
    write_wav(zh_pytorch_path, zh_pytorch_audio)
    print(f"  Saved: {zh_pytorch_path}")

    # Save Chinese token IDs for Swift
    np.save(os.path.join(OUTPUT_DIR, "zh_text_tokens.npy"), np.array(zh_text_tokens))
    print(f"  Chinese token IDs: {zh_text_tokens}")

    # ── Phase 2: CoreML pipeline ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 2: CoreML Pipeline Generation")
    print("=" * 70)

    # English
    np.random.seed(42)
    en_coreml_audio, en_coreml_codes = generate_coreml_audio(
        en_text_tokens, "EN-CoreML", models, embeddings
    )
    en_coreml_path = os.path.join(OUTPUT_DIR, "en_coreml_output.wav")
    write_wav(en_coreml_path, en_coreml_audio)
    print(f"  Saved: {en_coreml_path}")

    # Chinese
    np.random.seed(42)
    zh_coreml_audio, zh_coreml_codes = generate_coreml_audio(
        zh_text_tokens, "ZH-CoreML", models, embeddings
    )
    zh_coreml_path = os.path.join(OUTPUT_DIR, "zh_coreml_output.wav")
    write_wav(zh_coreml_path, zh_coreml_audio)
    print(f"  Saved: {zh_coreml_path}")

    # ── Phase 3: Comparison ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 3: Spectral Similarity & ASR Evaluation")
    print("=" * 70)

    # Spectral comparison
    for lang, pt_audio, cm_audio, label in [
        ("English", en_pytorch_audio, en_coreml_audio, "EN"),
        ("Chinese", zh_pytorch_audio, zh_coreml_audio, "ZH"),
    ]:
        print(f"\n  --- {lang}: PyTorch vs CoreML ---")
        metrics = spectral_similarity(pt_audio, cm_audio)
        print(f"  Cosine Similarity:        {metrics['cosine_similarity']:.4f}")
        print(f"  Mel Cepstral Distortion:  {metrics['mel_cepstral_distortion']:.4f}")
        print(f"  Correlation:              {metrics['correlation']:.4f}")
        print(f"  RMS Ratio:                {metrics['rms_ratio']:.4f}")
        print(f"  RMS (PyTorch):            {metrics['rms_a']:.4f}")
        print(f"  RMS (CoreML):             {metrics['rms_b']:.4f}")

    # Codebook comparison
    for lang, pt_codes, cm_codes, label in [
        ("English", en_pytorch_codes, en_coreml_codes, "EN"),
        ("Chinese", zh_pytorch_codes, zh_coreml_codes, "ZH"),
    ]:
        print(f"\n  --- {lang}: Codebook Comparison ---")
        min_len = min(len(pt_codes), len(cm_codes))
        if min_len > 0:
            cb0_match = np.sum(pt_codes[:min_len, 0] == cm_codes[:min_len, 0])
            print(f"  CB0 match: {cb0_match}/{min_len} ({100*cb0_match/min_len:.1f}%)")
            total_match = np.sum(pt_codes[:min_len] == cm_codes[:min_len])
            total = min_len * 16
            print(f"  Total match: {total_match}/{total} ({100*total_match/total:.1f}%)")
        print(f"  Frames: PyTorch={len(pt_codes)}, CoreML={len(cm_codes)}")

    # ASR evaluation
    print(f"\n  --- ASR Evaluation (Whisper) ---")
    try:
        import whisper
        whisper_model = whisper.load_model("base")

        for path, expected, label in [
            (en_pytorch_path, ENGLISH_TEXT, "EN-PyTorch"),
            (en_coreml_path, ENGLISH_TEXT, "EN-CoreML"),
            (zh_pytorch_path, CHINESE_TEXT, "ZH-PyTorch"),
            (zh_coreml_path, CHINESE_TEXT, "ZH-CoreML"),
        ]:
            result = whisper_model.transcribe(path, language=("en" if label.startswith("EN") else "zh"))
            transcription = result["text"].strip()
            print(f"  [{label}] ASR: '{transcription}'")
            print(f"  [{label}] Expected: '{expected}'")
    except ImportError:
        print("  Whisper not installed, skipping ASR. Install: pip install openai-whisper")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  English text tokens ({len(en_text_tokens)}): {en_text_tokens}")
    print(f"  Chinese text tokens ({len(zh_text_tokens)}): {zh_text_tokens}")
    print(f"\n  Output files in: {OUTPUT_DIR}/")
    print(f"    en_pytorch_reference.wav  - English PyTorch reference")
    print(f"    en_coreml_output.wav      - English CoreML output")
    print(f"    zh_pytorch_reference.wav  - Chinese PyTorch reference")
    print(f"    zh_coreml_output.wav      - Chinese CoreML output")
    print(f"    en_text_tokens.npy        - English token IDs for Swift")
    print(f"    zh_text_tokens.npy        - Chinese token IDs for Swift")
    print("=" * 70)


if __name__ == "__main__":
    main()
