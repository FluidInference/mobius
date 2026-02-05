#!/usr/bin/env python3
"""
Compare Swift, CoreML (Python), and PyTorch audio outputs.
Computes spectral similarity and runs Whisper ASR on all outputs.
"""

import numpy as np
import struct
import os
import warnings
warnings.filterwarnings('ignore')

SAMPLE_RATE = 24000

ENGLISH_TEXT = "Hello world, this is a test of the text to speech system."
CHINESE_TEXT = "你好世界，这是一个文字转语音系统的测试。"

# File paths
FILES = {
    "EN-PyTorch": "bilingual_test_outputs/en_pytorch_reference.wav",
    "EN-CoreML":  "bilingual_test_outputs/en_coreml_output.wav",
    "EN-Swift":   "/tmp/qwen3_en_swift.wav",
    "ZH-PyTorch": "bilingual_test_outputs/zh_pytorch_reference.wav",
    "ZH-CoreML":  "bilingual_test_outputs/zh_coreml_output.wav",
    "ZH-Swift":   "/tmp/qwen3_zh_swift.wav",
}


def read_wav(path):
    """Read WAV file and return float32 samples."""
    with open(path, "rb") as f:
        data = f.read()
    # Parse WAV header
    assert data[:4] == b"RIFF"
    assert data[8:12] == b"WAVE"

    # Find data chunk
    pos = 12
    while pos < len(data):
        chunk_id = data[pos:pos + 4]
        chunk_size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        if chunk_id == b"fmt ":
            fmt_data = data[pos + 8:pos + 8 + chunk_size]
            audio_fmt = struct.unpack("<H", fmt_data[0:2])[0]
            channels = struct.unpack("<H", fmt_data[2:4])[0]
            sample_rate = struct.unpack("<I", fmt_data[4:8])[0]
            bits_per_sample = struct.unpack("<H", fmt_data[14:16])[0]
        elif chunk_id == b"data":
            audio_data = data[pos + 8:pos + 8 + chunk_size]
            break
        pos += 8 + chunk_size

    if bits_per_sample == 16:
        n_samples = len(audio_data) // 2
        samples = np.array(struct.unpack(f"<{n_samples}h", audio_data[:n_samples * 2]), dtype=np.float32)
        samples = samples / 32768.0
    elif bits_per_sample == 32:
        n_samples = len(audio_data) // 4
        samples = np.array(struct.unpack(f"<{n_samples}f", audio_data[:n_samples * 4]), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported bits_per_sample: {bits_per_sample}")

    return samples, sample_rate


def compute_mel_spectrogram(audio, sr=24000, n_fft=1024, hop_length=256, n_mels=80):
    """Compute log mel spectrogram."""
    window = np.hanning(n_fft)
    frames = []
    for i in range(0, len(audio) - n_fft, hop_length):
        frame = audio[i:i + n_fft] * window
        spectrum = np.fft.rfft(frame)
        frames.append(np.abs(spectrum) ** 2)
    if not frames:
        return np.zeros((n_mels, 1))
    power_spec = np.array(frames).T

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
    """Compute spectral similarity metrics."""
    max_len = max(len(audio_a), len(audio_b))
    a = np.pad(audio_a, (0, max_len - len(audio_a)))
    b = np.pad(audio_b, (0, max_len - len(audio_b)))

    mel_a = compute_mel_spectrogram(a, sr)
    mel_b = compute_mel_spectrogram(b, sr)

    min_t = min(mel_a.shape[1], mel_b.shape[1])
    mel_a = mel_a[:, :min_t]
    mel_b = mel_b[:, :min_t]

    # Cosine similarity
    cos_sims = []
    for t in range(min_t):
        a_frame = mel_a[:, t]
        b_frame = mel_b[:, t]
        norm_a = np.linalg.norm(a_frame)
        norm_b = np.linalg.norm(b_frame)
        if norm_a > 0 and norm_b > 0:
            cos_sims.append(np.dot(a_frame, b_frame) / (norm_a * norm_b))
    avg_cosine = np.mean(cos_sims) if cos_sims else 0.0

    # Correlation
    flat_a = mel_a.flatten()
    flat_b = mel_b.flatten()
    if np.std(flat_a) > 0 and np.std(flat_b) > 0:
        correlation = np.corrcoef(flat_a, flat_b)[0, 1]
    else:
        correlation = 0.0

    rms_a = np.sqrt(np.mean(audio_a ** 2))
    rms_b = np.sqrt(np.mean(audio_b ** 2))
    rms_ratio = min(rms_a, rms_b) / max(rms_a, rms_b) if max(rms_a, rms_b) > 0 else 0.0

    return {
        'cosine': avg_cosine,
        'correlation': correlation,
        'rms_a': rms_a,
        'rms_b': rms_b,
        'rms_ratio': rms_ratio,
    }


def main():
    print("=" * 75)
    print("Swift vs PyTorch vs CoreML: Bilingual Audio Comparison")
    print("=" * 75)

    # Load all audio
    audio = {}
    print("\n1. Loading audio files...")
    for label, path in FILES.items():
        if not os.path.exists(path):
            print(f"  [{label}] MISSING: {path}")
            continue
        samples, sr = read_wav(path)
        rms = np.sqrt(np.mean(samples ** 2))
        duration = len(samples) / sr
        audio[label] = samples
        print(f"  [{label}] {duration:.2f}s, RMS={rms:.4f}, range=[{samples.min():.3f}, {samples.max():.3f}]")

    # Spectral comparison
    print("\n" + "=" * 75)
    print("2. Spectral Similarity (Mel Spectrogram Cosine Sim / Correlation)")
    print("=" * 75)

    comparisons = [
        ("EN-Swift", "EN-PyTorch", "English: Swift vs PyTorch"),
        ("EN-Swift", "EN-CoreML", "English: Swift vs CoreML(Py)"),
        ("EN-CoreML", "EN-PyTorch", "English: CoreML(Py) vs PyTorch"),
        ("ZH-Swift", "ZH-PyTorch", "Chinese: Swift vs PyTorch"),
        ("ZH-Swift", "ZH-CoreML", "Chinese: Swift vs CoreML(Py)"),
        ("ZH-CoreML", "ZH-PyTorch", "Chinese: CoreML(Py) vs PyTorch"),
    ]

    print(f"\n  {'Comparison':<35} {'Cosine':>8} {'Corr':>8} {'RMS_A':>8} {'RMS_B':>8} {'RMS%':>7}")
    print("  " + "-" * 73)

    for a_label, b_label, desc in comparisons:
        if a_label not in audio or b_label not in audio:
            print(f"  {desc:<35} {'SKIP':>8}")
            continue
        m = spectral_similarity(audio[a_label], audio[b_label])
        print(f"  {desc:<35} {m['cosine']:>8.4f} {m['correlation']:>8.4f} "
              f"{m['rms_a']:>8.4f} {m['rms_b']:>8.4f} {m['rms_ratio']:>6.1%}")

    # ASR evaluation
    print("\n" + "=" * 75)
    print("3. ASR Evaluation (Whisper)")
    print("=" * 75)

    try:
        import whisper
        model = whisper.load_model("base")

        expected = {
            "EN": ENGLISH_TEXT,
            "ZH": CHINESE_TEXT,
        }

        print(f"\n  {'Source':<15} {'ASR Transcription'}")
        print("  " + "-" * 73)

        for label, path in FILES.items():
            if not os.path.exists(path):
                continue
            lang_code = "en" if label.startswith("EN") else "zh"
            result = model.transcribe(path, language=lang_code)
            text = result["text"].strip()
            lang = label[:2]
            exp = expected[lang]
            match_str = "MATCH" if text.lower().rstrip('.') == exp.lower().rstrip('.') else ""
            print(f"  {label:<15} '{text}'  {match_str}")

        print(f"\n  Expected EN: '{ENGLISH_TEXT}'")
        print(f"  Expected ZH: '{CHINESE_TEXT}'")

    except ImportError:
        print("  Whisper not installed. Install: pip install openai-whisper")

    print("\n" + "=" * 75)
    print("Done.")
    print("=" * 75)


if __name__ == "__main__":
    main()
