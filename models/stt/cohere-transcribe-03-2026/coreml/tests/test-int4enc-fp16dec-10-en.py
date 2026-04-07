#!/usr/bin/env python3
"""Test INT4 encoder + FP16 decoder on 10 English FLEURS samples.

Expected:
- ~75% encoder size reduction (3.6 GB → ~900 MB)
- Total: ~1.2 GB (vs 2.1 GB hybrid INT8, vs 3.9 GB FP16)
- Quality: Unknown - INT4 is very aggressive quantization
- Stability: FP16 decoder should prevent loops
"""

import coremltools as ct
import numpy as np
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent / "f16"))
from cohere_mel_spectrogram import CohereMelSpectrogram

ENGLISH_PROMPT = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]
MAX_TOKENS = 108


def decode_with_prompt(decoder, encoder_hidden, vocab, prompt):
    """Decode with language prompt."""
    state = decoder.make_state()
    tokens = []
    last_token = None

    seq_len = encoder_hidden.shape[1]
    cross_mask = np.ones((1, 1, 1, seq_len), dtype=np.float16)

    for step in range(MAX_TOKENS):
        if step < len(prompt):
            current_token = prompt[step]
        else:
            current_token = last_token

        decoder_out = decoder.predict({
            "input_id": np.array([[current_token]], dtype=np.int32),
            "encoder_hidden_states": encoder_hidden.astype(np.float16),
            "attention_mask": np.zeros((1, 1, 1, step + 1), dtype=np.float16),
            "cross_attention_mask": cross_mask,
            "position_ids": np.array([[step]], dtype=np.int32),
        }, state=state)

        logits = decoder_out["logits"]
        next_token = int(np.argmax(logits[0]))
        last_token = next_token

        if step >= len(prompt) - 1:
            tokens.append(next_token)
            if next_token == 3:
                break

    text_tokens = []
    for t in tokens:
        if t <= 4 or t == 3:
            continue
        token_str = vocab.get(t, "")
        if token_str.startswith("<|"):
            continue
        text_tokens.append(token_str)

    return "".join(text_tokens).replace("▁", " ").strip().lower()


def detect_repetition(text, threshold=5):
    """Detect repetitive patterns."""
    words = text.split()
    if len(words) < 3:
        return False

    for i in range(len(words) - threshold):
        word = words[i]
        consecutive_count = 1
        for j in range(i + 1, min(i + 20, len(words))):
            if words[j] == word:
                consecutive_count += 1
            else:
                break
        if consecutive_count >= threshold:
            return True

    return False


def main():
    print("=" * 70)
    print("INT4 Encoder + FP16 Decoder Test")
    print("10 English FLEURS samples")
    print("=" * 70)
    print()

    # Load INT4 encoder + FP16 decoder
    print("Loading models...")
    print("  • INT4 encoder from int4/")
    print("  • FP16 decoder from f16/")

    encoder_path = "int4/cohere_encoder_int4.mlpackage"
    if not Path(encoder_path).exists():
        print(f"ERROR: INT4 encoder not found at {encoder_path}")
        print("Please run: uv run quantize_encoder_to_int4.py")
        return 1

    encoder = ct.models.MLModel(encoder_path)
    decoder = ct.models.MLModel("f16/cohere_decoder_stateful.mlpackage")

    with open("f16/vocab.json") as f:
        vocab = {int(k): v for k, v in json.load(f).items()}

    mel_processor = CohereMelSpectrogram()
    print("✓ Models loaded")
    print()

    # Print model sizes
    import subprocess
    int4_enc_size = subprocess.check_output(["du", "-sh", encoder_path]).decode().split()[0]
    fp16_dec_size = subprocess.check_output(["du", "-sh", "f16/cohere_decoder_stateful.mlpackage"]).decode().split()[0]
    print(f"Model sizes:")
    print(f"  INT4 encoder: {int4_enc_size}")
    print(f"  FP16 decoder: {fp16_dec_size}")
    print()

    from datasets import load_dataset
    from jiwer import wer

    dataset = load_dataset("google/fleurs", "en_us", split="test", streaming=True, trust_remote_code=True)
    samples = list(dataset.take(10))

    results = []
    good_count = 0
    loop_count = 0
    total_error = 0

    for idx, sample in enumerate(samples):
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        ground_truth = sample["transcription"].lower()

        # Encode with INT4
        mel = mel_processor(audio)
        if mel.shape[2] > 3500:
            mel_padded = mel[:, :, :3500]
            actual_length = 3500
        else:
            mel_padded = np.pad(mel, ((0, 0), (0, 0), (0, 3500 - mel.shape[2])))
            actual_length = mel.shape[2]

        encoder_out = encoder.predict({
            "input_features": mel_padded.astype(np.float32),
            "feature_length": np.array([actual_length], dtype=np.int32),
        })
        encoder_hidden = encoder_out["hidden_states"]

        # Decode with FP16
        hypothesis = decode_with_prompt(decoder, encoder_hidden, vocab, ENGLISH_PROMPT)

        error_rate = wer(ground_truth, hypothesis) * 100
        has_loop = detect_repetition(hypothesis)
        is_good = error_rate < 30
        is_perfect = error_rate < 1

        if has_loop:
            status = "🔁"
            loop_count += 1
        elif is_perfect:
            status = "✅"
        elif is_good:
            status = "🟢"
        else:
            status = "❌"

        if is_good:
            good_count += 1

        total_error += error_rate

        print(f"[{idx+1:2d}/10] {status} WER: {error_rate:6.2f}%", end="")
        if has_loop:
            print(" [LOOP]", end="")
        print()

        results.append({
            "ground_truth": ground_truth,
            "hypothesis": hypothesis,
            "wer": error_rate,
            "has_loop": has_loop,
            "is_good": is_good,
        })

    avg_wer = total_error / 10

    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)
    print(f"Good (<30% WER): {good_count}/10 ({good_count*10}%)")
    print(f"Loops: {loop_count}/10 ({loop_count*10}%)")
    print(f"Avg WER: {avg_wer:.2f}%")
    print()

    print("=" * 70)
    print("Comparison")
    print("=" * 70)
    print()
    print(f"{'Configuration':<25} {'Success':<10} {'Loops':<10} {'Size':<15} {'Encoder'}")
    print("-" * 75)
    print(f"{'Full FP16':<25} {'20%':<10} {'0%':<10} {'~3.9 GB':<15} {'3.6 GB FP16'}")
    print(f"{'Hybrid INT8+FP16':<25} {'20%':<10} {'0%':<10} {'~2.1 GB':<15} {'1.8 GB INT8'}")
    print(f"{'INT4+FP16':<25} {f'{good_count*10}%':<10} {f'{loop_count*10}%':<10} {f'~{int4_enc_size}+291M':<15} {f'{int4_enc_size} INT4'}")
    print(f"{'Full INT8':<25} {'14%':<10} {'71%':<10} {'~1.95 GB':<15} {'1.8 GB INT8'}")
    print()

    if loop_count == 0 and good_count >= 2:
        print("✅ INT4 encoder works! FP16 decoder prevents loops")
        print("   → Extreme memory savings with acceptable quality")
    elif loop_count == 0:
        print("⚠️  INT4 encoder stable but low quality")
        print("   → FP16 decoder prevents loops but encoder degraded")
    elif loop_count < 5:
        print("⚠️  INT4 encoder has moderate instability")
        print("   → Some encoder artifacts leak through to decoder")
    else:
        print("❌ INT4 encoder too aggressive")
        print("   → Quality degradation too severe")

    print()

    with open("test_int4enc_fp16dec_10_en_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Results saved to: test_int4enc_fp16dec_10_en_results.json")


if __name__ == "__main__":
    main()
