#!/usr/bin/env python3
"""Benchmark Cohere Transcribe models for speed, memory, and accuracy."""

import argparse
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Tuple

import coremltools as ct
import numpy as np
import soundfile as sf
import torch
from jiwer import wer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from cohere_mel_spectrogram import CohereMelSpectrogram


def measure_memory_and_time(func):
    """Decorator to measure memory usage and execution time."""
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        start_time = time.perf_counter()

        result = func(*args, **kwargs)

        elapsed = time.perf_counter() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return result, elapsed, peak / 1024**2  # Convert to MB
    return wrapper


class CoreMLPipeline:
    """CoreML inference pipeline."""

    def __init__(self, encoder_path: Path, decoder_path: Path, processor):
        print(f"Loading CoreML encoder from {encoder_path}...")
        self.encoder = ct.models.MLModel(str(encoder_path))
        print(f"Loading CoreML decoder from {decoder_path}...")
        self.decoder = ct.models.MLModel(str(decoder_path))
        self.processor = processor
        # EOS token ID from Cohere config
        self.eos_token_id = processor.eos_token_id if processor else 2
        self.mel_processor = CohereMelSpectrogram()

    def transcribe(self, audio_path: Path, max_new_tokens: int = 200) -> Tuple[str, Dict]:
        """Transcribe audio file and return text + metrics."""
        # Load audio
        audio, sr = sf.read(str(audio_path))
        if sr != 16000:
            raise ValueError(f"Expected 16kHz audio, got {sr}Hz")

        # Compute mel spectrogram
        mel_start = time.perf_counter()
        mel = self.mel_processor(audio)

        # Pad to 3001 frames (expected by encoder)
        mel_padded = np.pad(
            mel,
            ((0, 0), (0, 0), (0, 3001 - mel.shape[2])),
            mode='constant',
            constant_values=0
        )
        mel_features = mel_padded.astype(np.float32)
        mel_length = np.array([mel.shape[2]], dtype=np.int32)
        mel_time = time.perf_counter() - mel_start

        # Encoder inference
        enc_start = time.perf_counter()
        encoder_output = self.encoder.predict({
            "input_features": mel_features,
            "feature_length": mel_length
        })
        # Find encoder output (3D tensor)
        encoder_hidden = None
        for key, value in encoder_output.items():
            if hasattr(value, 'shape') and len(value.shape) == 3:
                encoder_hidden = value
                break
        if encoder_hidden is None:
            raise ValueError("Could not find encoder output")
        enc_time = time.perf_counter() - enc_start

        # Prepare decoder inputs
        num_layers = 8
        num_heads = 8
        head_dim = 128
        max_cache_len = 108

        cache_k = np.zeros((num_layers, num_heads, max_cache_len, head_dim), dtype=np.float32)
        cache_v = np.zeros((num_layers, num_heads, max_cache_len, head_dim), dtype=np.float32)

        # Start token
        current_token = np.array([[13764]], dtype=np.int32)
        generated_tokens = [13764]

        # Cross attention mask (all ones for encoder output)
        enc_seq_len = encoder_hidden.shape[1]
        cross_attention_mask = np.ones((1, 1, 1, enc_seq_len), dtype=np.float32)

        # Decode
        dec_start = time.perf_counter()
        for step in range(max_new_tokens):
            step_array = np.array([step], dtype=np.int32)

            decoder_output = self.decoder.predict({
                "input_id": current_token,
                "encoder_hidden_states": encoder_hidden,
                "cache_k": cache_k,
                "cache_v": cache_v,
                "step": step_array,
                "cross_attention_mask": cross_attention_mask,
            })

            # Handle different output names (reference vs our export)
            if "logits" in decoder_output:
                logits = decoder_output["logits"]
                cache_k = decoder_output["new_cache_k"]
                cache_v = decoder_output["new_cache_v"]
            else:
                # Reference model has var_* names
                output_values = list(decoder_output.values())
                logits = output_values[0]  # First output is logits
                cache_k = output_values[1]  # Second is cache_k
                cache_v = output_values[2]  # Third is cache_v

            next_token = int(np.argmax(logits, axis=-1)[0])
            generated_tokens.append(next_token)

            if next_token == self.eos_token_id:
                break

            current_token = np.array([[next_token]], dtype=np.int32)

        dec_time = time.perf_counter() - dec_start

        # Decode text
        if self.processor:
            text = self.processor.decode(generated_tokens, skip_special_tokens=True)
        else:
            # Just show token IDs if no tokenizer
            text = f"[Tokens: {generated_tokens}]"

        metrics = {
            "mel_time": mel_time,
            "encoder_time": enc_time,
            "decoder_time": dec_time,
            "total_time": mel_time + enc_time + dec_time,
            "tokens_generated": len(generated_tokens),
            "tokens_per_sec": len(generated_tokens) / dec_time if dec_time > 0 else 0,
        }

        return text, metrics


def load_test_audio() -> List[Tuple[Path, str]]:
    """Load test audio files with reference transcripts."""
    # Check if we have the LibriSpeech test file
    test_file = Path("test-audio/1089-134686-0000.flac")
    reference = "he hoped there would be stew for dinner turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick peppered flour fattened sauce"

    if test_file.exists():
        return [(test_file, reference)]

    print("⚠️  Warning: Test audio not found. Using any available .flac or .wav files...")

    # Try to find any audio files
    audio_dir = Path("test-audio")
    if not audio_dir.exists():
        audio_dir = Path(".")

    audio_files = list(audio_dir.glob("*.flac")) + list(audio_dir.glob("*.wav"))

    if not audio_files:
        print("❌ No audio files found. Please provide test audio.")
        return []

    # Return first file without reference
    return [(audio_files[0], None)]


def benchmark_model_type(
    model_type: str,
    encoder_path: Path,
    decoder_path: Path,
    test_files: List[Tuple[Path, str]],
    processor,
) -> Dict:
    """Benchmark a specific model configuration."""
    print(f"\n{'='*70}")
    print(f"Benchmarking: {model_type}")
    print(f"{'='*70}")

    # Load models
    pipeline = CoreMLPipeline(encoder_path, decoder_path, processor)

    results = []

    for audio_path, reference in test_files:
        print(f"\nProcessing: {audio_path.name}")

        # Transcribe
        tracemalloc.start()
        hypothesis, metrics = pipeline.transcribe(audio_path)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        metrics["peak_memory_mb"] = peak / 1024**2

        print(f"  Hypothesis: {hypothesis}")
        if reference:
            print(f"  Reference:  {reference}")
            error_rate = wer(reference, hypothesis)
            metrics["wer"] = error_rate
            print(f"  WER: {error_rate*100:.2f}%")

        print(f"  Mel time: {metrics['mel_time']:.3f}s")
        print(f"  Encoder time: {metrics['encoder_time']:.3f}s")
        print(f"  Decoder time: {metrics['decoder_time']:.3f}s")
        print(f"  Total time: {metrics['total_time']:.3f}s")
        print(f"  Tokens: {metrics['tokens_generated']}")
        print(f"  Tokens/sec: {metrics['tokens_per_sec']:.1f}")
        print(f"  Peak memory: {metrics['peak_memory_mb']:.1f} MB")

        results.append({
            "audio": audio_path.name,
            "hypothesis": hypothesis,
            "reference": reference,
            **metrics
        })

    # Compute averages
    avg_metrics = {
        "model_type": model_type,
        "num_samples": len(results),
        "avg_mel_time": np.mean([r["mel_time"] for r in results]),
        "avg_encoder_time": np.mean([r["encoder_time"] for r in results]),
        "avg_decoder_time": np.mean([r["decoder_time"] for r in results]),
        "avg_total_time": np.mean([r["total_time"] for r in results]),
        "avg_tokens": np.mean([r["tokens_generated"] for r in results]),
        "avg_tokens_per_sec": np.mean([r["tokens_per_sec"] for r in results]),
        "avg_peak_memory": np.mean([r["peak_memory_mb"] for r in results]),
    }

    if any(r.get("wer") is not None for r in results):
        avg_metrics["avg_wer"] = np.mean([r["wer"] for r in results if r.get("wer") is not None])

    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Benchmark Cohere Transcribe models")
    parser.add_argument(
        "--fp16-dir",
        type=Path,
        default=Path("build"),
        help="Directory containing FP16 models"
    )
    parser.add_argument(
        "--quantized-dir",
        type=Path,
        default=Path("build-quantized"),
        help="Directory containing quantized models"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["fp16", "quantized", "all"],
        default=["all"],
        help="Which models to benchmark"
    )

    args = parser.parse_args()

    print("="*70)
    print("Cohere Transcribe Model Benchmarking")
    print("="*70)

    # Load processor (optional - for decoding tokens to text)
    print("\nLoading tokenizer...")
    try:
        from transformers import AutoTokenizer
        processor = AutoTokenizer.from_pretrained(
            "CohereLabs/cohere-transcribe-03-2026",
            trust_remote_code=True
        )
        print("   ✓ Loaded tokenizer")
    except Exception as e:
        print(f"   ⚠️  Could not load tokenizer ({e})")
        print("   Will output token IDs only")
        processor = None

    # Load test files
    print("\nLoading test audio...")
    test_files = load_test_audio()
    if not test_files:
        print("❌ No test files available. Exiting.")
        return

    print(f"  Found {len(test_files)} test file(s)")

    # Benchmark configurations
    configs = []

    if "all" in args.models or "fp16" in args.models:
        configs.append({
            "name": "FP16",
            "encoder": args.fp16_dir / "cohere_encoder.mlpackage",
            "decoder": args.fp16_dir / "cohere_decoder_cached.mlpackage",
        })

    if "all" in args.models or "quantized" in args.models:
        configs.append({
            "name": "6-bit Quantized",
            "encoder": args.quantized_dir / "cohere_encoder.mlpackage",
            "decoder": args.quantized_dir / "cohere_decoder_cached.mlpackage",
        })

    # Run benchmarks
    all_results = []

    for config in configs:
        if not config["encoder"].exists() or not config["decoder"].exists():
            print(f"\n⚠️  Skipping {config['name']}: Models not found")
            continue

        result = benchmark_model_type(
            config["name"],
            config["encoder"],
            config["decoder"],
            test_files,
            processor
        )
        all_results.append(result)

    # Summary comparison
    if len(all_results) >= 2:
        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}")

        fp16 = next((r for r in all_results if "FP16" in r["model_type"]), None)
        quant = next((r for r in all_results if "Quantized" in r["model_type"]), None)

        if fp16 and quant:
            print(f"\n{'Metric':<25} {'FP16':<15} {'Quantized':<15} {'Speedup':<10}")
            print("-" * 70)

            metrics = [
                ("Encoder time (s)", "avg_encoder_time"),
                ("Decoder time (s)", "avg_decoder_time"),
                ("Total time (s)", "avg_total_time"),
                ("Tokens/sec", "avg_tokens_per_sec"),
                ("Peak memory (MB)", "avg_peak_memory"),
            ]

            for label, key in metrics:
                fp16_val = fp16[key]
                quant_val = quant[key]

                if "time" in key or "memory" in key:
                    speedup = fp16_val / quant_val if quant_val > 0 else 0
                    print(f"{label:<25} {fp16_val:<15.3f} {quant_val:<15.3f} {speedup:<10.2f}x")
                else:
                    speedup = quant_val / fp16_val if fp16_val > 0 else 0
                    print(f"{label:<25} {fp16_val:<15.1f} {quant_val:<15.1f} {speedup:<10.2f}x")

            if "avg_wer" in fp16 and "avg_wer" in quant:
                print(f"{'WER (%)':<25} {fp16['avg_wer']*100:<15.2f} {quant['avg_wer']*100:<15.2f}")

    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
