#!/usr/bin/env python3
"""Measure actual memory usage of CoreML models during inference."""

import argparse
import subprocess
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import psutil
import soundfile as sf

from cohere_mel_spectrogram import CohereMelSpectrogram


def get_memory_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024**2


def measure_model_memory(encoder_path: Path, decoder_path: Path, audio_path: Path):
    """Measure memory usage during model loading and inference."""

    print(f"\n{'='*70}")
    print(f"Memory Profiling: {encoder_path.parent.name}")
    print(f"{'='*70}")

    # Baseline
    baseline_mem = get_memory_mb()
    print(f"\n[Baseline] Process memory: {baseline_mem:.1f} MB")

    # Load encoder
    print(f"\n[1/5] Loading encoder...")
    mem_before = get_memory_mb()
    encoder = ct.models.MLModel(str(encoder_path))
    mem_after = get_memory_mb()
    encoder_load_mem = mem_after - mem_before
    print(f"  Encoder loaded: +{encoder_load_mem:.1f} MB")
    print(f"  Total memory: {mem_after:.1f} MB")

    # Load decoder
    print(f"\n[2/5] Loading decoder...")
    mem_before = get_memory_mb()
    decoder = ct.models.MLModel(str(decoder_path))
    mem_after = get_memory_mb()
    decoder_load_mem = mem_after - mem_before
    print(f"  Decoder loaded: +{decoder_load_mem:.1f} MB")
    print(f"  Total memory: {mem_after:.1f} MB")

    total_load_mem = mem_after - baseline_mem
    print(f"\n  Combined model load: +{total_load_mem:.1f} MB")

    # Load audio
    print(f"\n[3/5] Loading audio...")
    audio, sr = sf.read(str(audio_path))
    if sr != 16000:
        raise ValueError(f"Expected 16kHz audio, got {sr}Hz")

    # Compute mel
    print(f"\n[4/5] Computing mel spectrogram...")
    mel_processor = CohereMelSpectrogram()
    mel = mel_processor(audio)
    mel_padded = np.pad(mel, ((0, 0), (0, 0), (0, 3001 - mel.shape[2])), mode='constant')
    mel_features = mel_padded.astype(np.float32)
    mel_length = np.array([mel.shape[2]], dtype=np.int32)

    mem_before_inference = get_memory_mb()
    print(f"  Memory before inference: {mem_before_inference:.1f} MB")

    # Encoder inference
    print(f"\n[5/5] Running inference...")
    print(f"  Encoder inference...")
    mem_before = get_memory_mb()
    encoder_output = encoder.predict({
        "input_features": mel_features,
        "feature_length": mel_length
    })
    mem_after = get_memory_mb()
    encoder_inference_mem = mem_after - mem_before

    # Find encoder output
    encoder_hidden = None
    for key, value in encoder_output.items():
        if hasattr(value, 'shape') and len(value.shape) == 3:
            encoder_hidden = value
            break

    print(f"    Encoder inference: +{encoder_inference_mem:.1f} MB")
    print(f"    Total memory: {mem_after:.1f} MB")

    # Decoder inference (first few steps)
    print(f"  Decoder inference (10 steps)...")

    num_layers = 8
    num_heads = 8
    head_dim = 128
    max_cache_len = 108

    cache_k = np.zeros((num_layers, num_heads, max_cache_len, head_dim), dtype=np.float32)
    cache_v = np.zeros((num_layers, num_heads, max_cache_len, head_dim), dtype=np.float32)
    current_token = np.array([[13764]], dtype=np.int32)
    cross_attention_mask = np.ones((1, 1, 1, encoder_hidden.shape[1]), dtype=np.float32)

    mem_before = get_memory_mb()
    peak_decoder_mem = mem_before

    for step in range(10):
        step_array = np.array([step], dtype=np.int32)
        decoder_output = decoder.predict({
            "input_id": current_token,
            "encoder_hidden_states": encoder_hidden,
            "cache_k": cache_k,
            "cache_v": cache_v,
            "step": step_array,
            "cross_attention_mask": cross_attention_mask,
        })

        # Handle different output formats
        if "logits" in decoder_output:
            logits = decoder_output["logits"]
            cache_k = decoder_output["new_cache_k"]
            cache_v = decoder_output["new_cache_v"]
        else:
            output_values = list(decoder_output.values())
            logits = output_values[0]
            cache_k = output_values[1]
            cache_v = output_values[2]

        current_mem = get_memory_mb()
        peak_decoder_mem = max(peak_decoder_mem, current_mem)

        next_token = int(np.argmax(logits, axis=-1)[0])
        current_token = np.array([[next_token]], dtype=np.int32)

    mem_after = get_memory_mb()
    decoder_inference_mem = peak_decoder_mem - mem_before

    print(f"    Decoder inference peak: +{decoder_inference_mem:.1f} MB")
    print(f"    Peak memory: {peak_decoder_mem:.1f} MB")

    # Final summary
    print(f"\n{'='*70}")
    print(f"MEMORY SUMMARY")
    print(f"{'='*70}")
    print(f"Baseline (empty process):        {baseline_mem:.1f} MB")
    print(f"After encoder load:              +{encoder_load_mem:.1f} MB")
    print(f"After decoder load:              +{decoder_load_mem:.1f} MB")
    print(f"After encoder inference:         +{encoder_inference_mem:.1f} MB")
    print(f"Peak during decoder inference:   +{decoder_inference_mem:.1f} MB")
    print(f"─" * 70)
    print(f"Total peak memory:               {peak_decoder_mem:.1f} MB")
    print(f"Total memory overhead:           +{peak_decoder_mem - baseline_mem:.1f} MB")
    print(f"{'='*70}")

    return {
        "baseline": baseline_mem,
        "encoder_load": encoder_load_mem,
        "decoder_load": decoder_load_mem,
        "encoder_inference": encoder_inference_mem,
        "decoder_inference": decoder_inference_mem,
        "peak_total": peak_decoder_mem,
        "total_overhead": peak_decoder_mem - baseline_mem,
    }


def main():
    parser = argparse.ArgumentParser(description="Measure CoreML model memory usage")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["fp16", "quantized", "reference", "all"],
        default=["all"],
        help="Which models to profile"
    )
    args = parser.parse_args()

    print("="*70)
    print("CoreML Memory Profiling")
    print("="*70)

    # Find test audio
    test_audio = Path("test-audio/synthetic-test.wav")
    if not test_audio.exists():
        print(f"❌ Test audio not found: {test_audio}")
        return

    configs = []

    if "all" in args.models or "reference" in args.models:
        configs.append({
            "name": "Reference (Barathwaj)",
            "encoder": Path("barathwaj-models/cohere_encoder.mlpackage"),
            "decoder": Path("barathwaj-models/cohere_decoder_cached.mlpackage"),
        })

    if "all" in args.models or "fp16" in args.models:
        configs.append({
            "name": "Our FP16",
            "encoder": Path("build/cohere_encoder.mlpackage"),
            "decoder": Path("build/cohere_decoder_cached.mlpackage"),
        })

    if "all" in args.models or "quantized" in args.models:
        configs.append({
            "name": "Our Quantized (6-bit)",
            "encoder": Path("build-quantized/cohere_encoder.mlpackage"),
            "decoder": Path("build-quantized/cohere_decoder_cached.mlpackage"),
        })

    results = []

    for config in configs:
        if not config["encoder"].exists() or not config["decoder"].exists():
            print(f"\n⚠️  Skipping {config['name']}: Models not found")
            continue

        result = measure_model_memory(config["encoder"], config["decoder"], test_audio)
        result["name"] = config["name"]
        results.append(result)

    # Comparison table
    if len(results) >= 2:
        print(f"\n{'='*70}")
        print("COMPARISON")
        print(f"{'='*70}")
        print(f"{'Model':<25} {'Model Load':<15} {'Peak Total':<15} {'Overhead':<15}")
        print("─" * 70)

        for r in results:
            model_load = r["encoder_load"] + r["decoder_load"]
            print(f"{r['name']:<25} {model_load:<15.1f} {r['peak_total']:<15.1f} {r['total_overhead']:<15.1f}")

    print(f"\n{'='*70}")
    print("PROFILING COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
