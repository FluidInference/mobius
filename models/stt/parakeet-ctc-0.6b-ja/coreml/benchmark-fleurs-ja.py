#!/usr/bin/env python3
"""Benchmark Japanese Parakeet CTC CoreML model on FLEURS Japanese (650 samples).

Uses FluidInference/fleurs-full dataset for comprehensive Japanese ASR evaluation.

IMPORTANT: This model outputs RAW logits (not log-softmax). The script applies
log_softmax before CTC decoding.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import coremltools as ct
import librosa
import numpy as np
import typer
from datasets import load_dataset
from rich.console import Console
from rich.progress import track
from rich.table import Table

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
console = Console()


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER) for Japanese text.

    Japanese doesn't have clear word boundaries, so we use character-level metrics.
    Removes all spaces before comparison.
    """
    # Remove all spaces for character-level comparison
    ref_chars = reference.replace(' ', '')
    hyp_chars = hypothesis.replace(' ', '')

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    # Compute edit distance
    import editdistance
    distance = editdistance.eval(ref_chars, hyp_chars)
    return distance / len(ref_chars)


def load_coreml_models(build_dir: Path) -> tuple[ct.models.MLModel, ct.models.MLModel, ct.models.MLModel]:
    """Load the three CoreML models."""
    console.print(f"Loading CoreML models from {build_dir}...")

    preprocessor = ct.models.MLModel(str(build_dir / "Preprocessor.mlpackage"))
    encoder = ct.models.MLModel(str(build_dir / "Encoder.mlpackage"))
    ctc_decoder = ct.models.MLModel(str(build_dir / "CtcDecoder.mlpackage"))

    console.print("✓ All CoreML models loaded")
    return preprocessor, encoder, ctc_decoder


def load_vocabulary(vocab_path: Path) -> list[str]:
    """Load vocabulary as list (index -> token)."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    # Handle both list and dict formats
    if isinstance(vocab_data, list):
        return vocab_data
    else:
        # Dict format: {index_str: token}
        max_idx = max(int(k) for k in vocab_data.keys())
        vocab_list = [""] * (max_idx + 1)
        for idx_str, token in vocab_data.items():
            vocab_list[int(idx_str)] = token
        return vocab_list


def greedy_ctc_decode(log_probs: np.ndarray, vocabulary: list[str], blank_id: int) -> str:
    """Greedy CTC decoding with blank collapse.

    Args:
        log_probs: [1, time_steps, vocab_size] log-probabilities
        vocabulary: List of tokens (index -> token)
        blank_id: Index of the blank token

    Returns:
        Decoded text string
    """
    if log_probs.shape[0] != 1:
        raise ValueError("Batch size must be 1")

    log_probs = log_probs[0]  # [time_steps, vocab_size]
    labels = np.argmax(log_probs, axis=-1)  # [time_steps]

    # Collapse repeats and remove blanks
    decoded = []
    prev = None
    for label in labels:
        if label != blank_id and label != prev:
            decoded.append(label)
        prev = label

    # Convert to text
    tokens = [vocabulary[i] for i in decoded if i < len(vocabulary)]
    text = "".join(tokens)
    # Replace SentencePiece marker with space
    text = text.replace("▁", " ").strip()
    return text


def coreml_full_pipeline(
    audio: np.ndarray,
    sample_rate: int,
    preprocessor: ct.models.MLModel,
    encoder: ct.models.MLModel,
    ctc_decoder: ct.models.MLModel,
    vocabulary: list[str],
    blank_id: int,
    max_audio_samples: int = 240000,  # 15 seconds at 16kHz
) -> tuple[str, float]:
    """Run full CoreML pipeline: audio -> text.

    IMPORTANT: The Japanese CTC decoder outputs RAW logits. This function
    applies log_softmax before CTC decoding.
    """
    start_time = time.time()

    # Ensure audio is float32 and shape [1, samples]
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]

    # Pad or truncate audio to exactly max_audio_samples
    current_samples = audio.shape[1]
    if current_samples < max_audio_samples:
        # Pad with zeros
        pad_width = ((0, 0), (0, max_audio_samples - current_samples))
        audio = np.pad(audio, pad_width, mode='constant', constant_values=0)
    elif current_samples > max_audio_samples:
        # Truncate
        audio = audio[:, :max_audio_samples]

    audio_length = np.array([max_audio_samples], dtype=np.int32)

    # Step 1: Preprocessor (audio -> mel)
    preproc_out = preprocessor.predict({
        "audio_signal": audio,
        "length": audio_length,
    })
    mel = preproc_out["mel_features"]
    mel_length = preproc_out["mel_length"]

    # Step 2: Encoder (mel -> encoder features)
    encoder_out = encoder.predict({
        "mel_features": mel,
        "mel_length": mel_length,
    })
    encoder_output = encoder_out["encoder_output"]  # [1, 1024, T]

    # Step 3: CTC Decoder (encoder features -> RAW logits)
    ctc_out = ctc_decoder.predict({
        "encoder_output": encoder_output,
    })
    raw_logits = ctc_out["ctc_logits"]  # [1, T, vocab_size] - RAW logits

    # Step 4: Apply log_softmax (CRITICAL for Japanese model!)
    # The Japanese model outputs raw logits, not log-probabilities
    # We need to apply log_softmax before CTC decoding
    logits_max = np.max(raw_logits, axis=-1, keepdims=True)
    logits_shifted = raw_logits - logits_max  # Numerical stability
    exp_logits = np.exp(logits_shifted)
    sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
    log_probs = logits_shifted - np.log(sum_exp)

    # Step 5: Greedy CTC decode
    hypothesis = greedy_ctc_decode(log_probs, vocabulary, blank_id)

    latency = time.time() - start_time
    return hypothesis, latency


@app.command()
def benchmark(
    build_dir: Path = typer.Option(
        Path("build"),
        "--build-dir",
        help="Directory containing CoreML models",
    ),
    num_samples: Optional[int] = typer.Option(
        None,
        "--num-samples",
        help="Number of Japanese samples to test (default: all 650 samples)",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        help="Save detailed results to JSON",
    ),
) -> None:
    """Benchmark Japanese Parakeet CTC on FluidInference/fleurs-full (650 Japanese samples)."""

    # Load models
    preprocessor, encoder, ctc_decoder = load_coreml_models(build_dir)

    # Load vocabulary
    vocab_path = build_dir / "vocab.json"
    vocabulary = load_vocabulary(vocab_path)
    blank_id = len(vocabulary)  # Blank is last token

    console.print(f"\nVocabulary: {len(vocabulary)} tokens (blank_id={blank_id})")

    # Load FLEURS dataset
    console.print(f"\nLoading FluidInference/fleurs-full Japanese (ja_jp) samples...")
    console.print("Note: First run will download ~650 Japanese audio files + transcriptions\n")

    # Load the dataset with Japanese data directory filter
    # This downloads only ja_jp folder instead of all 30 languages
    dataset = load_dataset(
        "FluidInference/fleurs-full",
        data_dir="ja_jp",
        split="train",
        trust_remote_code=True,
    )

    console.print(f"✓ Loaded {len(dataset)} Japanese audio samples")

    # Load transcriptions from ja_jp.trans.txt
    console.print("Loading transcriptions from ja_jp.trans.txt...")
    from huggingface_hub import hf_hub_download

    trans_file = hf_hub_download(
        repo_id="FluidInference/fleurs-full",
        filename="ja_jp/ja_jp.trans.txt",
        repo_type="dataset",
    )

    # Parse transcriptions (format: file_id transcription)
    transcriptions = {}
    with open(trans_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                file_id, text = parts
                transcriptions[file_id] = text

    console.print(f"✓ Loaded {len(transcriptions)} transcriptions")

    # Create samples with transcriptions
    japanese_samples = []
    for i in range(len(dataset)):
        audio_data = dataset[i]["audio"]
        file_id = f"ja_jp_{i:04d}"

        sample = {
            "audio": audio_data,
            "id": file_id,
            "transcription": transcriptions.get(file_id, ""),
        }
        japanese_samples.append(sample)

    # Limit samples if requested
    if num_samples is not None and num_samples < len(japanese_samples):
        japanese_samples = japanese_samples[:num_samples]
        console.print(f"Using first {num_samples} samples for benchmarking\n")
    else:
        console.print(f"Benchmarking all {len(japanese_samples)} Japanese samples\n")

    # Benchmark
    results = []
    total_audio_duration = 0.0
    total_latency = 0.0

    for idx, item in enumerate(track(japanese_samples, description="Processing audio")):
        # Extract audio
        audio_data = item["audio"]
        audio_array = np.array(audio_data["array"], dtype=np.float32)
        audio_sr = audio_data["sampling_rate"]

        # Get transcription and file ID
        reference = item["transcription"]
        file_id = item["id"]

        # Resample if needed
        if audio_sr != 16000:
            audio_array = librosa.resample(
                audio_array, orig_sr=audio_sr, target_sr=16000
            )

        audio_duration = len(audio_array) / 16000
        total_audio_duration += audio_duration

        # Run CoreML pipeline
        try:
            hypothesis, latency = coreml_full_pipeline(
                audio_array,
                16000,
                preprocessor,
                encoder,
                ctc_decoder,
                vocabulary,
                blank_id,
            )
        except Exception as e:
            console.print(f"[red]Error processing {file_id}: {e}[/red]")
            continue

        total_latency += latency

        # Compute CER
        cer = compute_cer(reference, hypothesis)

        rtfx = audio_duration / latency if latency > 0 else 0

        results.append({
            "id": file_id,
            "reference": reference,
            "hypothesis": hypothesis,
            "cer": cer,
            "audio_duration": audio_duration,
            "latency": latency,
            "rtfx": rtfx,
        })

    # Summary statistics
    if len(results) == 0:
        console.print("[red]No results to analyze[/red]")
        return

    avg_cer = np.mean([r["cer"] for r in results])
    avg_latency = np.mean([r["latency"] for r in results])
    avg_rtfx = np.mean([r["rtfx"] for r in results])
    total_rtfx = total_audio_duration / total_latency if total_latency > 0 else 0

    # Display results table
    table = Table(title="Benchmark Results - Parakeet CTC Japanese (FLEURS)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Samples", str(len(results)))
    table.add_row("Average CER", f"{avg_cer*100:.2f}%")
    table.add_row("Total Audio Duration", f"{total_audio_duration:.2f}s")
    table.add_row("Total Latency", f"{total_latency:.2f}s")
    table.add_row("Average Latency/Sample", f"{avg_latency*1000:.2f}ms")
    table.add_row("Average RTFx", f"{avg_rtfx:.2f}x")
    table.add_row("Total RTFx", f"{total_rtfx:.2f}x")

    console.print("\n")
    console.print(table)

    # Show some example predictions
    console.print("\n[bold]Example Predictions:[/bold]")
    for i in range(min(5, len(results))):
        r = results[i]
        console.print(f"\n[cyan]Sample {i+1}:[/cyan] {r['id']}")
        console.print(f"  [green]Reference:[/green]  {r['reference']}")
        console.print(f"  [yellow]Hypothesis:[/yellow] {r['hypothesis']}")
        console.print(f"  [magenta]CER:[/magenta] {r['cer']*100:.2f}%")

    # Save detailed results
    if output_file:
        output_data = {
            "model": "nvidia/parakeet-tdt_ctc-0.6b-ja",
            "dataset": "FluidInference/fleurs-full (ja_jp, 650 samples)",
            "num_samples": len(results),
            "summary": {
                "average_cer": avg_cer,
                "average_latency": avg_latency,
                "average_rtfx": avg_rtfx,
                "total_rtfx": total_rtfx,
                "total_audio_duration": total_audio_duration,
                "total_latency": total_latency,
            },
            "results": results,
        }

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        console.print(f"\n✓ Detailed results saved to {output_file}")


if __name__ == "__main__":
    app()
