#!/usr/bin/env python3
"""Benchmark full CoreML pipeline (Preprocessor + Encoder + CTC) on FLEURS Mandarin.

Tests the complete pure-CoreML ASR pipeline without PyTorch dependencies.
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

from text_normalizer import compute_cer_normalized
import editdistance


def compute_cer_raw(reference: str, hypothesis: str) -> float:
    """Compute raw CER without normalization."""
    ref_chars = reference.replace(' ', '')
    hyp_chars = hypothesis.replace(' ', '')

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    distance = editdistance.eval(ref_chars, hyp_chars)
    return distance / len(ref_chars)

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
console = Console()


def load_coreml_models(build_dir: Path) -> tuple[ct.models.MLModel, ct.models.MLModel, ct.models.MLModel]:
    """Load the three CoreML models."""
    console.print(f"Loading CoreML models from {build_dir}...")

    preprocessor = ct.models.MLModel(str(build_dir / "Preprocessor.mlpackage"))
    encoder = ct.models.MLModel(str(build_dir / "Encoder.mlpackage"))
    ctc_head = ct.models.MLModel(str(build_dir / "CtcHeadZhCn.mlpackage"))

    console.print("✓ All CoreML models loaded")
    return preprocessor, encoder, ctc_head


def load_vocabulary(vocab_path: Path) -> list[str]:
    """Load vocabulary as list (index -> token)."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_dict = json.load(f)

    # Convert dict to list (handle both formats)
    if isinstance(vocab_dict, list):
        return vocab_dict
    else:
        # Dict format: {index_str: token}
        max_idx = max(int(k) for k in vocab_dict.keys())
        vocab_list = [""] * (max_idx + 1)
        for idx_str, token in vocab_dict.items():
            vocab_list[int(idx_str)] = token
        return vocab_list


def greedy_ctc_decode(log_probs: np.ndarray, vocabulary: list[str], blank_id: int) -> str:
    """Greedy CTC decoding with blank collapse."""
    # log_probs: [1, time_steps, vocab_size]
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
    ctc_head: ct.models.MLModel,
    vocabulary: list[str],
    blank_id: int,
    target_time_steps: int = 188,
    max_audio_samples: int = 240000,  # 15 seconds at 16kHz
) -> tuple[str, float]:
    """Run full CoreML pipeline: audio -> text."""
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
        "audio_length": audio_length,
    })
    mel = preproc_out["mel"]
    mel_length = preproc_out["mel_length"]

    # Step 2: Encoder (mel -> encoder features)
    encoder_out = encoder.predict({
        "audio_signal": mel,
        "length": mel_length,
    })
    encoder_output = encoder_out["encoder_output"]  # [1, 1024, T]

    # Pad or truncate encoder output to fixed time steps
    current_time_steps = encoder_output.shape[2]
    if current_time_steps != target_time_steps:
        if current_time_steps > target_time_steps:
            encoder_output = encoder_output[:, :, :target_time_steps]
        else:
            pad_width = ((0, 0), (0, 0), (0, target_time_steps - current_time_steps))
            encoder_output = np.pad(encoder_output, pad_width, mode='constant', constant_values=0)

    # Step 3: CTC Head (encoder features -> logits)
    ctc_out = ctc_head.predict({
        "encoder_output": encoder_output,
    })
    ctc_logits = ctc_out["ctc_logits"]  # [1, 188, vocab_size]

    # Step 4: Greedy CTC decode
    hypothesis = greedy_ctc_decode(ctc_logits, vocabulary, blank_id)

    latency = time.time() - start_time
    return hypothesis, latency


@app.command()
def benchmark(
    build_dir: Path = typer.Option(
        Path("build-full"),
        "--build-dir",
        help="Directory containing CoreML models",
    ),
    num_samples: int = typer.Option(
        100,
        "--num-samples",
        help="Number of FLEURS samples to test",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output-file",
        help="Save detailed results to JSON",
    ),
) -> None:
    """Benchmark full CoreML pipeline on FLEURS Mandarin test set."""

    # Load models
    preprocessor, encoder, ctc_head = load_coreml_models(build_dir)

    # Load vocabulary
    vocab_path = build_dir / "vocab.json"
    vocabulary = load_vocabulary(vocab_path)
    blank_id = len(vocabulary)

    console.print(f"\nVocabulary: {len(vocabulary)} tokens (blank_id={blank_id})")

    # Load FLEURS dataset
    console.print(f"\nDownloading {num_samples} FLEURS Mandarin (cmn_hans_cn) test samples...")
    dataset = load_dataset(
        "google/fleurs",
        "cmn_hans_cn",
        split=f"test[:{num_samples}]",
        trust_remote_code=True,
    )
    console.print(f"✓ Downloaded {len(dataset)} samples\n")

    # Benchmark
    results = []
    total_audio_duration = 0.0

    for idx, item in enumerate(track(dataset, description="Processing audio")):
        audio_array = np.array(item["audio"]["array"], dtype=np.float32)
        audio_sr = item["audio"]["sampling_rate"]
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
                ctc_head,
                vocabulary,
                blank_id,
            )
        except Exception as e:
            console.print(f"[red]Error processing {file_id}: {e}[/red]")
            continue

        # Compute CER
        cer_raw = compute_cer_raw(reference, hypothesis)
        cer_norm = compute_cer_normalized(reference, hypothesis)

        rtfx = audio_duration / latency if latency > 0 else 0

        results.append({
            "id": file_id,
            "reference": reference,
            "hypothesis": hypothesis,
            "cer_raw": cer_raw,
            "cer_normalized": cer_norm,
            "audio_duration": audio_duration,
            "latency": latency,
            "rtfx": rtfx,
        })

    # ========== Summary Statistics ==========
    if not results:
        console.print("[red]No results to report[/red]")
        return

    cer_raw_values = [r["cer_raw"] for r in results]
    cer_norm_values = [r["cer_normalized"] for r in results]
    latencies = [r["latency"] for r in results]
    rtfx_values = [r["rtfx"] for r in results]

    mean_cer_raw = np.mean(cer_raw_values) * 100
    median_cer_raw = np.median(cer_raw_values) * 100
    mean_cer_norm = np.mean(cer_norm_values) * 100
    median_cer_norm = np.median(cer_norm_values) * 100
    std_cer_norm = np.std(cer_norm_values) * 100

    mean_latency = np.mean(latencies) * 1000  # ms
    mean_rtfx = np.mean(rtfx_values)
    median_rtfx = np.median(rtfx_values)

    # CER distribution
    cer_bins = [0, 0.05, 0.10, 0.20, 0.30, float('inf')]
    cer_counts = [
        sum(1 for c in cer_norm_values if cer_bins[i] <= c < cer_bins[i+1])
        for i in range(len(cer_bins) - 1)
    ]

    # Print summary
    console.print("\n" + "="*60)
    console.print("═══ Full CoreML Pipeline Benchmark ═══\n")

    table = Table(title="Results Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Files Processed", str(len(results)))
    table.add_row("Total Audio Duration", f"{total_audio_duration:.1f}s")
    table.add_row("", "")
    table.add_row("CER (Raw - with formatting diffs)", "")
    table.add_row("  Mean CER", f"{mean_cer_raw:.2f}%")
    table.add_row("  Median CER", f"{median_cer_raw:.2f}%")
    table.add_row("", "")
    table.add_row("CER (Normalized - fair comparison)", "")
    table.add_row("  Mean CER", f"{mean_cer_norm:.2f}%")
    table.add_row("  Median CER", f"{median_cer_norm:.2f}%")
    table.add_row("  Std Dev", f"{std_cer_norm:.2f}%")
    table.add_row("", "")
    table.add_row("Performance", "")
    table.add_row("  Mean Latency (full pipeline)", f"{mean_latency:.1f}ms")
    table.add_row("  Mean RTFx", f"{mean_rtfx:.2f}x")
    table.add_row("  Median RTFx", f"{median_rtfx:.2f}x")

    console.print(table)

    # CER distribution
    console.print("\nCER Distribution (Normalized):")
    labels = ["<5%", "5-10%", "10-20%", "20-30%", ">30%"]
    for label, count in zip(labels, cer_counts):
        pct = (count / len(results)) * 100
        bar = "█" * int(pct / 2)
        console.print(f"  {label:8} {bar:30} {count:4} ({pct:5.1f}%)")

    # Top worst cases
    worst_cases = sorted(results, key=lambda x: x["cer_normalized"], reverse=True)[:10]
    console.print("\nTop 10 Worst Cases (Normalized CER):")
    worst_table = Table(show_header=True)
    worst_table.add_column("ID", style="cyan", width=15)
    worst_table.add_column("CER", style="red", width=6)
    worst_table.add_column("Reference", style="yellow", width=30)
    worst_table.add_column("Hypothesis", style="green", width=30)

    for case in worst_cases:
        ref_trunc = case["reference"][:27] + "..." if len(case["reference"]) > 30 else case["reference"]
        hyp_trunc = case["hypothesis"][:27] + "..." if len(case["hypothesis"]) > 30 else case["hypothesis"]
        file_id = str(case["id"])[:12] + "…" if len(str(case["id"])) > 12 else str(case["id"])
        worst_table.add_row(
            file_id,
            f"{case['cer_normalized']*100:.1f}%",
            ref_trunc,
            hyp_trunc,
        )

    console.print(worst_table)

    # Save detailed results
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "summary": {
                        "files_processed": len(results),
                        "total_audio_duration": total_audio_duration,
                        "mean_cer_raw": mean_cer_raw,
                        "median_cer_raw": median_cer_raw,
                        "mean_cer_normalized": mean_cer_norm,
                        "median_cer_normalized": median_cer_norm,
                        "std_cer_normalized": std_cer_norm,
                        "mean_latency_ms": mean_latency,
                        "mean_rtfx": mean_rtfx,
                        "median_rtfx": median_rtfx,
                    },
                    "results": results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        console.print(f"\n✓ Detailed results saved to {output_file}")


if __name__ == "__main__":
    app()
