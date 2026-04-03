#!/usr/bin/env python3
"""Benchmark zh-CN CTC model on AISHELL-1 test set with CER evaluation.

Downloads AISHELL-1 test samples and measures Character Error Rate (CER) for both
NeMo and CoreML implementations.

Usage:
    uv run python validation/benchmark-cer.py --num-files 100 --nemo-path ../parakeet-ctc-riva-0-6b-unified-zh-cn_vtrainable_v3.0/*.nemo
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import coremltools as ct
import numpy as np
import soundfile as sf
import torch
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

import nemo.collections.asr as nemo_asr

# Add parent directory to path to import from benchmark/
sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmark.text_normalizer import compute_cer_normalized

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
console = Console()


@dataclass
class BenchmarkResult:
    sample_id: str
    reference: str
    nemo_hypothesis: str
    coreml_hypothesis: str
    nemo_cer: float
    coreml_cer: float
    nemo_cer_norm: float  # Normalized CER (punctuation/numbers handled)
    coreml_cer_norm: float
    nemo_time: float
    coreml_time: float
    audio_duration: float
    nemo_rtfx: float
    coreml_rtfx: float


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate."""
    import editdistance

    # Remove spaces for character-level comparison
    ref_chars = reference.replace(" ", "")
    hyp_chars = hypothesis.replace(" ", "")

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    distance = editdistance.eval(ref_chars, hyp_chars)
    return distance / len(ref_chars)


def decode_ctc_greedy(log_probs: np.ndarray, vocab: list[str], blank_id: int) -> str:
    """Greedy CTC decoding."""
    best_path = np.argmax(log_probs, axis=-1)

    collapsed = []
    prev = None
    for token_id in best_path:
        if token_id != prev and token_id != blank_id:
            collapsed.append(int(token_id))
        prev = token_id

    tokens = [vocab[i] for i in collapsed]
    text = "".join(tokens).replace("▁", " ").strip()
    return text


def pad_or_truncate_encoder_output(encoder_output: np.ndarray, target_time_steps: int) -> np.ndarray:
    """Pad or truncate encoder output to match CoreML model's expected input shape.

    Args:
        encoder_output: Shape [1, encoder_dim, time_steps]
        target_time_steps: Target time dimension (e.g., 188)

    Returns:
        Padded/truncated array of shape [1, encoder_dim, target_time_steps]
    """
    current_time_steps = encoder_output.shape[2]

    if current_time_steps == target_time_steps:
        return encoder_output
    elif current_time_steps > target_time_steps:
        # Truncate
        return encoder_output[:, :, :target_time_steps]
    else:
        # Pad with zeros
        pad_width = ((0, 0), (0, 0), (0, target_time_steps - current_time_steps))
        return np.pad(encoder_output, pad_width, mode='constant', constant_values=0)



def download_fleurs_mandarin(num_files: int, cache_dir: Path) -> List[Tuple[str, str, str]]:
    """Download FLEURS Mandarin Chinese test samples.

    Returns: List of (audio_path, transcript, sample_id)
    """
    from datasets import load_dataset

    console.print(f"[cyan]Downloading {num_files} FLEURS Mandarin (cmn_hans_cn) test samples...[/cyan]")

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Use Google FLEURS dataset with Mandarin Chinese (Simplified)
    dataset = load_dataset("google/fleurs", "cmn_hans_cn", split="test", streaming=True, trust_remote_code=True)

    samples = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading audio...", total=num_files)

        for idx, sample in enumerate(dataset):
            if idx >= num_files:
                break

            # FLEURS format: sample["audio"]["array"], sample["audio"]["sampling_rate"], sample["transcription"]
            audio = sample["audio"]["array"]
            sr = sample["audio"]["sampling_rate"]
            transcript = sample["transcription"]

            # Save audio
            audio_path = cache_dir / f"fleurs_cmn_{idx:03d}.wav"
            sf.write(str(audio_path), audio, sr)

            samples.append((str(audio_path), transcript, f"fleurs_cmn_{idx:03d}"))
            progress.update(task, advance=1)

    console.print(f"[green]✓[/green] Downloaded {len(samples)} samples")
    return samples


@app.command()
def benchmark(
    num_files: int = typer.Option(
        100,
        "--num-files",
        help="Number of test files to process",
    ),
    nemo_path: Path = typer.Option(
        ...,
        "--nemo-path",
        exists=True,
        resolve_path=True,
        help="Path to zh-CN .nemo checkpoint",
    ),
    coreml_dir: Path = typer.Option(
        Path("build"),
        help="Directory containing CoreML model",
    ),
    cache_dir: Path = typer.Option(
        Path("test_audio_100"),
        help="Directory to cache downloaded audio",
    ),
    output_file: Path = typer.Option(
        Path("benchmark_results.json"),
        help="Output JSON file for detailed results",
    ),
) -> None:
    """Run CER benchmark on AISHELL-1 test set."""

    console.print("\n[bold blue]═══ zh-CN Parakeet CTC Benchmark ═══[/bold blue]\n")

    # Load vocabulary and metadata
    vocab_path = coreml_dir / "vocab.json"
    vocab = json.loads(vocab_path.read_text())
    blank_id = len(vocab)
    console.print(f"Vocabulary: {len(vocab)} tokens + blank\n")

    # Load CoreML model metadata for expected input shape
    metadata_path = coreml_dir / "ctc_head_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    expected_time_steps = metadata["time_steps"]
    console.print(f"CoreML model expects time_steps: {expected_time_steps}\n")

    # Load NeMo model
    console.print("[cyan]Loading NeMo model...[/cyan]")
    asr_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(
        str(nemo_path), map_location="cpu"
    )
    asr_model.eval()
    console.print("[green]✓[/green] NeMo model loaded\n")

    # Load CoreML model
    console.print("[cyan]Loading CoreML model...[/cyan]")
    mlmodel_path = coreml_dir / "CtcHeadZhCn.mlpackage"
    mlmodel = ct.models.MLModel(str(mlmodel_path), compute_units=ct.ComputeUnit.ALL)
    console.print("[green]✓[/green] CoreML model loaded\n")

    # Download test samples
    samples = download_fleurs_mandarin(num_files, cache_dir)
    console.print()

    # Run benchmark
    results: List[BenchmarkResult] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing audio...", total=len(samples))

        for audio_path, reference, sample_id in samples:
            # Load audio
            audio, sr = sf.read(audio_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            audio_duration = len(audio) / sr

            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            audio_length = torch.tensor([audio.shape[0]], dtype=torch.int32)

            # NeMo inference
            nemo_start = time.perf_counter()
            with torch.inference_mode():
                mel, mel_length = asr_model.preprocessor(
                    input_signal=audio_tensor, length=audio_length.long()
                )
                encoded, encoded_length = asr_model.encoder(
                    audio_signal=mel, length=mel_length.long()
                )
                ctc_logits_nemo = asr_model.ctc_decoder(encoder_output=encoded)

            nemo_log_probs = torch.nn.functional.log_softmax(
                ctc_logits_nemo[0], dim=-1
            ).numpy()
            nemo_hypothesis = decode_ctc_greedy(nemo_log_probs, vocab, blank_id)
            nemo_time = time.perf_counter() - nemo_start

            # CoreML inference (decoder only)
            coreml_start = time.perf_counter()
            encoder_output_np = encoded.numpy()

            # Pad or truncate to match CoreML model's expected input shape
            encoder_output_fixed = pad_or_truncate_encoder_output(encoder_output_np, expected_time_steps)

            coreml_output = mlmodel.predict({"encoder_output": encoder_output_fixed})
            ctc_logits_coreml = coreml_output["ctc_logits"]

            coreml_log_probs = torch.nn.functional.log_softmax(
                torch.from_numpy(ctc_logits_coreml[0]), dim=-1
            ).numpy()
            coreml_hypothesis = decode_ctc_greedy(coreml_log_probs, vocab, blank_id)
            coreml_time = time.perf_counter() - coreml_start

            # Compute CER (raw and normalized)
            nemo_cer = compute_cer(reference, nemo_hypothesis)
            coreml_cer = compute_cer(reference, coreml_hypothesis)
            nemo_cer_norm = compute_cer_normalized(reference, nemo_hypothesis)
            coreml_cer_norm = compute_cer_normalized(reference, coreml_hypothesis)

            # RTFx (Real-Time Factor)
            nemo_rtfx = audio_duration / nemo_time
            coreml_rtfx = audio_duration / coreml_time

            results.append(
                BenchmarkResult(
                    sample_id=sample_id,
                    reference=reference,
                    nemo_hypothesis=nemo_hypothesis,
                    coreml_hypothesis=coreml_hypothesis,
                    nemo_cer=nemo_cer,
                    coreml_cer=coreml_cer,
                    nemo_cer_norm=nemo_cer_norm,
                    coreml_cer_norm=coreml_cer_norm,
                    nemo_time=nemo_time,
                    coreml_time=coreml_time,
                    audio_duration=audio_duration,
                    nemo_rtfx=nemo_rtfx,
                    coreml_rtfx=coreml_rtfx,
                )
            )

            progress.update(task, advance=1)

    # Compute statistics
    console.print("\n[bold green]═══ Results ═══[/bold green]\n")

    nemo_cers = [r.nemo_cer for r in results]
    coreml_cers = [r.coreml_cer for r in results]
    nemo_cers_norm = [r.nemo_cer_norm for r in results]
    coreml_cers_norm = [r.coreml_cer_norm for r in results]
    nemo_rtfxs = [r.nemo_rtfx for r in results]
    coreml_rtfxs = [r.coreml_rtfx for r in results]

    total_duration = sum(r.audio_duration for r in results)

    # Summary table
    table = Table(title="Benchmark Summary", show_header=True)
    table.add_column("Metric", style="cyan", justify="left")
    table.add_column("NeMo", style="green", justify="right")
    table.add_column("CoreML", style="magenta", justify="right")

    table.add_row("Files Processed", f"{len(results)}", f"{len(results)}")
    table.add_row("Total Audio Duration", f"{total_duration:.1f}s", f"{total_duration:.1f}s")
    table.add_row("", "", "")
    table.add_row("[bold]CER (Raw - with punctuation/format diffs)[/bold]", "", "")
    table.add_row("  Mean CER", f"{np.mean(nemo_cers)*100:.2f}%", f"{np.mean(coreml_cers)*100:.2f}%")
    table.add_row("  Median CER", f"{np.median(nemo_cers)*100:.2f}%", f"{np.median(coreml_cers)*100:.2f}%")
    table.add_row("", "", "")
    table.add_row("[bold]CER (Normalized - fair comparison)[/bold]", "", "")
    table.add_row("  Mean CER", f"{np.mean(nemo_cers_norm)*100:.2f}%", f"{np.mean(coreml_cers_norm)*100:.2f}%")
    table.add_row("  Median CER", f"{np.median(nemo_cers_norm)*100:.2f}%", f"{np.median(coreml_cers_norm)*100:.2f}%")
    table.add_row("  Std Dev", f"{np.std(nemo_cers_norm)*100:.2f}%", f"{np.std(coreml_cers_norm)*100:.2f}%")
    table.add_row("", "", "")
    table.add_row("[bold]Performance[/bold]", "", "")
    table.add_row("  Mean RTFx", f"{np.mean(nemo_rtfxs):.2f}x", f"{np.mean(coreml_rtfxs):.2f}x")
    table.add_row("  Median RTFx", f"{np.median(nemo_rtfxs):.2f}x", f"{np.median(coreml_rtfxs):.2f}x")
    table.add_row("  Mean Latency", f"{np.mean([r.nemo_time for r in results])*1000:.1f}ms", f"{np.mean([r.coreml_time for r in results])*1000:.1f}ms")

    console.print(table)

    # CER distribution (normalized)
    console.print("\n[bold cyan]CER Distribution (CoreML Normalized):[/bold cyan]")
    cer_bins = [
        ("<5%", sum(1 for c in coreml_cers_norm if c < 0.05)),
        ("5-10%", sum(1 for c in coreml_cers_norm if 0.05 <= c < 0.10)),
        ("10-20%", sum(1 for c in coreml_cers_norm if 0.10 <= c < 0.20)),
        ("20-30%", sum(1 for c in coreml_cers_norm if 0.20 <= c < 0.30)),
        (">30%", sum(1 for c in coreml_cers_norm if c >= 0.30)),
    ]
    for label, count in cer_bins:
        pct = count / len(results) * 100
        bar = "█" * int(pct / 2)
        console.print(f"  {label:8s} {bar:40s} {count:3d} ({pct:5.1f}%)")

    # Worst cases (by normalized CER)
    console.print("\n[bold yellow]Top 10 Worst Cases (CoreML Normalized CER):[/bold yellow]")
    worst_cases = sorted(results, key=lambda r: r.coreml_cer_norm, reverse=True)[:10]

    worst_table = Table(show_header=True)
    worst_table.add_column("ID", style="cyan", width=15)
    worst_table.add_column("CER", style="red", justify="right", width=8)
    worst_table.add_column("Reference", style="white", width=30)
    worst_table.add_column("Hypothesis", style="yellow", width=30)

    for r in worst_cases:
        ref_short = r.reference[:30] + "..." if len(r.reference) > 30 else r.reference
        hyp_short = r.coreml_hypothesis[:30] + "..." if len(r.coreml_hypothesis) > 30 else r.coreml_hypothesis
        worst_table.add_row(
            r.sample_id,
            f"{r.coreml_cer_norm*100:.1f}%",
            ref_short,
            hyp_short,
        )

    console.print(worst_table)

    # Save detailed results
    output_data = {
        "summary": {
            "num_files": len(results),
            "total_duration": total_duration,
            "nemo": {
                "mean_cer": float(np.mean(nemo_cers)),
                "median_cer": float(np.median(nemo_cers)),
                "std_cer": float(np.std(nemo_cers)),
                "mean_cer_norm": float(np.mean(nemo_cers_norm)),
                "median_cer_norm": float(np.median(nemo_cers_norm)),
                "std_cer_norm": float(np.std(nemo_cers_norm)),
                "mean_rtfx": float(np.mean(nemo_rtfxs)),
            },
            "coreml": {
                "mean_cer": float(np.mean(coreml_cers)),
                "median_cer": float(np.median(coreml_cers)),
                "std_cer": float(np.std(coreml_cers)),
                "mean_cer_norm": float(np.mean(coreml_cers_norm)),
                "median_cer_norm": float(np.median(coreml_cers_norm)),
                "std_cer_norm": float(np.std(coreml_cers_norm)),
                "mean_rtfx": float(np.mean(coreml_rtfxs)),
            },
        },
        "results": [
            {
                "sample_id": r.sample_id,
                "reference": r.reference,
                "nemo_hypothesis": r.nemo_hypothesis,
                "coreml_hypothesis": r.coreml_hypothesis,
                "nemo_cer": r.nemo_cer,
                "coreml_cer": r.coreml_cer,
                "nemo_cer_norm": r.nemo_cer_norm,
                "coreml_cer_norm": r.coreml_cer_norm,
                "nemo_time": r.nemo_time,
                "coreml_time": r.coreml_time,
                "audio_duration": r.audio_duration,
            }
            for r in results
        ],
    }

    output_file.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
    console.print(f"\n[green]✓[/green] Detailed results saved to {output_file}")


if __name__ == "__main__":
    app()
