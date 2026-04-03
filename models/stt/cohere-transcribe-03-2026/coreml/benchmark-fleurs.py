#!/usr/bin/env python3
"""Benchmark Cohere Transcribe CoreML on FLEURS multilingual dataset.

Evaluates CoreML conversion quality by comparing WER (Word Error Rate) between
PyTorch and CoreML implementations across multiple languages.

Usage:
  # Test all supported languages with 100 samples each
  uv run python benchmark-fleurs.py --languages all --samples 100

  # Test specific languages
  uv run python benchmark-fleurs.py --languages en_us,fr_fr,de_de --samples 50

  # PyTorch only (baseline)
  uv run python benchmark-fleurs.py --pytorch-only --languages en_us --samples 10

  # CoreML only (fast validation)
  uv run python benchmark-fleurs.py --coreml-only --languages en_us --samples 10
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import coremltools as ct
import numpy as np
import torch
import typer
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

DEFAULT_MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
DATASET_ID = "google/fleurs"
SAMPLE_RATE = 16000

# Map FLEURS language codes to Cohere language codes
LANGUAGE_MAP = {
    "en_us": "en",              # English
    "fr_fr": "fr",              # French
    "de_de": "de",              # German
    "es_419": "es",             # Spanish
    "it_it": "it",              # Italian
    "pt_br": "pt",              # Portuguese
    "nl_nl": "nl",              # Dutch
    "pl_pl": "pl",              # Polish
    "el_gr": "el",              # Greek
    "ar_eg": "ar",              # Arabic
    "ja_jp": "ja",              # Japanese
    "cmn_hans_cn": "zh",        # Chinese (Mandarin Simplified) - Note: FLEURS uses cmn_hans_cn
    "vi_vn": "vi",              # Vietnamese
    "ko_kr": "ko",              # Korean
}


@dataclass
class BenchmarkResult:
    language: str
    samples_processed: int
    wer: float
    cer: float
    rtfx: float
    total_duration: float
    processing_time: float
    avg_latency: float


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    # Levenshtein distance
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]

    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1]) + 1

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate."""
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0

    # Levenshtein distance on characters
    d = [[0] * (len(hypothesis) + 1) for _ in range(len(reference) + 1)]

    for i in range(len(reference) + 1):
        d[i][0] = i
    for j in range(len(hypothesis) + 1):
        d[0][j] = j

    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1]) + 1

    return d[len(reference)][len(hypothesis)] / len(reference)


def normalize_text(text: str, language: str) -> str:
    """Normalize text for comparison."""
    import unicodedata

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = "".join(c for c in text if c.isalnum() or c.isspace())

    # Normalize unicode
    text = unicodedata.normalize("NFKD", text)

    # Collapse whitespace
    text = " ".join(text.split())

    return text


def benchmark_pytorch(
    model,
    processor,
    dataset,
    language: str,
    max_samples: int,
    typer_echo,
) -> BenchmarkResult:
    """Benchmark PyTorch model on FLEURS dataset."""
    typer_echo(f"\n[PyTorch] Benchmarking {language}...")

    total_wer = 0.0
    total_cer = 0.0
    total_duration = 0.0
    total_processing_time = 0.0
    samples_processed = 0

    cohere_lang = LANGUAGE_MAP.get(language, language.split("_")[0])

    for i, sample in enumerate(dataset):
        if i >= max_samples:
            break

        try:
            audio = sample["audio"]["array"]
            reference = sample["transcription"]
            duration = len(audio) / SAMPLE_RATE

            # Process
            start = time.time()
            inputs = processor(
                audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
            )

            # Initialize decoder with start token (required for generation)
            batch_size = inputs["input_features"].shape[0]
            decoder_input_ids = torch.full(
                (batch_size, 1),
                model.generation_config.decoder_start_token_id,
                dtype=torch.long
            )

            with torch.no_grad():
                # Generate transcription
                outputs = model.generate(
                    input_features=inputs["input_features"],
                    length=inputs.get("length"),
                    decoder_input_ids=decoder_input_ids,
                    max_new_tokens=512,
                )
                hypothesis = processor.batch_decode(outputs, skip_special_tokens=True)[0]

            processing_time = time.time() - start

            # Normalize and compute metrics
            ref_norm = normalize_text(reference, language)
            hyp_norm = normalize_text(hypothesis, language)

            wer = compute_wer(ref_norm, hyp_norm)
            cer = compute_cer(ref_norm, hyp_norm)

            total_wer += wer
            total_cer += cer
            total_duration += duration
            total_processing_time += processing_time
            samples_processed += 1

            if (i + 1) % 10 == 0:
                typer_echo(f"  Processed {i + 1}/{max_samples} samples...")

        except Exception as e:
            typer_echo(f"  Error processing sample {i}: {e}")
            continue

    avg_wer = total_wer / samples_processed if samples_processed > 0 else 0.0
    avg_cer = total_cer / samples_processed if samples_processed > 0 else 0.0
    rtfx = total_duration / total_processing_time if total_processing_time > 0 else 0.0
    avg_latency = total_processing_time / samples_processed if samples_processed > 0 else 0.0

    return BenchmarkResult(
        language=language,
        samples_processed=samples_processed,
        wer=avg_wer * 100,  # Convert to percentage
        cer=avg_cer * 100,
        rtfx=rtfx,
        total_duration=total_duration,
        processing_time=total_processing_time,
        avg_latency=avg_latency,
    )


def benchmark_coreml(
    coreml_encoder,
    coreml_decoder,
    coreml_lm_head,
    processor,
    dataset,
    language: str,
    max_samples: int,
    typer_echo,
) -> BenchmarkResult:
    """Benchmark CoreML model on FLEURS dataset."""
    typer_echo(f"\n[CoreML] Benchmarking {language}...")
    typer_echo("  Note: CoreML benchmark requires full decoder implementation")
    typer_echo("  Skipping for now - encoder-only validation in compare-models.py")

    # TODO: Implement full CoreML inference pipeline with decoder and beam search
    # For now, we only validate encoder outputs in compare-models.py

    return BenchmarkResult(
        language=language,
        samples_processed=0,
        wer=0.0,
        cer=0.0,
        rtfx=0.0,
        total_duration=0.0,
        processing_time=0.0,
        avg_latency=0.0,
    )


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def benchmark(
    languages: str = typer.Option(
        "en_us",
        "--languages",
        help="Comma-separated language codes or 'all' for all supported languages",
    ),
    samples: int = typer.Option(
        100,
        "--samples",
        help="Number of samples per language",
    ),
    model_id: str = typer.Option(
        DEFAULT_MODEL_ID,
        "--model-id",
        help="HuggingFace model ID",
    ),
    coreml_dir: Optional[Path] = typer.Option(
        None,
        "--coreml-dir",
        help="Directory containing CoreML packages (optional)",
    ),
    output_file: str = typer.Option(
        "fleurs_benchmark_results.json",
        "--output",
        help="Output JSON file for results",
    ),
    pytorch_only: bool = typer.Option(
        False,
        "--pytorch-only",
        help="Only benchmark PyTorch (baseline)",
    ),
    coreml_only: bool = typer.Option(
        False,
        "--coreml-only",
        help="Only benchmark CoreML",
    ),
) -> None:
    """Benchmark Cohere Transcribe on FLEURS multilingual dataset."""

    typer.echo("=" * 80)
    typer.echo("Cohere Transcribe FLEURS Benchmark")
    typer.echo("=" * 80)

    # Parse languages
    if languages == "all":
        selected_languages = list(LANGUAGE_MAP.keys())
    else:
        selected_languages = [lang.strip() for lang in languages.split(",")]

    typer.echo(f"\nLanguages: {', '.join(selected_languages)}")
    typer.echo(f"Samples per language: {samples}")
    typer.echo(f"Model: {model_id}")

    # Load PyTorch model if needed
    pytorch_model = None
    processor = None

    if not coreml_only:
        typer.echo(f"\nLoading PyTorch model: {model_id}")
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        pytorch_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        pytorch_model.eval()
        typer.echo("  ✓ PyTorch model loaded")

    # Load CoreML models if needed
    coreml_encoder = None
    coreml_decoder = None
    coreml_lm_head = None

    if not pytorch_only and coreml_dir:
        typer.echo(f"\nLoading CoreML models from: {coreml_dir}")
        try:
            coreml_encoder = ct.models.MLModel(str(coreml_dir / "cohere_audio_encoder.mlpackage"))
            coreml_decoder = ct.models.MLModel(str(coreml_dir / "cohere_decoder.mlpackage"))
            coreml_lm_head = ct.models.MLModel(str(coreml_dir / "cohere_lm_head.mlpackage"))
            typer.echo("  ✓ CoreML models loaded")
        except Exception as e:
            typer.echo(f"  ✗ Failed to load CoreML models: {e}")
            coreml_encoder = None

    # Run benchmarks
    results = {}

    for language in selected_languages:
        typer.echo(f"\n{'=' * 80}")
        typer.echo(f"Language: {language.upper()} ({LANGUAGE_MAP.get(language, language)})")
        typer.echo(f"{'=' * 80}")

        # Load dataset for this language
        typer.echo(f"Loading FLEURS dataset for {language}...")
        try:
            dataset = load_dataset(
                DATASET_ID,
                language,  # Language code as config name
                split="test",
                streaming=False,
            )
            typer.echo(f"  ✓ Loaded {len(dataset)} test samples")
        except Exception as e:
            typer.echo(f"  ✗ Failed to load dataset: {e}")
            continue

        lang_results = {}

        # PyTorch benchmark
        if not coreml_only and pytorch_model:
            pytorch_result = benchmark_pytorch(
                pytorch_model,
                processor,
                dataset,
                language,
                samples,
                typer.echo,
            )
            lang_results["pytorch"] = asdict(pytorch_result)

            typer.echo(f"\n  PyTorch Results:")
            typer.echo(f"    WER: {pytorch_result.wer:.2f}%")
            typer.echo(f"    CER: {pytorch_result.cer:.2f}%")
            typer.echo(f"    RTFx: {pytorch_result.rtfx:.2f}x")
            typer.echo(f"    Avg Latency: {pytorch_result.avg_latency:.3f}s")

        # CoreML benchmark
        if not pytorch_only and coreml_encoder:
            coreml_result = benchmark_coreml(
                coreml_encoder,
                coreml_decoder,
                coreml_lm_head,
                processor,
                dataset,
                language,
                samples,
                typer.echo,
            )
            lang_results["coreml"] = asdict(coreml_result)

        results[language] = lang_results

    # Save results
    typer.echo(f"\n{'=' * 80}")
    typer.echo("Summary")
    typer.echo(f"{'=' * 80}")

    output_path = Path(output_file)
    with open(output_path, "w") as f:
        json.dump(
            {
                "model_id": model_id,
                "samples_per_language": samples,
                "languages": selected_languages,
                "results": results,
            },
            f,
            indent=2,
        )

    typer.echo(f"\nResults saved to: {output_path}")

    # Print summary table
    typer.echo("\nLanguage Results:")
    typer.echo(f"{'Language':<12} {'WER (PyTorch)':<15} {'WER (CoreML)':<15} {'Samples':<10}")
    typer.echo("-" * 60)

    for language in selected_languages:
        if language in results:
            pytorch_wer = results[language].get("pytorch", {}).get("wer", 0.0)
            coreml_wer = results[language].get("coreml", {}).get("wer", 0.0)
            samples_count = results[language].get("pytorch", {}).get("samples_processed", 0)

            pytorch_str = f"{pytorch_wer:.2f}%" if pytorch_wer > 0 else "N/A"
            coreml_str = f"{coreml_wer:.2f}%" if coreml_wer > 0 else "N/A"

            typer.echo(f"{language:<12} {pytorch_str:<15} {coreml_str:<15} {samples_count:<10}")


if __name__ == "__main__":
    app()
