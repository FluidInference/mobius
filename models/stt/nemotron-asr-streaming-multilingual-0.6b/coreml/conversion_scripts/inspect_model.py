#!/usr/bin/env python3
"""Discovery helper for Nemotron-3.5-ASR-Streaming-Multilingual 0.6B.

Loads the .nemo file via the kingformatty/NeMo fork and prints:
- Top-level model attribute names
- Signature of `model.forward`, `model.encoder.forward`, and any
  `prompt_kernel` / `prompt_encoder` / `apply_prompt` callables found
- Where the language prompt is consumed in `EncDecRNNTBPEModelWithPrompt`
- A dry-run forward with a tiny mel + a fixed `prompt_id`

Used to confirm the correct wrapper API before exporting CoreML.

Run:
    uv run python inspect_model.py --nemo-path /path/to/multilingual.nemo
"""
from __future__ import annotations

import inspect as stdlib_inspect
from pathlib import Path

import torch
import typer

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def _print_section(title: str) -> None:
    typer.echo("\n" + "=" * 70)
    typer.echo(title)
    typer.echo("=" * 70)


def _print_signature(label: str, fn) -> None:
    try:
        sig = stdlib_inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        typer.echo(f"  {label}: <unavailable: {exc}>")
        return
    typer.echo(f"  {label}{sig}")


@app.command()
def inspect(
    nemo_path: Path = typer.Option(
        ...,
        "--nemo-path",
        help="Path to nemotron-asr-streaming-multilingual-0.6b.nemo",
    ),
    target_lang: str = typer.Option("en-US", "--target-lang"),
) -> None:
    """Print model structure to pick the right conversion wrapper."""

    import nemo.collections.asr as nemo_asr

    _print_section("Loading model")
    typer.echo(f"  path: {nemo_path}")
    # restore_from picks the class from the bundled config's `target:` field
    model = nemo_asr.models.ASRModel.restore_from(
        restore_path=str(nemo_path), map_location="cpu"
    )
    model.eval()
    typer.echo(f"  class: {type(model).__module__}.{type(model).__name__}")

    _print_section("Top-level child modules")
    for name, child in model.named_children():
        typer.echo(f"  {name:20s} {type(child).__name__}")

    _print_section("Key signatures")
    _print_signature("model.forward", model.forward)
    _print_signature("model.encoder.forward", model.encoder.forward)
    if hasattr(model, "preprocessor"):
        _print_signature("model.preprocessor.forward", model.preprocessor.forward)
    if hasattr(model, "prompt_kernel"):
        _print_signature("model.prompt_kernel.forward", model.prompt_kernel.forward)
        typer.echo(f"  prompt_kernel module: {model.prompt_kernel}")
    for hook_name in (
        "apply_prompt",
        "compute_prompt",
        "prompt_encoder",
        "encode_prompt",
        "_get_prompt_tensor",
    ):
        if hasattr(model, hook_name):
            _print_signature(f"model.{hook_name}", getattr(model, hook_name))

    _print_section("Prompt dictionary (subset)")
    pd = getattr(model.cfg, "prompt_dictionary", None) or getattr(
        getattr(model.cfg, "model_defaults", None), "prompt_dictionary", None
    )
    if pd is None:
        typer.echo("  <not found on model.cfg>")
    else:
        as_dict = dict(pd)
        typer.echo(f"  size: {len(as_dict)}")
        for k in sorted(as_dict.keys())[:8]:
            typer.echo(f"  {k} -> {as_dict[k]}")
        if target_lang in as_dict:
            typer.echo(f"\n  target_lang={target_lang!r} -> {as_dict[target_lang]}")
        else:
            typer.echo(f"\n  target_lang={target_lang!r} NOT FOUND")

    _print_section("Tokenizer")
    tok = getattr(model, "tokenizer", None)
    if tok is None:
        typer.echo("  <no tokenizer>")
    else:
        typer.echo(f"  class: {type(tok).__name__}")
        typer.echo(f"  vocab_size: {tok.vocab_size}")
        first_ids = list(range(min(45, tok.vocab_size)))
        pieces = [tok.ids_to_tokens([i])[0] for i in first_ids]
        for i, p in zip(first_ids, pieces):
            typer.echo(f"    {i:5d} {p!r}")

    _print_section("Source: model.forward")
    try:
        typer.echo(stdlib_inspect.getsource(type(model).forward))
    except (OSError, TypeError):
        typer.echo("  <source unavailable>")

    _print_section("Source: model.encoder.forward")
    try:
        typer.echo(stdlib_inspect.getsource(type(model.encoder).forward))
    except (OSError, TypeError):
        typer.echo("  <source unavailable>")

    _print_section("Dry-run inference")
    sample_rate = int(model.cfg.preprocessor.sample_rate)
    audio = torch.zeros(1, sample_rate)  # 1s of silence
    audio_len = torch.tensor([sample_rate], dtype=torch.long)
    try:
        model.encoder.setup_streaming_params()
    except Exception as exc:
        typer.echo(f"  setup_streaming_params(): {exc!r}")

    with torch.no_grad():
        mel, mel_len = model.preprocessor(input_signal=audio, length=audio_len)
        typer.echo(f"  mel shape: {tuple(mel.shape)}  mel_len: {mel_len.tolist()}")

        # Try the most likely high-level API first
        try:
            out = model.transcribe(
                audio=[audio.squeeze(0).numpy()], target_lang=target_lang
            )
            typer.echo(f"  model.transcribe(target_lang={target_lang}): {out!r}")
        except Exception as exc:
            typer.echo(f"  model.transcribe(): {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    app()
