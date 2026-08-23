#!/usr/bin/env python3
"""Slice the 44 GB VoiceChat-11B fp32 checkpoint into per-component safetensors.

Components (see ../README.md):
  encoder.safetensors  stt_model.perception.*                 fp32   (~2.5 GB)
  rnnt.safetensors     stt_model.rnnt_*                       fp32   (~50 MB)
  llm.safetensors      stt_model.llm.* / embed_tokens /
                       lm_head / function_head                bf16   (~19 GB)
  tts.safetensors      tts_model.tts_model.*                  fp32   (~3.2 GB)
  codec.safetensors    tts_model.audio_codec.*                fp32   (~0.8 GB)

Processes one bucket at a time so peak RSS is bounded by the largest single
bucket, not the whole checkpoint. Fails if any key matches no bucket.
"""
from __future__ import annotations

from pathlib import Path

import torch
import typer
from safetensors import safe_open
from safetensors.torch import save_file

app = typer.Typer(add_completion=False)

# First matching prefix wins.
BUCKETS: list[tuple[str, str, torch.dtype | None]] = [
    ("stt_model.perception.", "encoder", None),
    ("stt_model.rnnt_", "rnnt", None),
    ("stt_model.llm.", "llm", torch.bfloat16),
    ("stt_model.embed_tokens.", "llm", torch.bfloat16),
    ("stt_model.lm_head.", "llm", torch.bfloat16),
    ("stt_model.function_head.", "llm", torch.bfloat16),
    ("tts_model.audio_codec.", "codec", None),
    ("tts_model.", "tts", None),
]


def bucket_of(key: str) -> tuple[str, torch.dtype | None] | None:
    for prefix, bucket, dtype in BUCKETS:
        if key.startswith(prefix):
            return bucket, dtype
    return None


@app.command()
def slice_checkpoint(
    checkpoint: Path = typer.Option(
        Path.home() / "Documents/models/voicechat-11b/model.safetensors",
        help="Path to model.safetensors",
    ),
    output_dir: Path = typer.Option(
        Path.home() / "Documents/models/voicechat-11b/components",
        help="Output directory for per-component safetensors",
    ),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with safe_open(checkpoint, framework="pt", device="cpu") as f:
        keys = list(f.keys())
    typer.echo(f"{len(keys)} tensors in {checkpoint}")

    unmatched = [k for k in keys if bucket_of(k) is None]
    if unmatched:
        typer.echo(f"ERROR: {len(unmatched)} unmatched keys:")
        for k in unmatched[:40]:
            typer.echo(f"  {k}")
        raise typer.Exit(1)

    bucket_names = sorted({b for k in keys for b, _ in [bucket_of(k)]})
    for bucket in bucket_names:
        bucket_keys = [k for k in keys if bucket_of(k)[0] == bucket]
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(checkpoint, framework="pt", device="cpu") as f:
            for key in bucket_keys:
                t = f.get_tensor(key)
                dtype = bucket_of(key)[1]
                if dtype is not None and t.is_floating_point():
                    t = t.to(dtype)
                tensors[key] = t
        out = output_dir / f"{bucket}.safetensors"
        nbytes = sum(t.numel() * t.element_size() for t in tensors.values())
        typer.echo(f"writing {out} ({len(bucket_keys)} tensors, {nbytes / 1e9:.2f} GB)...")
        save_file(tensors, out)
        del tensors

    typer.echo("done.")


if __name__ == "__main__":
    app()
