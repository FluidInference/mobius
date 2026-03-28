#!/usr/bin/env python3
"""Compare PyTorch vs CoreML at every pipeline stage for numerical equivalence.

Tests fbank → encoder_embed → encoder → encoder_proj → decoder → joiner
and reports cosine similarity and max absolute difference at each stage.

Usage:
    uv run python compare-pipeline.py \
        --checkpoint /path/to/epoch-56-avg-4.pt \
        --tokens /path/to/tokens.txt \
        --coreml-dir ./build/vosk-0.62-atc-fused-fp32 \
        --audio-file /path/to/test.wav \
        --reference "expected transcription"
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf
import torch
import typer

# Load the convert script module (contains load_model, wrapper classes)
_convert_spec = importlib.util.spec_from_file_location(
    "convert_coreml", Path(__file__).parent / "convert-coreml.py"
)
_convert_mod = importlib.util.module_from_spec(_convert_spec)
_convert_spec.loader.exec_module(_convert_mod)

from fused_fbank import KaldiFbank
from icefall.utils import make_pad_mask
from rnnt_decode import greedy_decode_coreml, greedy_decode_pytorch, tokens_to_text

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

MAX_AUDIO = 239120


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_flat, b_flat = a.flatten(), b.flatten()
    return float(np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-10))


def report(name: str, pt: np.ndarray, cm: np.ndarray):
    cos = cosine(pt, cm)
    mad = float(np.max(np.abs(pt - cm)))
    mean_diff = float(np.mean(np.abs(pt - cm)))
    status = "✓" if cos > 0.999 else ("⚠" if cos > 0.99 else "✗")
    typer.echo(f"  {status} {name:30s} cosine={cos:.6f}  max_diff={mad:.6f}  mean_diff={mean_diff:.6f}  shape={pt.shape}")


def per_frame_cosine(pt: np.ndarray, cm: np.ndarray) -> tuple[float, float, float]:
    """Per-frame cosine for (T, D) arrays."""
    T = min(pt.shape[0], cm.shape[0])
    vals = []
    for t in range(T):
        c = np.dot(pt[t], cm[t]) / (np.linalg.norm(pt[t]) * np.linalg.norm(cm[t]) + 1e-10)
        vals.append(c)
    return float(min(vals)), float(np.mean(vals)), float(max(vals))


def word_error_rate(hyp: str, ref: str) -> tuple[int, int]:
    h, r = hyp.lower().split(), ref.lower().split()
    n, m = len(r), len(h)
    if n == 0:
        return m, 0
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if h[i - 1] == r[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n], n


@app.command()
def compare(
    checkpoint: Path = typer.Option(..., "--checkpoint", exists=True, resolve_path=True),
    tokens: Path = typer.Option(..., "--tokens", exists=True, resolve_path=True),
    coreml_dir: Path = typer.Option(..., "--coreml-dir", exists=True, resolve_path=True),
    audio_file: Path = typer.Option(..., "--audio-file", exists=True, resolve_path=True),
    reference: str = typer.Option("", "--reference", help="Reference transcription for WER"),
    compute_units: str = typer.Option("ALL", "--compute-units"),
):
    cu_map = {"CPU_ONLY": ct.ComputeUnit.CPU_ONLY, "ALL": ct.ComputeUnit.ALL}
    cu = cu_map.get(compute_units, ct.ComputeUnit.ALL)

    # Use MPS (Metal GPU) if available for PyTorch
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    typer.echo(f"PyTorch device: {device}")

    typer.echo(f"Loading PyTorch model from {checkpoint.name}...")
    ckpt, enc_embed, encoder, decoder, joiner = _convert_mod.load_model(checkpoint)

    fbank = KaldiFbank()
    for m in [encoder, enc_embed, fbank]:
        m.eval()
        m.to(device)

    typer.echo(f"Loading CoreML models from {coreml_dir}...")
    cm_prep = ct.models.MLModel(str(coreml_dir / "Preprocessor.mlpackage"), compute_units=cu)
    cm_dec = ct.models.MLModel(str(coreml_dir / "decoder.mlpackage"), compute_units=cu)
    cm_join = ct.models.MLModel(str(coreml_dir / "joiner.mlpackage"), compute_units=cu)

    vocab_path = coreml_dir / "vocab.json"
    vocab = json.loads(vocab_path.read_text())

    typer.echo(f"Loading audio: {audio_file.name}")
    audio, sr = sf.read(str(audio_file))
    count = len(audio)
    padded = np.zeros(MAX_AUDIO, dtype=np.float32)
    padded[:min(count, MAX_AUDIO)] = audio[:MAX_AUDIO].astype(np.float32)
    typer.echo(f"  samples={count} ({count/16000:.2f}s)")

    typer.echo("\n── Stage-by-stage comparison ──")

    with torch.no_grad():
        signal_t = torch.tensor(padded).unsqueeze(0).to(device)

        # Stage 1: Fbank
        pt_mel = fbank(signal_t)  # (T, 80)
        pt_mel_2d = pt_mel.unsqueeze(0)  # (1, T, 80)
        typer.echo(f"  [fbank] PT mel shape: {pt_mel.shape}")

        # Stage 2: Encoder embed (Conv2dSubsampling)
        x_lens = torch.tensor([pt_mel_2d.shape[1]], device=device)
        pt_embed, pt_embed_lens = enc_embed(pt_mel_2d, x_lens)
        typer.echo(f"  [embed] PT shape: {pt_embed.shape}, lens={pt_embed_lens.tolist()}")

        # Stage 3: Full encoder
        mask = make_pad_mask(pt_embed_lens, pt_embed.shape[1]).to(device)
        pt_enc_out, pt_enc_lens = encoder(pt_embed.permute(1, 0, 2), pt_embed_lens, mask)
        pt_enc_out = pt_enc_out.permute(1, 0, 2)
        typer.echo(f"  [encoder] PT shape: {pt_enc_out.shape}, lens={pt_enc_lens.tolist()}")

        # Stage 4: Encoder projection
        joiner.encoder_proj.to(device)
        pt_proj = joiner.encoder_proj(pt_enc_out)
        pt_vT = pt_enc_lens[0].item()
        pt_valid = pt_proj[0, :pt_vT, :].cpu().numpy()
        typer.echo(f"  [enc_proj] PT valid frames: {pt_vT}")

    # CoreML fused preprocessor
    cm_out = cm_prep.predict({
        "audio_signal": padded.reshape(1, -1),
        "audio_length": np.array([count], dtype=np.int32),
    })
    cm_enc = cm_out["encoder_out"]  # (1, T, 512)
    cm_enc_lens = cm_out.get("encoder_out_lens")
    typer.echo(f"  [CoreML prep] shape: {cm_enc.shape}, lens={cm_enc_lens}")

    # Compute valid frame count
    mel_frames = max(1, (count - 200) // 160 + 1)
    cm_vT = min(cm_enc.shape[1], max(1, (mel_frames - 7) // 4))
    cm_valid = cm_enc[0, :cm_vT, :]
    typer.echo(f"  [CoreML] valid frames (computed): {cm_vT}")

    # Compare encoder output
    T = min(pt_vT, cm_vT)
    typer.echo(f"\n── Encoder output comparison (first {T} frames) ──")
    report("fused_encoder_output", pt_valid[:T], cm_valid[:T])

    pf_min, pf_mean, pf_max = per_frame_cosine(pt_valid[:T], cm_valid[:T])
    typer.echo(f"  Per-frame cosine: min={pf_min:.4f} mean={pf_mean:.4f} max={pf_max:.4f}")

    # Stage 5: Decoder
    typer.echo("\n── Decoder comparison ──")
    dec_export = _convert_mod.DecoderForExport(decoder, joiner.decoder_proj)
    dec_export.eval()

    test_y = torch.tensor([[0, 0]], dtype=torch.int64)
    with torch.no_grad():
        pt_dec = dec_export(test_y).numpy()
    cm_dec_out = cm_dec.predict({"y": np.array([[0, 0]], dtype=np.int32)})
    cm_dec_val = cm_dec_out["decoder_out"]
    report("decoder(blank,blank)", pt_dec, cm_dec_val)

    # Stage 6: Joiner
    typer.echo("\n── Joiner comparison ──")
    join_export = _convert_mod.JoinerForExport(joiner.output_linear)
    join_export.eval()

    enc_frame_pt = pt_valid[0:1]  # (1, 512)
    dec_frame_pt = pt_dec  # (1, 512)
    with torch.no_grad():
        pt_logit = join_export(torch.tensor(enc_frame_pt), torch.tensor(dec_frame_pt)).numpy()
    cm_logit = cm_join.predict({
        "encoder_out": cm_valid[0:1],
        "decoder_out": cm_dec_val,
    })["logit"]
    report("joiner(frame0, blank)", pt_logit, cm_logit)

    # Stage 7: Greedy decode
    typer.echo("\n── Greedy decode ──")
    with torch.no_grad():
        pt_toks = greedy_decode_pytorch(
            torch.tensor(pt_valid[:T]).unsqueeze(0), dec_export, join_export,
            blank_id=0, context_size=2, joiner_dim=512,
        )
    pt_text = tokens_to_text(pt_toks, vocab)
    typer.echo(f"  PT greedy:  {pt_text}")

    # CoreML greedy - only compare if encoder output is close enough
    if pf_mean > 0.95:
        cm_toks = greedy_decode_coreml(
            cm_enc[:, :cm_vT, :], cm_dec, cm_join,
            blank_id=0, context_size=2,
        )
        cm_text = tokens_to_text(cm_toks, vocab)
        typer.echo(f"  CM greedy:  {cm_text}")
    else:
        cm_text = "(skipped - encoder divergence too high)"
        typer.echo(f"  CM greedy:  {cm_text}")

    if reference:
        typer.echo(f"  Reference:  {reference}")
        pt_edits, pt_words = word_error_rate(pt_text, reference)
        typer.echo(f"  PT WER: {pt_edits}/{pt_words} = {pt_edits/max(pt_words,1)*100:.1f}%")
        if cm_text and not cm_text.startswith("("):
            cm_edits, cm_words = word_error_rate(cm_text, reference)
            typer.echo(f"  CM WER: {cm_edits}/{cm_words} = {cm_edits/max(cm_words,1)*100:.1f}%")


if __name__ == "__main__":
    app()
