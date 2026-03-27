#!/usr/bin/env python3
"""Step-by-step comparison of fused_fbank.py vs torchaudio kaldi.fbank.

Traces through each processing stage and reports cosine similarity at every step.
Useful for identifying exactly where our implementation diverges from the reference.

Usage:
    uv run python debug-fbank.py
    uv run python debug-fbank.py --samples 160000
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi
import typer

from fused_fbank import KaldiFbank

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
    ).item()


def report(label: str, ours: torch.Tensor, ref: torch.Tensor) -> None:
    n = min(ours.shape[0], ref.shape[0])
    a, b = ours[:n], ref[:n]
    cos = cosine(a, b)
    maxd = (a - b).abs().max().item()
    meand = (a - b).abs().mean().item()
    status = "PASS" if cos > 0.9999 else ("CLOSE" if cos > 0.999 else "FAIL")
    typer.echo(f"  [{status}] {label}")
    typer.echo(f"         cosine={cos:.8f}  max_diff={maxd:.6f}  mean_diff={meand:.6f}")
    if ours.shape != ref.shape:
        typer.echo(f"         shapes: ours={list(ours.shape)} ref={list(ref.shape)}")


@app.command()
def compare(
    samples: int = typer.Option(16000, help="Number of audio samples to test with"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
) -> None:
    """Compare fused_fbank.py vs torchaudio kaldi.fbank step by step."""
    torch.manual_seed(seed)
    x = torch.randn(1, samples)
    x0 = x[0]

    typer.echo(f"Audio: {samples} samples ({samples/16000:.2f}s)")
    typer.echo("")

    # ---------------------------------------------------------------
    # Step 0: Reference output (end-to-end)
    # ---------------------------------------------------------------
    fbank = KaldiFbank()
    our_full = fbank(x)

    ref_full = kaldi.fbank(
        x, sample_frequency=16000, num_mel_bins=80,
        frame_length=25.0, frame_shift=10.0,
        window_type="povey", dither=0.0, energy_floor=1.0,
        snip_edges=False,
    )

    typer.echo("=== End-to-end comparison ===")
    report("fused_fbank vs torchaudio.kaldi.fbank", our_full, ref_full)
    typer.echo("")

    # ---------------------------------------------------------------
    # Step 1: Padding
    # ---------------------------------------------------------------
    typer.echo("=== Step 1: Padding (snip_edges=False) ===")

    # Our padding
    our_left_pad = fbank.win_length // 2 - fbank.hop_length // 2  # should be 120
    our_rev = x0.flip(0)
    our_pad_left = our_rev[-our_left_pad:]
    our_padded = torch.cat([our_pad_left, x0, our_rev])

    # Kaldi padding (from _get_strided source)
    kaldi_rev = torch.flip(x0, [0])
    kaldi_pad = fbank.win_length // 2 - fbank.hop_length // 2  # 120
    kaldi_pad_left = kaldi_rev[-kaldi_pad:]
    kaldi_padded = torch.cat([kaldi_pad_left, x0, kaldi_rev])

    typer.echo(f"  Our left_pad={our_left_pad}, Kaldi pad={kaldi_pad}")
    report("Padded signal", our_padded, kaldi_padded)
    typer.echo("")

    # ---------------------------------------------------------------
    # Step 2: Frame extraction
    # ---------------------------------------------------------------
    typer.echo("=== Step 2: Frame extraction ===")
    num_frames = (samples + fbank.hop_length // 2) // fbank.hop_length

    # Our framing (index gather)
    starts = torch.arange(num_frames) * fbank.hop_length
    offsets = torch.arange(fbank.win_length)
    indices = starts.unsqueeze(1) + offsets.unsqueeze(0)
    our_frames = our_padded[indices]

    # Kaldi framing (as_strided)
    kaldi_frames = kaldi_padded.as_strided(
        (num_frames, fbank.win_length),
        (fbank.hop_length, 1)
    )

    report(f"Raw frames ({num_frames} x {fbank.win_length})", our_frames, kaldi_frames)
    typer.echo("")

    # ---------------------------------------------------------------
    # Step 3: DC offset removal
    # ---------------------------------------------------------------
    typer.echo("=== Step 3: DC offset removal ===")
    our_dc = our_frames - our_frames.mean(dim=1, keepdim=True)
    kaldi_dc = kaldi_frames - kaldi_frames.mean(dim=1, keepdim=True)
    report("After DC removal", our_dc, kaldi_dc)
    typer.echo("")

    # ---------------------------------------------------------------
    # Step 4: Preemphasis (per-frame)
    # ---------------------------------------------------------------
    typer.echo("=== Step 4: Preemphasis (per-frame) ===")

    # Our preemph
    first_col = our_dc[:, :1]
    shifted = torch.cat([first_col, our_dc[:, :-1]], dim=1)
    our_preemph = our_dc - fbank.preemph * shifted

    # Kaldi preemph
    kaldi_offset = F.pad(kaldi_dc.unsqueeze(0), (1, 0), mode="replicate").squeeze(0)
    kaldi_preemph = kaldi_dc - fbank.preemph * kaldi_offset[:, :-1]

    report("After preemphasis", our_preemph, kaldi_preemph)
    typer.echo("")

    # ---------------------------------------------------------------
    # Step 5: Windowing (povey = hann^0.85)
    # ---------------------------------------------------------------
    typer.echo("=== Step 5: Povey window ===")
    our_windowed = our_preemph * fbank.window
    kaldi_window = torch.hann_window(fbank.win_length, periodic=False).pow(0.85)
    kaldi_windowed = kaldi_preemph * kaldi_window
    report("After windowing", our_windowed, kaldi_windowed)
    typer.echo("")

    # ---------------------------------------------------------------
    # Step 6: Zero-pad to n_fft
    # ---------------------------------------------------------------
    typer.echo("=== Step 6: Zero-pad to n_fft ===")
    pad_right = fbank.n_fft - fbank.win_length
    our_padded_fft = F.pad(our_windowed, (0, pad_right))
    kaldi_padded_fft = F.pad(kaldi_windowed, (0, pad_right))
    report(f"Padded to {fbank.n_fft}", our_padded_fft, kaldi_padded_fft)
    typer.echo("")

    # ---------------------------------------------------------------
    # Step 7: FFT → power spectrum
    # ---------------------------------------------------------------
    typer.echo("=== Step 7: FFT → power spectrum ===")
    our_spec = torch.fft.rfft(our_padded_fft, n=fbank.n_fft)
    our_power = our_spec.abs().pow(2)

    kaldi_spec = torch.fft.rfft(kaldi_padded_fft, n=fbank.n_fft)
    kaldi_power = kaldi_spec.abs().pow(2)
    report("Power spectrum", our_power, kaldi_power)
    typer.echo("")

    # ---------------------------------------------------------------
    # Step 8: Mel filterbank
    # ---------------------------------------------------------------
    typer.echo("=== Step 8: Mel filterbank ===")

    # Our filterbank (from init)
    our_mel = torch.matmul(our_power, fbank.mel_filterbank.t())

    # Kaldi filterbank — using fbank() defaults: high_freq=0.0 means Nyquist
    kaldi_fb, _ = kaldi.get_mel_banks(
        80, fbank.n_fft, fbank.sample_rate,
        low_freq=20.0, high_freq=0.0,  # fbank() default
        vtln_low=100.0, vtln_high=-500.0, vtln_warp_factor=1.0,
    )
    kaldi_fb = F.pad(kaldi_fb, (0, 1))
    kaldi_mel = torch.mm(kaldi_power, kaldi_fb.T)

    report("Mel energies (before log)", our_mel, kaldi_mel)

    # Also check filterbank itself
    fb_cos = cosine(fbank.mel_filterbank, kaldi_fb)
    fb_diff = (fbank.mel_filterbank - kaldi_fb).abs().max().item()
    typer.echo(f"  Filterbank comparison: cosine={fb_cos:.8f}  max_diff={fb_diff:.8f}")
    if fb_cos < 0.9999:
        typer.echo(f"  WARNING: Filterbanks differ! Our high_freq may not match fbank() default.")
        typer.echo(f"  Our fb was built with high_freq={fbank.sample_rate/2 - 400}")
        typer.echo(f"  fbank() default high_freq=0.0 means Nyquist={fbank.sample_rate/2}")
    typer.echo("")

    # ---------------------------------------------------------------
    # Step 9: Log with floor
    # ---------------------------------------------------------------
    typer.echo("=== Step 9: Log with epsilon floor ===")
    eps = torch.finfo(torch.float32).eps
    our_log = torch.max(our_mel, torch.tensor(eps)).log()
    kaldi_log = torch.max(kaldi_mel, torch.tensor(eps)).log()
    report("Log mel (our fb)", our_log, kaldi_log)
    typer.echo("")

    # ---------------------------------------------------------------
    # Step 10: Using kaldi's filterbank on our power
    # ---------------------------------------------------------------
    typer.echo("=== Step 10: Cross-check — kaldi filterbank on our power ===")
    cross_mel = torch.mm(our_power, kaldi_fb.T)
    cross_log = torch.max(cross_mel, torch.tensor(eps)).log()
    report("Kaldi fb on our power vs kaldi ref", cross_log, ref_full)
    typer.echo("")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    typer.echo("=== Summary ===")
    typer.echo(f"  End-to-end cosine: {cosine(our_full, ref_full):.8f}")
    typer.echo(f"  With kaldi fb:     {cosine(cross_log, ref_full):.8f}")
    if fb_cos < 0.9999:
        typer.echo("")
        typer.echo("  FIX: Update KaldiFbank.__init__ to use high_freq=0.0 (Nyquist)")
        typer.echo("  instead of high_freq=sample_rate/2 - 400 (kaldi CLI default)")


if __name__ == "__main__":
    app()
