"""Parity check: Inflect v2 CoreML (fixed-shape split) vs PyTorch reference.

Stages compared on the valid (unpadded) region:
  1. encoder: m_p / logs_p / logw
  2. synthesizer: waveform given identical z_p
  3. end-to-end audio written to build/ for listening
"""

import argparse
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent
HOP = 256


def load_reference(checkpoint_dir: Path):
    sys.path.insert(0, str(checkpoint_dir / "runtime"))
    sys.path.insert(0, str(checkpoint_dir))
    import commons
    import utils
    from inflect_vits_frontend import run_vits_frontend
    from models import SynthesizerTrn
    from text import cleaned_text_to_sequence
    from text.symbols import symbols

    hps = utils.get_hparams_from_file(str(checkpoint_dir / "config.json"))
    model = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).eval()
    utils.load_checkpoint(str(checkpoint_dir / "model.pth"), model, None)
    import contextlib, io

    with contextlib.redirect_stdout(io.StringIO()):
        model.dec.remove_weight_norm()
    for flow in model.flow.flows:
        if hasattr(getattr(flow, "enc", None), "remove_weight_norm"):
            flow.enc.remove_weight_norm()

    def tokenize(text: str) -> list[int]:
        phonemes = run_vits_frontend(text).phoneme_text
        sequence = cleaned_text_to_sequence(phonemes)
        if hps.data.add_blank:
            sequence = commons.intersperse(sequence, 0)
        return sequence

    return model, hps, tokenize


def report(name: str, ref: np.ndarray, got: np.ndarray) -> None:
    ref = ref.astype(np.float32).ravel()
    got = got.astype(np.float32).ravel()
    max_abs = float(np.max(np.abs(ref - got))) if ref.size else 0.0
    denom = float(np.linalg.norm(ref)) or 1.0
    rel = float(np.linalg.norm(ref - got)) / denom
    corr = float(np.corrcoef(ref, got)[0, 1]) if ref.size > 1 else 1.0
    print(f"  {name:12s} max_abs={max_abs:.5f}  rel_l2={rel:.5f}  corr={corr:.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["micro", "nano"], default="micro")
    parser.add_argument("--coreml-dir", type=Path, default=None)
    parser.add_argument("--text", default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-scale", type=float, default=0.667)
    parser.add_argument("--compute-units", choices=["all", "cpu", "cpu_and_ne"], default="all")
    args = parser.parse_args()

    coreml_dir = args.coreml_dir or ROOT / "build" / f"inflect-{args.variant}-v2-fp16-t256-f1024"
    checkpoint_dir = ROOT / "checkpoints" / f"inflect-{args.variant}-v2"
    model, hps, tokenize = load_reference(checkpoint_dir)

    units = {
        "all": ct.ComputeUnit.ALL,
        "cpu": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    }[args.compute_units]
    ml_encoder = ct.models.MLModel(str(coreml_dir / "encoder.mlpackage"), compute_units=units)
    ml_synth = ct.models.MLModel(str(coreml_dir / "synthesizer.mlpackage"), compute_units=units)
    enc_shape = ml_encoder.get_spec().description.input[0].type.multiArrayType.shape
    synth_shape = ml_synth.get_spec().description.input[0].type.multiArrayType.shape
    t_text, t_frames = int(enc_shape[1]), int(synth_shape[2])

    sequence = tokenize(args.text)
    n_tok = len(sequence)
    assert n_tok <= t_text, f"{n_tok} tokens exceed fixed t_text={t_text}"
    print(f"variant={args.variant}  tokens={n_tok}/{t_text}  frames_budget={t_frames}")

    tokens = torch.LongTensor(sequence).unsqueeze(0)
    lengths = torch.LongTensor([n_tok])

    # --- PyTorch reference (unpadded) ---
    with torch.inference_mode():
        x, m_p, logs_p, x_mask = model.enc_p(tokens, lengths)
        logw = model.dp(x, x_mask)

    # --- CoreML encoder (padded) ---
    tokens_pad = np.zeros((1, t_text), dtype=np.int32)
    tokens_pad[0, :n_tok] = sequence
    mask_pad = np.zeros((1, 1, t_text), dtype=np.float32)
    mask_pad[0, 0, :n_tok] = 1.0
    t0 = time.perf_counter()
    enc_out = ml_encoder.predict({"tokens": tokens_pad, "x_mask": mask_pad})
    enc_ms = (time.perf_counter() - t0) * 1000
    print(f"encoder: {enc_ms:.1f} ms")
    report("m_p", m_p.numpy(), enc_out["m_p"][:, :, :n_tok])
    report("logs_p", logs_p.numpy(), enc_out["logs_p"][:, :, :n_tok])
    report("logw", logw.numpy(), enc_out["logw"][:, :, :n_tok])

    # --- host expansion + shared noise ---
    w_ceil_ref = torch.ceil(torch.exp(logw) * x_mask)
    w_ceil_ml = np.ceil(np.exp(enc_out["logw"][:, :, :n_tok]))
    dur_diff = int(np.abs(w_ceil_ref.numpy() - w_ceil_ml).sum())
    y_len = int(w_ceil_ref.sum())
    print(f"  durations: y_len={y_len} frames ({y_len * HOP / hps.data.sampling_rate:.2f}s), ml_dur_diff={dur_diff}")
    assert y_len <= t_frames, f"{y_len} frames exceed fixed t_frames={t_frames}"

    idx = np.repeat(np.arange(n_tok), w_ceil_ref.numpy().astype(np.int64).ravel())
    m_exp = m_p.numpy()[:, :, idx]
    logs_exp = logs_p.numpy()[:, :, idx]
    torch.manual_seed(args.seed)
    noise = torch.randn(1, m_exp.shape[1], y_len).numpy()
    z_p = m_exp + noise * np.exp(logs_exp) * args.noise_scale

    # --- PyTorch flow + decoder (unpadded) ---
    y_mask = torch.ones(1, 1, y_len)
    with torch.inference_mode():
        z = model.flow(torch.from_numpy(z_p).float(), y_mask, reverse=True)
        audio_ref = model.dec(z * y_mask)[0, 0].numpy()

    # --- CoreML synthesizer (padded) ---
    z_p_pad = np.zeros((1, z_p.shape[1], t_frames), dtype=np.float32)
    z_p_pad[:, :, :y_len] = z_p
    y_mask_pad = np.zeros((1, 1, t_frames), dtype=np.float32)
    y_mask_pad[0, 0, :y_len] = 1.0
    t0 = time.perf_counter()
    synth_out = ml_synth.predict({"z_p": z_p_pad, "y_mask": y_mask_pad})
    synth_ms = (time.perf_counter() - t0) * 1000
    audio_ml = synth_out["audio"][0, 0, : y_len * HOP]
    audio_s = y_len * HOP / hps.data.sampling_rate
    print(f"synthesizer: {synth_ms:.1f} ms  ({audio_s / (synth_ms / 1000):.1f}x realtime for {audio_s:.2f}s)")
    report("audio", audio_ref, audio_ml)

    out_dir = ROOT / "build"
    sf.write(out_dir / f"parity_{args.variant}_ref.wav", audio_ref, hps.data.sampling_rate)
    sf.write(out_dir / f"parity_{args.variant}_coreml.wav", audio_ml, hps.data.sampling_rate)
    print(f"wrote build/parity_{args.variant}_{{ref,coreml}}.wav")


if __name__ == "__main__":
    main()
