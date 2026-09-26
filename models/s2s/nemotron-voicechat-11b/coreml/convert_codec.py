#!/usr/bin/env python3
"""Phase 2c: VoiceChat-11B audio codec decoder (Latent2Wav) -> CoreML.

The codec is an RVQ-VAE at 12.5 Hz / 22.05 kHz (wav_to_token_ratio 1764).
Decode path: codes [B,T,31] -> PRVQ embedding-sum -> latent [B,T,512] ->
ConvT(512->1536,k9 s9) -> 3x ConvNeXt(1536) -> ConvT(1536->768,k7 s7) ->
3x ConvNeXt(768) -> ConvT(768->384,k7 s7) -> 3x ConvNeXt(384) ->
Conv1d(384->18) -> mag/phase -> complex spec -> iSTFT(n_fft 16, hop 4).

CoreML split: the conv stack (all the compute, 441x upsample) runs in CoreML
with a flexible T; the PRVQ code->latent lookup (31 embedding sums) and the
tiny 16-point iSTFT tail (complex-valued, not traceable) run host-side.
PRVQ codebooks are exported as codec_prvq_mus.npy and verified identical to
the TTS model's rvq_embs buffer.

Commands: convert | parity | bench   (parity exits nonzero on failure)
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import typer
from safetensors.torch import load_file

CHECKPOINT_DIR = Path.home() / "Documents/models/voicechat-11b"
COMPONENTS_DIR = CHECKPOINT_DIR / "components"
SAMPLE_WAV = CHECKPOINT_DIR / "Speech/examples/speechlm2/sample_audio/sample_general.wav"
BUILD = Path("build/codec")

N_FFT, HOP = 16, 4
UPSAMPLE = 441  # 9*7*7; x hop 4 = 1764 samples/frame

app = typer.Typer(add_completion=False)


def codec_config() -> dict:
    cfg = json.loads((CHECKPOINT_DIR / "config.json").read_text())
    return cfg["model"]["speech_generation"]["model"]["codec_config"]


def build_codec():
    from nemo.collections.speechlm2.modules.ear_tts_vae_codec import RVQVAEModel
    from omegaconf import DictConfig

    model = RVQVAEModel(DictConfig(codec_config()))
    sd = load_file(COMPONENTS_DIR / "codec.safetensors")
    prefix = "tts_model.audio_codec."
    stripped = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    real_missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    assert not real_missing, f"missing keys: {real_missing[:5]}"
    model.eval()
    return model


class DecoderConvWrapper(torch.nn.Module):
    """latent [1, T, 512] -> raw spec precursor [1, 18, 441*T] (pre-complex)."""

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, latent):
        x = latent.transpose(-1, -2)
        for layer in self.decoder.layers:
            if hasattr(layer, "dwconv"):  # ConvNeXt1d — bypass cache kwargs
                residual = x
                y = torch.nn.functional.pad(x, [layer.kernel_size - 1, 0])
                y = layer.dwconv(y)
                y = layer.norm(y)
                y = layer.pwconv1(y)
                y = layer.act(y)
                y = layer.pwconv2(y)
                x = residual + y
            else:
                x = layer(x)
        return x


def prvq_decode_np(mus: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """codes [T, 31] int -> latent [1, T, 512] via sum of per-depth embeddings."""
    T = codes.shape[0]
    z = np.zeros((T, mus.shape[2]), dtype=np.float32)
    for i in range(mus.shape[0]):
        z += mus[i, codes[:, i]]
    return z[None]


def istft_tail_np(raw: np.ndarray, constrain: bool = True) -> np.ndarray:
    """Numpy port of Latent2Wav's complex tail: raw [1,18,T'] -> wav [1, T'*4]."""
    mag, ph = raw[:, :9], raw[:, 9:]
    max_mag = 100.0
    mag = max_mag * np.exp(-np.logaddexp(0.0, -(mag.astype(np.float64)) + math.log(max_mag)))
    cos, sin = np.cos(ph), np.sin(ph)
    spec = mag * cos + 1j * (mag * sin)
    spec[:, 0] = mag[:, 0] * cos[:, 0]  # DC and Nyquist are real
    spec[:, -1] = mag[:, -1] * cos[:, -1]
    ifft = np.fft.irfft(spec, n=N_FFT, axis=1)  # [1, 16, T']
    window = np.hanning(N_FFT + 1)[:-1]  # torch.hann_window = periodic
    if constrain:
        w = window[None, :, None]
        ifft = np.where(ifft >= 0, np.minimum(ifft, w), np.maximum(ifft, -w))
    ifft = ifft * window[None, :, None]
    T = raw.shape[2]
    out_len = (T - 1) * HOP + N_FFT
    wav = np.zeros((raw.shape[0], out_len))
    env = np.zeros(out_len)
    wsq = window**2
    for t in range(T):
        wav[:, t * HOP : t * HOP + N_FFT] += ifft[:, :, t]
        env[t * HOP : t * HOP + N_FFT] += wsq
    pad = (N_FFT - HOP) // 2
    return (wav[:, pad:-pad] / env[pad:-pad]).astype(np.float32)


def coreml_decode(mlmodel, latent: np.ndarray) -> np.ndarray:
    raw = mlmodel.predict({"latent": latent.astype(np.float32)})["spec_raw"]
    return istft_tail_np(raw.astype(np.float64))


@app.command()
def convert(default_t: int = typer.Option(25), max_t: int = typer.Option(500)) -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    model = build_codec()

    mus = torch.stack(list(model.prvq.mus_list)).numpy()  # [31, 1024, 512]
    np.save(BUILD / "codec_prvq_mus.npy", mus)
    tts_sd = load_file(COMPONENTS_DIR / "tts.safetensors")
    rvq_embs = tts_sd["tts_model.tts_model.rvq_embs"].numpy()
    assert np.allclose(mus, rvq_embs, atol=1e-6), "prvq mus != tts rvq_embs"
    typer.echo(f"prvq codebooks exported {mus.shape}; identical to tts rvq_embs")

    wrapper = DecoderConvWrapper(model.decoder).eval()
    ex = torch.randn(1, default_t, 512)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, ex)

    t_dim = ct.RangeDim(lower_bound=1, upper_bound=max_t, default=default_t)
    for precision, name in ((ct.precision.FLOAT32, "decoder_fp32"), (ct.precision.FLOAT16, "decoder_fp16")):
        mlm = ct.convert(
            traced,
            inputs=[ct.TensorType(name="latent", shape=(1, t_dim, 512), dtype=np.float32)],
            outputs=[ct.TensorType(name="spec_raw", dtype=np.float32)],
            minimum_deployment_target=ct.target.iOS17,
            compute_precision=precision,
            compute_units=ct.ComputeUnit.CPU_ONLY,
        )
        mlm.save(str(BUILD / f"{name}.mlpackage"))
        typer.echo(f"saved {BUILD}/{name}.mlpackage")


@app.command()
def parity(seconds: float = typer.Option(2.0)) -> None:
    import soundfile as sf
    from scipy.signal import resample_poly

    model = build_codec()
    mus = np.load(BUILD / "codec_prvq_mus.npy")

    audio, sr = sf.read(SAMPLE_WAV, dtype="float32")
    assert sr == 16000
    audio = resample_poly(audio, 441, 320)  # 16 kHz -> 22.05 kHz
    n = int(seconds * 22050) // 1764 * 1764
    wav_in = torch.from_numpy(audio[:n].astype(np.float32))[None, None]  # [1, 1, N] mono

    with torch.no_grad():
        codes_t, _ = model.encode(wav_in, torch.tensor([wav_in.shape[2]]))
    codes = codes_t[0].numpy()  # [T, 31]
    typer.echo(f"encoded {seconds:.1f}s -> codes {codes.shape}")

    latent_np = prvq_decode_np(mus, codes)
    latent_torch = model.prvq.decode(list(codes_t.permute(2, 0, 1)))
    assert np.allclose(latent_np, latent_torch.numpy(), atol=1e-5), "host prvq decode != torch"

    with torch.no_grad():
        wav_ref = model.decoder(latent_torch).squeeze(1).numpy()  # full torch path incl. iSTFT

    failed = False
    # fp16 raw gate is loose on purpose: the pre-iSTFT tensor holds log-domain
    # magnitudes and radian phases where fp16 deltas are big but audio-irrelevant;
    # the waveform gates are what matter.
    for name, gate_raw, gate_wav in (("decoder_fp32", 1e-4, 1e-4), ("decoder_fp16", 2.0, 0.02)):
        mlm = ct.models.MLModel(str(BUILD / f"{name}.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)
        with torch.no_grad():
            raw_ref = DecoderConvWrapper(model.decoder)(latent_torch).numpy()
        raw_cm = mlm.predict({"latent": latent_np})["spec_raw"]
        d_raw = float(np.abs(raw_cm - raw_ref).max())
        wav_cm = coreml_decode(mlm, latent_np)
        d_wav = float(np.abs(wav_cm - wav_ref).max())
        corr = float(np.corrcoef(wav_cm.ravel(), wav_ref.ravel())[0, 1])
        ok = d_raw <= gate_raw and d_wav <= gate_wav and corr >= 0.999
        typer.echo(
            f"{name}: conv max|Δ| {d_raw:.3e} (gate {gate_raw:g}), wav max|Δ| {d_wav:.3e} "
            f"(gate {gate_wav:g}), corr {corr:.6f} -> {'OK' if ok else 'FAIL'}"
        )
        failed |= not ok

    # round-trip sanity on real audio: decoded wav should correlate with input
    src = wav_in.numpy().ravel()
    dec = wav_ref.ravel()
    m = min(src.shape[0], dec.shape[0])
    rt = float(np.corrcoef(dec[:m], src[:m])[0, 1])
    typer.echo(f"codec round-trip corr vs source audio: {rt:.4f}")
    if failed:
        raise typer.Exit(1)
    typer.echo("CODEC PARITY OK")


@app.command()
def bench(t: int = typer.Option(13), runs: int = typer.Option(30)) -> None:
    latent = np.random.randn(1, t, 512).astype(np.float32) * 0.1
    for units in (ct.ComputeUnit.CPU_AND_GPU, ct.ComputeUnit.CPU_AND_NE):
        mlm = ct.models.MLModel(str(BUILD / "decoder_fp16.mlpackage"), compute_units=units)
        mlm.predict({"latent": latent})
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            mlm.predict({"latent": latent})
            times.append((time.perf_counter() - t0) * 1e3)
        med = sorted(times)[len(times) // 2]
        audio_ms = t * 1764 / 22050 * 1e3
        typer.echo(f"{units.name}: T={t} ({audio_ms:.0f} ms audio) -> {med:.2f} ms median ({audio_ms / med:.0f}x RT)")


if __name__ == "__main__":
    app()
