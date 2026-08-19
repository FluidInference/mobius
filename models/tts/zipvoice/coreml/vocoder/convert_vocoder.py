"""Convert the LuxTTS 48 kHz dual-head Vocos vocoder to CoreML, fully in-graph.

The linacodec vocoder is: VocosBackbone (8 ConvNeXt blocks, dim 512) ->
{ISTFTHead @24k on L frames, UpSamplerBlock [2,1] -> ISTFTHead @48k on 2L+1
frames}, 24k path resampled to 48k, then an FFT-domain brickwall crossover at
12 kHz. Everything is reimplemented with fixed shapes so the whole decode is
one CoreML graph:

- ISTFT (center padding): irfft as a constant DFT matmul with the hann window
  folded in, overlap-add as 4 shifted chunk adds (hop 256, n_fft 1024), then a
  precomputed 1/window-envelope constant. Validated vs torch.istft at >120 dB.
- 24k->48k resample: torchaudio's sinc kernel (2 phases x 15 taps) applied as
  conv1d + interleave, bit-exact vs AF.resample.
- Crossover: torch does rfft over the whole waveform and blends with a ~2.6 Hz
  brickwall fade at 12 kHz (linkwitz.py). Reimplemented as
  merged = path_24k + FIR_hp(path_48k - path_24k) with a 511-tap linear-phase
  highpass (spectral inversion of scipy firwin lowpass, so the two branches
  stay exactly complementary). ~62 dB vs the torch crossover on real signals.

Usage:
    .venv/bin/python -m coreml.vocoder.convert_vocoder --frames 282 \
        --output-dir build/coreml-vocoder --validate
"""

import argparse
import math
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from torch import Tensor, nn
from torch.nn.utils import parametrize

from coreml.convert_coreml import patch_coremltools_int

N_FFT = 1024
HOP = 256
N_BINS = N_FFT // 2 + 1
FEAT_DIM = 100
FIR_TAPS = 511
CUTOFF_HZ = 12000
SR = 48000


def load_vocos():
    """LuxTTS linacodec Vocos, weight-norm parametrizations removed (as the
    reference loader does) so the upsampler convs are plain tensors."""
    from linacodec.vocoder.vocos import Vocos

    model_path = snapshot_download("YatharthS/LuxTTS")
    vocos = Vocos.from_hparams(f"{model_path}/vocoder/config.yaml")
    parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[0], "weight")
    parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[1], "weight")
    vocos.load_state_dict(torch.load(f"{model_path}/vocoder/vocos.bin", map_location="cpu"))
    vocos.eval()
    vocos.freq_range = CUTOFF_HZ
    vocos.return_48k = True
    return vocos


def patch_snake():
    """Snake1d calls a jit-scripted helper whose reshape shape-math trips the
    converter; the reshapes are no-ops for 3D input, so compute directly."""
    from linacodec.vocoder import upsampler_block

    def forward(self, x: Tensor) -> Tensor:
        return x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)

    upsampler_block.Snake1d.forward = forward


def istft_constants(num_frames: int):
    """Constant tensors for a fixed-shape center-padded ISTFT.

    Returns (basis_cos (N_BINS, N_FFT), basis_sin, env_inv (1, (T-1)*HOP)):
    frame = real @ basis_cos + imag @ basis_sin gives window * irfft(spec).
    """
    n = np.arange(N_FFT)[:, None]
    k = np.arange(N_BINS)[None, :]
    herm = np.where((k == 0) | (k == N_FFT // 2), 1.0, 2.0) / N_FFT
    cos_b = herm * np.cos(2 * np.pi * k * n / N_FFT)  # (N_FFT, N_BINS)
    sin_b = -herm * np.sin(2 * np.pi * k * n / N_FFT)
    window = torch.hann_window(N_FFT).numpy().astype(np.float64)
    basis_cos = (window[:, None] * cos_b).T  # (N_BINS, N_FFT)
    basis_sin = (window[:, None] * sin_b).T

    full = num_frames * HOP + (N_FFT - HOP)
    env = np.zeros(full)
    for t in range(num_frames):
        env[t * HOP : t * HOP + N_FFT] += window**2
    env = env[N_FFT // 2 : -(N_FFT // 2)]  # center trim; hann/75% NOLA holds
    assert env.min() > 1e-3
    return (
        torch.from_numpy(basis_cos).float(),
        torch.from_numpy(basis_sin).float(),
        torch.from_numpy(1.0 / env[None, :]).float(),
    )


class DftIstftHead(nn.Module):
    """ISTFTHead (linear -> exp/cos/sin -> center ISTFT) with the ISTFT done as
    DFT matmul + shifted-chunk overlap-add, fixed frame count."""

    def __init__(self, head: nn.Module, num_frames: int):
        super().__init__()
        self.out = head.out
        self.num_frames = num_frames
        self.out_len = (num_frames - 1) * HOP
        self.full_len = num_frames * HOP + (N_FFT - HOP)
        basis_cos, basis_sin, env_inv = istft_constants(num_frames)
        self.register_buffer("basis_cos", basis_cos)
        self.register_buffer("basis_sin", basis_sin)
        self.register_buffer("env_inv", env_inv)
        self.log_mag_max = float(np.log(100.0))  # == upstream clip(mag, max=1e2)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, 512) -> audio (B, (T-1)*HOP)
        z = self.out(x)  # (B, T, 2*N_BINS)
        log_mag = z[:, :, :N_BINS].clamp(max=self.log_mag_max)  # pre-exp: fp16 safe
        phase = z[:, :, N_BINS:]
        mag = torch.exp(log_mag)
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        frames = real @ self.basis_cos + imag @ self.basis_sin  # (B, T, N_FFT)

        # Overlap-add: chunk c of every frame lands at flat offset c*HOP.
        flat = self.num_frames * HOP
        y = None
        for c in range(N_FFT // HOP):
            chunk = frames[:, :, c * HOP : (c + 1) * HOP].reshape(1, flat)
            part = F.pad(chunk, (c * HOP, self.full_len - flat - c * HOP))
            y = part if y is None else y + part
        y = y[:, N_FFT // 2 : self.full_len - N_FFT // 2]
        return y * self.env_inv


class Resample2x(nn.Module):
    """torchaudio sinc_interp_hann 24k->48k as conv1d + phase interleave."""

    def __init__(self, in_len: int):
        super().__init__()
        from torchaudio.functional.functional import _get_sinc_resample_kernel

        kernel, width = _get_sinc_resample_kernel(
            24000, 48000, math.gcd(24000, 48000), 6, 0.99, "sinc_interp_hann", None,
            torch.device("cpu"), torch.float32,
        )
        self.register_buffer("kernel", kernel)  # (2, 1, taps)
        self.width = int(width)
        self.out_len = 2 * in_len

    def forward(self, wav: Tensor) -> Tensor:
        # wav: (B, L) -> (B, 2L); mirrors _apply_sinc_resample_kernel, orig_freq=1
        x = F.pad(wav.unsqueeze(1), (self.width, self.width + 1))
        y = F.conv1d(x, self.kernel)  # (B, 2, L+1)
        y = y.transpose(1, 2).reshape(1, -1)
        return y[:, : self.out_len]


class FirCrossover(nn.Module):
    """FIR stand-in for linkwitz.py's whole-signal FFT brickwall at 12 kHz:
    merged = low_path + hp(high_path - low_path). Linear-phase highpass by
    spectral inversion keeps the branches exactly complementary."""

    def __init__(self, taps: int = FIR_TAPS):
        super().__init__()
        from scipy.signal import firwin

        lp = firwin(taps, CUTOFF_HZ, fs=SR)
        hp = -lp
        hp[taps // 2] += 1.0
        self.taps = taps
        self.register_buffer("weight", torch.from_numpy(hp[None, None, :]).float())

    def forward(self, high: Tensor, low: Tensor) -> Tensor:
        d = (high - low).unsqueeze(1)
        hp = F.conv1d(d, self.weight, padding=self.taps // 2)
        return low + hp.squeeze(1)


class CoreMLVocoder(nn.Module):
    """Whole vocoder decode as one fixed-shape graph: mel (1,100,S) -> (1,(S-1)*512)."""

    def __init__(self, vocos: nn.Module, frames: int, fir_taps: int = FIR_TAPS):
        super().__init__()
        self.backbone = vocos.backbone
        self.upsampler = vocos.upsampler
        self.head_24k = DftIstftHead(vocos.head, frames)
        self.head_48k = DftIstftHead(vocos.head_48k, 2 * frames + 1)  # upsampler emits 2S+1
        self.resample = Resample2x(self.head_24k.out_len)
        self.crossover = FirCrossover(fir_taps)
        self.out_len = (frames - 1) * 2 * HOP

    def forward(self, mel: Tensor) -> Tensor:
        features = self.backbone(mel)  # (B, S, 512)
        upsampled = self.upsampler(features.transpose(1, 2))  # (B, 512, 2S+1)
        audio_hi = self.head_48k(upsampled.transpose(1, 2))  # (B, 2S*HOP)
        audio_lo = self.resample(self.head_24k(features))  # (B, (S-1)*2*HOP)
        return self.crossover(audio_hi[:, : self.out_len], audio_lo)


def snr_db(ref, test) -> float:
    ref = np.asarray(ref, dtype=np.float64).ravel()
    test = np.asarray(test, dtype=np.float64).ravel()
    return float(10 * np.log10(np.sum(ref**2) / (np.sum((ref - test) ** 2) + 1e-30)))


def validate_istft():
    """Gate: DFT-matmul ISTFT vs torch.istft on random mag/phase."""
    torch.manual_seed(0)
    for frames in (282, 565):
        log_mag = torch.randn(1, frames, N_BINS).clamp(max=np.log(100.0))
        phase = torch.randn(1, frames, N_BINS) * 3.0
        mag = torch.exp(log_mag)
        real, imag = mag * torch.cos(phase), mag * torch.sin(phase)

        spec = torch.complex(real, imag).transpose(1, 2)
        ref = torch.istft(spec, N_FFT, HOP, N_FFT, torch.hann_window(N_FFT), center=True)

        basis_cos, basis_sin, env_inv = istft_constants(frames)
        fr = real @ basis_cos + imag @ basis_sin
        flat, full = frames * HOP, frames * HOP + (N_FFT - HOP)
        y = sum(
            F.pad(fr[:, :, c * HOP : (c + 1) * HOP].reshape(1, flat), (c * HOP, full - flat - c * HOP))
            for c in range(N_FFT // HOP)
        )
        mine = y[:, N_FFT // 2 : full - N_FFT // 2] * env_inv

        s = snr_db(ref.numpy(), mine.numpy())
        print(f"[istft standalone] frames={frames}: SNR {s:.1f} dB")
        assert s > 60, f"ISTFT reimplementation SNR {s:.1f} dB < 60"


def validate_wrapper(vocos, wrapper, frames: int, mel_path: str | None):
    """Gate: eager fp32 wrapper vs vocos.decode (isolates the FIR crossover)."""
    if mel_path:
        mel = torch.from_numpy(np.load(mel_path)).float()
        assert mel.shape == (1, FEAT_DIM, frames), mel.shape
    else:
        torch.manual_seed(1)
        mel = torch.randn(1, FEAT_DIM, frames) * 0.5
    with torch.no_grad():
        ref = vocos.decode(mel)
        mine = wrapper(mel)
    s = snr_db(ref.numpy(), mine.numpy())
    print(f"[wrapper eager fp32] frames={frames} src={'oracle mel' if mel_path else 'random mel'}: "
          f"SNR {s:.1f} dB (vs torch decode; residual = FIR-vs-FFT crossover)")
    assert s > 40, f"eager wrapper SNR {s:.1f} dB < 40"


def convert(vocos, frames: int, out_dir: Path, fir_taps: int, validate: bool, mel_path: str | None):
    wrapper = CoreMLVocoder(vocos, frames, fir_taps).eval()
    if validate:
        validate_istft()
        validate_wrapper(vocos, wrapper, frames, mel_path)

    mel = torch.randn(1, FEAT_DIM, frames) * 0.5
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (mel,))

    t0 = time.perf_counter()
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="mel", shape=(1, FEAT_DIM, frames), dtype=np.float32)],
        outputs=[ct.TensorType(name="audio", dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "Vocoder.mlpackage"
    mlmodel.save(str(path))
    print(f"saved {path} ({(time.perf_counter() - t0):.1f} s) — "
          f"mel (1,{FEAT_DIM},{frames}) -> audio (1,{(frames - 1) * 2 * HOP}) @48k")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=282, help="fixed mel frame count S (gen region)")
    parser.add_argument("--output-dir", default="build/coreml-vocoder")
    parser.add_argument("--fir-taps", type=int, default=FIR_TAPS)
    parser.add_argument("--validate", action="store_true", help="run eager ISTFT + wrapper gates first")
    parser.add_argument("--mel", default=None, help="optional .npy mel (1,100,frames) for the eager gate")
    args = parser.parse_args()

    patch_coremltools_int()
    patch_snake()
    vocos = load_vocos()
    convert(vocos, args.frames, Path(args.output_dir), args.fir_taps, args.validate, args.mel)


if __name__ == "__main__":
    main()
