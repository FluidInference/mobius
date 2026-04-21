"""Reproduce FP32 vs FP16 benchmark with proper warmup."""
import sys
from pathlib import Path
import time
import torch
import torch.nn as nn
import coremltools as ct
import numpy as np


class ResidualStack(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation,
                                padding=(kernel_size - 1) * dilation // 2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=1,
                                padding=(kernel_size - 1) // 2)

    def forward(self, x):
        residual = x
        x = nn.functional.leaky_relu(x, 0.2)
        x = self.conv1(x)
        x = nn.functional.leaky_relu(x, 0.2)
        x = self.conv2(x)
        return x + residual


class MelGANGenerator(nn.Module):
    def __init__(self, in_channels=80, out_channels=4, kernel_size=7, channels=384,
                 upsample_scales=[5, 5, 3], stack_kernel_size=3, stacks=4):
        super().__init__()
        layers = []
        layers.append(nn.ReflectionPad1d((kernel_size - 1) // 2))
        layers.append(nn.Conv1d(in_channels, channels, kernel_size))
        for i, u in enumerate(upsample_scales):
            layers.append(nn.LeakyReLU(0.2))
            in_ch = channels // (2 ** i)
            out_ch = channels // (2 ** (i + 1))
            layers.append(nn.ConvTranspose1d(in_ch, out_ch, u * 2, stride=u,
                                              padding=u // 2 + u % 2,
                                              output_padding=u % 2))
            for j in range(stacks):
                layers.append(ResidualStack(out_ch, kernel_size=stack_kernel_size,
                                             dilation=stack_kernel_size ** j))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.ReflectionPad1d((kernel_size - 1) // 2))
        final_channels = channels // (2 ** len(upsample_scales))
        layers.append(nn.Conv1d(final_channels, out_channels, kernel_size))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def convert(model, example, precision, name):
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    t0 = time.time()
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="mel", shape=example.shape)],
        outputs=[ct.TensorType(name="audio")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=precision,
        convert_to="mlprogram",
    )
    t_conv = time.time() - t0
    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(exist_ok=True)
    p = out_dir / f"mbmelgan_{name}.mlpackage"
    ml.save(str(p))
    size_mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 / 1024
    return ml, size_mb, t_conv


def bench(model_pt, model_ml, mel, n_warmup=5, n_iters=20):
    mel_np = mel.numpy()

    # PyTorch
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model_pt(mel)
        pt_times = []
        for _ in range(n_iters):
            t = time.perf_counter()
            pt_out = model_pt(mel).numpy()
            pt_times.append(time.perf_counter() - t)

    # CoreML
    for _ in range(n_warmup):
        _ = model_ml.predict({"mel": mel_np})
    ml_times = []
    for _ in range(n_iters):
        t = time.perf_counter()
        ml_out = model_ml.predict({"mel": mel_np})["audio"]
        ml_times.append(time.perf_counter() - t)

    mae = np.abs(pt_out - ml_out).mean()
    max_diff = np.abs(pt_out - ml_out).max()
    return {
        "pt_median_ms": np.median(pt_times) * 1000,
        "pt_p05_ms": np.percentile(pt_times, 5) * 1000,
        "ml_median_ms": np.median(ml_times) * 1000,
        "ml_p05_ms": np.percentile(ml_times, 5) * 1000,
        "mae": mae,
        "max_diff": max_diff,
    }


def main():
    print("=" * 80)
    print("MB-MelGAN FP32 vs FP16 — proper benchmark")
    print("=" * 80)

    torch.manual_seed(0)
    model = MelGANGenerator(in_channels=80, out_channels=4, channels=384, kernel_size=7,
                             upsample_scales=[5, 5, 3], stack_kernel_size=3, stacks=4)
    model.eval()

    mel = torch.randn(1, 80, 125)
    print(f"Input: {tuple(mel.shape)}  |  Output: {tuple(model(mel).shape)}")

    # FP16
    print("\nConverting FP16...")
    ml16, sz16, tc16 = convert(model, mel, ct.precision.FLOAT16, "fp16")
    print(f"  size: {sz16:.2f} MB   convert: {tc16:.2f}s")

    # FP32
    print("\nConverting FP32...")
    ml32, sz32, tc32 = convert(model, mel, ct.precision.FLOAT32, "fp32")
    print(f"  size: {sz32:.2f} MB   convert: {tc32:.2f}s")

    print("\nBenchmarking FP16 (5 warmup, 20 iters)...")
    r16 = bench(model, ml16, mel)
    print("\nBenchmarking FP32 (5 warmup, 20 iters)...")
    r32 = bench(model, ml32, mel)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"{'':12s} {'FP16':>15s} {'FP32':>15s}  Claim")
    print(f"{'size(MB)':12s} {sz16:>15.2f} {sz32:>15.2f}  (4.50 / 8.94)")
    print(f"{'MAE':12s} {r16['mae']:>15.6f} {r32['mae']:>15.6f}  (0.056 / 0.000)")
    print(f"{'max_diff':12s} {r16['max_diff']:>15.6f} {r32['max_diff']:>15.6f}")
    print(f"{'ml_median':12s} {r16['ml_median_ms']:>13.1f}ms {r32['ml_median_ms']:>13.1f}ms  (129 / 1664)")
    print(f"{'ml_p05':12s} {r16['ml_p05_ms']:>13.1f}ms {r32['ml_p05_ms']:>13.1f}ms")
    print(f"{'pt_median':12s} {r16['pt_median_ms']:>13.1f}ms {r32['pt_median_ms']:>13.1f}ms")
    ratio = r32['ml_median_ms'] / r16['ml_median_ms']
    print(f"\nFP32/FP16 time ratio: {ratio:.2f}x  (claimed: 12.9x)")


if __name__ == "__main__":
    main()
