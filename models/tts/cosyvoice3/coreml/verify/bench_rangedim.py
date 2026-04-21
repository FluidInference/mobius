"""Reproduce RangeDim vs EnumeratedShapes benchmark."""
import sys
from pathlib import Path
import time
import torch
import torch.nn as nn
import coremltools as ct


class ResidualStack(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation,
                                padding=(kernel_size - 1) * dilation // 2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=1,
                                padding=(kernel_size - 1) // 2)

    def forward(self, x):
        r = x
        x = nn.functional.leaky_relu(x, 0.2)
        x = self.conv1(x)
        x = nn.functional.leaky_relu(x, 0.2)
        x = self.conv2(x)
        return x + r


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
                                              padding=u // 2 + u % 2, output_padding=u % 2))
            for j in range(stacks):
                layers.append(ResidualStack(out_ch, kernel_size=stack_kernel_size,
                                             dilation=stack_kernel_size ** j))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.ReflectionPad1d((kernel_size - 1) // 2))
        final = channels // (2 ** len(upsample_scales))
        layers.append(nn.Conv1d(final, out_channels, kernel_size))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def main():
    print("=" * 80)
    print("RangeDim vs EnumeratedShapes")
    print("=" * 80)

    torch.manual_seed(0)
    model = MelGANGenerator().eval()
    example = torch.randn(1, 80, 125)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)

    out_dir = Path(__file__).parent / "out"
    out_dir.mkdir(exist_ok=True)

    # EnumeratedShapes
    print("\n1. EnumeratedShapes [(1,80,125),(1,80,250),(1,80,500)]")
    # Run 3 times to get stable timing
    enum_times = []
    for i in range(3):
        t0 = time.time()
        ml_enum = ct.convert(
            traced,
            inputs=[ct.TensorType(name="mel",
                                   shape=ct.EnumeratedShapes(shapes=[(1, 80, 125), (1, 80, 250), (1, 80, 500)]))],
            outputs=[ct.TensorType(name="audio")],
            minimum_deployment_target=ct.target.iOS17,
            compute_precision=ct.precision.FLOAT16,
            convert_to="mlprogram",
        )
        enum_times.append(time.time() - t0)
    p = out_dir / "mbmelgan_enum.mlpackage"
    ml_enum.save(str(p))
    enum_size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"   Convert times: {[f'{t:.2f}s' for t in enum_times]}  median={sorted(enum_times)[1]:.2f}s")
    print(f"   Size: {enum_size:.2f} MB")

    # Test inference on 259 (not in enum) -- should fail
    import numpy as np
    print("   Testing inference:")
    for f in [125, 250, 500, 259]:
        mel = np.random.randn(1, 80, f).astype(np.float32)
        try:
            r = ml_enum.predict({"mel": mel})
            print(f"     {f} frames: OK  out={r['audio'].shape}")
        except Exception as e:
            print(f"     {f} frames: FAIL  {str(e)[:80]}")

    # RangeDim
    print("\n2. RangeDim (50–500)")
    range_times = []
    for i in range(3):
        t0 = time.time()
        ml_rd = ct.convert(
            traced,
            inputs=[ct.TensorType(name="mel",
                                   shape=(1, 80, ct.RangeDim(lower_bound=50, upper_bound=500, default=125)))],
            outputs=[ct.TensorType(name="audio")],
            minimum_deployment_target=ct.target.iOS17,
            compute_precision=ct.precision.FLOAT16,
            convert_to="mlprogram",
        )
        range_times.append(time.time() - t0)
    p = out_dir / "mbmelgan_rangedim.mlpackage"
    ml_rd.save(str(p))
    rd_size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"   Convert times: {[f'{t:.2f}s' for t in range_times]}  median={sorted(range_times)[1]:.2f}s")
    print(f"   Size: {rd_size:.2f} MB")

    print("   Testing inference:")
    for f in [50, 125, 259, 500]:
        mel = np.random.randn(1, 80, f).astype(np.float32)
        try:
            r = ml_rd.predict({"mel": mel})
            print(f"     {f} frames: OK  out={r['audio'].shape}")
        except Exception as e:
            print(f"     {f} frames: FAIL  {str(e)[:80]}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    em = sorted(enum_times)[1]
    rm = sorted(range_times)[1]
    print(f"EnumeratedShapes convert: {em:.2f}s  (claimed 8.45s)")
    print(f"RangeDim        convert: {rm:.2f}s  (claimed 3.93s)")
    print(f"Speedup ratio: {em/rm:.2f}x  (claimed 2.1x)")
    print(f"Sizes: enum={enum_size:.2f} MB  rangedim={rd_size:.2f} MB  (claimed both 4.49 MB)")


if __name__ == "__main__":
    main()
