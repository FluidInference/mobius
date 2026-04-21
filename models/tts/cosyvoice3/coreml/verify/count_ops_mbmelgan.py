"""Verify MB-MelGAN op count claim (202 ops)."""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import coremltools as ct


class ResidualStack(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        residual = x
        x = nn.functional.leaky_relu(x, 0.2)
        x = self.conv1(x)
        x = nn.functional.leaky_relu(x, 0.2)
        x = self.conv2(x)
        return x + residual


class MelGANGenerator(nn.Module):
    def __init__(self, in_channels=80, out_channels=4, kernel_size=7,
                 channels=384, upsample_scales=[5, 5, 3],
                 stack_kernel_size=3, stacks=4):
        super().__init__()
        layers = []
        layers.append(nn.ReflectionPad1d((kernel_size - 1) // 2))
        layers.append(nn.Conv1d(in_channels, channels, kernel_size))
        for i, upsample_scale in enumerate(upsample_scales):
            layers.append(nn.LeakyReLU(0.2))
            in_ch = channels // (2 ** i)
            out_ch = channels // (2 ** (i + 1))
            layers.append(nn.ConvTranspose1d(
                in_ch, out_ch, upsample_scale * 2,
                stride=upsample_scale,
                padding=upsample_scale // 2 + upsample_scale % 2,
                output_padding=upsample_scale % 2,
            ))
            for j in range(stacks):
                layers.append(ResidualStack(out_ch, kernel_size=stack_kernel_size, dilation=stack_kernel_size ** j))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.ReflectionPad1d((kernel_size - 1) // 2))
        final_channels = channels // (2 ** len(upsample_scales))
        layers.append(nn.Conv1d(final_channels, out_channels, kernel_size))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def count_mil_ops(mlmodel):
    """Count ops in the CoreML MIL program."""
    spec = mlmodel.get_spec()
    # For ML Program models
    if spec.WhichOneof("Type") == "mlProgram":
        program = spec.mlProgram
        total_ops = 0
        ops_by_type = {}
        for fname, func in program.functions.items():
            for bname, block in func.block_specializations.items():
                for op in block.operations:
                    total_ops += 1
                    ops_by_type[op.type] = ops_by_type.get(op.type, 0) + 1
        return total_ops, ops_by_type
    # Legacy neural network path
    elif spec.WhichOneof("Type") == "neuralNetwork":
        nn_layers = spec.neuralNetwork.layers
        total = len(nn_layers)
        ops_by_type = {}
        for layer in nn_layers:
            t = layer.WhichOneof("layer")
            ops_by_type[t] = ops_by_type.get(t, 0) + 1
        return total, ops_by_type
    return 0, {}


def main():
    print("=" * 80)
    print("Verifying MB-MelGAN op count claim: 202 ops")
    print("=" * 80)

    model = MelGANGenerator(
        in_channels=80, out_channels=4, channels=384, kernel_size=7,
        upsample_scales=[5, 5, 3], stack_kernel_size=3, stacks=4,
    )
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel params: {total_params:,}")

    example_mel = torch.randn(1, 80, 125)
    with torch.no_grad():
        traced = torch.jit.trace(model, example_mel)
        out = model(example_mel)
    print(f"Input shape:  {tuple(example_mel.shape)}")
    print(f"Output shape: {tuple(out.shape)}")

    print("\nConverting to CoreML (FP16, iOS17)...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="mel_spectrogram", shape=example_mel.shape)],
        outputs=[ct.TensorType(name="audio_bands")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )

    total, ops_by_type = count_mil_ops(mlmodel)
    print(f"\nTotal MIL ops: {total}")
    print(f"Claimed:       202")
    print(f"Match:         {'YES' if total == 202 else 'NO'}")

    print("\nOp breakdown (top 20):")
    for op_type, count in sorted(ops_by_type.items(), key=lambda x: -x[1])[:20]:
        print(f"  {op_type:30s} {count:4d}")

    out_dir = Path("verify/out")
    out_dir.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_dir / "mbmelgan.mlpackage"))
    print(f"\nSaved: {out_dir / 'mbmelgan.mlpackage'}")


if __name__ == "__main__":
    main()
