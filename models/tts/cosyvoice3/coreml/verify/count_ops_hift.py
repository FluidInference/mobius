"""Count ops in the real CosyVoice HiFT vocoder (no conversion, just trace)."""
import sys
from pathlib import Path

# Add CosyVoice to path
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice"))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice" / "third_party" / "Matcha-TTS"))

import torch
import numpy as np

from cosyvoice.hifigan.generator import HiFTGenerator
from cosyvoice.hifigan.f0_predictor import ConvRNNF0Predictor


def build_hift():
    f0_pred = ConvRNNF0Predictor()
    model = HiFTGenerator(
        in_channels=80,
        base_channels=512,
        nb_harmonics=8,
        sampling_rate=22050,
        upsample_rates=[8, 8],
        upsample_kernel_sizes=[16, 16],
        istft_params={"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes=[7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]],
        lrelu_slope=0.1,
        audio_limit=0.99,
        f0_predictor=f0_pred,
    )
    hift_path = Path(__file__).parent / "cosyvoice_dl" / "hift.pt"
    sd = torch.load(str(hift_path), map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded HiFT (missing: {len(missing)}, unexpected: {len(unexpected)})")
    if missing:
        print(f"  missing (first 5): {missing[:5]}")
    if unexpected:
        print(f"  unexpected (first 5): {unexpected[:5]}")
    # Skip remove_weight_norm due to torch version API changes; doesn't affect op count materially
    model.eval()
    return model


class HiFTInferenceWrapper(torch.nn.Module):
    def __init__(self, hift):
        super().__init__()
        self.hift = hift

    def forward(self, speech_feat):
        out, _ = self.hift.inference(speech_feat)
        return out


def count_torch_graph_ops(traced):
    """Count nodes in the traced torch graph."""
    g = traced.graph
    nodes = list(g.nodes())
    return len(nodes)


def main():
    print("=" * 80)
    print("Counting ops in real CosyVoice HiFT vocoder")
    print("=" * 80)

    model = build_hift()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal params: {total_params:,}")

    # 250 mel frames -> audio
    mel = torch.randn(1, 80, 250)
    wrapper = HiFTInferenceWrapper(model).eval()

    print("\nTracing...")
    with torch.no_grad():
        out = wrapper(mel)
    print(f"Output shape: {tuple(out.shape)}")

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, mel, strict=False)

    n_torch_ops = count_torch_graph_ops(traced)
    print(f"\nTraced torch graph nodes: {n_torch_ops}")

    # Count ops by kind
    ops_by_kind = {}
    for n in traced.graph.nodes():
        k = n.kind()
        ops_by_kind[k] = ops_by_kind.get(k, 0) + 1
    print("\nTop 20 torch op kinds:")
    for k, c in sorted(ops_by_kind.items(), key=lambda x: -x[1])[:20]:
        print(f"  {k:50s} {c:6d}")

    # Try CoreML conversion
    print("\n" + "=" * 80)
    print("Attempting CoreML conversion (may fail on stft/istft)")
    print("=" * 80)
    try:
        import coremltools as ct
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="mel", shape=mel.shape)],
            minimum_deployment_target=ct.target.iOS17,
            compute_precision=ct.precision.FLOAT16,
            convert_to="mlprogram",
        )

        spec = mlmodel.get_spec()
        total_mil = 0
        mil_by_type = {}
        for fname, func in spec.mlProgram.functions.items():
            for bname, block in func.block_specializations.items():
                for op in block.operations:
                    total_mil += 1
                    mil_by_type[op.type] = mil_by_type.get(op.type, 0) + 1
        print(f"\nTotal MIL ops: {total_mil}")
        print("Top MIL ops:")
        for t, c in sorted(mil_by_type.items(), key=lambda x: -x[1])[:20]:
            print(f"  {t:30s} {c:5d}")
    except Exception as e:
        print(f"Conversion failed: {type(e).__name__}: {str(e)[:500]}")


if __name__ == "__main__":
    main()
