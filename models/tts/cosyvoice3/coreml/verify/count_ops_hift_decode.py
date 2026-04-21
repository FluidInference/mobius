"""Count ops in HiFT decode() only (skip f0 predictor + SineGen)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "CosyVoice"))
sys.path.insert(0, str(Path(__file__).parent / "CosyVoice" / "third_party" / "Matcha-TTS"))

import torch
import coremltools as ct
import numpy as np

from cosyvoice.hifigan.generator import HiFTGenerator
from cosyvoice.hifigan.f0_predictor import ConvRNNF0Predictor


def build_hift():
    f0_pred = ConvRNNF0Predictor()
    model = HiFTGenerator(
        in_channels=80, base_channels=512, nb_harmonics=8, sampling_rate=22050,
        upsample_rates=[8, 8], upsample_kernel_sizes=[16, 16],
        istft_params={"n_fft": 16, "hop_len": 4},
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        source_resblock_kernel_sizes=[7, 11],
        source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]],
        lrelu_slope=0.1, audio_limit=0.99, f0_predictor=f0_pred,
    )
    sd = torch.load(str(Path(__file__).parent / "cosyvoice_dl" / "hift.pt"),
                    map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


class DecodeWrapper(torch.nn.Module):
    def __init__(self, hift):
        super().__init__()
        self.hift = hift

    def forward(self, mel, source):
        return self.hift.decode(mel, source)


def main():
    print("=" * 80)
    print("Counting ops in HiFT decode() only (excludes f0 predictor + SineGen)")
    print("=" * 80)

    model = build_hift()
    # 250 mel frames -> 250*64=16000 samples audio (64 = 8*8 upsample, times istft hop=4 = 256 total; 250*256=64000)
    mel = torch.randn(1, 80, 250)
    source = torch.randn(1, 1, 64000)
    wrapper = DecodeWrapper(model).eval()

    with torch.no_grad():
        out = wrapper(mel, source)
    print(f"Input mel:    {tuple(mel.shape)}")
    print(f"Input source: {tuple(source.shape)}")
    print(f"Output:       {tuple(out.shape)}")

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (mel, source), strict=False)

    n_torch = len(list(traced.graph.nodes()))
    print(f"\nTraced torch graph nodes: {n_torch}")

    try:
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="mel", shape=mel.shape),
                ct.TensorType(name="source", shape=source.shape),
            ],
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
        print(f"\nMIL ops in HiFT decode(): {total_mil}")
        print("Top MIL ops:")
        for t, c in sorted(mil_by_type.items(), key=lambda x: -x[1])[:15]:
            print(f"  {t:30s} {c:5d}")

        out_dir = Path(__file__).parent / "out"
        out_dir.mkdir(exist_ok=True)
        mlmodel.save(str(out_dir / "hift_decode.mlpackage"))
        print(f"\nSaved: {out_dir / 'hift_decode.mlpackage'}")
    except Exception as e:
        print(f"Conversion failed: {type(e).__name__}: {str(e)[:500]}")


if __name__ == "__main__":
    main()
