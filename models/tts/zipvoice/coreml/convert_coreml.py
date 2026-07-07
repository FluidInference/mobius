"""Convert LuxTTS (ZipVoice-Distill) text_encoder + fm_decoder to CoreML.

Component boundaries mirror upstream zipvoice/bin/onnx_export.py, except the
data-dependent token-duration expansion stays on the host (CoreML cannot
express dynamic expand), so the text encoder emits per-token embeddings.
The 4-step anchor-Euler solver loop and the vocoder also stay on the host
for this trial.

Usage:
    .venv/bin/python coreml/convert_coreml.py --output-dir build/coreml
"""

import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch import Tensor, nn

from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.scaling_converter import convert_scaled_to_non_scaled

MAX_TOKENS = 256  # default fixed token bucket (prompt + text)
MAX_FRAMES = 1024  # default fixed feature bucket (~10.9 s at 93.75 Hz)
FEAT_DIM = 100


class FrozenPosEmb(nn.Module):
    """Constant replacement for CompactRelPositionalEncoding under fixed shapes."""

    def __init__(self, pos_emb: Tensor):
        super().__init__()
        self.register_buffer("pos_emb", pos_emb)

    def forward(self, x: Tensor, left_context_len: int = 0) -> Tensor:
        return self.pos_emb


def patch_pos_encodings(module: nn.Module, example_inputs: tuple):
    """Freeze positional-encoding outputs as constants for tracing.

    convert_scaled_to_non_scaled jit-scripts encoder_pos, whose in-graph
    shape math becomes a slice_by_index with fp32 begin that CoreML rejects.
    Shapes are fixed here, so probe each Zipformer2Encoder's input length
    eagerly and swap encoder_pos for a constant-buffer module.
    """
    parents = [m for m in module.modules() if hasattr(m, "encoder_pos")]
    captured = {}

    hooks = [
        m.register_forward_pre_hook(lambda mod, args: captured.__setitem__(id(mod), args[0].detach().clone()))
        for m in parents
    ]
    with torch.no_grad():
        module(*example_inputs)
    for h in hooks:
        h.remove()

    for m in parents:
        assert id(m) in captured, "probe pass did not reach an encoder stack"
        with torch.no_grad():
            pos_emb = m.encoder_pos(captured[id(m)]).detach().clone()
        m.encoder_pos = FrozenPosEmb(pos_emb)


def patch_coremltools_int():
    """Let aten::Int handle 1-element constant arrays (shape math folds to size-1
    arrays under fixed shapes; stock handler only accepts 0-d)."""
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch import ops as torch_ops

    reg = torch_ops._TORCH_OPS_REGISTRY
    orig = reg.get_func("int")

    def _int(context, node):
        x = context[node.inputs[0]]
        val = getattr(x, "val", None)
        if val is not None and np.asarray(val).size == 1:
            res = mb.const(val=int(np.asarray(val).reshape(-1)[0]), name=node.name)
            context.add(res)
            return
        return orig(context, node)

    reg.set_func_by_name(_int, "int")


def patch_simple_downsample():
    """Skip the repeat-last-row padding when seq_len is already a multiple of
    the downsample factor. The zero-size expand+cat that upstream emits for
    pad=0 becomes a spurious extra row under coremltools, breaking the
    downstream reshape. MAX_FRAMES is chosen divisible by all ds factors."""
    from zipvoice.models.modules.zipformer import SimpleDownsample

    def forward(self, src: Tensor) -> Tensor:
        seq_len, batch_size, in_channels = src.shape
        ds = self.downsample
        d_seq_len = (seq_len + ds - 1) // ds
        pad = d_seq_len * ds - seq_len
        if pad > 0:  # python int under fixed-shape trace
            src_extra = src[src.shape[0] - 1 :].expand(pad, src.shape[1], src.shape[2])
            src = torch.cat((src, src_extra), dim=0)
        src = src.reshape(d_seq_len, ds, batch_size, in_channels)
        weights = self.bias.softmax(dim=0)
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        return (src * weights).sum(dim=1)

    SimpleDownsample.forward = forward


class CoreMLTextEncoder(nn.Module):
    """embed + Zipformer text encoder -> per-token embeddings (B, S, feat_dim)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.embed = model.embed
        self.text_encoder = model.text_encoder

    def forward(self, tokens: Tensor, padding_mask: Tensor) -> Tensor:
        # tokens: int32 (1, S); padding_mask: float (1, S), 1.0 = padded
        embed = self.embed(tokens)
        mask = padding_mask > 0.5
        return self.text_encoder(x=embed, t=None, padding_mask=mask)


class CoreMLFmDecoder(nn.Module):
    """Single flow-matching step: velocity prediction (distill, guidance embedded)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.fm_decoder = model.fm_decoder

    def forward(
        self,
        t: Tensor,
        x: Tensor,
        text_condition: Tensor,
        speech_condition: Tensor,
        guidance_scale: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        # t, guidance_scale: float (1,); x/conditions: (1, T, feat); mask: float (1, T)
        xt = torch.cat([x, text_condition, speech_condition], dim=2)
        mask = padding_mask > 0.5
        return self.fm_decoder(x=xt, t=t, padding_mask=mask, guidance_scale=guidance_scale)


def load_model():
    model_path = snapshot_download("YatharthS/LuxTTS")
    tokenizer = EmiliaTokenizer(token_file=f"{model_path}/tokens.txt")
    with open(f"{model_path}/config.json") as f:
        config = json.load(f)
    model = ZipVoiceDistill(
        **config["model"], vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id
    )
    load_checkpoint(filename=f"{model_path}/model.pt", model=model, strict=True)
    model.eval()
    convert_scaled_to_non_scaled(model, inplace=True, is_onnx=True)
    return model, tokenizer


def convert_text_encoder(model, out_dir: Path, max_tokens: int = MAX_TOKENS):
    wrapper = CoreMLTextEncoder(model).eval()
    tokens = torch.zeros(1, max_tokens, dtype=torch.int32)
    tokens[0, :10] = torch.arange(2, 12, dtype=torch.int32)
    mask = torch.zeros(1, max_tokens)
    mask[0, 10:] = 1.0

    patch_pos_encodings(wrapper, (tokens, mask))
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (tokens, mask))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="tokens", shape=(1, max_tokens), dtype=np.int32),
            ct.TensorType(name="padding_mask", shape=(1, max_tokens), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="token_embeds", dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )
    path = out_dir / "TextEncoder.mlpackage"
    mlmodel.save(str(path))
    print(f"saved {path}")


def convert_fm_decoder(model, out_dir: Path, max_frames: int = MAX_FRAMES):
    wrapper = CoreMLFmDecoder(model).eval()
    t = torch.tensor([0.5])
    x = torch.randn(1, max_frames, FEAT_DIM)
    text_cond = torch.randn(1, max_frames, FEAT_DIM)
    speech_cond = torch.randn(1, max_frames, FEAT_DIM)
    guidance = torch.tensor([3.0])
    mask = torch.zeros(1, max_frames)
    mask[0, max_frames - 124:] = 1.0

    patch_pos_encodings(wrapper, (t, x, text_cond, speech_cond, guidance, mask))
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (t, x, text_cond, speech_cond, guidance, mask))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="t", shape=(1,), dtype=np.float32),
            ct.TensorType(name="x", shape=(1, max_frames, FEAT_DIM), dtype=np.float32),
            ct.TensorType(name="text_condition", shape=(1, max_frames, FEAT_DIM), dtype=np.float32),
            ct.TensorType(name="speech_condition", shape=(1, max_frames, FEAT_DIM), dtype=np.float32),
            ct.TensorType(name="guidance_scale", shape=(1,), dtype=np.float32),
            ct.TensorType(name="padding_mask", shape=(1, max_frames), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="v", dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )
    path = out_dir / "FmDecoder.mlpackage"
    mlmodel.save(str(path))
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="build/coreml")
    parser.add_argument("--skip-text-encoder", action="store_true")
    parser.add_argument("--skip-fm-decoder", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patch_coremltools_int()
    patch_simple_downsample()
    model, _ = load_model()
    if not args.skip_text_encoder:
        convert_text_encoder(model, out_dir, max_tokens=args.max_tokens)
    if not args.skip_fm_decoder:
        convert_fm_decoder(model, out_dir, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
