"""E7: Build a B3+B1 fused `decoder_joint_noencproj.mlpackage`.

Variant of `fuse_decjoint_engprune.py` that bypasses `joint.enc` (the
1024→640 encoder-projection layer) and takes a PRE-PROJECTED
`encoder_proj` (640-d, [1, 1, 640]) as input instead of `encoder`
(1024-d, [1, 1024, 1]).

Mechanism:
  Standard B1 (decoder_joint.mlpackage):
    encoder[1024] -> joint.enc -> 640
    dec_out[640]  -> joint.pred -> 640
    joint_after_projection(f, g) -> logits[V]

  B3+B1 (decoder_joint_noencproj.mlpackage):
    encoder_proj[640] (already projected externally) -> bypass joint.enc
    dec_out[640]  -> joint.pred -> 640
    joint_after_projection(f, g) -> logits[V]

Saves one 1024→640 matmul per emitted token. Per-token win is small,
but on Earnings22-1h where decoder is 85% of wall time, this is the
most likely remaining card.

The encoder_proj is computed externally either:
- By the encoder mlpackage emitting it as a 6th output (B3 split), or
- By Swift cblas_sgemm using the joint.enc weights from native_weights/
  (current production swiftencproj path).

Swift integration: the path at Pipeline.swift:449 already activates
on `decoderJointNoEncProj != nil`. Drop this mlpackage into the build
dir to engage.

Usage:
    .venv/bin/python conversion_scripts/fuse_decjoint_noencproj_tokenizer.py \
        --nemo-path /path/to/nemotron-asr-streaming-multilingual-0.6b.nemo \
        --reference-tokenizer-json /path/to/build/tokenizer.json \
        --output-dir build_test_e7_djne
"""
from __future__ import annotations

import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import typer

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

_ENG_COREML = (
    THIS_DIR.parent.parent.parent
    / "nemotron-speech-streaming-0.6b"
    / "coreml"
    / "conversion_scripts"
)
sys.path.insert(0, str(_ENG_COREML))

from individual_components import DecoderWrapper, ExportSettings, _coreml_convert  # type: ignore
from patch_uniquebias import patch_and_log  # type: ignore
from patch_vocab_prune import prune_vocab_english  # type: ignore
from build_joint_noencproj_batched_from_tokenizer import (  # type: ignore
    recover_keep_ids_from_tokenizer_json,
)


def _tensor_shape(t):
    return tuple(int(s) for s in t.shape)


class DecoderJointNoEncProjFusedWrapper(torch.nn.Module):
    """Decoder + joint (without enc.proj) fused into a single forward.

    Mirrors `DecoderJointFusedWrapper` but takes `encoder_proj` (640-d)
    instead of `encoder` (1024-d). Bypasses `joint.project_encoder` /
    `joint.enc` — caller is responsible for pre-projecting.
    """

    def __init__(self, dec: torch.nn.Module, jnt: torch.nn.Module) -> None:
        super().__init__()
        self.decoder = DecoderWrapper(dec)
        self.joint_module = jnt

    def forward(self, token, token_length, h_in, c_in, encoder_proj):
        # encoder_proj: [B, T, joint_hidden=640]
        dec_out, h_out, c_out = self.decoder(token, token_length, h_in, c_in)
        # dec_out from DecoderWrapper: [B, D=640, U]
        # joint_after_projection expects (B, T, H) and (B, U, H)
        # → transpose dec_out from [B, D, U] to [B, U, D]
        g = dec_out.transpose(1, 2)  # [B, U=1, 640]
        # project_prednet
        g = self.joint_module.project_prednet(g)  # [B, U, joint_hidden=640]
        # encoder_proj is already projected; pass through
        logits = self.joint_module.joint_after_projection(encoder_proj, g)
        return logits, h_out, c_out


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def fuse(
    nemo_path: Path = typer.Option(..., "--nemo-path"),
    reference_tokenizer_json: Path = typer.Option(..., "--reference-tokenizer-json"),
    output_dir: Path = typer.Option(..., "--output-dir"),
):
    import nemo.collections.asr as nemo_asr

    typer.echo(f"Loading {nemo_path}...")
    m = nemo_asr.models.ASRModel.restore_from(str(nemo_path), map_location="cpu")
    m.eval()
    patch_and_log(m.encoder)

    old_blank = int(m.decoder.blank_idx)
    keep_ids = recover_keep_ids_from_tokenizer_json(
        m.tokenizer, reference_tokenizer_json, old_blank_idx=old_blank
    )
    typer.echo(f"  [vocab-prune] recovered {len(keep_ids)} ids from tokenizer.json")
    prune_vocab_english(m, keep_ids)
    m.decoder._rnnt_export = True

    fused = DecoderJointNoEncProjFusedWrapper(m.decoder.eval(), m.joint.eval()).eval()

    decoder_hidden = int(m.decoder.pred_hidden)
    decoder_layers = int(m.decoder.pred_rnn_layers)
    joint_hidden = int(m.joint.joint_hidden)

    targets = torch.tensor([[m.decoder.blank_idx]], dtype=torch.int32)
    target_len = torch.tensor([1], dtype=torch.int32)
    h = torch.zeros(decoder_layers, 1, decoder_hidden)
    c = torch.zeros(decoder_layers, 1, decoder_hidden)
    enc_proj_step = torch.randn(1, 1, joint_hidden)  # [B, T=1, H=640]

    typer.echo(f"Tracing fused decoder + joint(no enc proj) — joint_hidden={joint_hidden}...")
    traced = torch.jit.trace(fused, (targets, target_len, h, c, enc_proj_step), strict=False)
    settings = ExportSettings(
        output_dir=output_dir,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        max_audio_seconds=30.0,
        max_symbol_steps=1,
        chunk_size_frames=14,
        cache_size=42,
    )
    mlmodel = _coreml_convert(
        traced,
        inputs=[
            ct.TensorType(name="token", shape=(1, 1), dtype=np.int32),
            ct.TensorType(name="token_length", shape=(1,), dtype=np.int32),
            ct.TensorType(name="h_in", shape=_tensor_shape(h), dtype=np.float32),
            ct.TensorType(name="c_in", shape=_tensor_shape(c), dtype=np.float32),
            ct.TensorType(name="encoder_proj", shape=_tensor_shape(enc_proj_step), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float32),
            ct.TensorType(name="h_out", dtype=np.float32),
            ct.TensorType(name="c_out", dtype=np.float32),
        ],
        settings=settings,
        compute_units_override=ct.ComputeUnit.CPU_ONLY,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_pkg = output_dir / "decoder_joint_noencproj.mlpackage"
    mlmodel.save(str(out_pkg))
    typer.echo(f"Saved {out_pkg}")
    typer.echo(f"  Inputs:  token, token_length, h_in, c_in, encoder_proj[1, 1, {joint_hidden}]")
    typer.echo(f"  Outputs: logits, h_out, c_out")


if __name__ == "__main__":
    app()
