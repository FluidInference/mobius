#!/usr/bin/env python3
"""B1 fusion: export a single decoder+joint CoreML model for Nemotron streaming.

Merges the RNNT prediction network (decoder LSTM) and the joint network into one
mlpackage so the streaming decode loop makes ONE CoreML dispatch per step instead
of two. Argmax stays in Swift (NOT fused in) — fusing argmax over the vocab forces
ANE->CPU and regresses RTFx (the investigation's "B2" dead-end).

  inputs : token[1,1] i32, token_length[1] i32, h_in[2,1,640], c_in[2,1,640],
           encoder_step[1,1024,1]
  outputs: logits[1,1,1,vocab], h_out[2,1,640], c_out[2,1,640]

decoder/joint are chunk-independent, so one build serves every chunk tier.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Tuple

import coremltools as ct
import numpy as np
import torch
import typer

import nemo.collections.asr as nemo_asr
from individual_components import DecoderWrapper, JointWrapper, ExportSettings, _coreml_convert

DEFAULT_MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"


class DecoderJointFusedWrapper(torch.nn.Module):
    def __init__(self, decoder: DecoderWrapper, joint: JointWrapper) -> None:
        super().__init__()
        self.decoder = decoder
        self.joint = joint

    def forward(self, token, token_length, h_in, c_in, encoder_step):
        dec_out, h_out, c_out = self.decoder(token, token_length, h_in, c_in)  # [1,640,1]
        logits = self.joint(encoder_step, dec_out)  # [1,1,1,vocab]
        return logits, h_out, c_out


app = typer.Typer(add_completion=False)


@app.command()
def convert(
    output: Path = typer.Option(..., help="Output .mlpackage path"),
    precision: str = typer.Option("FLOAT16", help="FLOAT16 or FLOAT32"),
) -> None:
    typer.echo("Loading model...")
    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(DEFAULT_MODEL_ID, map_location="cpu")
    model.eval()
    model.decoder._rnnt_export = True

    decoder = DecoderWrapper(model.decoder.eval())
    joint = JointWrapper(model.joint.eval())
    fused = DecoderJointFusedWrapper(decoder, joint).eval()

    dh = int(model.decoder.pred_hidden)
    dl = int(model.decoder.pred_rnn_layers)
    enc_dim = int(model.joint.enc.in_features) if hasattr(model.joint, "enc") else 1024

    token = torch.tensor([[model.decoder.blank_idx]], dtype=torch.int32)
    token_len = torch.tensor([1], dtype=torch.int32)
    h = torch.zeros(dl, 1, dh)
    c = torch.zeros(dl, 1, dh)
    enc_step = torch.randn(1, enc_dim, 1)

    settings = ExportSettings(
        output_dir=output.parent,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16 if precision.upper() == "FLOAT16" else ct.precision.FLOAT32,
        max_audio_seconds=30.0,
        max_symbol_steps=1,
        chunk_size_frames=14,
        cache_size=70,
    )

    traced = torch.jit.trace(fused, (token, token_len, h, c, enc_step), strict=False)
    inputs = [
        ct.TensorType(name="token", shape=(1, 1), dtype=np.int32),
        ct.TensorType(name="token_length", shape=(1,), dtype=np.int32),
        ct.TensorType(name="h_in", shape=tuple(h.shape), dtype=np.float32),
        ct.TensorType(name="c_in", shape=tuple(c.shape), dtype=np.float32),
        ct.TensorType(name="encoder", shape=tuple(enc_step.shape), dtype=np.float32),
    ]
    outputs = [
        ct.TensorType(name="logits", dtype=np.float32),
        ct.TensorType(name="h_out", dtype=np.float32),
        ct.TensorType(name="c_out", dtype=np.float32),
    ]
    mlmodel = _coreml_convert(traced, inputs, outputs, settings, ct.ComputeUnit.CPU_AND_NE)
    mlmodel.save(str(output))
    typer.echo(f"Done! Fused decoder+joint -> {output}")


if __name__ == "__main__":
    app()
