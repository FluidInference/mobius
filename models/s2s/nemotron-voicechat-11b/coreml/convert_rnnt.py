#!/usr/bin/env python3
"""Export the VoiceChat-11B RNNT user-transcription head to CoreML.

Two single-step models (same split FluidAudio's Nemotron streaming stack uses):
  decoder.mlpackage: token [1,1] + LSTM state (h,c [2,1,640]) -> decoder_out [1,640,1] + new state
  joint.mlpackage:   asr_emb frame [1,1024,1] + decoder_out [1,640,1] -> logits [1,1,1,1025]

Configs come from config.json `_rnnt_merge_info`; weights from components/rnnt.safetensors.
"""
from __future__ import annotations

import json
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import typer
from safetensors.torch import load_file

CHECKPOINT_DIR = Path.home() / "Documents/models/voicechat-11b"
COMPONENTS_DIR = CHECKPOINT_DIR / "components"

app = typer.Typer(add_completion=False)


class DecoderStepWrapper(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(self, token: torch.Tensor, h_in: torch.Tensor, c_in: torch.Tensor):
        out, _, state = self.decoder(
            targets=token.to(dtype=torch.long),
            target_length=torch.ones(1, dtype=torch.long),
            states=[h_in, c_in],
        )
        return out, state[0], state[1]


class JointStepWrapper(torch.nn.Module):
    def __init__(self, joint: torch.nn.Module) -> None:
        super().__init__()
        self.joint = joint

    def forward(self, encoder_step: torch.Tensor, decoder_step: torch.Tensor):
        enc = self.joint.enc(encoder_step.transpose(1, 2))  # [1,1,640]
        dec = self.joint.pred(decoder_step.transpose(1, 2))  # [1,1,640]
        x = enc.unsqueeze(2) + dec.unsqueeze(1)  # [1,1,1,640]
        x = self.joint.joint_net[0](x)
        x = self.joint.joint_net[1](x)
        return self.joint.joint_net[2](x)  # [1,1,1,1025]


def build_models():
    from nemo.collections.asr.modules import RNNTDecoder, RNNTJoint

    cfg = json.loads((CHECKPOINT_DIR / "config.json").read_text())
    info = cfg["_rnnt_merge_info"]
    dec_cfg, joint_cfg = info["decoder_config"], info["joint_config"]

    decoder = RNNTDecoder(
        prednet=dec_cfg["prednet"],
        vocab_size=dec_cfg["vocab_size"],
        blank_as_pad=dec_cfg["blank_as_pad"],
    )
    joint = RNNTJoint(
        jointnet=joint_cfg["jointnet"],
        num_classes=joint_cfg["num_classes"],
        vocabulary=joint_cfg["vocabulary"],
    )

    weights = load_file(COMPONENTS_DIR / "rnnt.safetensors")
    dec_state = {k[len("stt_model.rnnt_decoder."):]: v for k, v in weights.items() if k.startswith("stt_model.rnnt_decoder.")}
    joint_state = {k[len("stt_model.rnnt_joint."):]: v for k, v in weights.items() if k.startswith("stt_model.rnnt_joint.")}
    decoder.load_state_dict(dec_state, strict=True)
    joint.load_state_dict(joint_state, strict=True)
    decoder.eval()
    joint.eval()
    decoder._rnnt_export = True
    return decoder, joint


@app.command()
def convert(output_dir: Path = typer.Option(Path("build/rnnt"), help="Output directory")) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    decoder, joint = build_models()
    pred_hidden = decoder.pred_hidden

    dec_wrap = DecoderStepWrapper(decoder)
    joint_wrap = JointStepWrapper(joint)

    token = torch.tensor([[42]], dtype=torch.int32)
    h = torch.zeros(2, 1, pred_hidden)
    c = torch.zeros(2, 1, pred_hidden)
    enc_step = torch.randn(1, 1024, 1)

    with torch.no_grad():
        dec_ref = dec_wrap(token, h, c)
        joint_ref = joint_wrap(enc_step, dec_ref[0])
    typer.echo(f"decoder_out {tuple(dec_ref[0].shape)}, logits {tuple(joint_ref.shape)}")

    traced_dec = torch.jit.trace(dec_wrap, (token, h, c), strict=False)
    mldec = ct.convert(
        traced_dec,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="token", shape=(1, 1), dtype=np.int32),
            ct.TensorType(name="h_in", shape=(2, 1, pred_hidden), dtype=np.float32),
            ct.TensorType(name="c_in", shape=(2, 1, pred_hidden), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="decoder_out", dtype=np.float32),
            ct.TensorType(name="h_out", dtype=np.float32),
            ct.TensorType(name="c_out", dtype=np.float32),
        ],
        compute_units=ct.ComputeUnit.CPU_ONLY,
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS17,
    )
    mldec.save(str(output_dir / "decoder.mlpackage"))

    traced_joint = torch.jit.trace(joint_wrap, (enc_step, dec_ref[0]), strict=False)
    mljoint = ct.convert(
        traced_joint,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="encoder_step", shape=(1, 1024, 1), dtype=np.float32),
            ct.TensorType(name="decoder_step", shape=(1, pred_hidden, 1), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        compute_units=ct.ComputeUnit.CPU_ONLY,
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS17,
    )
    mljoint.save(str(output_dir / "joint.mlpackage"))

    # Parity: 20 chained decoder steps + joint on each.
    max_dec, max_logit = 0.0, 0.0
    t_h, t_c = h.clone(), c.clone()
    c_h, c_c = h.numpy().copy(), c.numpy().copy()
    for step in range(20):
        tok = torch.tensor([[step * 37 % 1025]], dtype=torch.int32)
        step_enc = torch.randn(1, 1024, 1)
        with torch.no_grad():
            t_dec = dec_wrap(tok, t_h, t_c)
            t_log = joint_wrap(step_enc, t_dec[0])
        c_dec = mldec.predict({"token": tok.numpy(), "h_in": c_h, "c_in": c_c})
        c_log = mljoint.predict(
            {"encoder_step": step_enc.numpy(), "decoder_step": c_dec["decoder_out"].astype(np.float32)}
        )
        max_dec = max(max_dec, float(np.abs(t_dec[0].numpy() - c_dec["decoder_out"]).max()))
        max_logit = max(max_logit, float(np.abs(t_log.numpy() - c_log["logits"]).max()))
        t_h, t_c = t_dec[1], t_dec[2]
        c_h, c_c = c_dec["h_out"].astype(np.float32), c_dec["c_out"].astype(np.float32)
    typer.echo(f"fp32 parity over 20 chained steps: max|Δ| decoder_out={max_dec:.3e}, logits={max_logit:.3e}")
    if max_dec > 1e-3 or max_logit > 5e-3:  # measured: 1.4e-05 / 5.5e-04
        typer.echo("PARITY FAIL")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
