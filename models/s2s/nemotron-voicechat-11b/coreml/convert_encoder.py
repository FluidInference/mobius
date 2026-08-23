#!/usr/bin/env python3
"""Export the VoiceChat-11B perception encoder to CoreML.

Builds NeMo ConformerEncoder from the checkpoint's config.json, loads the
sliced weights (components/encoder.safetensors), wraps it with the 1024->4480
LLM projection and the raw-encoder asr_emb tap, and exports a cache-aware
streaming model:

  inputs:  mel [1, 128, total_mel], mel_len [1],
           cache_last_channel [1, L, cache, d], cache_last_time [1, L, d, k],
           cache_last_channel_len [1]
  outputs: audio_embeds [1, T, 4480]  (LLM channel),
           asr_emb      [1, T, 1024]  (RNNT channel),
           enc_len + updated caches

Parity: torch wrapper vs converted CoreML on the same streaming chunk loop
(identical inputs including evolving caches).
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


class VoiceChatEncoderWrapper(torch.nn.Module):
    """Streaming conformer + proj; emits both the LLM channel and the RNNT tap."""

    def __init__(self, encoder: torch.nn.Module, proj: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.proj = proj

    def forward(
        self,
        mel: torch.Tensor,
        mel_len: torch.Tensor,
        cache_last_channel: torch.Tensor,
        cache_last_time: torch.Tensor,
        cache_last_channel_len: torch.Tensor,
    ):
        encoded, enc_len, cache_ch, cache_t, cache_ch_len = self.encoder(
            audio_signal=mel,
            length=mel_len.to(dtype=torch.long),
            cache_last_channel=cache_last_channel.transpose(0, 1),
            cache_last_time=cache_last_time.transpose(0, 1),
            cache_last_channel_len=cache_last_channel_len.to(dtype=torch.int64),
        )
        asr_emb = encoded.transpose(1, 2)  # [B, T, 1024]
        audio_embeds = self.proj(asr_emb)  # [B, T, 4480]
        return (
            audio_embeds,
            asr_emb,
            enc_len.to(dtype=torch.int32),
            cache_ch.transpose(0, 1),
            cache_t.transpose(0, 1),
            cache_ch_len.to(dtype=torch.int32),
        )


def build_model() -> tuple[VoiceChatEncoderWrapper, dict]:
    from nemo.collections.asr.modules import ConformerEncoder

    cfg = json.loads((CHECKPOINT_DIR / "config.json").read_text())
    enc_cfg = dict(cfg["model"]["stt"]["model"]["perception"]["encoder"])
    enc_cfg.pop("_target_")
    encoder = ConformerEncoder(**enc_cfg)
    proj = torch.nn.Linear(enc_cfg["d_model"], cfg["model"]["stt"]["model"]["perception"]["output_dim"])

    weights = load_file(COMPONENTS_DIR / "encoder.safetensors")
    enc_state, proj_state, leftover = {}, {}, []
    for k, v in weights.items():
        if k.startswith("stt_model.perception.encoder."):
            enc_state[k[len("stt_model.perception.encoder."):]] = v
        elif k.startswith("stt_model.perception.proj."):
            proj_state[k[len("stt_model.perception.proj."):]] = v
        else:
            leftover.append(k)
    if leftover:
        typer.echo(f"NOTE: {len(leftover)} unused perception keys: {leftover[:10]}")
    encoder.load_state_dict(enc_state, strict=True)
    proj.load_state_dict(proj_state, strict=True)
    encoder.eval()
    proj.eval()
    return VoiceChatEncoderWrapper(encoder, proj), enc_cfg


@app.command()
def convert(
    output_dir: Path = typer.Option(Path("build/encoder"), help="Output directory"),
    chunk_enc_frames: int = typer.Option(1, help="Encoder frames per step (1 = 80 ms full-duplex cadence)"),
    parity_steps: int = typer.Option(25, help="Streaming steps for the parity loop"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    wrapper, enc_cfg = build_model()
    encoder = wrapper.encoder
    encoder.setup_streaming_params()
    typer.echo(f"streaming_cfg: {encoder.streaming_cfg}")

    cache_ch, cache_t, cache_len = encoder.get_initial_cache_state(batch_size=1, device="cpu")
    cache_ch_b = cache_ch.transpose(0, 1)
    cache_t_b = cache_t.transpose(0, 1)
    cache_len = cache_len.to(torch.int32)
    typer.echo(f"cache shapes: channel={tuple(cache_ch_b.shape)}, time={tuple(cache_t_b.shape)}")

    pre_encode_cache = int(encoder.streaming_cfg.pre_encode_cache_size[1])
    chunk_mel = 8 * chunk_enc_frames
    total_mel = chunk_mel + pre_encode_cache
    typer.echo(f"chunk geometry: chunk_mel={chunk_mel} (+{pre_encode_cache} pre-encode cache) = {total_mel} mel frames/step")

    mel = torch.randn(1, int(enc_cfg["feat_in"]), total_mel) * 2.0
    mel_len = torch.tensor([total_mel], dtype=torch.int32)

    with torch.no_grad():
        ref = wrapper(mel, mel_len, cache_ch_b, cache_t_b, cache_len)
    typer.echo(f"audio_embeds {tuple(ref[0].shape)}, asr_emb {tuple(ref[1].shape)}, enc_len {ref[2].tolist()}")

    traced = torch.jit.trace(
        wrapper, (mel, mel_len, cache_ch_b, cache_t_b, cache_len), strict=False
    )

    inputs = [
        ct.TensorType(name="mel", shape=tuple(mel.shape), dtype=np.float32),
        ct.TensorType(name="mel_length", shape=(1,), dtype=np.int32),
        ct.TensorType(name="cache_last_channel", shape=tuple(cache_ch_b.shape), dtype=np.float32),
        ct.TensorType(name="cache_last_time", shape=tuple(cache_t_b.shape), dtype=np.float32),
        ct.TensorType(name="cache_last_channel_len", shape=(1,), dtype=np.int32),
    ]
    outputs = [
        ct.TensorType(name="audio_embeds", dtype=np.float32),
        ct.TensorType(name="asr_emb", dtype=np.float32),
        ct.TensorType(name="encoded_length", dtype=np.int32),
        ct.TensorType(name="new_cache_last_channel", dtype=np.float32),
        ct.TensorType(name="new_cache_last_time", dtype=np.float32),
        ct.TensorType(name="new_cache_last_channel_len", dtype=np.int32),
    ]

    for precision, tag in ((ct.precision.FLOAT32, "fp32"), (ct.precision.FLOAT16, "fp16")):
        typer.echo(f"converting {tag}...")
        mlmodel = ct.convert(
            traced,
            convert_to="mlprogram",
            inputs=inputs,
            outputs=outputs,
            compute_units=ct.ComputeUnit.CPU_ONLY,
            compute_precision=precision,
            minimum_deployment_target=ct.target.iOS17,
        )
        path = output_dir / f"encoder_{tag}.mlpackage"
        mlmodel.save(str(path))
        typer.echo(f"saved {path}")

        # Streaming parity loop: identical evolving caches through both stacks.
        t_cch, t_ct, t_clen = cache_ch_b.clone(), cache_t_b.clone(), cache_len.clone()
        c_cch, c_ct, c_clen = (
            cache_ch_b.numpy().copy(),
            cache_t_b.numpy().copy(),
            cache_len.numpy().astype(np.int32).copy(),
        )
        max_emb, max_asr = 0.0, 0.0
        for step in range(parity_steps):
            step_mel = torch.randn(1, int(enc_cfg["feat_in"]), total_mel) * 2.0
            with torch.no_grad():
                t_out = wrapper(step_mel, mel_len, t_cch, t_ct, t_clen)
            c_out = mlmodel.predict(
                {
                    "mel": step_mel.numpy(),
                    "mel_length": mel_len.numpy().astype(np.int32),
                    "cache_last_channel": c_cch,
                    "cache_last_time": c_ct,
                    "cache_last_channel_len": c_clen,
                }
            )
            max_emb = max(max_emb, float(np.abs(t_out[0].numpy() - c_out["audio_embeds"]).max()))
            max_asr = max(max_asr, float(np.abs(t_out[1].numpy() - c_out["asr_emb"]).max()))
            t_cch, t_ct, t_clen = t_out[3], t_out[4], t_out[5]
            c_cch, c_ct, c_clen = (
                c_out["new_cache_last_channel"].astype(np.float32),
                c_out["new_cache_last_time"].astype(np.float32),
                c_out["new_cache_last_channel_len"].astype(np.int32),
            )
        typer.echo(f"{tag} parity over {parity_steps} steps: max|Δ| audio_embeds={max_emb:.3e}, asr_emb={max_asr:.3e}")
        thr = 1e-5 if tag == "fp32" else 5e-2  # measured: fp32 8e-07, fp16 1.05e-02
        if max_emb > thr or max_asr > thr:
            typer.echo(f"PARITY FAIL: {tag} exceeds {thr:g}")
            raise typer.Exit(1)


if __name__ == "__main__":
    app()
