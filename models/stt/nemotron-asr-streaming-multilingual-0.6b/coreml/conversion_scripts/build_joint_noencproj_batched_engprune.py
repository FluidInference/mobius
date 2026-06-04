"""Build a vocab-pruned `joint_noencproj_batched.mlpackage`.

Smart speculative blank decode: the prior `joint_batched` failed because
it re-ran the full joint (encoder_proj 1024→640 + post-projection) on
every emission. The encoder_proj matmul is 56× more work in the batched
case, defeating any speculative win.

This variant skips encoder_proj — encoder already produces
`encoder_proj [1, 640, T]` (B3 split), so the joint just broadcasts +
ReLU + final linear. Cost per batched call ≈ K × cheap final-linear vs
K × full joint.

Joint forward (mirrors `JointWithoutEncProjBatched` semantics):
  encoder_proj [B=1, T=K, joint_dim=640] (already projected)
  dec_out     [B=1, U=1, decoder_hidden=640]
  → dec_proj = joint.pred(dec_out)         [B, U, 640]
  → x = encoder_proj.unsqueeze(2) + dec_proj.unsqueeze(1)  # [B, T, U, 640]
  → x = joint.joint_net[0..2](x)           # ReLU → Dropout → Linear
  → logits [B, T=K, U=1, V]

Caller (Swift) batched-skip loop:
  dec_out = decoder(token, state)
  logits_K = joint_noencproj_batched(encoder_proj[t:t+K], dec_out)
  for k in 0..<K:
      if argmax(logits_K[k]) != blank:
          emit, advance state, jump to t+k+1, recompute dec_out
          break
  else:
      t += K  # all blank — fast skip

Usage:
    .venv/bin/python conversion_scripts/build_joint_noencproj_batched_engprune.py \\
        --nemo-path /path/to/nemotron-asr-streaming-multilingual-0.6b.nemo \\
        --prune-corpus-jsonl /path/to/corpus.jsonl \\
        --output-dir build_lp_engprune_42_13_4480ms_v3 \\
        --batch-frames 8
"""
from __future__ import annotations

import json as _json
import sys
from pathlib import Path
from typing import List

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

from individual_components import ExportSettings, _coreml_convert  # type: ignore
from patch_uniquebias import patch_and_log  # type: ignore
from patch_vocab_prune import (  # type: ignore
    build_english_keep_set,
    prune_vocab_english,
)

_LANG_TAG_RE = __import__("re").compile(r"^<[A-Za-z]{2,4}-[A-Za-z]{2,4}>$")


def _lang_tag_token_ids(model) -> List[int]:
    ids: List[int] = []
    vocab_size = int(model.tokenizer.vocab_size)
    for i in range(vocab_size):
        tok = model.tokenizer.ids_to_tokens([i])[0]
        if _LANG_TAG_RE.match(tok):
            ids.append(i)
    return ids


def _tensor_shape(t):
    return tuple(int(s) for s in t.shape)


class JointNoEncProjBatched(torch.nn.Module):
    """Batched joint over K already-projected encoder frames.

    Skips the 1024→640 encoder projection (caller passes `encoder_proj`
    direct from the encoder's `encoder_proj` output). The remaining
    operations (pred projection of dec_out, broadcast add, ReLU, output
    linear) are cheap per-frame, so batched-K cost ≈ K × cheap op rather
    than K × full joint.
    """

    def __init__(self, joint_module: torch.nn.Module) -> None:
        super().__init__()
        self.joint_module = joint_module

    def forward(self, encoder_proj: torch.Tensor, dec_out: torch.Tensor):
        # encoder_proj: [B=1, T=K, joint_dim=640] (pre-projected)
        # dec_out:     [B=1, decoder_hidden=640, U=1]
        decoder_outputs = dec_out.transpose(1, 2)  # [B, U, D]
        dec_proj = self.joint_module.pred(decoder_outputs)  # [B, U, joint_dim]
        x = encoder_proj.unsqueeze(2) + dec_proj.unsqueeze(1)  # [B, T, U, joint_dim]
        x = self.joint_module.joint_net[0](x)  # ReLU
        x = self.joint_module.joint_net[1](x)  # Dropout
        logits = self.joint_module.joint_net[2](x)  # Linear → [B, T, U, V]
        return logits


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def build(
    nemo_path: Path = typer.Option(..., "--nemo-path"),
    prune_corpus_jsonl: Path = typer.Option(..., "--prune-corpus-jsonl"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    batch_frames: int = typer.Option(
        8,
        "--batch-frames",
        help="K encoder_proj frames per joint call (speculative skip window).",
    ),
):
    import nemo.collections.asr as nemo_asr

    typer.echo(f"Loading {nemo_path}...")
    m = nemo_asr.models.ASRModel.restore_from(str(nemo_path), map_location="cpu")
    m.eval()

    patch_and_log(m.encoder)

    texts = []
    with open(prune_corpus_jsonl) as f:
        for line in f:
            r = _json.loads(line)
            for k in ("hyp_raw", "ref_raw"):
                t = r.get(k)
                if t:
                    texts.append(t)
    old_blank = int(m.decoder.blank_idx)
    old_lang_tag_ids = _lang_tag_token_ids(m)
    keep_ids = build_english_keep_set(
        m.tokenizer, texts, lang_tag_ids=old_lang_tag_ids, old_blank_idx=old_blank
    )
    typer.echo(f"  [vocab-prune] keeping {len(keep_ids)} of 13088 tokens")
    prune_vocab_english(m, keep_ids)

    joint_batched = JointNoEncProjBatched(m.joint.eval()).eval()

    # Trace inputs — encoder_proj already in 640 dim, dec_out [1, 640, 1]
    enc_proj_batch = torch.randn(1, batch_frames, 640)
    dec_step = torch.randn(1, 640, 1)

    traced = torch.jit.trace(joint_batched, (enc_proj_batch, dec_step), strict=False)
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
            ct.TensorType(
                name="encoder_proj",
                shape=_tensor_shape(enc_proj_batch),
                dtype=np.float32,
            ),
            ct.TensorType(name="decoder", shape=_tensor_shape(dec_step), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        settings=settings,
        compute_units_override=ct.ComputeUnit.CPU_ONLY,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_pkg = output_dir / "joint_noencproj_batched.mlpackage"
    mlmodel.save(str(out_pkg))
    typer.echo(f"Saved {out_pkg}")
    typer.echo(f"  encoder_proj input shape: [1, {batch_frames}, 640]")
    typer.echo(f"  output logits shape:      [1, {batch_frames}, 1, {len(keep_ids)}]")


if __name__ == "__main__":
    app()
