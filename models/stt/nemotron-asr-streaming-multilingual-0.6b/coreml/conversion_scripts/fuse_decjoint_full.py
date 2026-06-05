"""One-shot script: produce a vocab-pruned, B1-fused decoder_joint.mlpackage
to drop into an existing engprune build directory.

Mirrors the engprune convert path's prune-and-trace setup so the resulting
fused mlpackage matches the engprune build's joint vocab (e.g. 992) instead
of the unpruned 13088 vocab that the generic decoder_joint_fusion.py emits.

Usage:
    .venv/bin/python conversion_scripts/fuse_decjoint_engprune.py \
        --nemo-path /path/to/nemotron-asr-streaming-multilingual-0.6b.nemo \
        --prune-corpus-jsonl /path/to/english_corpus.jsonl \
        --output-dir build_fp16_engprune_42_13_4480ms_onehotfix
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

from multilingual_components import (  # type: ignore
    EncoderStreamingWithPostPrompt,
    NUM_PROMPTS,
)

_ENG_COREML = (
    THIS_DIR.parent.parent.parent
    / "nemotron-speech-streaming-0.6b"
    / "coreml"
    / "conversion_scripts"
)
sys.path.insert(0, str(_ENG_COREML))

from individual_components import (  # type: ignore
    DecoderWrapper,
    JointWrapper,
    ExportSettings,
    _coreml_convert,
)

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


class DecoderJointFusedWrapper(torch.nn.Module):
    def __init__(self, dec: torch.nn.Module, jnt: torch.nn.Module) -> None:
        super().__init__()
        self.decoder = DecoderWrapper(dec)
        self.joint = JointWrapper(jnt)

    def forward(self, token, token_length, h_in, c_in, encoder):
        dec_out, h_out, c_out = self.decoder(token, token_length, h_in, c_in)
        logits = self.joint(encoder, dec_out)
        return logits, h_out, c_out


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def fuse(
    nemo_path: Path = typer.Option(..., "--nemo-path"),
    prune_corpus_jsonl: Path = typer.Option(..., "--prune-corpus-jsonl"),
    output_dir: Path = typer.Option(..., "--output-dir"),
):
    import nemo.collections.asr as nemo_asr

    typer.echo(f"Loading {nemo_path}...")
    m = nemo_asr.models.ASRModel.restore_from(str(nemo_path), map_location="cpu")
    m.eval()

    # Same patches the engprune convert applies.
    patch_and_log(m.encoder)

    # SCRIPT-BASED keep-set (matches convert_nemotron_multilingual_scriptprune):
    import unicodedata as _ud
    _OTHER = ("CJK", "HIRAGANA", "KATAKANA", "HANGUL", "CYRILLIC", "ARABIC",
              "HEBREW", "DEVANAGARI", "THAI", "BENGALI", "TAMIL", "TELUGU",
              "GEORGIAN", "ARMENIAN")

    def _latin_or_shared(piece: str) -> bool:
        for ch in piece.replace("▁", ""):
            cat = _ud.category(ch)
            if ch.isdigit() or ch.isspace() or cat[0] in ("P", "S", "Z", "C"):
                continue
            try:
                nm = _ud.name(ch)
            except ValueError:
                continue
            if any(s in nm for s in _OTHER):
                return False
        return True

    old_blank = int(m.decoder.blank_idx)
    old_lang_tag_ids = _lang_tag_token_ids(m)
    _vsz = int(m.tokenizer.vocab_size)
    # FULL vocab (no prune): keep every token id (blank appended last).
    _keep = set(range(_vsz))
    _keep.discard(old_blank)
    keep_ids = sorted(_keep) + [old_blank]
    typer.echo(f"  [vocab-prune] keeping {len(keep_ids)} of 13088 tokens")
    prune_vocab_english(m, keep_ids)
    m.decoder._rnnt_export = True

    fused = DecoderJointFusedWrapper(m.decoder.eval(), m.joint.eval()).eval()

    decoder_hidden = int(m.decoder.pred_hidden)
    decoder_layers = int(m.decoder.pred_rnn_layers)
    targets = torch.tensor([[m.decoder.blank_idx]], dtype=torch.int32)
    target_len = torch.tensor([1], dtype=torch.int32)
    h = torch.zeros(decoder_layers, 1, decoder_hidden)
    c = torch.zeros(decoder_layers, 1, decoder_hidden)
    enc_step = torch.randn(1, 1024, 1)

    traced = torch.jit.trace(fused, (targets, target_len, h, c, enc_step), strict=False)
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
            ct.TensorType(name="encoder", shape=_tensor_shape(enc_step), dtype=np.float32),
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
    out_pkg = output_dir / "decoder_joint.mlpackage"
    mlmodel.save(str(out_pkg))
    typer.echo(f"Saved {out_pkg}")


if __name__ == "__main__":
    app()
