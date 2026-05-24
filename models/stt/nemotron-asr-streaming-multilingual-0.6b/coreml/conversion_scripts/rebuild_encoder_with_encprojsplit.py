"""Rebuild ONLY the encoder.mlpackage to additionally emit `encoder_proj`.

B3 split: encoder emits `joint.enc(encoded)` once per chunk (computed
once, ANE-resident), so the decoder loop's joint can skip the
1024→640 enc_proj matmul on every token.

Pairs with the existing `joint_noencproj_batched.mlpackage` (built in
the previous step) — and unlocks the smarter speculative blank
decode path.

This script ONLY rebuilds the encoder for a given output_dir, leaving
the rest of the build (preprocessor, decoder, joint, decoder_joint,
tokenizer, metadata) untouched. Backup the old encoder before run.

Outputs (encoder.mlpackage emits these tensors):
  encoded             [1, 1024, T_enc]  — unchanged from prior
  encoded_length      [1]                — unchanged
  cache_channel_out   [1, ...]           — unchanged
  cache_time_out      [1, ...]           — unchanged
  cache_len_out       [1]                — unchanged
  encoder_proj        [1, T_enc, 640]    — NEW (pre-projected via joint.enc)

After this, must also rebuild a vocab-matched `joint_noencproj.mlpackage`
(single-frame) and re-fuse `decoder_joint_noencproj.mlpackage` to match
the keep-set. The batched joint_noencproj_batched can stay if its
keep-set was built from the same corpus.

Usage:
    .venv/bin/python conversion_scripts/rebuild_encoder_with_encprojsplit.py \\
        --nemo-path ... --prune-corpus-jsonl ... \\
        --output-dir build_lp_engprune_42_13_4480ms_v3_encprojsplit \\
        --att-context 56,13 --chunk-mel-frames 448
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
from multilingual_components_encproj import (  # type: ignore
    EncoderStreamingWithPostPromptAndEncProj,
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


def _parse_cu(s: str) -> ct.ComputeUnit:
    return {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    }[s.upper()]


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def build(
    nemo_path: Path = typer.Option(..., "--nemo-path"),
    prune_corpus_jsonl: Path = typer.Option(..., "--prune-corpus-jsonl"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    att_context: str = typer.Option("56,13", "--att-context"),
    chunk_mel_frames: int = typer.Option(448, "--chunk-mel-frames"),
    pre_encode_cache: int = typer.Option(9, "--pre-encode-cache"),
    encoder_cu: str = typer.Option("CPU_AND_NE", "--encoder-cu"),
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

    # Configure attention context
    left, right = [int(x) for x in att_context.split(",")]
    m.encoder.set_default_att_context_size([left, right])

    NUM_PROMPTS = 128
    encoder_streaming = EncoderStreamingWithPostPromptAndEncProj(
        m.encoder.eval(),
        m.prompt_kernel.eval(),
        joint_enc=m.joint.enc.eval(),
        num_prompts=NUM_PROMPTS,
    )

    # Initial cache state
    sample_rate = 16000
    mel_features = 128
    total_mel_frames = chunk_mel_frames + pre_encode_cache

    cc, ct_, cl = m.encoder.get_initial_cache_state(batch_size=1, device="cpu")
    cache_channel_b = cc.transpose(0, 1).contiguous()
    cache_time_b = ct_.transpose(0, 1).contiguous()
    cache_len = cl.to(torch.int32)

    mel = torch.randn(1, mel_features, total_mel_frames)
    mel_len = torch.tensor([total_mel_frames], dtype=torch.int32)
    prompt_id = torch.tensor([0], dtype=torch.int32)

    typer.echo("Tracing encoder with encoder_proj output...")
    traced = torch.jit.trace(
        encoder_streaming,
        (mel, mel_len, cache_channel_b, cache_time_b, cache_len, prompt_id),
        strict=False,
    )

    settings = ExportSettings(
        output_dir=output_dir,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        max_audio_seconds=80.0,
        max_symbol_steps=1,
        chunk_size_frames=chunk_mel_frames // 8,
        cache_size=cache_channel_b.shape[2],
    )

    typer.echo("Converting encoder mlprogram (with encoder_proj output)...")
    mlmodel = _coreml_convert(
        traced,
        inputs=[
            ct.TensorType(name="mel", shape=_tensor_shape(mel), dtype=np.float32),
            ct.TensorType(name="mel_length", shape=(1,), dtype=np.int32),
            ct.TensorType(
                name="cache_channel",
                shape=_tensor_shape(cache_channel_b),
                dtype=np.float32,
            ),
            ct.TensorType(
                name="cache_time",
                shape=_tensor_shape(cache_time_b),
                dtype=np.float32,
            ),
            ct.TensorType(name="cache_len", shape=(1,), dtype=np.int32),
            ct.TensorType(name="prompt_id", shape=(1,), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="encoded", dtype=np.float32),
            ct.TensorType(name="encoded_length", dtype=np.int32),
            ct.TensorType(name="cache_channel_out", dtype=np.float32),
            ct.TensorType(name="cache_time_out", dtype=np.float32),
            ct.TensorType(name="cache_len_out", dtype=np.int32),
            ct.TensorType(name="encoder_proj", dtype=np.float32),  # NEW
        ],
        settings=settings,
        compute_units_override=_parse_cu(encoder_cu),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_pkg = output_dir / "encoder.mlpackage"
    mlmodel.save(str(out_pkg))
    typer.echo(f"Saved {out_pkg}")
    typer.echo(f"  Outputs: encoded, encoded_length, cache_channel_out, cache_time_out, cache_len_out, encoder_proj")
    typer.echo(f"  encoder_proj shape: [1, T_enc={chunk_mel_frames // 8}, 640]")


if __name__ == "__main__":
    app()
