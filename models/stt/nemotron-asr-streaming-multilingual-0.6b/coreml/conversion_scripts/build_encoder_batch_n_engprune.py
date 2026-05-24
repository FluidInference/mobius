"""Build a batch-N encoder.mlpackage for multi-stream parallel inference.

Current multi-stream issues N independent encoder predictions against
shared MLModel handles (N managers each calling encoder.prediction()).
ANE saturates at N=3 (170 RTFx aggregate, +1% from N=4 — flat).

This variant retraces the encoder with batch_dim=N so a single
encoder.prediction() handles N streams at once. Hypothesis: amortizes
ANE dispatch overhead → push past N=3 saturation, or get N=2-like
throughput from a single call (lower per-stream wall, same aggregate).

Inputs (all batched):
    mel           [N, mel_features, total_mel_frames]
    mel_length    [N]
    cache_channel [N, layers, channel_dim, encoder_dim]
    cache_time    [N, layers, encoder_dim, time_dim]
    cache_len     [N]
    prompt_id     [N]

Outputs (all batched):
    encoded             [N, encoder_dim, chunk_frames]
    encoded_length      [N]
    cache_channel_out   [N, layers, channel_dim, encoder_dim]
    cache_time_out      [N, layers, encoder_dim, time_dim]
    cache_len_out       [N]

NOTE: NVIDIA's conformer encoder supports batched forward natively
(training was batched). This converter mostly just changes the trace
input shapes; the PyTorch graph is unchanged. The risk is that
specific ops (chunked-attention masks, prompt MLP) may have N=1
assumptions in the wrapper.

Swift integration (separate work, not done here):
- Multi-stream manager gathers N streams' inputs into one batched call
- Synchronize to a single chunk boundary OR pad short streams
- Scatter N outputs back to each stream's cache state

Usage:
    .venv/bin/python conversion_scripts/build_encoder_batch_n_engprune.py \\
        --nemo-path ... --prune-corpus-jsonl ... \\
        --output-dir build_lp_engprune_42_13_4480ms_v3 \\
        --batch-size 2 \\
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
from patch_uniquebias import patch_and_log  # type: ignore
from patch_vocab_prune import (  # type: ignore
    build_english_keep_set,
    prune_vocab_english,
)

# Import the prompt-aware encoder wrapper from the existing engprune converter
sys.path.insert(0, str(THIS_DIR))
# (EncoderStreamingWithPostPrompt lives in mobius's local conversion_scripts;
# the convert_nemotron_multilingual_engprune.py imports it.)


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


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def build(
    nemo_path: Path = typer.Option(..., "--nemo-path"),
    prune_corpus_jsonl: Path = typer.Option(..., "--prune-corpus-jsonl"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    batch_size: int = typer.Option(2, "--batch-size", help="N streams batched per call."),
    att_context: str = typer.Option("56,13", "--att-context"),
    chunk_mel_frames: int = typer.Option(448, "--chunk-mel-frames"),
    pre_encode_cache: int = typer.Option(9, "--pre-encode-cache"),
    encoder_cu: str = typer.Option("CPU_AND_NE", "--encoder-cu"),
):
    import nemo.collections.asr as nemo_asr
    from convert_nemotron_multilingual_engprune import (
        EncoderStreamingWithPostPrompt,
    )

    typer.echo(f"Loading {nemo_path}...")
    m = nemo_asr.models.ASRModel.restore_from(str(nemo_path), map_location="cpu")
    m.eval()

    # Apply same patches the production engprune build uses, so the
    # batch-N encoder matches the single-batch graph exactly.
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
    encoder_streaming = EncoderStreamingWithPostPrompt(
        m.encoder.eval(),
        m.prompt_kernel.eval(),
        num_prompts=NUM_PROMPTS,
    )

    # Get initial cache shapes and batch them to N
    sample_rate = 16000
    mel_features = 128
    total_mel_frames = chunk_mel_frames + pre_encode_cache

    cc, ct_, cl = m.encoder.get_initial_cache_state(batch_size=batch_size, device="cpu")
    cache_channel_b = cc.transpose(0, 1).contiguous()  # → [N, layers, channel_dim, dim]
    cache_time_b = ct_.transpose(0, 1).contiguous()    # → [N, layers, dim, time_dim]
    cache_len = cl.to(torch.int32)                     # [N]
    typer.echo(
        f"  Batched cache shapes: cache_channel={tuple(cache_channel_b.shape)}, "
        f"cache_time={tuple(cache_time_b.shape)}, cache_len={tuple(cache_len.shape)}"
    )

    mel = torch.randn(batch_size, mel_features, total_mel_frames)
    mel_len = torch.tensor([total_mel_frames] * batch_size, dtype=torch.int32)
    prompt_id = torch.tensor([0] * batch_size, dtype=torch.int32)

    typer.echo("Tracing batched encoder...")
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

    def _parse_cu(s: str) -> ct.ComputeUnit:
        return {
            "ALL": ct.ComputeUnit.ALL,
            "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
            "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
            "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        }[s.upper()]

    typer.echo("Converting batched encoder to mlprogram...")
    mlmodel = _coreml_convert(
        traced,
        inputs=[
            ct.TensorType(name="mel", shape=_tensor_shape(mel), dtype=np.float32),
            ct.TensorType(name="mel_length", shape=(batch_size,), dtype=np.int32),
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
            ct.TensorType(name="cache_len", shape=(batch_size,), dtype=np.int32),
            ct.TensorType(name="prompt_id", shape=(batch_size,), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="encoded", dtype=np.float32),
            ct.TensorType(name="encoded_length", dtype=np.int32),
            ct.TensorType(name="cache_channel_out", dtype=np.float32),
            ct.TensorType(name="cache_time_out", dtype=np.float32),
            ct.TensorType(name="cache_len_out", dtype=np.int32),
        ],
        settings=settings,
        compute_units_override=_parse_cu(encoder_cu),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_pkg = output_dir / f"encoder_batch{batch_size}.mlpackage"
    mlmodel.save(str(out_pkg))
    typer.echo(f"Saved {out_pkg}")
    typer.echo(f"  Inputs all batched at dim 0 = {batch_size}")
    typer.echo(f"  encoded output shape: [{batch_size}, encoder_dim, {chunk_mel_frames // 8}]")


if __name__ == "__main__":
    app()
