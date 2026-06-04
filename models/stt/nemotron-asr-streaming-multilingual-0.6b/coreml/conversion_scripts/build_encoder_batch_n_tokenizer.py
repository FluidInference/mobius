"""Build a batch-N encoder.mlpackage (true batch-dim=N, single dispatch).

STATUS: Achieved-but-aggregate-only. Validated on macOS 26.5 with
coremltools 9.0 — runtime load works (previous attempt's failure is
gone), and the GPU dispatch wall shrinks substantially at batch=N:

    batch=1:  13.49 ms/call
    batch=2:  16.56 ms/call  ( 8.28 ms/stream, -38.6% vs serial)
    batch=4:  24.44 ms/call  ( 6.11 ms/stream, -54.7%)
    batch=8:  42.27 ms/call  ( 5.28 ms/stream, -60.8%)
    batch=16: 75.56 ms/call  ( 4.72 ms/stream, -65.0%)

WHY THIS IS NOT IN PRODUCTION: per-file test-clean RTFx (the canonical
metric — single audio file, single stream) is unaffected. A single file's
chunks are sequentially cache-dependent — chunk t+1's encoder needs
chunk t's output cache — so true batch dispatch only helps when running
N INDEPENDENT files concurrently. That's an aggregate-throughput win,
not a per-file-latency win.

If aggregate (operator-side fleet throughput) ever becomes a tracked
metric, the Swift integration TODO is:
  1. New SharedBatchedNemotronModels with batch-N encoder mlpackage
  2. Multi-stream coordinator that gathers N streams at chunk boundaries
  3. Scatter outputs back to per-stream cache state
  4. Pad / shrink batch when streams complete

Until then, this script just builds the asset so the win is reproducible.

Mirrors the production v4_swiftencproj_layerpos build with batch_size=N
instead of 1. Same prompt/encoder_proj output schema, FLOAT16 precision,
iOS18 target.
"""
from __future__ import annotations

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
from patch_vocab_prune import prune_vocab_english  # type: ignore
from build_joint_noencproj_batched_from_tokenizer import (  # type: ignore
    recover_keep_ids_from_tokenizer_json,
)


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
    reference_tokenizer_json: Path = typer.Option(..., "--reference-tokenizer-json"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    batch_size: int = typer.Option(2, "--batch-size"),
    att_context: str = typer.Option("42,13", "--att-context"),
    chunk_mel_frames: int = typer.Option(448, "--chunk-mel-frames"),
    pre_encode_cache: int = typer.Option(9, "--pre-encode-cache"),
    encoder_cu: str = typer.Option("CPU_AND_GPU", "--encoder-cu"),
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

    left, right = [int(x) for x in att_context.split(",")]
    m.encoder.set_default_att_context_size([left, right])

    NUM_PROMPTS = 128
    encoder_streaming = EncoderStreamingWithPostPromptAndEncProj(
        m.encoder.eval(),
        m.prompt_kernel.eval(),
        joint_enc=m.joint.enc.eval(),
        num_prompts=NUM_PROMPTS,
    )

    mel_features = 128
    total_mel_frames = chunk_mel_frames + pre_encode_cache

    cc, ct_, cl = m.encoder.get_initial_cache_state(batch_size=batch_size, device="cpu")
    cache_channel_b = cc.transpose(0, 1).contiguous()  # [N, layers, T, D]
    cache_time_b = ct_.transpose(0, 1).contiguous()    # [N, layers, D, W]
    cache_len = cl.to(torch.int32)                     # [N]
    typer.echo(f"  Batched cache_channel: {_tensor_shape(cache_channel_b)}")
    typer.echo(f"  Batched cache_time: {_tensor_shape(cache_time_b)}")

    mel = torch.randn(batch_size, mel_features, total_mel_frames)
    mel_len = torch.tensor([total_mel_frames] * batch_size, dtype=torch.int32)
    prompt_id = torch.tensor([0] * batch_size, dtype=torch.int32)

    typer.echo(f"Tracing batched encoder (N={batch_size}, encoder_proj output)...")
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

    typer.echo("Converting batched encoder → mlprogram...")
    mlmodel = _coreml_convert(
        traced,
        inputs=[
            ct.TensorType(name="mel", shape=_tensor_shape(mel), dtype=np.float32),
            ct.TensorType(name="mel_length", shape=(batch_size,), dtype=np.int32),
            ct.TensorType(name="cache_channel", shape=_tensor_shape(cache_channel_b), dtype=np.float32),
            ct.TensorType(name="cache_time", shape=_tensor_shape(cache_time_b), dtype=np.float32),
            ct.TensorType(name="cache_len", shape=(batch_size,), dtype=np.int32),
            ct.TensorType(name="prompt_id", shape=(batch_size,), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="encoded", dtype=np.float32),
            ct.TensorType(name="encoded_length", dtype=np.int32),
            ct.TensorType(name="cache_channel_out", dtype=np.float32),
            ct.TensorType(name="cache_time_out", dtype=np.float32),
            ct.TensorType(name="cache_len_out", dtype=np.int32),
            ct.TensorType(name="encoder_proj", dtype=np.float32),
        ],
        settings=settings,
        compute_units_override=_parse_cu(encoder_cu),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_pkg = output_dir / "encoder.mlpackage"
    mlmodel.save(str(out_pkg))
    typer.echo(f"Saved {out_pkg}")
    typer.echo(f"  batch_size={batch_size}, encoded output: [{batch_size}, 1024, {chunk_mel_frames // 8}]")


if __name__ == "__main__":
    app()
