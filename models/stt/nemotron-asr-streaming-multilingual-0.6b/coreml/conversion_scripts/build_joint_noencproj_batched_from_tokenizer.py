"""Build `joint_noencproj_batched.mlpackage` using an existing pruned
tokenizer.json as the keep-set source.

This avoids ANY fresh corpus involvement — the keep-set is recovered
verbatim from the shipped tokenizer.json of an already-built bundle
(e.g. build_lp_engprune_42_13_1120ms_v3). The resulting joint vocab
layout is byte-identical to that bundle's decoder/joint, so the new
smart-spec asset slots into the existing 1120ms build without any
vocab-policy change and without any new corpus passing through the
build.

Use this when:
  - You already have a shipped (decoder + joint) build at one chunk
    size and want to add the smart-spec companion (joint_noencproj_
    batched.mlpackage + native_weights) at the SAME vocab.
  - You want to validate smart-spec at a different chunk size without
    re-pruning the vocab from a corpus (which could leak test set or
    drift policy).

Usage:
    .venv/bin/python conversion_scripts/build_joint_noencproj_batched_from_tokenizer.py \\
        --nemo-path /path/to/nemotron-asr-streaming-multilingual-0.6b.nemo \\
        --reference-tokenizer-json /path/to/build_lp_engprune_42_13_1120ms_v3/tokenizer.json \\
        --output-dir build_lp_engprune_42_13_1120ms_v3 \\
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
from patch_vocab_prune import prune_vocab_english  # type: ignore


def _tensor_shape(t):
    return tuple(int(s) for s in t.shape)


def recover_keep_ids_from_tokenizer_json(
    nemo_tokenizer,
    reference_tokenizer_json: Path,
    old_blank_idx: int,
) -> List[int]:
    """Reverse the prune mapping: read surface strings from the shipped
    tokenizer.json (new_id → surface) and look each up in the full
    NeMo vocab (surface → original_id).

    Returns sorted keep_ids with old_blank_idx appended last — exactly
    the shape that `prune_vocab_english` expects.
    """
    with open(reference_tokenizer_json) as f:
        new_id_to_surface = _json.load(f)

    # Build the inverse map from the full NeMo vocab.
    full_vocab_size = int(nemo_tokenizer.vocab_size)
    surface_to_original = {}
    for original_id in range(full_vocab_size):
        surface = nemo_tokenizer.ids_to_tokens([original_id])[0]
        # Multiple original_ids could share a surface; we'd hit that
        # case only for special tokens. Track the first occurrence
        # (matches sorted-order recovery).
        surface_to_original.setdefault(surface, original_id)

    recovered: List[int] = []
    missing: List[str] = []
    for new_id_str, surface in sorted(new_id_to_surface.items(), key=lambda kv: int(kv[0])):
        new_id = int(new_id_str)
        if surface == "<blank>":
            # Skip — we'll append the canonical blank at the end.
            continue
        if surface not in surface_to_original:
            missing.append(f"new_id={new_id} surface={surface!r}")
            continue
        recovered.append(surface_to_original[surface])

    if missing:
        raise RuntimeError(
            "Could not reverse-map every shipped surface to an original "
            f"vocab id. Missing {len(missing)} entries: first few = "
            f"{missing[:5]}"
        )

    recovered_sorted = sorted(set(recovered))
    if len(recovered_sorted) != len(recovered):
        raise RuntimeError(
            f"Duplicate surface mappings — got {len(recovered)} entries "
            f"but only {len(recovered_sorted)} unique ids."
        )

    # Append blank last so new_blank_idx = len(keep)-1 (mirrors
    # build_english_keep_set in patch_vocab_prune).
    recovered_sorted.append(old_blank_idx)
    return recovered_sorted


class JointNoEncProjBatched(torch.nn.Module):
    """Batched joint over K already-projected encoder frames."""

    def __init__(self, joint_module: torch.nn.Module) -> None:
        super().__init__()
        self.joint_module = joint_module

    def forward(self, encoder_proj: torch.Tensor, dec_out: torch.Tensor):
        # encoder_proj: [B=1, T=K, joint_dim=640] (pre-projected)
        # dec_out:     [B=1, decoder_hidden=640, U=1]
        decoder_outputs = dec_out.transpose(1, 2)  # [B, U, D]
        dec_proj = self.joint_module.pred(decoder_outputs)  # [B, U, joint_dim]
        x = encoder_proj.unsqueeze(2) + dec_proj.unsqueeze(1)
        x = self.joint_module.joint_net[0](x)  # ReLU
        x = self.joint_module.joint_net[1](x)  # Dropout
        logits = self.joint_module.joint_net[2](x)  # Linear → [B, T, U, V]
        return logits


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def build(
    nemo_path: Path = typer.Option(..., "--nemo-path"),
    reference_tokenizer_json: Path = typer.Option(
        ...,
        "--reference-tokenizer-json",
        help="Path to the shipped tokenizer.json whose keep-set we recover.",
    ),
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

    old_blank = int(m.decoder.blank_idx)
    typer.echo(
        f"Recovering keep-set from {reference_tokenizer_json}..."
    )
    keep_ids = recover_keep_ids_from_tokenizer_json(
        m.tokenizer, reference_tokenizer_json, old_blank_idx=old_blank
    )
    typer.echo(
        f"  [vocab-prune] recovered {len(keep_ids)} ids from tokenizer.json "
        f"(blank at new_id={len(keep_ids) - 1})"
    )
    prune_vocab_english(m, keep_ids)

    # Sanity check — pruned vocab matches the reference exactly.
    with open(reference_tokenizer_json) as f:
        ref = _json.load(f)
    expected_n = len(ref)  # includes blank
    actual_n = len(keep_ids)
    if expected_n != actual_n:
        raise RuntimeError(
            f"Vocab size mismatch — reference tokenizer.json has "
            f"{expected_n} entries (incl. blank), recovered {actual_n}."
        )

    joint_batched = JointNoEncProjBatched(m.joint.eval()).eval()

    enc_proj_batch = torch.randn(1, batch_frames, 640)
    dec_step = torch.randn(1, 640, 1)

    traced = torch.jit.trace(
        joint_batched, (enc_proj_batch, dec_step), strict=False
    )
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
            ct.TensorType(
                name="decoder",
                shape=_tensor_shape(dec_step),
                dtype=np.float32,
            ),
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
    typer.echo(
        f"  output logits shape:      [1, {batch_frames}, 1, {len(keep_ids)}]"
    )


if __name__ == "__main__":
    app()
