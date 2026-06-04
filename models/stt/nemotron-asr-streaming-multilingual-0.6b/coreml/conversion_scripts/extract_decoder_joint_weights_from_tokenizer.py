"""Extract decoder + joint weights as a raw binary blob, with the
keep-set recovered from an existing shipped tokenizer.json (instead of
a fresh corpus).

Mirrors `extract_decoder_joint_weights.py` exactly, but the vocab
prune is driven by `--reference-tokenizer-json` so the output
`native_weights/` matches the shipped bundle's vocab layout
byte-for-byte. Use this to add smart-spec assets to an already-built
decoder/joint bundle at a different chunk size, without re-running
any corpus through the build process.

Usage:
    .venv/bin/python conversion_scripts/extract_decoder_joint_weights_from_tokenizer.py \\
        --nemo-path /path/to/nemotron-asr-streaming-multilingual-0.6b.nemo \\
        --reference-tokenizer-json /path/to/build_lp_engprune_42_13_1120ms_v3/tokenizer.json \\
        --output-dir build_lp_engprune_42_13_1120ms_v3/native_weights
"""
from __future__ import annotations

import json as _json
import struct
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import typer

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from patch_uniquebias import patch_and_log  # type: ignore
from patch_vocab_prune import prune_vocab_english  # type: ignore
from build_joint_noencproj_batched_from_tokenizer import (  # type: ignore
    recover_keep_ids_from_tokenizer_json,
)


def _write_blob(
    tensors: Dict[str, torch.Tensor],
    out_dir: Path,
) -> Dict[str, Dict]:
    blob_path = out_dir / "weights.bin"
    index: Dict[str, Dict] = {}
    offset = 0
    with open(blob_path, "wb") as f:
        for name, t in tensors.items():
            t16 = t.detach().to(torch.float16).contiguous().cpu().numpy()
            assert t16.dtype == np.float16
            data = t16.tobytes()
            shape = list(t16.shape)
            index[name] = {"offset": offset, "shape": shape, "dtype": "float16"}
            f.write(data)
            offset += len(data)
    typer.echo(f"  wrote {offset} bytes ({offset/1e6:.1f} MB) to {blob_path}")
    return index


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def extract(
    nemo_path: Path = typer.Option(..., "--nemo-path"),
    reference_tokenizer_json: Path = typer.Option(
        ...,
        "--reference-tokenizer-json",
        help="Path to the shipped tokenizer.json whose keep-set we recover.",
    ),
    output_dir: Path = typer.Option(..., "--output-dir"),
):
    import nemo.collections.asr as nemo_asr

    typer.echo(f"Loading {nemo_path}...")
    m = nemo_asr.models.ASRModel.restore_from(str(nemo_path), map_location="cpu")
    m.eval()
    patch_and_log(m.encoder)

    old_blank = int(m.decoder.blank_idx)
    typer.echo(f"Recovering keep-set from {reference_tokenizer_json}...")
    keep_ids = recover_keep_ids_from_tokenizer_json(
        m.tokenizer, reference_tokenizer_json, old_blank_idx=old_blank
    )
    typer.echo(
        f"  [vocab-prune] recovered {len(keep_ids)} ids from tokenizer.json "
        f"(blank at new_id={len(keep_ids) - 1})"
    )
    prune_vocab_english(m, keep_ids)
    pruned_vocab = len(keep_ids)
    pruned_blank = pruned_vocab - 1

    output_dir.mkdir(parents=True, exist_ok=True)

    pred = m.decoder.prediction
    embed = pred["embed"]
    lstm = pred["dec_rnn"].lstm

    tensors: Dict[str, torch.Tensor] = {}
    tensors["decoder.embed.weight"] = embed.weight

    for layer in range(lstm.num_layers):
        tensors[f"decoder.lstm.weight_ih_l{layer}"] = getattr(lstm, f"weight_ih_l{layer}")
        tensors[f"decoder.lstm.weight_hh_l{layer}"] = getattr(lstm, f"weight_hh_l{layer}")
        tensors[f"decoder.lstm.bias_ih_l{layer}"] = getattr(lstm, f"bias_ih_l{layer}")
        tensors[f"decoder.lstm.bias_hh_l{layer}"] = getattr(lstm, f"bias_hh_l{layer}")

    jnt = m.joint
    tensors["joint.enc.weight"] = jnt.enc.weight
    tensors["joint.enc.bias"] = jnt.enc.bias
    tensors["joint.pred.weight"] = jnt.pred.weight
    tensors["joint.pred.bias"] = jnt.pred.bias
    tensors["joint.out.weight"] = jnt.joint_net[2].weight
    tensors["joint.out.bias"] = jnt.joint_net[2].bias

    typer.echo("Extracted tensors:")
    for name, t in tensors.items():
        typer.echo(f"  {name:<35} {tuple(t.shape)}  {t.dtype}")

    index = _write_blob(tensors, output_dir)
    meta = {
        "version": 1,
        "decoder_hidden": 640,
        "decoder_layers": int(lstm.num_layers),
        "decoder_input_dim": 640,
        "encoder_dim": 1024,
        "joint_inner_dim": 640,
        "vocab_size": pruned_vocab,
        "blank_idx": pruned_blank,
        "tensors": index,
    }
    idx_path = output_dir / "weights_index.json"
    idx_path.write_text(_json.dumps(meta, indent=2))
    typer.echo(f"  wrote {idx_path}")

    # Parity reference (same shape as the corpus-driven extractor).
    torch.manual_seed(42)
    test_token = torch.tensor([[pruned_blank]], dtype=torch.long)
    test_enc_step = torch.randn(1, 1024, 1, dtype=torch.float32)

    with torch.no_grad():
        emb = embed(test_token)
        emb = emb.transpose(0, 1)
        h0 = torch.zeros(lstm.num_layers, 1, 640)
        c0 = torch.zeros(lstm.num_layers, 1, 640)
        lstm_out, (h_n, c_n) = lstm(emb, (h0, c0))
        dec_out = lstm_out.transpose(0, 1)

        enc_in = test_enc_step.transpose(1, 2)
        dec_in = dec_out
        enc_proj = jnt.enc(enc_in)
        dec_proj = jnt.pred(dec_in)
        combined = enc_proj.unsqueeze(2) + dec_proj.unsqueeze(1)
        combined = jnt.joint_net[0](combined)
        logits = jnt.joint_net[2](combined)
        argmax = int(logits.argmax(dim=-1).item())

    parity = {
        "test_input": {
            "token_id": int(pruned_blank),
            "enc_step_seed": 42,
            "enc_step_shape": [1, 1024, 1],
        },
        "reference_output": {
            "decoder_out_shape": list(dec_out.shape),
            "decoder_out_sample": dec_out[0, 0, :5].tolist(),
            "h_n_sample": h_n[0, 0, :5].tolist(),
            "c_n_sample": c_n[0, 0, :5].tolist(),
            "logits_shape": list(logits.shape),
            "logits_argmax": argmax,
            "logits_max_value": float(logits.max().item()),
        },
    }
    (output_dir / "parity_reference.json").write_text(_json.dumps(parity, indent=2))
    typer.echo(f"  wrote {output_dir / 'parity_reference.json'}")

    # Also save the enc step bytes so Swift parity can use the exact
    # same input bytes.
    enc_bytes = test_enc_step.detach().contiguous().cpu().numpy().astype(np.float32).tobytes()
    (output_dir / "test_enc_step.bin").write_bytes(enc_bytes)
    typer.echo(f"  wrote {output_dir / 'test_enc_step.bin'}")


if __name__ == "__main__":
    app()
