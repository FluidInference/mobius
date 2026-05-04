#!/usr/bin/env python3
"""
Convert the upstream g2pW ONNX checkpoint to a CoreML mlpackage.

Pipeline:
    1. Download `G2PWModel-v2-onnx.zip` from upstream Google Cloud Storage.
    2. Extract `g2pw.onnx` plus the label / lexicon side files.
    3. Round-trip ONNX → PyTorch nn.Module via `onnx2torch`.
    4. Trace with fixed shapes (batch=1, seq_len=512) and convert to
       CoreML via the unified `coremltools.convert` API.
    5. Emit `build/<name>/g2pw.mlpackage` plus copies of the side files
       (`POLYPHONIC_CHARS.txt`, `MONOPHONIC_CHARS.txt`, `config.py`,
       `version`) so downstream Swift code has the labelling artefacts
       colocated.

Trace target: `.CpuOnly` (per mobius CLAUDE.md). The conversion script
is reproducible — no checked-in checkpoints — and the build dir is
gitignored.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np

UPSTREAM_ZIP_URL = (
    "https://storage.googleapis.com/esun-ai/g2pW/G2PWModel-v2-onnx.zip"
)
UPSTREAM_DIR_NAME = "G2PWModel"
ONNX_FILE_NAME = "g2pw.onnx"
# Side files included in the upstream v2 ONNX archive. The bopomofo
# pinyin / char dicts that ship with the original (non-v2) GitYCC repo
# are NOT in this zip — downstream Swift code maintains its own
# bopomofo→pinyin lookup tables.
SIDE_FILES = [
    "POLYPHONIC_CHARS.txt",
    "MONOPHONIC_CHARS.txt",
    "config.py",
    "version",
]

# Fixed CoreML input shapes. Upstream defaults to max_len=512 and the
# ONNX export uses batch=1, so traced shapes match production.
BATCH_SIZE = 1
SEQ_LEN = 512


def _ensure_zip(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "G2PWModel-v2-onnx.zip"
    if zip_path.exists():
        print(f"[cache] using {zip_path}")
        return zip_path
    print(f"[download] {UPSTREAM_ZIP_URL}")
    urllib.request.urlretrieve(UPSTREAM_ZIP_URL, zip_path)
    return zip_path


def _extract(zip_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    extracted = dest / UPSTREAM_DIR_NAME
    if extracted.exists():
        print(f"[cache] using extracted {extracted}")
        return extracted
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    if not extracted.exists():
        raise RuntimeError(
            f"Expected {extracted} after unzip; archive layout changed?"
        )
    return extracted


def _inspect_onnx(onnx_path: Path) -> Tuple[list, list]:
    """Return (input_names, output_names) of the ONNX graph in order."""
    import onnx

    model = onnx.load(str(onnx_path))
    inputs = [i.name for i in model.graph.input]
    outputs = [o.name for o in model.graph.output]
    print(f"[onnx] inputs: {inputs}")
    print(f"[onnx] outputs: {outputs}")
    return inputs, outputs


def _build_dummy_inputs(num_labels: int) -> dict:
    """Construct dummy tensors that match the ONNX input contract.

    The upstream forward signature (g2pw.module.G2PW.forward) is:
        input_ids, token_type_ids, attention_mask,
        phoneme_mask, char_ids, position_ids[, pos_ids]

    We omit `pos_ids` for the inference graph (matches the released
    ONNX export — POS conditioning is baked into the weights, not a
    runtime input).
    """
    import torch

    input_ids = torch.zeros((BATCH_SIZE, SEQ_LEN), dtype=torch.long)
    token_type_ids = torch.zeros((BATCH_SIZE, SEQ_LEN), dtype=torch.long)
    attention_mask = torch.zeros((BATCH_SIZE, SEQ_LEN), dtype=torch.long)
    attention_mask[:, :8] = 1  # pretend we have 8 real tokens
    phoneme_mask = torch.ones((BATCH_SIZE, num_labels), dtype=torch.float32)
    char_ids = torch.zeros((BATCH_SIZE,), dtype=torch.long)
    position_ids = torch.zeros((BATCH_SIZE,), dtype=torch.long)
    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "phoneme_mask": phoneme_mask,
        "char_ids": char_ids,
        "position_ids": position_ids,
    }


def _infer_num_labels(onnx_path: Path) -> int:
    """Read the phoneme_mask input dim from the ONNX graph."""
    import onnx

    model = onnx.load(str(onnx_path))
    for inp in model.graph.input:
        if inp.name == "phoneme_mask":
            shape = inp.type.tensor_type.shape.dim
            # phoneme_mask is [batch, num_labels].
            label_dim = shape[-1]
            if label_dim.dim_value > 0:
                return int(label_dim.dim_value)
    raise RuntimeError(
        "Could not infer num_labels from ONNX graph (no concrete "
        "phoneme_mask dim). Inspect with onnx.helper.printable_graph."
    )


def _convert(extracted_dir: Path, output_dir: Path) -> None:
    """Run the actual ONNX → torch → CoreML pipeline."""
    import torch
    import coremltools as ct
    from onnx2torch import convert as onnx2torch_convert

    onnx_path = extracted_dir / ONNX_FILE_NAME
    if not onnx_path.exists():
        raise FileNotFoundError(f"missing {onnx_path}")

    input_names, output_names = _inspect_onnx(onnx_path)
    num_labels = _infer_num_labels(onnx_path)
    print(f"[onnx] num_labels = {num_labels}")

    print("[onnx2torch] converting graph to torch.nn.Module …")
    torch_model = onnx2torch_convert(str(onnx_path))
    torch_model.eval()

    dummy = _build_dummy_inputs(num_labels)
    # Reorder dummy inputs to match ONNX graph input order.
    ordered_inputs = tuple(dummy[name] for name in input_names)

    with torch.no_grad():
        ref_out = torch_model(*ordered_inputs)
    if isinstance(ref_out, (tuple, list)):
        ref_out = ref_out[0]
    print(f"[trace] ref output shape: {tuple(ref_out.shape)}")

    print("[trace] tracing torch.jit …")
    traced = torch.jit.trace(torch_model, ordered_inputs, strict=False)

    ct_inputs = []
    for name in input_names:
        t = dummy[name]
        if t.dtype in (torch.long, torch.int64, torch.int32):
            ct_inputs.append(
                ct.TensorType(name=name, shape=tuple(t.shape), dtype=np.int32)
            )
        else:
            ct_inputs.append(
                ct.TensorType(name=name, shape=tuple(t.shape), dtype=np.float32)
            )

    ct_outputs = [ct.TensorType(name=output_names[0])]

    print("[coreml] converting traced graph …")
    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=ct_outputs,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY,  # trace target — runtime can override.
        minimum_deployment_target=ct.target.iOS17,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    mlpkg = output_dir / "g2pw.mlpackage"
    if mlpkg.exists():
        shutil.rmtree(mlpkg)
    mlmodel.save(str(mlpkg))
    print(f"[coreml] wrote {mlpkg}")

    for fname in SIDE_FILES:
        src = extracted_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            print(f"[copy] {fname}")
        else:
            print(f"[warn] missing side file {fname}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "g2pw-coreml",
        help="Where to keep the upstream zip + extraction (default: ~/.cache/g2pw-coreml)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./build/g2pw"),
        help="Where the mlpackage + side files land (default: ./build/g2pw)",
    )
    args = p.parse_args()

    zip_path = _ensure_zip(args.cache_dir)
    extracted = _extract(zip_path, args.cache_dir)
    _convert(extracted, args.output_dir)
    print("[done] g2pw mlpackage ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
