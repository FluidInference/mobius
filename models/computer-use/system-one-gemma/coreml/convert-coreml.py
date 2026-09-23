"""Convert the pinned trained System One scorer after the Gemma gate is accepted."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from assets import LOCK, ROOT, check_base_access, sha256
from native_reference import MAX_LENGTH, encode_options, pad_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build")
    args = parser.parse_args()
    check_base_access()  # Fail before importing Torch/Core ML or acquiring hundreds of MB.

    import coremltools as ct
    import numpy as np
    import torch

    from export_model import export_wrapper, load_trained_scorer

    torch.set_num_threads(4)
    tokenizer, native = load_trained_scorer()
    source = ROOT / "build" / "source"
    example = json.loads((source / "demos.json").read_text())[0]
    sequences = encode_options(tokenizer, example["state"], example["question"], example["options"])
    # The trace uses real upstream demo tokens. Repeat them to fill the fixed K16 graph.
    sequences = (sequences * 16)[:16]
    ids, mask = pad_batch(sequences, tokenizer.pad_token_id, MAX_LENGTH)
    example_tensors = (torch.tensor(ids, dtype=torch.int32), torch.tensor(mask, dtype=torch.int32))
    wrapper = export_wrapper(native)
    with torch.inference_mode():
        original = native(input_ids=example_tensors[0].long(), attention_mask=example_tensors[1].long()).logits
        wrapped = wrapper(*example_tensors)
    wrapper_error = float((original - wrapped).abs().max())
    if wrapper_error > 1e-5:
        raise ValueError(f"wrapper changed the trained scorer: {wrapper_error}")
    traced = torch.jit.trace(wrapper, example_tensors)

    started = time.perf_counter()
    coreml = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=[
            ct.TensorType(name="input_ids", shape=(16, MAX_LENGTH), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(16, MAX_LENGTH), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
    )
    coreml.short_description = "Trained System One Gemma scalar scorer; choice temperature 2.35 in host"
    coreml.author = "Akash Kamat (trained weights); Fluid Inference (Core ML conversion)"
    coreml.license = "Gemma Terms of Use; trained scorer noncommercial restriction"
    coreml.user_defined_metadata.update(
        {
            "source_repo": LOCK["source_repo"],
            "source_revision": LOCK["source_revision"],
            "base_repo": LOCK["base_repo"],
            "base_revision": LOCK["base_revision"],
            "trained_adapter_sha256": LOCK["files"]["pretrained-scorer/adapter_model.safetensors"],
            "output_contract": "16 scalar logits; apply temperature 2.35 and softmax to real candidates in host",
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = args.output_dir / "system_one_gemma_fp16_L256_K16.mlpackage"
    coreml.save(str(package))
    report = {
        "package": package.name,
        "source_revision": LOCK["source_revision"],
        "base_revision": LOCK["base_revision"],
        "parameters": sum(p.numel() for p in native.parameters()),
        "trained_score_head_verified": True,
        "wrapper_max_logit_error": wrapper_error,
        "conversion_seconds": time.perf_counter() - started,
        "package_bytes": sum(path.stat().st_size for path in package.rglob("*") if path.is_file()),
        "package_files_sha256": {
            str(path.relative_to(package)): sha256(path) for path in package.rglob("*") if path.is_file()
        },
        "parity_verified": False,
    }
    (args.output_dir / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
