"""Validate Jeff's trained classifier graph and export its complete decision path."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from jeff.backends.torch_backend import TorchBackend
from jeff.core.backend import Group

from jeff_decision import JeffDecision, marker_positions
from trace_compat import finite_fp16_mask, install_trace_compatibility

SOURCE = "knowledgator/gliformer-large-v1"
REVISION = "d0a4e53d09cebe6bc963dd9be319d4279084bb2d"
BUCKET = 128
MAX_CATEGORIES = 8

FIXTURES = (
    (
        "billing",
        "The invoice was charged twice and the customer asks for a refund.",
        Group(key="route", labels=("billing: invoice or payment issue", "support: technical product issue"),
              name="Choose the correct support queue"),
    ),
    (
        "technical",
        "The app crashes when I save my project. Please help me recover the file.",
        Group(key="route", labels=("billing: invoice or payment issue", "support: technical product issue"),
              name="Choose the correct support queue"),
    ),
    (
        "three_way",
        "Tomorrow at 9 a.m. works well for the appointment.",
        Group(key="intent", labels=("schedule: appointment request", "billing: payment issue",
                                     "support: technical issue"), name="Classify the user intent"),
    ),
    (
        "boolean",
        "I cannot sign in after resetting my password.",
        Group(key="answer", labels=("yes", "no"), name="Is this a technical support request?"),
    ),
)


def make_batch(backend: TorchBackend, text: str, group: Group) -> dict:
    tokens, _, _ = backend.model.prepare_inputs([text])
    return backend._collator([{
        "tokenized_text": tokens[0],
        "classification": [{
            "name": group.name,
            "description": group.description,
            "all_labels": list(group.labels),
            "true_labels": [],
        }],
    }])


def model_inputs(batch: dict, config) -> tuple[torch.Tensor, ...]:
    ids = batch["input_ids"]
    mask = batch["attention_mask"]
    if ids.shape[1] > BUCKET:
        raise ValueError(f"input has {ids.shape[1]} tokens; L{BUCKET} cannot serve it")
    parent, children, count = marker_positions(ids, config, MAX_CATEGORIES)
    if count != len(batch["classes_mapping"].cat_mapping[0].cat_class_to_id[0].class_to_id):
        raise ValueError("collator category mapping does not match marker count")
    pad = BUCKET - ids.shape[1]
    return (
        F.pad(ids.to(torch.int32), (0, pad)),
        F.pad(mask.to(torch.int32), (0, pad)),
        parent,
        children,
    )


@torch.inference_mode()
def native_report(backend: TorchBackend, decision: JeffDecision) -> tuple[list[dict], tuple[torch.Tensor, ...]]:
    records = []
    first = None
    for name, text, group in FIXTURES:
        batch = make_batch(backend, text, group)
        tensors = model_inputs(batch, backend.model.config)
        native = backend.model.model(**batch, include_media=False).cat_logits.detach().float().numpy()[0]
        converted = decision(*tensors).detach().float().numpy()[0, :len(group.labels)]
        error = float(np.max(np.abs(native - converted)))
        records.append({
            "name": name,
            "token_count": int(batch["attention_mask"].sum()),
            "labels": list(group.labels),
            "native_logits": native.tolist(),
            "decision_logits": converted.tolist(),
            "max_logit_error": error,
        })
        if error > 1e-3:
            raise AssertionError(f"{name}: trained graph differs from native logits by {error}")
        if first is None:
            first = tensors
    assert first is not None
    return records, first


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--convert", action="store_true")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint = snapshot_download(SOURCE, revision=REVISION, local_files_only=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"attn_kernel=.*flashdeberta")
        backend = TorchBackend(checkpoint, device="cpu", dtype="float32", attn_kernel="eager", batch_size=1)
    decision = JeffDecision(backend.model.model).eval()
    report, first = native_report(backend, decision)
    out = Path("build")
    out.mkdir(exist_ok=True)
    (out / "native-parity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"native_parity": report}, indent=2), flush=True)
    if not args.convert:
        return

    import coremltools as ct
    install_trace_compatibility()
    with finite_fp16_mask():
        patched_report, _ = native_report(backend, decision)
        patched_error = max(
            abs(before - after)
            for baseline, patched in zip(report, patched_report)
            for before, after in zip(baseline["decision_logits"], patched["decision_logits"])
        )
        if patched_error > 1e-3:
            raise AssertionError(f"tracing-only mask/attention scale changes native logits by {patched_error}")
        print(f"Patched mask/attention max native logit error: {patched_error:.8f}", flush=True)
        traced = torch.jit.trace(decision, first, check_trace=False).eval()
    traced.save(str(out / "jeff-decision-L128.pt"))
    with torch.inference_mode():
        expected = decision(*first).detach().numpy()
        actual = traced(*first).detach().numpy()
    trace_error = float(np.max(np.abs(expected - actual)))
    if trace_error > 1e-3:
        raise AssertionError(f"TorchScript trace mismatch: {trace_error}")
    print(f"TorchScript trace max logit error: {trace_error:.8f}", flush=True)
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, BUCKET), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, BUCKET), dtype=np.int32),
            ct.TensorType(name="parent_position", shape=(1, 1), dtype=np.int32),
            ct.TensorType(name="category_positions", shape=(1, MAX_CATEGORIES), dtype=np.int32),
        ],
    )
    package = out / f"JeffDecision-L128-{args.precision.upper()}.mlpackage"
    mlmodel.save(str(package))
    print(f"Saved {package}", flush=True)


if __name__ == "__main__":
    main()
