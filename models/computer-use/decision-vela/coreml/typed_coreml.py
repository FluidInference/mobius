"""Core ML export of the complete Kai/Lex decision paths.

The upstream model has independent Choice, Noul, and Score encoders.  Each
export keeps the trained encoder and the corresponding interaction head.  The
host applies the upstream candidate mask and probability calculation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import torch
from torch import nn

KINDS = ("choice", "noul", "score")


def package_sha256(package: Path) -> str:
    """Hash relative paths and bytes for an immutable .mlpackage inventory."""
    digest = hashlib.sha256()
    for path in sorted(item for item in package.rglob("*") if item.is_file()):
        digest.update(path.relative_to(package).as_posix().encode() + b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load_native(native_dir: Path):
    """Load the pinned upstream runtime on CPU after its manifest checks."""
    native_dir = native_dir.resolve()
    namespace = "_decision_coreml_source_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(
        namespace, native_dir / "__init__.py", submodule_search_locations=[str(native_dir)]
    )
    if spec is None or spec.loader is None:
        raise ValueError("Native runtime package is missing")
    package = importlib.util.module_from_spec(spec)
    sys.modules[namespace] = package
    spec.loader.exec_module(package)
    artifacts = __import__(namespace + ".artifacts", fromlist=["load_export"])
    model, collator, config = artifacts.load_export(native_dir, device="cpu")
    if (config["arm"], config["training_arm"]) != ("all22", "S22"):
        raise ValueError("Expected the full three-path all22/S22 release")
    if sum(p.numel() for p in model.parameters()) != 571_909_635:
        raise ValueError("Incomplete Kai/Lex decision model")
    model.eval()
    return model, collator, config


class TypedPath(nn.Module):
    """One upstream typed path, retaining trained embeddings, blocks, and head."""

    def __init__(self, model: nn.Module, kind: str, *, marker_map: bool = False):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"Unknown decision kind: {kind}")
        self.kind = kind
        self.marker_map = marker_map
        self.kind_index = KINDS.index(kind)
        # Keep the mask helper without registering unused encoder layers in
        # Choice/Score exports.  The trained embeddings and active blocks are
        # registered separately below.
        object.__setattr__(self, "_encoder", model.encoder)
        self.embeddings = model.encoder.embeddings
        self.type_embedding = model.type_embedding
        self.head = model.heads[kind]
        self.scorer = model.scorers[kind]
        if kind == "choice":
            self.blocks = model.choice_blocks
            self.final_norm = model.choice_final_norm
        elif kind == "score":
            self.blocks = model.score_blocks
            self.final_norm = model.score_final_norm
        else:
            self.blocks = model.encoder.layers
            self.final_norm = model.encoder.final_norm
        self.eval()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, markers: torch.Tensor):
        # The native collator supplies exactly these token IDs and positions.
        ids = input_ids.long()
        mask = attention_mask.bool()
        position_ids = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        global_mask, local_mask = self._encoder._update_attention_mask(mask, output_attentions=False)
        kwargs = {
            "attention_mask": global_mask,
            "sliding_window_mask": local_mask,
            "position_ids": position_ids,
            "cu_seqlens": None,
            "max_seqlen": None,
            "output_attentions": False,
        }
        hidden = self.embeddings(input_ids=ids, inputs_embeds=None)
        for layer in self.blocks:
            hidden = layer(hidden, **kwargs)[0]
        hidden = self.final_norm(hidden)
        hidden = hidden + self.type_embedding.weight[self.kind_index].view(1, 1, -1).to(hidden.dtype)
        padding = ~mask
        for layer in self.head:
            hidden = layer(hidden, src_key_padding_mask=padding)
        if self.marker_map:
            selected = torch.matmul(markers.to(hidden.dtype), hidden)
        else:
            selected = torch.gather(
                hidden, 1, markers.long()[:, :, None].expand(-1, -1, hidden.shape[-1])
            )
        return self.scorer(selected).squeeze(-1).float()


def marker_tensor(batch: dict, *, marker_map: bool) -> torch.Tensor:
    """Use an FP16 one-hot map for ANE-friendly marker selection when requested."""
    positions = batch["marker_positions"]
    if not marker_map:
        return positions.to(torch.int32)
    selected = torch.nn.functional.one_hot(positions.long(), num_classes=batch["input_ids"].shape[1])
    return (selected * batch["valid_candidates"].unsqueeze(-1)).to(torch.float16)


def padded_batch(collator, records: list[dict], tokens: int | None, candidates: int | None):
    """Right-pad real native records; reject every truncating shape."""
    batch, encoded = collator(records, device="cpu")
    actual_tokens = batch["input_ids"].shape[1]
    actual_candidates = batch["marker_positions"].shape[1]
    tokens = actual_tokens if tokens is None else tokens
    candidates = actual_candidates if candidates is None else candidates
    if tokens < actual_tokens or candidates < actual_candidates:
        raise ValueError("Core ML shape would truncate a native request")
    token_pad = tokens - actual_tokens
    candidate_pad = candidates - actual_candidates
    if token_pad:
        batch["input_ids"] = torch.nn.functional.pad(
            batch["input_ids"], (0, token_pad), value=collator.pad
        )
        batch["attention_mask"] = torch.nn.functional.pad(batch["attention_mask"], (0, token_pad))
    if candidate_pad:
        for key in ("marker_positions", "valid_candidates", "targets", "values"):
            batch[key] = torch.nn.functional.pad(batch[key], (0, candidate_pad))
    return batch, encoded


def compare_native(
    model, collator, records: list[dict], kind: str, *, tokens: int | None = None,
    candidates: int | None = None, atol: float = 1e-4, marker_map: bool = False
) -> dict:
    """Real-record parity with upstream's complete typed forward."""
    batch, encoded = padded_batch(collator, records, tokens, candidates)
    if any(row["kind"] != kind for row in encoded):
        raise ValueError("All records must use the selected kind")
    wrapper = TypedPath(model, kind, marker_map=marker_map)
    with torch.inference_mode():
        expected = model(batch)
        actual = wrapper(batch["input_ids"], batch["attention_mask"], marker_tensor(batch, marker_map=marker_map))
    valid = batch["valid_candidates"]
    delta = (expected[valid] - actual[valid]).abs()
    maximum = float(delta.max()) if delta.numel() else 0.0
    if maximum > atol:
        raise AssertionError(f"{kind} wrapper differs from native by {maximum:.6g}")
    return {
        "kind": kind,
        "record_count": len(records),
        "tokens": batch["input_ids"].shape[1],
        "candidates": batch["marker_positions"].shape[1],
        "max_abs_logit_error": maximum,
        "marker_map": marker_map,
    }


def export_coreml(
    model: nn.Module,
    collator,
    records: list[dict],
    kind: str,
    output_dir: Path,
    *,
    source_repo: str,
    source_revision: str,
    tokens: int | None = None,
    candidates: int | None = None,
    flexible: bool = False,
    marker_map: bool = False,
) -> dict:
    """Trace a real native-shaped batch and convert its trained typed path."""
    import coremltools as ct

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if marker_map and flexible:
        raise ValueError("marker-map export currently requires a fixed shape")
    parity = compare_native(model, collator, records, kind, tokens=tokens, candidates=candidates,
                            marker_map=marker_map)
    batch, _ = padded_batch(collator, records, tokens, candidates)
    ids = batch["input_ids"].to(torch.int32)
    mask = batch["attention_mask"].to(torch.int32)
    positions = batch["marker_positions"].to(torch.int32)
    markers = marker_tensor(batch, marker_map=marker_map)
    wrapper = TypedPath(model, kind, marker_map=marker_map).eval()
    # Avoid PyTorch's fused inference fast path: the unfused graph is traceable.
    torch.backends.mha.set_fastpath_enabled(False)
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, (ids, mask, markers), strict=True, check_trace=True)
        traced_logits = traced(ids, mask, markers)
        native_logits = wrapper(ids, mask, markers)
    trace_delta = float((traced_logits - native_logits).abs().max())
    if trace_delta > 1e-4:
        raise AssertionError(f"Trace differs from wrapper by {trace_delta:.6g}")
    target = ct.target.macOS14
    token_shape = tokens if not flexible else ct.RangeDim(lower_bound=16, upper_bound=1024, default=tokens)
    candidate_shape = positions.shape[1] if not flexible else ct.RangeDim(
        lower_bound=2, upper_bound=255, default=positions.shape[1]
    )
    marker_shape = (ids.shape[0], candidate_shape, token_shape) if marker_map else (ids.shape[0], candidate_shape)
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=target,
        compute_precision=ct.precision.FLOAT16,
        inputs=[
            ct.TensorType(name="input_ids", shape=(ids.shape[0], token_shape), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(mask.shape[0], token_shape), dtype=np.int32),
            ct.TensorType(name="marker_map" if marker_map else "marker_positions", shape=marker_shape,
                          dtype=np.float16 if marker_map else np.int32),
        ],
        outputs=[ct.TensorType(name="logits")],
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_dir))
    reference = wrapper(ids, mask, markers).detach().cpu().numpy()
    prediction = mlmodel.predict(
        {
            "input_ids": ids.cpu().numpy(),
            "attention_mask": mask.cpu().numpy(),
            "marker_map" if marker_map else "marker_positions": markers.cpu().numpy(),
        }
    )["logits"]
    valid = batch["valid_candidates"].cpu().numpy()
    coreml_delta = float(np.max(np.abs(reference[valid] - prediction[valid])))
    flexible_checks = []
    if flexible:
        for next_tokens, next_candidates in ((256, positions.shape[1]), (128, positions.shape[1] + 2)):
            test_batch, _ = padded_batch(collator, records, next_tokens, next_candidates)
            test_ids = test_batch["input_ids"].to(torch.int32)
            test_mask = test_batch["attention_mask"].to(torch.int32)
            test_markers = marker_tensor(test_batch, marker_map=marker_map)
            with torch.inference_mode():
                expected = wrapper(test_ids, test_mask, test_markers).cpu().numpy()
            actual = mlmodel.predict({
                "input_ids": test_ids.cpu().numpy(),
                "attention_mask": test_mask.cpu().numpy(),
                "marker_map" if marker_map else "marker_positions": test_markers.cpu().numpy(),
            })["logits"]
            test_valid = test_batch["valid_candidates"].cpu().numpy()
            delta = float(np.max(np.abs(expected[test_valid] - actual[test_valid])))
            flexible_checks.append({
                "tokens": next_tokens,
                "candidates": next_candidates,
                "max_abs_logit_error": delta,
                "same_choice": bool(np.array_equal(
                    np.where(test_valid, expected, -1e30).argmax(-1),
                    np.where(test_valid, actual, -1e30).argmax(-1),
                )),
            })
    report = {
        "source_repo": source_repo,
        "source_revision": source_revision,
        "kind": kind,
        "native_parameter_count": 571_909_635,
        "shape": {"batch": ids.shape[0], "tokens": ids.shape[1], "candidates": positions.shape[1]},
        "wrapper_parity": parity,
        "trace_max_abs_logit_error": trace_delta,
        "package_sha256": package_sha256(output_dir),
        "package_bytes": sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file()),
        "coreml_max_abs_logit_error": coreml_delta,
        "coreml_choice_agreement": bool(
            np.array_equal(np.where(valid, reference, -1e30).argmax(-1),
                           np.where(valid, prediction, -1e30).argmax(-1))
        ),
        "flexible": flexible,
        "marker_map": marker_map,
        "flexible_checks": flexible_checks,
        "historical_tracker_revision_verified": False,
    }
    (output_dir.parent / (output_dir.stem + ".json")).write_text(json.dumps(report, indent=2) + "\n")
    return report


def verify_coreml(model, collator, records: list[dict], kind: str, package: Path) -> dict:
    """Compare a saved Core ML package with the full native model on real rows."""
    import coremltools as ct

    metadata = json.loads((package.parent / (package.stem + ".json")).read_text())
    if metadata["kind"] != kind:
        raise ValueError("Core ML package kind differs from the requested kind")
    shape = metadata["shape"]
    marker_map = metadata.get("marker_map", False)
    if shape["batch"] != 1:
        raise ValueError("This verifier handles one real request per Core ML call")
    mlmodel = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_AND_NE)
    comparisons = []
    for record in records:
        batch, encoded = padded_batch(collator, [record], shape["tokens"], shape["candidates"])
        if encoded[0]["kind"] != kind:
            raise ValueError("Wrong fixture kind")
        with torch.inference_mode():
            reference = model(batch)[0].cpu().numpy()
        prediction = mlmodel.predict(
            {
                "input_ids": batch["input_ids"].to(torch.int32).cpu().numpy(),
                "attention_mask": batch["attention_mask"].to(torch.int32).cpu().numpy(),
                "marker_map" if marker_map else "marker_positions": marker_tensor(
                    batch, marker_map=marker_map
                ).cpu().numpy(),
            }
        )["logits"][0]
        valid = batch["valid_candidates"][0].cpu().numpy()
        native_valid, coreml_valid = reference[valid], prediction[valid]
        native_prob = torch.from_numpy(native_valid).softmax(-1).numpy()
        coreml_prob = torch.from_numpy(coreml_valid).softmax(-1).numpy()
        comparisons.append(
            {
                "record_id": record.get("id"),
                "valid_candidates": int(valid.sum()),
                "max_abs_logit_error": float(np.max(np.abs(native_valid - coreml_valid))),
                "max_abs_probability_error": float(np.max(np.abs(native_prob - coreml_prob))),
                "same_choice": bool(native_valid.argmax() == coreml_valid.argmax()),
            }
        )
    return {
        "kind": kind,
        "source_repo": metadata["source_repo"],
        "source_revision": metadata["source_revision"],
        "shape": shape,
        "request_count": len(comparisons),
        "max_abs_logit_error": max(row["max_abs_logit_error"] for row in comparisons),
        "max_abs_probability_error": max(row["max_abs_probability_error"] for row in comparisons),
        "choice_agreement": all(row["same_choice"] for row in comparisons),
        "rows": comparisons,
    }
