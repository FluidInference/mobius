#!/usr/bin/env python3
"""Fused decoder+joint_decision export for Parakeet TDT v3 (B1-style fusion).

Combines the RNNT prediction network (2-layer LSTM decoder) and the
joint+decision head (token/duration argmax + top-K) into ONE CoreML model,
eliminating one MLModel dispatch per decode step (~89 calls -> ~49 per
7.8 s utterance). Precedent: Nemotron Multilingual B1 fusion (+15%).

The decoder LSTM has no ANE kernel (`ios17.lstm` -- categorical dead end,
see Kokoro PostAlbert), so this targets CPU dispatch savings, not ANE
placement.

I/O contract (drop-in superset of the shipped Decoder + JointDecisionv3):
  inputs:  targets [1,1] i32, target_length [1] i32,
           h_in [2,1,640] f32, c_in [2,1,640] f32,
           encoder_step [1,1024,1] f32
  outputs: token_id [1,1,1] i32, token_prob [1,1,1] f32,
           duration [1,1,1] i32, top_k_ids [1,1,1,64] i32,
           top_k_logits [1,1,1,64] f32, h_out [2,1,640] f32,
           c_out [2,1,640] f32

Host semantics: call once per joint step. On blank emission the host
re-feeds the previous (targets, h_in, c_in) unchanged; the LSTM recompute
is deterministic so behavior is identical to the two-model loop.

Usage:
    uv run python fuse_decoder_joint.py export --output-dir build/fused
    uv run python fuse_decoder_joint.py parity --build-dir build/fused
    uv run python fuse_decoder_joint.py bench --build-dir build/fused
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Dict, Optional

import coremltools as ct
import numpy as np
import torch
import typer

from individual_components import (
    DecoderWrapper,
    ExportSettings,
    JointWrapper,
    JointDecisionSingleStep,
    _coreml_convert,
)

DEFAULT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


class FusedDecoderJointDecision(torch.nn.Module):
    """Prediction network + joint + decision head in a single forward."""

    def __init__(self, decoder: torch.nn.Module, joint: torch.nn.Module, vocab_size: int, num_extra: int) -> None:
        super().__init__()
        self.decoder = DecoderWrapper(decoder)
        self.joint_decision = JointDecisionSingleStep(JointWrapper(joint), vocab_size=vocab_size, num_extra=num_extra)

    def forward(self, targets, target_length, h_in, c_in, encoder_step):
        dec_out, h_out, c_out = self.decoder(targets, target_length, h_in, c_in)  # [1, 640, 1]
        token_id, token_prob, duration, topk_ids, topk_logits = self.joint_decision(encoder_step, dec_out)
        return token_id, token_prob, duration, topk_ids, topk_logits, h_out, c_out


def _load_model(model_id: str, nemo_path: Optional[Path]):
    import nemo.collections.asr as nemo_asr

    if nemo_path is not None:
        m = nemo_asr.models.EncDecRNNTBPEModel.restore_from(str(nemo_path), map_location="cpu")
    else:
        m = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_id, map_location="cpu")
    m.eval()
    m.decoder._rnnt_export = True
    return m


def _settings(output_dir: Path, precision: Optional[ct.precision]) -> ExportSettings:
    return ExportSettings(
        output_dir=output_dir,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        deployment_target=ct.target.iOS17,
        compute_precision=precision,
        max_audio_seconds=15.0,
        max_symbol_steps=1,
    )


@app.command()
def export(
    model_id: str = typer.Option(DEFAULT_MODEL_ID, "--model-id"),
    nemo_path: Optional[Path] = typer.Option(None, "--nemo-path", exists=True, resolve_path=True),
    output_dir: Path = typer.Option(Path("build/fused"), "--output-dir"),
) -> None:
    """Export fp32 reference pair (decoder, jd_single) + fused model (fp32 and fp16)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    m = _load_model(model_id, nemo_path)

    vocab_size = int(m.tokenizer.vocab_size)
    num_extra = int(m.joint.num_extra_outputs)
    decoder_hidden = int(m.decoder.pred_hidden)
    decoder_layers = int(m.decoder.pred_rnn_layers)
    typer.echo(f"vocab={vocab_size} extra={num_extra} hidden={decoder_hidden} layers={decoder_layers}")

    targets = torch.tensor([[m.decoder.blank_idx]], dtype=torch.int32)
    target_len = torch.tensor([1], dtype=torch.int32)
    h = torch.zeros(decoder_layers, 1, decoder_hidden)
    c = torch.zeros(decoder_layers, 1, decoder_hidden)
    enc_step = torch.randn(1, 1024, 1)

    decoder = DecoderWrapper(m.decoder.eval()).eval()
    jd_single = JointDecisionSingleStep(
        JointWrapper(m.joint.eval()), vocab_size=vocab_size, num_extra=num_extra
    ).eval()
    fused = FusedDecoderJointDecision(m.decoder.eval(), m.joint.eval(), vocab_size, num_extra).eval()

    with torch.no_grad():
        dec_out, _, _ = decoder(targets, target_len, h, c)
    dec_out = dec_out.clone()

    decoder_inputs = [
        ct.TensorType(name="targets", shape=(1, 1), dtype=np.int32),
        ct.TensorType(name="target_length", shape=(1,), dtype=np.int32),
        ct.TensorType(name="h_in", shape=(decoder_layers, 1, decoder_hidden), dtype=np.float32),
        ct.TensorType(name="c_in", shape=(decoder_layers, 1, decoder_hidden), dtype=np.float32),
    ]
    decision_outputs = [
        ct.TensorType(name="token_id", dtype=np.int32),
        ct.TensorType(name="token_prob", dtype=np.float32),
        ct.TensorType(name="duration", dtype=np.int32),
        ct.TensorType(name="top_k_ids", dtype=np.int32),
        ct.TensorType(name="top_k_logits", dtype=np.float32),
    ]

    # fp32 reference pair (parity baseline)
    typer.echo("Exporting fp32 reference decoder…")
    traced_dec = torch.jit.trace(decoder, (targets, target_len, h, c), strict=False)
    dec_model = _coreml_convert(
        traced_dec,
        decoder_inputs,
        [
            ct.TensorType(name="decoder", dtype=np.float32),
            ct.TensorType(name="h_out", dtype=np.float32),
            ct.TensorType(name="c_out", dtype=np.float32),
        ],
        _settings(output_dir, ct.precision.FLOAT32),
    )
    dec_model.save(str(output_dir / "decoder_fp32.mlpackage"))

    typer.echo("Exporting fp32 reference joint_decision_single_step…")
    traced_jd = torch.jit.trace(jd_single, (enc_step, dec_out), strict=False)
    jd_model = _coreml_convert(
        traced_jd,
        [
            ct.TensorType(name="encoder_step", shape=(1, 1024, 1), dtype=np.float32),
            ct.TensorType(name="decoder_step", shape=(1, decoder_hidden, 1), dtype=np.float32),
        ],
        decision_outputs,
        _settings(output_dir, ct.precision.FLOAT32),
    )
    jd_model.save(str(output_dir / "joint_decision_fp32.mlpackage"))

    fused_inputs = decoder_inputs + [ct.TensorType(name="encoder_step", shape=(1, 1024, 1), dtype=np.float32)]
    fused_outputs = decision_outputs + [
        ct.TensorType(name="h_out", dtype=np.float32),
        ct.TensorType(name="c_out", dtype=np.float32),
    ]
    traced_fused = torch.jit.trace(fused, (targets, target_len, h, c, enc_step), strict=False)
    for prec, tag in ((ct.precision.FLOAT32, "fp32"), (ct.precision.FLOAT16, "fp16")):
        typer.echo(f"Exporting fused decoder_joint_decision ({tag})…")
        fused_model = _coreml_convert(traced_fused, fused_inputs, fused_outputs, _settings(output_dir, prec))
        fused_model.save(str(output_dir / f"decoder_joint_decision_{tag}.mlpackage"))

    typer.echo(f"Done. Packages in {output_dir}")


def _predict_separate(dec, jd, targets, h, c, enc_step):
    d = dec.predict({"targets": targets, "target_length": np.array([1], dtype=np.int32), "h_in": h, "c_in": c})
    j = jd.predict({"encoder_step": enc_step, "decoder_step": d["decoder"].astype(np.float32)})
    return d, j


def _predict_fused(fused, targets, h, c, enc_step):
    return fused.predict(
        {
            "targets": targets,
            "target_length": np.array([1], dtype=np.int32),
            "h_in": h,
            "c_in": c,
            "encoder_step": enc_step,
        }
    )


@app.command()
def parity(
    build_dir: Path = typer.Option(Path("build/fused"), "--build-dir", exists=True),
    steps: int = typer.Option(50, "--steps"),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Fused fp32 vs chained two-model fp32 reference, evolving state, random encoder frames."""
    cu = ct.ComputeUnit.CPU_ONLY
    dec = ct.models.MLModel(str(build_dir / "decoder_fp32.mlpackage"), compute_units=cu)
    jd = ct.models.MLModel(str(build_dir / "joint_decision_fp32.mlpackage"), compute_units=cu)
    fused = ct.models.MLModel(str(build_dir / "decoder_joint_decision_fp32.mlpackage"), compute_units=cu)

    rng = np.random.default_rng(seed)
    h = np.zeros((2, 1, 640), dtype=np.float32)
    c = np.zeros((2, 1, 640), dtype=np.float32)
    token = np.array([[8192]], dtype=np.int32)  # blank

    worst: Dict[str, float] = {}
    mismatches = 0
    for step in range(steps):
        enc_step = rng.standard_normal((1, 1024, 1)).astype(np.float32) * 2.0
        d, j = _predict_separate(dec, jd, token, h, c, enc_step)
        f = _predict_fused(fused, token, h, c, enc_step)

        for key, ref in (
            ("token_prob", j["token_prob"]),
            ("top_k_logits", j["top_k_logits"]),
            ("duration", j["duration"].astype(np.float64)),
            ("h_out", d["h_out"]),
            ("c_out", d["c_out"]),
        ):
            diff = float(np.max(np.abs(np.asarray(f[key], dtype=np.float64) - np.asarray(ref, dtype=np.float64))))
            worst[key] = max(worst.get(key, 0.0), diff)
        if int(f["token_id"].flatten()[0]) != int(j["token_id"].flatten()[0]):
            mismatches += 1

        # evolve state along the separate-path trajectory (the production reference)
        h, c = d["h_out"].astype(np.float32), d["c_out"].astype(np.float32)
        tok = int(j["token_id"].flatten()[0])
        token = np.array([[tok if tok < 8192 else 8192]], dtype=np.int32)

    typer.echo(f"steps={steps}  token_id mismatches={mismatches}")
    for key, val in worst.items():
        typer.echo(f"  max|Δ| {key:<14} {val:.3e}")
    gate = max(worst["token_prob"], worst["h_out"], worst["c_out"])
    typer.echo(f"PARITY {'PASS' if gate < 1e-5 and mismatches == 0 else 'FAIL'} (gate < 1e-5 on prob/state)")


CU_MAP = {
    "cpuOnly": ct.ComputeUnit.CPU_ONLY,
    "cpuAndGPU": ct.ComputeUnit.CPU_AND_GPU,
    "cpuAndNE": ct.ComputeUnit.CPU_AND_NE,
    "all": ct.ComputeUnit.ALL,
}


@app.command()
def bench(
    build_dir: Path = typer.Option(Path("build/fused"), "--build-dir", exists=True),
    warmup: int = typer.Option(10, "--warmup"),
    runs: int = typer.Option(200, "--runs"),
    precision: str = typer.Option("fp16", "--precision", help="fused/separate variant: fp32 or fp16"),
    shipped_dir: Optional[Path] = typer.Option(
        None, "--shipped-dir", help="Production mlmodelc dir (Decoder.mlmodelc + JointDecisionv3.mlmodelc) to bench as the separate arm"
    ),
    output_json: Optional[Path] = typer.Option(None, "--output-json"),
) -> None:
    """Interleaved A/B: fused single-dispatch vs separate decoder->joint chain (Trial 15/19-22 methodology)."""
    rng = np.random.default_rng(0)
    enc_step = rng.standard_normal((1, 1024, 1)).astype(np.float32)
    h = np.zeros((2, 1, 640), dtype=np.float32)
    c = np.zeros((2, 1, 640), dtype=np.float32)
    token = np.array([[8192]], dtype=np.int32)

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cu_name, cu in CU_MAP.items():
        fused = ct.models.MLModel(
            str(build_dir / f"decoder_joint_decision_{precision}.mlpackage"), compute_units=cu
        )
        if shipped_dir is not None:
            dec = ct.models.CompiledMLModel(str(shipped_dir / "Decoder.mlmodelc"), compute_units=cu)
            jd = ct.models.CompiledMLModel(str(shipped_dir / "JointDecisionv3.mlmodelc"), compute_units=cu)
        else:
            dec = ct.models.MLModel(str(build_dir / "decoder_fp32.mlpackage"), compute_units=cu)
            jd = ct.models.MLModel(str(build_dir / "joint_decision_fp32.mlpackage"), compute_units=cu)

        for _ in range(warmup):
            _predict_separate(dec, jd, token, h, c, enc_step)
            _predict_fused(fused, token, h, c, enc_step)

        sep_ms, fus_ms = [], []
        for _ in range(runs):
            t0 = time.perf_counter()
            _predict_separate(dec, jd, token, h, c, enc_step)
            t1 = time.perf_counter()
            _predict_fused(fused, token, h, c, enc_step)
            t2 = time.perf_counter()
            sep_ms.append((t1 - t0) * 1e3)
            fus_ms.append((t2 - t1) * 1e3)

        def stats(xs):
            xs = sorted(xs)
            return {
                "median_ms": statistics.median(xs),
                "p95_ms": xs[int(len(xs) * 0.95) - 1],
            }

        results[cu_name] = {"separate": stats(sep_ms), "fused": stats(fus_ms)}
        s, f = results[cu_name]["separate"], results[cu_name]["fused"]
        typer.echo(
            f"{cu_name:<10} separate {s['median_ms']:6.3f} ms (p95 {s['p95_ms']:6.3f})   "
            f"fused {f['median_ms']:6.3f} ms (p95 {f['p95_ms']:6.3f})   "
            f"speedup {s['median_ms'] / f['median_ms']:.2f}x"
        )

    if output_json is not None:
        output_json.write_text(json.dumps(results, indent=2))
        typer.echo(f"wrote {output_json}")


if __name__ == "__main__":
    app()
