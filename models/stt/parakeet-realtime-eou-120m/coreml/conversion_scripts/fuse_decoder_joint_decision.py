#!/usr/bin/env python3
"""Fuse the Parakeet EOU `decoder` + `joint_decision` into one CoreML graph.

Why: the EOU streaming decode loop dispatches 2 CoreML predictions per RNNT
step (229 steps for 7.8 s audio => 458 dispatches/utt, 58 ms/utt, 100% CPU).
The decoder contains `ios17.lstm` which has NO ANE kernel at any precision
(ANE_Candidates.md, categorical dead end #4), so this graph can never be
ANE-resident. The win available is dispatch-halving: fuse the two graphs into
one (Nemotron B1 precedent) so each RNNT inner step is 1 prediction call.

Mechanism: instead of re-tracing from the NeMo checkpoint (nemo-toolkit not
installed locally, multi-GB), we rebuild the fused graph directly with the MIL
builder using the *exact fp16 weight tensors* read out of the shipped
`.mlmodelc` weight blobs (the cached `.mlpackage` copies are stripped to
weights-only, so the blob offsets come from each `model.mil`). The op
sequence replicates the two source model.mil programs 1:1, except:
  - the decoder's output transpose [1,1,640]->[1,640,1] followed by the
    joint's transpose back [1,640,1]->[1,1,640] cancel (batch=seq=1, pure
    layout, no arithmetic change) and are elided;
  - the fp32<->fp16 boundary casts between the two models disappear (the
    decoder LSTM output stays fp16 into the joint projection). NOTE: this
    means the fused joint sees the un-rounded fp16 value rather than
    fp16->fp32->fp16 (identity for fp16 values, so still exact).

Inputs  (fp32/int32 IO contract, same names as the source models):
  targets       [1, 1]      int32   last emitted token (blank=1026 initially)
  h_in          [1, 1, 640] fp32    LSTM hidden state
  c_in          [1, 1, 640] fp32    LSTM cell state
  encoder_step  [1, 512, 1] fp32    one encoder frame
Outputs:
  token_id      [1, 1, 1]   int32   argmax token
  token_prob    [1, 1, 1]   fp32    softmax prob of token_id
  h_out         [1, 1, 640] fp32
  c_out         [1, 1, 640] fp32
  (a --with-topk variant additionally emits top_k_ids/top_k_logits [1,1,1,64]
   to match joint_decision's full interface; the Swift RnntDecoder only reads
   token_id/h_out/c_out, so the default lean build drops the per-step
   1027-way sort.)

Usage:
    python fuse_decoder_joint_decision.py \
        --model-dir "$HOME/Library/Application Support/FluidAudio/Models/parakeet-eou-streaming/160ms" \
        --output-dir /tmp/eou_fused [--with-topk]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.libmilstoragepython import _BlobStorageReader

AUTHOR = "Fluid Inference"

# MIL const name -> (our name, expected shape). Offsets are parsed from each
# variant's model.mil so the script works for 160/320/1280ms alike.
DECODER_CONSTS = {
    "module_prediction_embed_weight_to_fp16": ("embed_w", (1027, 640)),
    "concat_1_to_fp16": ("lstm_w_ih", (2560, 640)),
    "concat_2_to_fp16": ("lstm_w_hh", (2560, 640)),
    "concat_0_to_fp16": ("lstm_bias", (2560,)),
}
JOINT_CONSTS = {
    "joint_module_enc_weight_to_fp16": ("enc_w", (640, 512)),
    "joint_module_enc_bias_to_fp16": ("enc_b", (640,)),
    "joint_module_pred_weight_to_fp16": ("pred_w", (640, 640)),
    "joint_module_pred_bias_to_fp16": ("pred_b", (640,)),
    "joint_module_joint_net_2_weight_to_fp16": ("out_w", (1027, 640)),
    "joint_module_joint_net_2_bias_to_fp16": ("out_b", (1027,)),
}

_CONST_RE = re.compile(
    r'tensor<fp16, \[[\d, ]+\]> (\w+) = const\(\).*?'
    r'offset = tensor<uint64, \[\]>\((\d+)\)',
)


def _load_blobs(mlmodelc: Path, table: dict) -> dict:
    mil_text = (mlmodelc / "model.mil").read_text()
    offsets = {m.group(1): int(m.group(2)) for m in _CONST_RE.finditer(mil_text)}
    reader = _BlobStorageReader(str(mlmodelc / "weights" / "weight.bin"))
    out = {}
    for mil_name, (name, shape) in table.items():
        assert mil_name in offsets, f"{mil_name} not found in {mlmodelc}/model.mil"
        raw = np.array(reader.read_fp16_data(offsets[mil_name]), copy=True)
        out[name] = raw.view(np.float16).reshape(shape)
    return out


def build(model_dir: Path, output_dir: Path, with_topk: bool, replicate_boundary: bool = False) -> Path:
    dec = _load_blobs(model_dir / "decoder.mlmodelc", DECODER_CONSTS)
    jd = _load_blobs(model_dir / "joint_decision.mlmodelc", JOINT_CONSTS)

    embed_w = dec["embed_w"]  # [1027, 640]
    lstm_w_ih = dec["lstm_w_ih"]  # [2560, 640]
    lstm_w_hh = dec["lstm_w_hh"]  # [2560, 640]
    lstm_bias = dec["lstm_bias"]  # [2560]

    enc_w = jd["enc_w"]  # [640, 512]
    enc_b = jd["enc_b"]  # [640]
    pred_w = jd["pred_w"]  # [640, 640]
    pred_b = jd["pred_b"]  # [640]
    out_w = jd["out_w"]  # [1027, 640]
    out_b = jd["out_b"]  # [1027]

    for name, arr, shape in [
        ("embed_w", embed_w, (1027, 640)),
        ("lstm_w_ih", lstm_w_ih, (2560, 640)),
        ("lstm_w_hh", lstm_w_hh, (2560, 640)),
        ("lstm_bias", lstm_bias, (2560,)),
        ("enc_w", enc_w, (640, 512)),
        ("enc_b", enc_b, (640,)),
        ("pred_w", pred_w, (640, 640)),
        ("pred_b", pred_b, (640,)),
        ("out_w", out_w, (1027, 640)),
        ("out_b", out_b, (1027,)),
    ]:
        assert tuple(arr.shape) == shape, f"{name}: {arr.shape} != {shape}"
        assert arr.dtype == np.float16, f"{name}: {arr.dtype}"

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, 1), dtype=types.int32),  # targets
            mb.TensorSpec(shape=(1, 1, 640), dtype=types.fp32),  # h_in
            mb.TensorSpec(shape=(1, 1, 640), dtype=types.fp32),  # c_in
            mb.TensorSpec(shape=(1, 512, 1), dtype=types.fp32),  # encoder_step
        ],
        opset_version=ct.target.iOS17,
    )
    def fused(targets, h_in, c_in, encoder_step):
        # ===== decoder (RNNT prediction network) =====
        targets_i16 = mb.cast(x=targets, dtype="int16")
        emb = mb.gather(
            x=embed_w, indices=targets_i16, axis=0, batch_dims=0, validate_indices=False
        )  # [1, 1, 640] fp16
        x_seq = mb.transpose(x=emb, perm=[1, 0, 2])  # [U=1, B=1, 640]

        h0 = mb.squeeze(x=mb.cast(x=h_in, dtype="fp16"), axes=[0])  # [1, 640]
        c0 = mb.squeeze(x=mb.cast(x=c_in, dtype="fp16"), axes=[0])  # [1, 640]

        lstm_out, lstm_h, lstm_c = mb.lstm(
            x=x_seq,
            initial_h=h0,
            initial_c=c0,
            weight_ih=lstm_w_ih,
            weight_hh=lstm_w_hh,
            bias=lstm_bias,
            direction="forward",
            output_sequence=True,
            recurrent_activation="sigmoid",
            cell_activation="tanh",
            activation="tanh",
        )  # out [1, 1, 640] fp16, h/c [1, 640] fp16

        h_out = mb.cast(x=mb.expand_dims(x=lstm_h, axes=[0]), dtype="fp32", name="h_out")
        c_out = mb.cast(x=mb.expand_dims(x=lstm_c, axes=[0]), dtype="fp32", name="c_out")

        # ===== joint_decision =====
        # Source models round-trip the decoder output through
        # transpose([1,2,0]) -> fp32 -> fp16 -> transpose([0,2,1]).
        # With batch=seq=1 both transposes are pure layout and the fp16->fp32
        # ->fp16 cast is the identity, so lstm_out [1, 1, 640] feeds the pred
        # projection directly.
        enc16 = mb.cast(x=encoder_step, dtype="fp16")
        enc_t = mb.transpose(x=enc16, perm=[0, 2, 1])  # [1, 1, 512]
        f = mb.linear(x=enc_t, weight=enc_w, bias=enc_b)  # [1, 1, 640]
        if replicate_boundary:
            # Reproduce the exact two-model boundary: the decoder emits
            # `decoder` [1, 640, 1] fp32 and the joint re-ingests it via
            # cast(fp16) -> transpose. The transpose->linear pattern lowers
            # to a different GEMM kernel than linear(lstm_out) and that
            # accumulation difference is what flips near-tie argmaxes.
            dec_t = mb.transpose(x=lstm_out, perm=[1, 2, 0])  # [1, 640, 1]
            dec_f32 = mb.cast(x=dec_t, dtype="fp32")
            dec_f16 = mb.cast(x=dec_f32, dtype="fp16")
            g_in = mb.transpose(x=dec_f16, perm=[0, 2, 1])  # [1, 1, 640]
        else:
            g_in = lstm_out
        g = mb.linear(x=g_in, weight=pred_w, bias=pred_b)  # [1, 1, 640]

        f4 = mb.expand_dims(x=f, axes=[2])  # [1, 1, 1, 640]
        g4 = mb.expand_dims(x=g, axes=[1])  # [1, 1, 1, 640]
        joint = mb.relu(x=mb.add(x=f4, y=g4))
        logits = mb.linear(x=joint, weight=out_w, bias=out_b)  # [1, 1, 1, 1027]

        token_id = mb.reduce_argmax(
            x=logits, axis=-1, keep_dims=False, output_dtype="int32", name="token_id"
        )  # [1, 1, 1]

        probs = mb.softmax(x=logits, axis=-1)
        idx = mb.cast(x=mb.expand_dims(x=token_id, axes=[-1]), dtype="int16")
        prob = mb.gather_along_axis(x=probs, indices=idx, axis=-1, validate_indices=False)
        token_prob = mb.cast(x=mb.squeeze(x=prob, axes=[-1]), dtype="fp32", name="token_prob")

        if with_topk:
            tk_logits, tk_ids = mb.topk(
                x=logits, k=64, axis=-1, ascending=False, sort=True,
                return_indices=True, output_indices_dtype="uint16",
            )
            top_k_logits = mb.cast(x=tk_logits, dtype="fp32", name="top_k_logits")
            top_k_ids = mb.cast(x=tk_ids, dtype="int32", name="top_k_ids")
            return token_id, token_prob, h_out, c_out, top_k_ids, top_k_logits

        return token_id, token_prob, h_out, c_out

    # The program is already explicit fp16 internally; FLOAT32 precision makes
    # the converter's fp16-cast pass a no-op so the graph ships byte-exact.
    convert_kwargs = {}
    if replicate_boundary:
        # Keep the boundary-replicating transpose/cast chain: stop the
        # optimizer from cancelling it back into the lean form.
        pipeline = ct.PassPipeline.DEFAULT
        for p in (
            "common::cast_optimization",
            "common::merge_consecutive_transposes",
            "common::reduce_transposes",
            "common::noop_elimination",
        ):
            if p in pipeline.passes:
                pipeline.remove_passes([p])
        convert_kwargs["pass_pipeline"] = pipeline
    mlmodel = ct.convert(
        fused,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        **convert_kwargs,
    )
    mlmodel.author = AUTHOR
    suffix = "" if not with_topk else "_topk"
    if replicate_boundary:
        suffix += "_boundary"
    mlmodel.short_description = (
        "Parakeet EOU fused decoder + joint decision (1 dispatch per RNNT step). "
        "LSTM has no ANE kernel; this graph is CPU-resident by design."
    )
    out_path = output_dir / f"decoder_joint_decision_fused{suffix}.mlpackage"
    output_dir.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))
    print(f"Saved {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--with-topk", action="store_true")
    ap.add_argument("--replicate-boundary", action="store_true")
    args = ap.parse_args()
    build(args.model_dir, args.output_dir, args.with_topk, args.replicate_boundary)


if __name__ == "__main__":
    main()
