#!/usr/bin/env python3
"""Export the parakeet-redux decoder (RNNT prediction net) and single-step JointDecision (v3 contract, top-K=64).

Mirrors the nvidia v3 export in ../../parakeet-tdt-v3-0.6b/coreml/convert-parakeet.py so the outputs are drop-in
replacements for Decoder.mlmodelc / JointDecisionv3.mlmodelc (same I/O names and shapes, iOS17, fp16 storage).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
V3_DIR = HERE.parent.parent / "parakeet-tdt-v3-0.6b" / "coreml"
sys.path.insert(0, str(V3_DIR))
sys.path.insert(0, str(HERE))

import coremltools as ct  # noqa: E402
import nemo.collections.asr as nemo_asr  # noqa: E402

from individual_components import (  # noqa: E402
    DecoderWrapper,
    ExportSettings,
    JointDecisionSingleStep,
    JointWrapper,
    _coreml_convert,
)
from redux_weights import load_into_nemo  # noqa: E402


def save(model: ct.models.MLModel, path: Path, desc: str) -> None:
    model.minimum_deployment_target = ct.target.iOS17
    model.short_description = desc
    model.author = "Fluid Inference"
    model.save(str(path))
    print("saved", path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", type=Path, default=Path("~/Documents/parakeet-redux-work/hf").expanduser())
    ap.add_argument("--out", type=Path, default=Path("~/Documents/parakeet-redux-work/components").expanduser())
    ap.add_argument("--name", default="redux", help="Model name used in the mlpackage descriptions")
    args = ap.parse_args()
    title = f"Parakeet-{args.name}"
    args.out.mkdir(parents=True, exist_ok=True)

    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3", map_location="cpu")
    model.eval()
    load_into_nemo(model, args.hf_dir)

    settings = ExportSettings(
        output_dir=args.out, compute_units=ct.ComputeUnit.CPU_ONLY, deployment_target=ct.target.iOS17,
        compute_precision=None, max_audio_seconds=15.0, max_symbol_steps=1,
    )
    decoder = DecoderWrapper(model.decoder.eval())
    joint = JointWrapper(model.joint.eval())
    model.decoder._rnnt_export = True

    hidden = int(model.decoder.pred_hidden)
    layers = int(model.decoder.pred_rnn_layers)
    targets = torch.full((1, 1), fill_value=model.decoder.blank_idx, dtype=torch.int32)
    target_lengths = torch.tensor([1], dtype=torch.int32)
    zero_state = torch.zeros(layers, 1, hidden, dtype=torch.float32)
    enc_step = torch.randn(1, 1024, 1)
    with torch.inference_mode():
        dec_ref, h_ref, c_ref = decoder(targets, target_lengths, zero_state, zero_state)
    dec_ref = dec_ref.clone()
    dec_step = dec_ref[:, :, :1].contiguous()

    traced_decoder = torch.jit.trace(decoder, (targets, target_lengths, zero_state, zero_state), strict=False).eval()
    decoder_model = _coreml_convert(
        traced_decoder,
        [
            ct.TensorType(name="targets", shape=(1, 1), dtype=np.int32),
            ct.TensorType(name="target_length", shape=(1,), dtype=np.int32),
            ct.TensorType(name="h_in", shape=(layers, 1, hidden), dtype=np.float32),
            ct.TensorType(name="c_in", shape=(layers, 1, hidden), dtype=np.float32),
        ],
        [
            ct.TensorType(name="decoder", dtype=np.float32),
            ct.TensorType(name="h_out", dtype=np.float32),
            ct.TensorType(name="c_out", dtype=np.float32),
        ],
        settings,
    )
    save(decoder_model, args.out / "Decoder.mlpackage", f"{title} decoder (RNNT prediction network)")

    vocab_size = int(model.tokenizer.vocab_size)
    num_extra = int(model.joint.num_extra_outputs)
    jd = JointDecisionSingleStep(joint, vocab_size=vocab_size, num_extra=num_extra)
    traced_jd = torch.jit.trace(jd, (enc_step, dec_step), strict=False).eval()
    jd_model = _coreml_convert(
        traced_jd,
        [
            ct.TensorType(name="encoder_step", shape=(1, 1024, 1), dtype=np.float32),
            ct.TensorType(name="decoder_step", shape=(1, hidden, 1), dtype=np.float32),
        ],
        [
            ct.TensorType(name="token_id", dtype=np.int32),
            ct.TensorType(name="token_prob", dtype=np.float32),
            ct.TensorType(name="duration", dtype=np.int32),
            ct.TensorType(name="top_k_ids", dtype=np.int32),
            ct.TensorType(name="top_k_logits", dtype=np.float32),
        ],
        settings,
    )
    save(jd_model, args.out / "JointDecisionv3.mlpackage", f"{title} single-step joint decision (top-K 64)")

    # Parity on the trace inputs (CPU)
    with torch.inference_mode():
        t_ids, t_prob, t_dur, t_topk_ids, t_topk_logits = jd(enc_step, dec_step)
    cm = jd_model.predict({"encoder_step": enc_step.numpy(), "decoder_step": dec_step.numpy()})
    print("joint parity: token_id", int(t_ids.flatten()[0]), int(cm["token_id"].flatten()[0]),
          "prob", float(t_prob.flatten()[0]), float(cm["token_prob"].flatten()[0]),
          "max|topk_logits diff|", float(np.abs(t_topk_logits.numpy() - cm["top_k_logits"]).max()))
    cd = decoder_model.predict({"targets": targets.numpy(), "target_length": target_lengths.numpy(),
                                "h_in": zero_state.numpy(), "c_in": zero_state.numpy()})
    print("decoder parity: max|decoder diff|", float(np.abs(dec_ref.numpy() - cd["decoder"]).max()),
          "max|h diff|", float(np.abs(h_ref.numpy() - cd["h_out"]).max()))


if __name__ == "__main__":
    main()
