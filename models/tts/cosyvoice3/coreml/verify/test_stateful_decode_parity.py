"""Parity check: stateful LLM-Decode mlpackage vs non-stateful pass-through.

Loads both compiled models, seeds them with the same prefill-synthesized KV
cache, and runs N_STEPS decode steps with the same token-embedding trajectory.
Asserts the logits match within fp16 tolerance — any divergence flags a bug in
the stateful conversion (wrong state seeding, wrong per-step update, etc.).

Usage:
    uv run python verify/test_stateful_decode_parity.py \
        --nonstateful build/upload/cosyvoice3-coreml/LLM-Decode-M768-fp16.mlpackage \
        --stateful    build/llm-fp16-stateful/LLM-Decode-M768-fp16-stateful.mlpackage \
        --steps 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE / "CosyVoice"))
sys.path.insert(0, str(HERE / "CosyVoice" / "third_party" / "Matcha-TTS"))

from src.llm_coreml import Qwen2Prefill  # noqa: E402


def load_prefill_kv(max_len: int = 768, t_prefill: int = 256):
    """Run PyTorch Qwen2Prefill on a short prompt → return kv_k / kv_v (fp32)."""
    from cosyvoice.cli.cosyvoice import CosyVoice3

    model_dir = HERE.parent / "cosyvoice3_dl"
    cv = CosyVoice3(str(model_dir), load_trt=False, load_vllm=False, fp16=False)
    cv.model.llm.float()
    llm_model = cv.model.llm
    qwen = llm_model.llm.model
    speech_head = llm_model.llm_decoder

    fe = cv.frontend
    mi = fe.frontend_zero_shot(
        "希望你以后能够做的比我还好用",
        "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
        str(HERE / "CosyVoice" / "asset" / "zero_shot_prompt.wav"),
        cv.sample_rate,
        zero_shot_spk_id="",
    )
    text = torch.concat([mi["prompt_text"], mi["text"]], dim=1).to(torch.int64)
    text_emb = llm_model.llm.model.model.embed_tokens(text)
    sos_emb = llm_model.speech_embedding.weight[llm_model.sos].reshape(1, 1, -1)
    task_id_emb = llm_model.speech_embedding.weight[llm_model.task_id].reshape(1, 1, -1)
    pst = mi["llm_prompt_speech_token"]
    pst_emb = (llm_model.speech_embedding(pst) if pst.shape[1] > 0
               else torch.zeros(1, 0, llm_model.speech_embedding.embedding_dim,
                                dtype=text_emb.dtype))
    lm_input = torch.concat([sos_emb, text_emb, task_id_emb, pst_emb], dim=1).to(torch.float32)

    t_real = lm_input.shape[1]
    if t_real >= t_prefill:
        lm_input = lm_input[:, :t_prefill, :]
        input_len_val = t_prefill
    else:
        pad = torch.zeros(1, t_prefill - t_real, lm_input.shape[-1], dtype=torch.float32)
        lm_input = torch.cat([lm_input, pad], dim=1)
        input_len_val = t_real

    pre = Qwen2Prefill(qwen, speech_head, max_len=max_len, t_prefill=t_prefill).eval()
    with torch.no_grad():
        last_hidden, speech_logits, kv_k, kv_v = pre(
            lm_input, torch.tensor([input_len_val], dtype=torch.int32))

    print(f"  prefill: input_len={input_len_val}  kv_k shape={tuple(kv_k.shape)}")
    return kv_k, kv_v, input_len_val, llm_model


def run_nonstateful(model, inputs_embeds, kv_k, kv_v, cur_len):
    out = model.predict({
        "inputs_embeds": inputs_embeds.astype(np.float32),
        "kv_k": kv_k.astype(np.float32),
        "kv_v": kv_v.astype(np.float32),
        "cur_len": np.array([cur_len], dtype=np.int32),
    })
    return out["speech_logits"], out["kv_k_out"], out["kv_v_out"]


def run_stateful(model, state, inputs_embeds, cur_len):
    out = model.predict(
        {
            "inputs_embeds": inputs_embeds.astype(np.float32),
            "cur_len": np.array([cur_len], dtype=np.int32),
        },
        state=state,
    )
    return out["speech_logits"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nonstateful", required=True)
    ap.add_argument("--stateful", required=True)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--tolerance", type=float, default=5e-3,
                    help="Max abs diff tolerance (fp16 ~5e-4 per op, accumulates).")
    args = ap.parse_args()

    print(f"[1/4] Running PyTorch prefill for realistic KV seed...")
    kv_k, kv_v, input_len, llm_model = load_prefill_kv()
    L, _, Hkv, M, D = kv_k.shape
    print(f"       L={L} Hkv={Hkv} M={M} D={D}  cur_len0={input_len}")

    print(f"[2/4] Loading non-stateful decode ({Path(args.nonstateful).name})...")
    ns = ct.models.MLModel(args.nonstateful, compute_units=ct.ComputeUnit.CPU_ONLY)

    print(f"[3/4] Loading stateful decode ({Path(args.stateful).name})...")
    sf = ct.models.MLModel(args.stateful, compute_units=ct.ComputeUnit.CPU_ONLY)
    state = sf.make_state()
    # Seed per-layer state buffers from prefill KV.
    for i in range(L):
        # write_state takes an fp32 ndarray and internally casts to the
        # state's fp16 storage. (read_state returns an fp32 copy, not a
        # view — in-place writes on the copy do not propagate.)
        state.write_state(f"kv_k_{i}", kv_k[i].cpu().numpy().astype(np.float32))
        state.write_state(f"kv_v_{i}", kv_v[i].cpu().numpy().astype(np.float32))

    print(f"[4/4] Running {args.steps} decode steps in lockstep...")
    torch.manual_seed(0)
    kv_k_ns = kv_k.numpy()
    kv_v_ns = kv_v.numpy()
    cur_len = input_len
    max_diff = 0.0
    tokens_match = 0
    for step in range(args.steps):
        # Fake next-step embedding from speech_embedding row (realistic magnitude).
        tok_id = (step * 137) % 6_561
        emb = llm_model.speech_embedding.weight[tok_id].reshape(1, 1, -1).to(torch.float32).detach().numpy()

        logits_ns, kv_k_ns, kv_v_ns = run_nonstateful(ns, emb, kv_k_ns, kv_v_ns, cur_len)
        logits_sf = run_stateful(sf, state, emb, cur_len)
        diff = np.abs(logits_ns - logits_sf).max()
        max_diff = max(max_diff, float(diff))
        arg_ns = int(logits_ns.argmax())
        arg_sf = int(logits_sf.argmax())
        match = "✓" if arg_ns == arg_sf else "✗"
        if arg_ns == arg_sf:
            tokens_match += 1
        print(f"  step {step:>3}  cur_len={cur_len}  max|Δlogits|={diff:.3e}  "
              f"argmax ns={arg_ns} sf={arg_sf} {match}")
        cur_len += 1

    print(f"\nsteps={args.steps}  max|Δlogits|={max_diff:.3e}  "
          f"argmax agreement={tokens_match}/{args.steps}")

    if max_diff > args.tolerance:
        print(f"FAIL: max_diff {max_diff:.3e} > tol {args.tolerance:.3e}")
        sys.exit(1)
    if tokens_match != args.steps:
        print(f"FAIL: {args.steps - tokens_match} step(s) had argmax mismatch")
        sys.exit(2)
    print("PASS: stateful and non-stateful decode paths match.")


if __name__ == "__main__":
    main()
