"""Convert CosyVoice3LM (Qwen2) to two CoreML mlpackages: Prefill + Decode.

Pipeline (mirrors convert-flow.py):
  1. Load CosyVoice3 end-to-end (for weights + frontend to build a real lm_input).
  2. Wrap Qwen2 weights with src.llm_coreml.Qwen2Prefill / Qwen2Decode.
  3. Sanity-check wrapper outputs vs each other (decode consumes prefill's KV).
  4. torch.jit.trace each wrapper separately.
  5. coremltools convert → two mlpackages (FP32 default, FP16 optional).

Static shapes
-------------
  * Qwen2Prefill input: inputs_embeds [1, T_prefill, 896], input_len [1]
      -> last_hidden [1, T_prefill, 896]
         speech_logits [1, T_prefill, 6761]
         kv_k, kv_v [L=24, 1, Hkv=2, max_len, D=64]
  * Qwen2Decode input: inputs_embeds [1, 1, 896], kv_k, kv_v, cur_len [1]
      -> speech_logits [1, 1, 6761]
         kv_k_out, kv_v_out (same shape as inputs)

Both share the same max_len so Swift can pass prefill's KV directly into decode.

Usage:
    uv run python convert-llm.py --output-dir ./build/llm-fp32
    uv run python convert-llm.py --output-dir ./build/llm-fp16 --fp16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "verify" / "CosyVoice"))
sys.path.insert(0, str(HERE / "verify" / "CosyVoice" / "third_party" / "Matcha-TTS"))

from src.llm_coreml import Qwen2Prefill, Qwen2Decode, Qwen2DecodeStateful  # noqa: E402


MODEL_DIR = HERE / "cosyvoice3_dl"


def load_llm():
    """Return (qwen_for_causal_lm, speech_lm_head, llm_model) all fp32 on CPU."""
    from cosyvoice.cli.cosyvoice import CosyVoice3
    cv = CosyVoice3(str(MODEL_DIR), load_trt=False, load_vllm=False, fp16=False)
    cv.model.llm.float()
    llm_model = cv.model.llm
    qwen = llm_model.llm.model          # Qwen2ForCausalLM
    speech_head = llm_model.llm_decoder  # nn.Linear(896, 6761)
    return qwen, speech_head, llm_model, cv


def build_dummy_lm_input(cv, llm_model, t_prefill: int) -> torch.Tensor:
    """Build a plausible prefill embed tensor of shape [1, t_prefill, 896].

    Uses the frontend to produce a real prefix of whatever length; if shorter
    than t_prefill we zero-pad on the right; if longer we truncate. This makes
    tracing exercise the real embedding distribution.
    """
    fe = cv.frontend
    mi = fe.frontend_zero_shot(
        "希望你以后能够做的比我还好用",
        "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
        str(HERE / "verify" / "CosyVoice" / "asset" / "zero_shot_prompt.wav"),
        cv.sample_rate,
        zero_shot_spk_id="",
    )
    text = torch.concat([mi["prompt_text"], mi["text"]], dim=1).to(torch.int64)
    text_emb = llm_model.llm.model.model.embed_tokens(text)
    sos_emb     = llm_model.speech_embedding.weight[llm_model.sos].reshape(1, 1, -1)
    task_id_emb = llm_model.speech_embedding.weight[llm_model.task_id].reshape(1, 1, -1)
    pst = mi["llm_prompt_speech_token"]
    if pst.shape[1] > 0:
        pst_emb = llm_model.speech_embedding(pst)
    else:
        pst_emb = torch.zeros(1, 0, llm_model.speech_embedding.embedding_dim,
                              dtype=text_emb.dtype)
    lm_input = torch.concat([sos_emb, text_emb, task_id_emb, pst_emb], dim=1).to(torch.float32)
    T_real = lm_input.shape[1]
    if T_real >= t_prefill:
        lm_input = lm_input[:, :t_prefill, :]
        input_len_val = t_prefill
    else:
        pad = torch.zeros(1, t_prefill - T_real, lm_input.shape[-1], dtype=torch.float32)
        lm_input = torch.cat([lm_input, pad], dim=1)
        input_len_val = T_real
    return lm_input, input_len_val


def convert_prefill(qwen, speech_head, lm_input, input_len_val, max_len, out_dir, fp16, min_dep):
    t_prefill = lm_input.shape[1]
    print(f"[prefill 1/4] Building Qwen2Prefill wrapper (T_prefill={t_prefill}, max_len={max_len})...")
    wrapper = Qwen2Prefill(qwen, speech_head, max_len=max_len, t_prefill=t_prefill).eval()

    input_len_t = torch.tensor([input_len_val], dtype=torch.int32)

    print(f"[prefill 2/4] Running PyTorch wrapper (warm-up)...")
    with torch.no_grad():
        last_hidden, speech_logits, kv_k, kv_v = wrapper(lm_input, input_len_t)
    print(f"      last_hidden: {tuple(last_hidden.shape)}  speech_logits: {tuple(speech_logits.shape)}")
    print(f"      kv_k: {tuple(kv_k.shape)}  kv_v: {tuple(kv_v.shape)}")

    print(f"[prefill 3/4] Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (lm_input, input_len_t), strict=False)

    print(f"[prefill 4/4] Converting to CoreML mlpackage...")
    precision = _make_precision(fp16)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="inputs_embeds", shape=(1, t_prefill, 896), dtype=np.float32),
            ct.TensorType(name="input_len", shape=(1,), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="last_hidden", dtype=np.float32),
            ct.TensorType(name="speech_logits", dtype=np.float32),
            ct.TensorType(name="kv_k", dtype=np.float32),
            ct.TensorType(name="kv_v", dtype=np.float32),
        ],
        compute_precision=precision,
        minimum_deployment_target=min_dep,
        convert_to="mlprogram",
    )
    tag = "fp16" if fp16 else "fp32"
    mlp = out_dir / f"LLM-Prefill-T{t_prefill}-M{max_len}-{tag}.mlpackage"
    mlmodel.save(str(mlp))
    print(f"      saved: {mlp}")

    return mlp, (lm_input, input_len_t, last_hidden, speech_logits, kv_k, kv_v)


def convert_decode(qwen, speech_head, max_len, prefill_kv_k, prefill_kv_v, cur_len_val,
                    out_dir, fp16, min_dep):
    print(f"[decode 1/4] Building Qwen2Decode wrapper (max_len={max_len})...")
    wrapper = Qwen2Decode(qwen, speech_head, max_len=max_len).eval()

    # Dummy next-token embedding: just take position 0 of the speech embedding table.
    # The actual embedding table lives in CosyVoice3LM.speech_embedding (6761, 896),
    # but we don't need it traced — Swift does the lookup and passes embeds in.
    torch.manual_seed(0)
    dummy_embed = torch.randn(1, 1, 896, dtype=torch.float32) * 0.02
    cur_len_t = torch.tensor([cur_len_val], dtype=torch.int32)

    print(f"[decode 2/4] Running PyTorch wrapper (warm-up)...")
    with torch.no_grad():
        speech_logits, kv_k_out, kv_v_out = wrapper(
            dummy_embed, prefill_kv_k, prefill_kv_v, cur_len_t
        )
    print(f"      speech_logits: {tuple(speech_logits.shape)}")
    print(f"      kv_k_out: {tuple(kv_k_out.shape)}  kv_v_out: {tuple(kv_v_out.shape)}")

    print(f"[decode 3/4] Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper,
            (dummy_embed, prefill_kv_k, prefill_kv_v, cur_len_t),
            strict=False,
        )

    print(f"[decode 4/4] Converting to CoreML mlpackage...")
    L, _, Hkv, M, D = prefill_kv_k.shape
    precision = _make_precision(fp16)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="inputs_embeds", shape=(1, 1, 896), dtype=np.float32),
            ct.TensorType(name="kv_k", shape=(L, 1, Hkv, M, D), dtype=np.float32),
            ct.TensorType(name="kv_v", shape=(L, 1, Hkv, M, D), dtype=np.float32),
            ct.TensorType(name="cur_len", shape=(1,), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="speech_logits", dtype=np.float32),
            ct.TensorType(name="kv_k_out", dtype=np.float32),
            ct.TensorType(name="kv_v_out", dtype=np.float32),
        ],
        compute_precision=precision,
        minimum_deployment_target=min_dep,
        convert_to="mlprogram",
    )
    tag = "fp16" if fp16 else "fp32"
    mlp = out_dir / f"LLM-Decode-M{max_len}-{tag}.mlpackage"
    mlmodel.save(str(mlp))
    print(f"      saved: {mlp}")

    return mlp, (dummy_embed, cur_len_t, speech_logits, kv_k_out, kv_v_out)


def convert_decode_stateful(qwen, speech_head, max_len, prefill_kv_k, prefill_kv_v,
                              cur_len_val, out_dir, fp16, min_dep):
    """Stateful variant: kv_k / kv_v live inside the model as StateType,
    mutated in place per call. macOS 15+ only. ~2-3× faster decode vs
    pass-through (no per-step MLMultiArray binding for the 18 MB KV).
    """
    print(f"[decode-stateful 1/4] Building Qwen2DecodeStateful wrapper (max_len={max_len})...")
    wrapper = Qwen2DecodeStateful(qwen, speech_head, max_len=max_len).eval()

    L, _, Hkv, M, D = prefill_kv_k.shape

    def _seed_from_prefill():
        with torch.no_grad():
            for i in range(L):
                getattr(wrapper, f"kv_k_{i}").copy_(prefill_kv_k[i])
                getattr(wrapper, f"kv_v_{i}").copy_(prefill_kv_v[i])

    _seed_from_prefill()

    torch.manual_seed(0)
    dummy_embed = torch.randn(1, 1, 896, dtype=torch.float32) * 0.02
    cur_len_t = torch.tensor([cur_len_val], dtype=torch.int32)

    print(f"[decode-stateful 2/4] Running PyTorch wrapper (warm-up)...")
    with torch.no_grad():
        speech_logits = wrapper(dummy_embed, cur_len_t)
    print(f"      speech_logits: {tuple(speech_logits.shape)}")

    # Restore state buffers for tracing (warm-up mutated them).
    _seed_from_prefill()

    print(f"[decode-stateful 3/4] Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_embed, cur_len_t), strict=False)

    print(f"[decode-stateful 4/4] Converting to CoreML mlpackage (stateful)...")
    precision = _make_precision(fp16)
    state_dtype = np.float16 if fp16 else np.float32
    states = []
    for i in range(L):
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(shape=(1, Hkv, M, D), dtype=state_dtype),
            name=f"kv_k_{i}",
        ))
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(shape=(1, Hkv, M, D), dtype=state_dtype),
            name=f"kv_v_{i}",
        ))
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="inputs_embeds", shape=(1, 1, 896), dtype=np.float32),
            ct.TensorType(name="cur_len", shape=(1,), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="speech_logits", dtype=np.float32),
        ],
        states=states,
        compute_precision=precision,
        minimum_deployment_target=min_dep,
        convert_to="mlprogram",
    )
    tag = "fp16" if fp16 else "fp32"
    mlp = out_dir / f"LLM-Decode-M{max_len}-{tag}-stateful.mlpackage"
    mlmodel.save(str(mlp))
    print(f"      saved: {mlp}")
    return mlp, (dummy_embed, cur_len_t, speech_logits)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--t-prefill", type=int, default=256,
                   help="Static prefill length (must be >= actual prompt length)")
    p.add_argument("--max-len", type=int, default=768,
                   help="Static KV cache length (>= t_prefill + max decode steps)")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--min-deployment", default="macOS14",
                   choices=["macOS13", "macOS14", "macOS15"])
    p.add_argument("--skip-prefill", action="store_true")
    p.add_argument("--skip-decode", action="store_true")
    p.add_argument("--stateful-decode", action="store_true",
                   help="Emit a stateful LLM-Decode mlpackage (macOS 15+, "
                        "~2-3× faster per-step vs pass-through KV).")
    args = p.parse_args()

    if args.t_prefill > args.max_len:
        raise SystemExit(f"--t-prefill ({args.t_prefill}) > --max-len ({args.max_len})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[0/2] Loading CosyVoice3 (CPU, fp32)...")
    qwen, speech_head, llm_model, cv = load_llm()
    n_params = sum(p.numel() for p in qwen.parameters())
    print(f"      Qwen2 params: {n_params:,}")

    print(f"      Building dummy lm_input (T_prefill={args.t_prefill})...")
    lm_input, input_len_val = build_dummy_lm_input(cv, llm_model, args.t_prefill)
    print(f"      lm_input shape: {tuple(lm_input.shape)}  input_len: {input_len_val}")

    ref = {"lm_input": lm_input, "input_len": torch.tensor([input_len_val], dtype=torch.int32)}

    if not args.skip_prefill:
        prefill_mlp, (lm_in, inp_len, last_hidden, speech_logits_pre, kv_k, kv_v) = \
            convert_prefill(qwen, speech_head, lm_input, input_len_val,
                             args.max_len, out_dir, args.fp16, _deployment(args.min_deployment))
        ref.update({
            "prefill_last_hidden": last_hidden,
            "prefill_speech_logits": speech_logits_pre,
            "prefill_kv_k": kv_k,
            "prefill_kv_v": kv_v,
        })
    else:
        print("[prefill] skipped — building KV via PyTorch for decode trace")
        pre_wrapper = Qwen2Prefill(qwen, speech_head, max_len=args.max_len,
                                    t_prefill=args.t_prefill).eval()
        with torch.no_grad():
            _, _, kv_k, kv_v = pre_wrapper(lm_input,
                                            torch.tensor([input_len_val], dtype=torch.int32))

    if not args.skip_decode:
        if args.stateful_decode:
            # Stateful path: prefill still outputs tensor KV (Swift seeds the
            # state on the first decode step). No kv_k_out / kv_v_out.
            if args.min_deployment != "macOS15":
                print(f"[decode-stateful] forcing min-deployment=macOS15 "
                      f"(was {args.min_deployment})")
            decode_mlp, (dummy_embed, cur_len_t, dec_logits) = \
                convert_decode_stateful(qwen, speech_head, args.max_len,
                                         kv_k, kv_v, input_len_val,
                                         out_dir, args.fp16,
                                         _deployment("macOS15"))
            ref.update({
                "decode_input_embed": dummy_embed,
                "decode_cur_len": cur_len_t,
                "decode_speech_logits": dec_logits,
                "decode_stateful": torch.tensor([1], dtype=torch.int32),
            })
        else:
            decode_mlp, (dummy_embed, cur_len_t, dec_logits, kv_k_out, kv_v_out) = \
                convert_decode(qwen, speech_head, args.max_len,
                                kv_k, kv_v, input_len_val,
                                out_dir, args.fp16, _deployment(args.min_deployment))
            ref.update({
                "decode_input_embed": dummy_embed,
                "decode_cur_len": cur_len_t,
                "decode_speech_logits": dec_logits,
                "decode_kv_k_out": kv_k_out,
                "decode_kv_v_out": kv_v_out,
            })

    tag = "fp16" if args.fp16 else "fp32"
    ref_pt = out_dir / f"ref-T{args.t_prefill}-M{args.max_len}-{tag}.pt"
    torch.save(ref, str(ref_pt))
    print(f"[done] ref: {ref_pt}")


def _make_precision(fp16: bool):
    """FP16 everywhere except RMSNorm ops (pow / reduce_mean / rsqrt) which
    need fp32 to avoid overflow on Qwen2's activation outliers.  Softmax is
    also kept in fp32 since fp16 softmax over long keys can underflow.
    """
    if not fp16:
        return ct.precision.FLOAT32
    FP32_OPS = {"pow", "reduce_mean", "rsqrt", "softmax"}
    return ct.transform.FP16ComputePrecision(
        op_selector=lambda op: op.op_type not in FP32_OPS
    )


def _deployment(name: str):
    return {
        "macOS13": ct.target.macOS13,
        "macOS14": ct.target.macOS14,
        "macOS15": ct.target.macOS15,
    }[name]


if __name__ == "__main__":
    main()
