"""Convert the NeuTTS-2E Qwen3 backbone to CoreML mlpackages.

Emits up to three models into --output-dir:

    LM-Prefill-T{t_prefill}-M{max_len}-{tag}.mlpackage      (macOS 14+)
    LM-Decode-M{max_len}-{tag}.mlpackage                     (macOS 14+, pass-through KV)
    LM-Decode-M{max_len}-{tag}-stateful.mlpackage            (macOS 15+, StateType KV)

Pipeline: load HF fp32 → wrap (src.lm_coreml) → PyTorch parity vs HF →
torch.jit.trace → ct.convert → CoreML parity vs the torch wrappers.

Usage:
    uv run python convert-lm.py --output-dir ./build/lm --fp16 --stateful-decode
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from src.lm_coreml import Qwen3Decode, Qwen3DecodeStateful, Qwen3Prefill  # noqa: E402

BACKBONE_REPO = "neuphonic/neutts-2e"


def load_backbone():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_REPO)
    model = AutoModelForCausalLM.from_pretrained(BACKBONE_REPO, dtype=torch.float32)
    model.eval()
    return model, tokenizer


def load_prompt_ids(tokenizer) -> list[int]:
    ref = HERE / "build" / "ref" / "prompt_ids.json"
    if ref.exists():
        return json.loads(ref.read_text())
    from src.prompt import build_prompt_ids

    return build_prompt_ids(tokenizer, "Hello there, this is a conversion smoke test.")


def _make_precision(fp16: bool):
    """FP16 everywhere except RMSNorm internals and softmax (fp32 for range)."""
    if not fp16:
        return ct.precision.FLOAT32
    FP32_OPS = {"pow", "reduce_mean", "rsqrt", "softmax"}
    return ct.transform.FP16ComputePrecision(op_selector=lambda op: op.op_type not in FP32_OPS)


def check(name: str, got: torch.Tensor, want: torch.Tensor, tol: float):
    diff = (got.float() - want.float()).abs().max().item()
    match = torch.argmax(got.reshape(-1)) == torch.argmax(want.reshape(-1))
    print(f"      {name}: max|Δ|={diff:.4e}  argmax match={bool(match)}")
    if diff > tol:
        raise SystemExit(f"parity failure: {name} max|Δ|={diff} > {tol}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--t-prefill", type=int, default=768)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--skip-prefill", action="store_true")
    p.add_argument("--skip-decode", action="store_true")
    p.add_argument("--stateful-decode", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "fp16" if args.fp16 else "fp32"
    precision = _make_precision(args.fp16)

    print(f"[0/3] Loading {BACKBONE_REPO} (fp32, cpu)...")
    hf_model, tokenizer = load_backbone()
    prompt_ids = load_prompt_ids(tokenizer)
    T_pre, M = args.t_prefill, args.max_len
    if len(prompt_ids) > T_pre:
        raise SystemExit(f"prompt ({len(prompt_ids)}) longer than --t-prefill ({T_pre})")
    n_valid = len(prompt_ids)
    print(f"      prompt: {n_valid} tokens, T_prefill={T_pre}, max_len={M}")

    # HF ground truth: last-position logits for the prompt, and one decode step.
    print("[0/3] HF reference forward...")
    with torch.no_grad():
        hf_out = hf_model(torch.tensor(prompt_ids)[None, :]).logits
        hf_last = hf_out[:, -1, :]  # [1, V]
        next_id = int(torch.argmax(hf_last, dim=-1))
        hf_out2 = hf_model(torch.tensor(prompt_ids + [next_id])[None, :]).logits
        hf_step = hf_out2[:, -1, :]

    padded = prompt_ids + [0] * (T_pre - n_valid)
    ids_t = torch.tensor(padded, dtype=torch.int32)[None, :]
    len_t = torch.tensor([n_valid], dtype=torch.int32)

    # ---------------- prefill ----------------
    print("[1/3] Prefill wrapper...")
    prefill = Qwen3Prefill(hf_model, max_len=M, t_prefill=T_pre).eval()
    with torch.no_grad():
        logits_last, kv_k, kv_v = prefill(ids_t, len_t)
    check("prefill logits vs HF", logits_last, hf_last, tol=2e-2)

    if not args.skip_prefill:
        with torch.no_grad():
            traced = torch.jit.trace(prefill, (ids_t, len_t), strict=False)
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="input_ids", shape=(1, T_pre), dtype=np.int32),
                ct.TensorType(name="input_len", shape=(1,), dtype=np.int32),
            ],
            outputs=[
                ct.TensorType(name="logits_last", dtype=np.float32),
                ct.TensorType(name="kv_k", dtype=np.float32),
                ct.TensorType(name="kv_v", dtype=np.float32),
            ],
            compute_precision=precision,
            minimum_deployment_target=ct.target.macOS14,
            convert_to="mlprogram",
        )
        mlp = out_dir / f"LM-Prefill-T{T_pre}-M{M}-{tag}.mlpackage"
        mlmodel.save(str(mlp))
        print(f"      saved: {mlp}")

        pred = mlmodel.predict(
            {"input_ids": ids_t.numpy().astype(np.int32), "input_len": len_t.numpy().astype(np.int32)}
        )
        check(
            "CoreML prefill logits",
            torch.from_numpy(pred["logits_last"]),
            logits_last,
            tol=1.0 if args.fp16 else 1e-2,
        )
    del prefill

    # ---------------- decode (pass-through KV) ----------------
    print("[2/3] Decode wrapper...")
    decode = Qwen3Decode(hf_model, max_len=M).eval()
    step_ids = torch.tensor([[next_id]], dtype=torch.int32)
    cur_len = torch.tensor([n_valid], dtype=torch.int32)
    with torch.no_grad():
        logits_step, kv_k2, kv_v2 = decode(step_ids, kv_k, kv_v, cur_len)
    check("decode logits vs HF", logits_step, hf_step, tol=2e-2)

    if not args.skip_decode:
        with torch.no_grad():
            traced = torch.jit.trace(decode, (step_ids, kv_k, kv_v, cur_len), strict=False)
        L, _, Hkv, _, D = kv_k.shape
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="input_ids", shape=(1, 1), dtype=np.int32),
                ct.TensorType(name="kv_k", shape=(L, 1, Hkv, M, D), dtype=np.float32),
                ct.TensorType(name="kv_v", shape=(L, 1, Hkv, M, D), dtype=np.float32),
                ct.TensorType(name="cur_len", shape=(1,), dtype=np.int32),
            ],
            outputs=[
                ct.TensorType(name="logits", dtype=np.float32),
                ct.TensorType(name="kv_k_out", dtype=np.float32),
                ct.TensorType(name="kv_v_out", dtype=np.float32),
            ],
            compute_precision=precision,
            minimum_deployment_target=ct.target.macOS14,
            convert_to="mlprogram",
        )
        mlp = out_dir / f"LM-Decode-M{M}-{tag}.mlpackage"
        mlmodel.save(str(mlp))
        print(f"      saved: {mlp}")

        pred = mlmodel.predict(
            {
                "input_ids": step_ids.numpy().astype(np.int32),
                "kv_k": kv_k.numpy().astype(np.float32),
                "kv_v": kv_v.numpy().astype(np.float32),
                "cur_len": cur_len.numpy().astype(np.int32),
            }
        )
        check(
            "CoreML decode logits",
            torch.from_numpy(pred["logits"]),
            logits_step,
            tol=1.0 if args.fp16 else 1e-2,
        )
    del decode

    # ---------------- decode (stateful) ----------------
    if args.stateful_decode:
        print("[3/3] Stateful decode wrapper (macOS 15+)...")
        stateful = Qwen3DecodeStateful(hf_model, max_len=M).eval()
        L, _, Hkv, _, D = kv_k.shape

        def seed_state():
            with torch.no_grad():
                for i in range(L):
                    getattr(stateful, f"kv_k_{i}").copy_(kv_k[i])
                    getattr(stateful, f"kv_v_{i}").copy_(kv_v[i])

        seed_state()
        with torch.no_grad():
            logits_sf = stateful(step_ids, cur_len)
        check("stateful vs pass-through", logits_sf, logits_step, tol=1e-4)

        seed_state()  # warm-up mutated the buffers; restore before tracing
        with torch.no_grad():
            traced = torch.jit.trace(stateful, (step_ids, cur_len), strict=False)

        state_dtype = np.float16 if args.fp16 else np.float32
        states = []
        for i in range(L):
            states.append(
                ct.StateType(
                    wrapped_type=ct.TensorType(shape=(1, Hkv, M, D), dtype=state_dtype),
                    name=f"kv_k_{i}",
                )
            )
            states.append(
                ct.StateType(
                    wrapped_type=ct.TensorType(shape=(1, Hkv, M, D), dtype=state_dtype),
                    name=f"kv_v_{i}",
                )
            )
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="input_ids", shape=(1, 1), dtype=np.int32),
                ct.TensorType(name="cur_len", shape=(1,), dtype=np.int32),
            ],
            outputs=[ct.TensorType(name="logits", dtype=np.float32)],
            states=states,
            compute_precision=precision,
            minimum_deployment_target=ct.target.macOS15,
            convert_to="mlprogram",
        )
        mlp = out_dir / f"LM-Decode-M{M}-{tag}-stateful.mlpackage"
        mlmodel.save(str(mlp))
        print(f"      saved: {mlp}")

        state = mlmodel.make_state()
        # Seed CoreML state from prefill KV, then run one step.
        # write_state only accepts float32 (converts to the fp16 state itself).
        for i in range(L):
            state.write_state(f"kv_k_{i}", kv_k[i].numpy().astype(np.float32))
            state.write_state(f"kv_v_{i}", kv_v[i].numpy().astype(np.float32))
        pred = mlmodel.predict(
            {
                "input_ids": step_ids.numpy().astype(np.int32),
                "cur_len": cur_len.numpy().astype(np.int32),
            },
            state=state,
        )
        check(
            "CoreML stateful logits",
            torch.from_numpy(pred["logits"]),
            logits_step,
            tol=1.0 if args.fp16 else 1e-2,
        )

    print("[done]")


if __name__ == "__main__":
    main()
