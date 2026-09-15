"""Convert the MOSS-TTS-Nano language model to CoreML.

Emits into --output-dir:
    MossNano-Prefill-T{t}-M{m}-{tag}.mlpackage   rows [1,T,17] + len → hidden [1,768], kv_k/kv_v [12,1,12,M,64]
    MossNano-Step-M{m}-{tag}.mlpackage           row [1,1,17] + kv + cur_len → hidden, kv out
    MossNano-Frame-{tag}.mlpackage               hidden + sampling params → should_continue, frame [1,16]

Every wrapper is checked against the upstream PyTorch model before tracing, and every
CoreML model is checked against its wrapper after conversion (greedy tokens must match).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.lm_coreml import MossFrame, MossPrefill, MossStep  # noqa: E402

TTS_REPO = "OpenMOSS-Team/MOSS-TTS-Nano-100M"
TEXT = "The quick brown fox jumps over the lazy dog near the riverbank."


def load_model():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(TTS_REPO, trust_remote_code=True, dtype=torch.float32)
    for module in model.modules():
        if hasattr(module, "attn_implementation"):
            module.attn_implementation = "eager"
    model.eval()
    tokenizer = model._load_text_tokenizer(text_tokenizer=None, text_tokenizer_path=None)
    return model, tokenizer


def precision(fp16: bool):
    if not fp16:
        return ct.precision.FLOAT32
    fp32_ops = {"softmax", "cumsum"}
    return ct.transform.FP16ComputePrecision(op_selector=lambda op: op.op_type not in fp32_ops)


def report(name: str, got: torch.Tensor, want: torch.Tensor, tol: float) -> None:
    diff = (got.float() - want.float()).abs().max().item()
    ok = "ok" if diff <= tol else "FAIL"
    print(f"      {name}: max|Δ|={diff:.3e} (tol {tol:g}) {ok}")
    if diff > tol:
        raise SystemExit(f"parity failure: {name}")


def frame_inputs(hidden: torch.Tensor, greedy: bool, seen=None):
    return {
        "global_hidden": hidden.float(),
        "text_u": torch.tensor([0.37]),
        "audio_u": torch.linspace(0.05, 0.95, 16)[None, :],
        "text_temperature": torch.tensor([1.5]),
        "audio_temperature": torch.tensor([1.7]),
        "audio_top_p": torch.tensor([0.8]),
        "repetition_penalty": torch.tensor([1.0]),
        "seen": torch.zeros(1, 16, 1024) if seen is None else seen,
        "greedy": torch.tensor([1.0 if greedy else 0.0]),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=str(HERE / "build" / "lm"))
    p.add_argument("--t-prefill", type=int, default=512)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--skip-prefill", action="store_true")
    p.add_argument("--skip-step", action="store_true")
    p.add_argument("--skip-frame", action="store_true")
    p.add_argument("--ref-tokens", default=str(HERE / "build" / "ref_greedy_audio_token_ids.npy"))
    p.add_argument("--ref-prompt-tokens", default=str(HERE / "build" / "ref_greedy_prompt_audio_token_ids.npy"))
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "fp16" if args.fp16 else "fp32"
    T_pre, M = args.t_prefill, args.max_len
    target = ct.target.macOS14

    print(f"[0] loading {TTS_REPO}")
    model, tokenizer = load_model()
    cfg = model.config
    prompt_codes = torch.from_numpy(np.load(args.ref_prompt_tokens)).long()  # [Tp,16]
    ref_frames = torch.from_numpy(np.load(args.ref_tokens)).long()  # [Tg,16]
    input_ids, attn = model.build_inference_input_ids(
        text=TEXT, text_tokenizer=tokenizer, mode="voice_clone", prompt_audio_codes=prompt_codes, device="cpu"
    )
    n_prompt = input_ids.shape[1]
    print(f"      prompt rows={n_prompt}  ref frames={ref_frames.shape[0]}  T_prefill={T_pre}  M={M}")
    if n_prompt > T_pre:
        raise SystemExit("prompt longer than --t-prefill")

    # Upstream ground truth: hidden after prompt, and after prompt + first generated row.
    row0 = model._build_generation_row(1, torch.device("cpu"), ref_frames[0][None])  # [1,1,17]
    with torch.no_grad():
        hf_h0 = model(input_ids=input_ids, attention_mask=attn, use_cache=False).global_hidden_states[:, -1]
        seq1 = torch.cat((input_ids, row0), dim=1)
        hf_h1 = model(input_ids=seq1, use_cache=False).global_hidden_states[:, -1]

    padded = torch.full((1, T_pre, cfg.n_vq + 1), cfg.audio_pad_token_id, dtype=torch.int32)
    padded[:, :, 0] = cfg.pad_token_id
    padded[:, :n_prompt] = input_ids.to(torch.int32)
    len_t = torch.tensor([n_prompt], dtype=torch.int32)

    # ---------------- prefill ----------------
    print("[1] prefill wrapper")
    prefill = MossPrefill(model, max_len=M).eval()
    with torch.no_grad():
        h0, kv_k, kv_v = prefill(padded, len_t)
    report("wrapper hidden vs upstream", h0, hf_h0, 1e-3)
    if not args.skip_prefill:
        t0 = time.perf_counter()
        with torch.no_grad():
            traced = torch.jit.trace(prefill, (padded, len_t), strict=False)
        ml = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="input_ids", shape=(1, T_pre, cfg.n_vq + 1), dtype=np.int32),
                ct.TensorType(name="input_len", shape=(1,), dtype=np.int32),
            ],
            outputs=[
                ct.TensorType(name="hidden", dtype=np.float32),
                ct.TensorType(name="kv_k", dtype=np.float32),
                ct.TensorType(name="kv_v", dtype=np.float32),
            ],
            compute_precision=precision(args.fp16),
            minimum_deployment_target=target,
            convert_to="mlprogram",
        )
        path = out_dir / f"MossNano-Prefill-T{T_pre}-M{M}-{tag}.mlpackage"
        ml.save(str(path))
        print(f"      saved {path.name} ({time.perf_counter() - t0:.0f}s)")
        pred = ml.predict({"input_ids": padded.numpy(), "input_len": len_t.numpy()})
        report("coreml prefill hidden", torch.from_numpy(pred["hidden"]), h0, 0.15 if args.fp16 else 1e-2)
        report("coreml prefill kv_k", torch.from_numpy(pred["kv_k"]), kv_k, 0.25 if args.fp16 else 1e-2)
        del ml
    del prefill

    # ---------------- step ----------------
    print("[2] step wrapper")
    step = MossStep(model, max_len=M).eval()
    cur_len = torch.tensor([n_prompt], dtype=torch.int32)
    with torch.no_grad():
        h1, kv_k1, kv_v1 = step(row0.to(torch.int32), kv_k, kv_v, cur_len)
    report("wrapper hidden vs upstream", h1, hf_h1, 1e-3)
    if not args.skip_step:
        t0 = time.perf_counter()
        with torch.no_grad():
            traced = torch.jit.trace(step, (row0.to(torch.int32), kv_k, kv_v, cur_len), strict=False)
        L, _, H, _, Dh = kv_k.shape
        ml = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="input_ids", shape=(1, 1, cfg.n_vq + 1), dtype=np.int32),
                ct.TensorType(name="kv_k", shape=(L, 1, H, M, Dh), dtype=np.float32),
                ct.TensorType(name="kv_v", shape=(L, 1, H, M, Dh), dtype=np.float32),
                ct.TensorType(name="cur_len", shape=(1,), dtype=np.int32),
            ],
            outputs=[
                ct.TensorType(name="hidden", dtype=np.float32),
                ct.TensorType(name="kv_k_out", dtype=np.float32),
                ct.TensorType(name="kv_v_out", dtype=np.float32),
            ],
            compute_precision=precision(args.fp16),
            minimum_deployment_target=target,
            convert_to="mlprogram",
        )
        path = out_dir / f"MossNano-Step-M{M}-{tag}.mlpackage"
        ml.save(str(path))
        print(f"      saved {path.name} ({time.perf_counter() - t0:.0f}s)")
        pred = ml.predict(
            {
                "input_ids": row0.numpy().astype(np.int32),
                "kv_k": kv_k.numpy(),
                "kv_v": kv_v.numpy(),
                "cur_len": cur_len.numpy(),
            }
        )
        report("coreml step hidden", torch.from_numpy(pred["hidden"]), h1, 0.15 if args.fp16 else 1e-2)
        del ml
    del step

    # ---------------- frame ----------------
    print("[3] frame wrapper")
    frame = MossFrame(model).eval()
    with torch.no_grad():
        cont, toks = frame(**frame_inputs(hf_h0, greedy=True))
    match = int((toks[0].long() == ref_frames[0]).sum())
    print(f"      greedy frame0 vs upstream: continue={int(cont)}  {match}/16 tokens match")
    if match != 16 or int(cont) != 1:
        raise SystemExit("frame wrapper greedy mismatch")
    with torch.no_grad():
        cont_s, toks_s = frame(**frame_inputs(hf_h0, greedy=False))
    print(f"      sampled frame0: continue={int(cont_s)} tokens={toks_s[0].tolist()}")
    if not args.skip_frame:
        t0 = time.perf_counter()
        ex = frame_inputs(hf_h0, greedy=True)
        with torch.no_grad():
            traced = torch.jit.trace(frame, tuple(ex.values()), strict=False)
        ml = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="global_hidden", shape=(1, cfg.hidden_size), dtype=np.float32),
                ct.TensorType(name="text_u", shape=(1,), dtype=np.float32),
                ct.TensorType(name="audio_u", shape=(1, cfg.n_vq), dtype=np.float32),
                ct.TensorType(name="text_temperature", shape=(1,), dtype=np.float32),
                ct.TensorType(name="audio_temperature", shape=(1,), dtype=np.float32),
                ct.TensorType(name="audio_top_p", shape=(1,), dtype=np.float32),
                ct.TensorType(name="repetition_penalty", shape=(1,), dtype=np.float32),
                ct.TensorType(name="seen", shape=(1, cfg.n_vq, cfg.audio_vocab_size), dtype=np.float32),
                ct.TensorType(name="greedy", shape=(1,), dtype=np.float32),
            ],
            outputs=[
                ct.TensorType(name="should_continue", dtype=np.int32),
                ct.TensorType(name="frame", dtype=np.int32),
            ],
            compute_precision=precision(args.fp16),
            minimum_deployment_target=target,
            convert_to="mlprogram",
        )
        path = out_dir / f"MossNano-Frame-{tag}.mlpackage"
        ml.save(str(path))
        print(f"      saved {path.name} ({time.perf_counter() - t0:.0f}s)")
        feed = {k: v.numpy().astype(np.float32) for k, v in ex.items()}
        pred = ml.predict(feed)
        got = torch.from_numpy(pred["frame"]).long()[0]
        m = int((got == ref_frames[0]).sum())
        print(f"      coreml greedy frame0: continue={int(pred['should_continue'].reshape(-1)[0])}  {m}/16 match")
        if m != 16:
            raise SystemExit("coreml frame greedy mismatch")
        feed["greedy"] = np.zeros(1, np.float32)
        pred = ml.predict(feed)
        print(f"      coreml sampled frame0: {pred['frame'].reshape(-1).tolist()}  (wrapper: {toks_s[0].tolist()})")
    print("[done]")


if __name__ == "__main__":
    main()
