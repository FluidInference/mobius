"""Convert Chatterbox Nano T3 (GPT2-small) to CoreML: Prefill + Decode.

Steps:
  1. Load ChatterboxTurboTTS (nano=True) from the local HF snapshot (fp32, CPU).
  2. Build the batch-1 prefill embed via t3.prepare_input_embeds (built-in
     voice from conds.pt, real BPE-tokenized text) + BOS speech embed —
     replicating inference_turbo exactly (single BOS, no CFG).
  3. PyTorch parity: run the stock GPT2 path (t3.tfmr with past_key_values,
     exactly what inference_turbo does) for N greedy steps; run the
     T3NanoPrefill/T3NanoDecode wrappers on the same inputs and compare.
  4. torch.jit.trace + ct.convert both wrappers, re-check parity on CoreML.

Usage:
    uv run python convert-t3-nano.py --output-dir build/t3-nano [--fp16] [--steps 20]
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

from src.t3_nano_coreml import (  # noqa: E402
    T3NanoDecode, T3NanoDecodeStateful, T3NanoPrefill)

T_PREFILL = 512   # cond (1 spkr + 375 prompt) + text BPE + 1 BOS
MAX_LEN = 1536    # prefill bucket + ~1024 generated tokens (~40 s audio)
HIDDEN = 768
TEXT = "The quick brown fox jumps over the lazy dog near the river bank."


def load_model():
    from src.nano_ckpt import load_nano_t3

    return load_nano_t3()


def build_prefill_embeds(model):
    """Replicate tts_turbo.generate() up to prepare_input_embeds (batch 1)."""
    from chatterbox.tts_turbo import punc_norm

    t3 = model.t3
    text = punc_norm(TEXT)
    text_tokens = model.tokenizer(text, return_tensors="pt").input_ids
    print(f"      text tokens: {text_tokens.shape[1]} ids "
          f"(first/last: {text_tokens[0, 0].item()}/{text_tokens[0, -1].item()})")

    with torch.no_grad():
        embeds, len_cond = t3.prepare_input_embeds(
            t3_cond=model.conds.t3,
            text_tokens=text_tokens,
            speech_tokens=t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1]),
            cfg_weight=0.0,
        )
    return embeds, len_cond, text_tokens.shape[1]


def stock_reference(model, embeds, steps, greedy_tokens):
    """Run the stock GPT2 path (inference_turbo's exact calls), capture logits."""
    t3 = model.t3
    logits_seq = []
    with torch.no_grad():
        out = t3.tfmr(inputs_embeds=embeds, use_cache=True)
        past = out.past_key_values
        logits_seq.append(t3.speech_head(out[0][:, -1:])[:, 0].clone())
        for i in range(steps):
            emb = t3.speech_emb(torch.tensor([[greedy_tokens[i]]]))
            out = t3.tfmr(inputs_embeds=emb, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits_seq.append(t3.speech_head(out[0][:, -1:])[:, 0].clone())
    return logits_seq


def wrapper_reference(model, embeds, steps):
    t3 = model.t3
    prefill = T3NanoPrefill(t3.tfmr, t3.speech_head, MAX_LEN, T_PREFILL).eval()
    decode = T3NanoDecode(t3.tfmr, t3.speech_head, MAX_LEN).eval()

    T0 = embeds.shape[1]
    padded = torch.nn.functional.pad(embeds, (0, 0, 0, T_PREFILL - T0))
    input_len = torch.tensor([T0], dtype=torch.int32)

    logits_seq = []
    with torch.no_grad():
        logits, kv_k, kv_v = prefill(padded, input_len)
        logits_seq.append(logits)
        for i in range(steps):
            tok = int(torch.argmax(logits_seq[-1], dim=-1))
            emb = t3.speech_emb(torch.tensor([[tok]]))
            cur_len = torch.tensor([T0 + i], dtype=torch.int32)
            logits, kv_k, kv_v = decode(emb, kv_k, kv_v, cur_len)
            logits_seq.append(logits)
    return prefill, decode, padded, input_len, logits_seq


def compare(tag, ref_logits, wrap_logits):
    worst = 0.0
    for rl, wl in zip(ref_logits, wrap_logits):
        worst = max(worst, (rl - wl).abs().max().item())
    print(f"[{tag}] max |dlogits| = {worst:.3e}")
    return worst


def convert(prefill, decode, padded, input_len, kv_shape, out_dir: Path, fp16: bool):
    precision = ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32
    tag = "fp16" if fp16 else "fp32"
    target = ct.target.iOS17

    with torch.no_grad():
        traced_p = torch.jit.trace(prefill, (padded, input_len), strict=False)
    mlp = ct.convert(
        traced_p,
        inputs=[
            ct.TensorType(name="inputs_embeds", shape=(1, T_PREFILL, HIDDEN), dtype=np.float32),
            ct.TensorType(name="input_len", shape=(1,), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float32),
            ct.TensorType(name="kv_k", dtype=np.float32),
            ct.TensorType(name="kv_v", dtype=np.float32),
        ],
        compute_precision=precision,
        minimum_deployment_target=target,
        convert_to="mlprogram",
    )
    p_path = out_dir / f"T3Nano-Prefill-T{T_PREFILL}-M{MAX_LEN}-{tag}.mlpackage"
    mlp.save(str(p_path))
    print(f"saved {p_path}")

    kv_k = torch.zeros(kv_shape)
    kv_v = torch.zeros(kv_shape)
    emb1 = torch.zeros(1, 1, HIDDEN)
    cur_len = torch.tensor([64], dtype=torch.int32)
    with torch.no_grad():
        traced_d = torch.jit.trace(decode, (emb1, kv_k, kv_v, cur_len), strict=False)
    mld = ct.convert(
        traced_d,
        inputs=[
            ct.TensorType(name="inputs_embeds", shape=(1, 1, HIDDEN), dtype=np.float32),
            ct.TensorType(name="kv_k", shape=tuple(kv_shape), dtype=np.float32),
            ct.TensorType(name="kv_v", shape=tuple(kv_shape), dtype=np.float32),
            ct.TensorType(name="cur_len", shape=(1,), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float32),
            ct.TensorType(name="kv_k_out", dtype=np.float32),
            ct.TensorType(name="kv_v_out", dtype=np.float32),
        ],
        compute_precision=precision,
        minimum_deployment_target=target,
        convert_to="mlprogram",
    )
    d_path = out_dir / f"T3Nano-Decode-M{MAX_LEN}-{tag}.mlpackage"
    mld.save(str(d_path))
    print(f"saved {d_path}")
    return p_path, d_path


def coreml_parity(p_path, d_path, model, padded, input_len, steps,
                  ref_logits, T0, compute_units="CPU_AND_GPU"):
    t3 = model.t3
    cu = getattr(ct.ComputeUnit, compute_units)
    mp = ct.models.MLModel(str(p_path), compute_units=cu)
    md = ct.models.MLModel(str(d_path), compute_units=cu)

    out = mp.predict({"inputs_embeds": padded.numpy(), "input_len": input_len.numpy()})
    logits = torch.from_numpy(out["logits"])
    kv_k, kv_v = out["kv_k"], out["kv_v"]
    worst = (logits - ref_logits[0]).abs().max().item()

    for i in range(steps):
        tok = int(torch.argmax(logits, dim=-1))
        with torch.no_grad():
            emb = t3.speech_emb(torch.tensor([[tok]])).numpy()
        out = md.predict({
            "inputs_embeds": emb, "kv_k": kv_k, "kv_v": kv_v,
            "cur_len": np.array([T0 + i], dtype=np.int32)})
        kv_k, kv_v = out["kv_k_out"], out["kv_v_out"]
        logits = torch.from_numpy(out["logits"])
        worst = max(worst, (logits - ref_logits[i + 1]).abs().max().item())
    print(f"[coreml] max |dlogits| = {worst:.3e}")


def convert_stateful(model, padded, input_len, T0, steps, ref_logits,
                     out_dir: Path, fp16: bool):
    """Convert T3NanoDecodeStateful (macOS 15+/iOS 18+ MLState KV).

    CoreML validation feeds the whole prefix through the stateful decode from
    an empty state (positions 0..T0-1), then greedy-decodes — Swift seeds
    state from prefill KV at runtime.
    """
    t3 = model.t3
    wrapper = T3NanoDecodeStateful(t3.tfmr, t3.speech_head, MAX_LEN).eval()

    emb1 = torch.zeros(1, 1, HIDDEN)
    cur_len = torch.tensor([0], dtype=torch.int32)
    with torch.no_grad():
        wrapper(emb1, cur_len)  # warm-up
    for i in range(wrapper.L):  # reset mutated state before trace
        getattr(wrapper, f"kv_k_{i}").zero_()
        getattr(wrapper, f"kv_v_{i}").zero_()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (emb1, cur_len), strict=False)

    precision = ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32
    state_dtype = np.float16 if fp16 else np.float32
    states = []
    for i in range(wrapper.L):
        for kv in ("k", "v"):
            states.append(ct.StateType(
                wrapped_type=ct.TensorType(shape=(1, wrapper.H, MAX_LEN, wrapper.D),
                                           dtype=state_dtype),
                name=f"kv_{kv}_{i}"))
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="inputs_embeds", shape=(1, 1, HIDDEN), dtype=np.float32),
            ct.TensorType(name="cur_len", shape=(1,), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        states=states,
        compute_precision=precision,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    tag = "fp16" if fp16 else "fp32"
    s_path = out_dir / f"T3Nano-Decode-M{MAX_LEN}-{tag}-stateful.mlpackage"
    mlmodel.save(str(s_path))
    print(f"saved {s_path}")

    import time
    md = ct.models.MLModel(str(s_path), compute_units=ct.ComputeUnit.CPU_AND_GPU)
    state = md.make_state()
    logits = None
    t_ctx = 0.0
    for pos in range(T0):
        t0 = time.perf_counter()
        out = md.predict({"inputs_embeds": padded[:, pos:pos + 1].numpy(),
                          "cur_len": np.array([pos], dtype=np.int32)}, state)
        t_ctx += time.perf_counter() - t0
        logits = torch.from_numpy(out["logits"])
    worst = (logits - ref_logits[0]).abs().max().item()

    step_times = []
    for i in range(steps):
        tok = int(torch.argmax(logits, dim=-1))
        with torch.no_grad():
            emb = t3.speech_emb(torch.tensor([[tok]])).numpy()
        t0 = time.perf_counter()
        out = md.predict({"inputs_embeds": emb,
                          "cur_len": np.array([T0 + i], dtype=np.int32)}, state)
        step_times.append(time.perf_counter() - t0)
        logits = torch.from_numpy(out["logits"])
        worst = max(worst, (logits - ref_logits[i + 1]).abs().max().item())
    med = sorted(step_times)[len(step_times) // 2]
    print(f"[coreml-stateful] max |dlogits| = {worst:.3e}")
    print(f"[coreml-stateful] median decode step = {med * 1000:.1f} ms "
          f"(ctx feed {t_ctx / T0 * 1000:.1f} ms/pos)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("build/t3-nano"))
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--skip-convert", action="store_true")
    ap.add_argument("--parity-only", action="store_true",
                    help="reuse existing mlpackages; skip trace/convert")
    ap.add_argument("--stateful", action="store_true",
                    help="convert + validate the MLState decode variant only")
    ap.add_argument("--compute-units", default="CPU_AND_GPU",
                    choices=["ALL", "CPU_ONLY", "CPU_AND_NE", "CPU_AND_GPU"])
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    print("[1/4] loading model...")
    model = load_model()
    embeds, len_cond, text_len = build_prefill_embeds(model)
    T0 = embeds.shape[1]
    print(f"      embeds {tuple(embeds.shape)}  len_cond={len_cond} text_len={text_len}")
    assert T0 <= T_PREFILL, f"prefill bucket too small: {T0} > {T_PREFILL}"

    print("[2/4] wrapper forward...")
    prefill, decode, padded, input_len, wrap_logits = \
        wrapper_reference(model, embeds, args.steps)

    print("[3/4] stock reference...")
    greedy = [int(torch.argmax(l, dim=-1)) for l in wrap_logits[:-1]]
    stock_logits = stock_reference(model, embeds, args.steps, greedy)
    compare("pytorch", stock_logits, wrap_logits)

    if args.skip_convert:
        return
    if args.stateful:
        convert_stateful(model, padded, input_len, T0, args.steps,
                         wrap_logits, args.output_dir, args.fp16)
        return
    kv_shape = (12, 1, 12, MAX_LEN, 64)
    tag = "fp16" if args.fp16 else "fp32"
    if args.parity_only:
        p_path = args.output_dir / f"T3Nano-Prefill-T{T_PREFILL}-M{MAX_LEN}-{tag}.mlpackage"
        d_path = args.output_dir / f"T3Nano-Decode-M{MAX_LEN}-{tag}.mlpackage"
    else:
        print("[4/4] converting...")
        p_path, d_path = convert(prefill, decode, padded, input_len, kv_shape,
                                 args.output_dir, args.fp16)
    coreml_parity(p_path, d_path, model, padded, input_len, args.steps,
                  wrap_logits, T0, compute_units=args.compute_units)


if __name__ == "__main__":
    main()
