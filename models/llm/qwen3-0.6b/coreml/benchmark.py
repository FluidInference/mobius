"""On-device decode benchmark for the Qwen3-0.6B CoreML decoder.

Measures prefill latency and steady-state decode tokens/s, and derives an RTFx-style
real-time factor: decode throughput divided by a real-time consumption rate.

RTFx here = (decode tokens/s) / RT_TOKENS_PER_SEC, where RT_TOKENS_PER_SEC is the rate a
downstream consumer drains tokens. Default 15 tok/s ≈ brisk TTS narration (~3-4 tokens per
spoken word at ~4 words/s). RTFx > 1 means generation outpaces consumption — real-time.

Host owns embeddings + RoPE (they live outside the decoder graph, matching the runtime
split); the CoreML model owns the 28 transformer layers + lm_head with a stateful KV cache.

Usage:
    uv run benchmark.py --model-dir ./build --compute-units CPU_AND_NE
    uv run benchmark.py --model-dir ./build --compute-units CPU_AND_GPU --gen-tokens 128
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

HEAD_DIM = 128
ROPE_THETA = 1_000_000.0
RT_TOKENS_PER_SEC = 15.0  # real-time consumption bar (see module docstring)


def rope_tables(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cos/sin for given absolute positions, concatenated-halves layout. [N, HEAD_DIM]."""
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float64) / HEAD_DIM))
    freqs = np.outer(positions.astype(np.float64), inv_freq)  # [N, HEAD_DIM/2]
    emb = np.concatenate([freqs, freqs], axis=-1)  # [N, HEAD_DIM]
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="./build")
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--compute-units", default="CPU_AND_NE",
                        choices=["ALL", "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"])
    parser.add_argument("--prompt", default="The key idea behind on-device inference is")
    parser.add_argument("--gen-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=8)
    args = parser.parse_args()

    import coremltools as ct
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mlpkg = Path(args.model_dir) / "qwen3_0_6b_decoder_stateful.mlpackage"
    print(f"Loading {mlpkg} on {args.compute_units} ...")
    cu = getattr(ct.ComputeUnit, args.compute_units)
    t0 = time.time()
    mlmodel = ct.models.MLModel(str(mlpkg), compute_units=cu)
    print(f"  loaded/compiled in {time.time() - t0:.1f}s")

    tok = AutoTokenizer.from_pretrained(args.model_id)
    hf = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32)
    embed = hf.model.embed_tokens.weight.detach().numpy()  # [vocab, hidden]

    ids = tok(args.prompt, return_tensors="np")["input_ids"][0].astype(np.int64)
    S = len(ids)
    print(f"Prompt tokens: {S}")

    def embed_ids(token_ids: np.ndarray) -> np.ndarray:
        return embed[token_ids][None, :, :].astype(np.float32)  # [1, N, hidden]

    NEG = -1e4

    def run_prefill(state):
        h = embed_ids(ids)  # [1, S, hidden]
        cos, sin = rope_tables(np.arange(S))
        mask = np.triu(np.full((1, 1, S, S), NEG, dtype=np.float32), k=1)
        out = mlmodel.predict({
            "hidden_states": h,
            "position_cos": cos[None],
            "position_sin": sin[None],
            "attention_mask": mask,
        }, state=state)
        return out["logits"][0, -1]  # [vocab]

    def run_decode_step(state, token_id: int, pos: int):
        h = embed_ids(np.array([token_id]))  # [1,1,hidden]
        cos, sin = rope_tables(np.array([pos]))
        mask = np.zeros((1, 1, 1, pos + 1), dtype=np.float32)  # attend all past + self
        out = mlmodel.predict({
            "hidden_states": h,
            "position_cos": cos[None],
            "position_sin": sin[None],
            "attention_mask": mask,
        }, state=state)
        return out["logits"][0, -1]

    # ---- Warmup (compile caches, ANE spin-up) ----
    for _ in range(2):
        st = mlmodel.make_state()
        logits = run_prefill(st)
        nxt = int(logits.argmax())
        pos = S
        for _ in range(args.warmup):
            logits = run_decode_step(st, nxt, pos)
            nxt = int(logits.argmax())
            pos += 1

    # ---- Timed: prefill ----
    st = mlmodel.make_state()
    t0 = time.time()
    logits = run_prefill(st)
    prefill_ms = (time.time() - t0) * 1000.0
    nxt = int(logits.argmax())
    pos = S

    # ---- Timed: decode ----
    decode_times = []
    generated = [nxt]
    for _ in range(args.gen_tokens):
        t0 = time.time()
        logits = run_decode_step(st, nxt, pos)
        decode_times.append(time.time() - t0)
        nxt = int(logits.argmax())
        generated.append(nxt)
        pos += 1

    dt = np.array(decode_times)
    tok_per_s = 1.0 / dt.mean()
    p50 = np.percentile(dt, 50) * 1000
    p99 = np.percentile(dt, 99) * 1000
    rtfx = tok_per_s / RT_TOKENS_PER_SEC

    print("\n=== Results ===")
    print(f"compute_units       : {args.compute_units}")
    print(f"prefill ({S} tok)    : {prefill_ms:.1f} ms  ({S / (prefill_ms/1000):.0f} tok/s)")
    print(f"decode p50 / p99    : {p50:.2f} / {p99:.2f} ms per token")
    print(f"decode throughput   : {tok_per_s:.1f} tok/s")
    print(f"RTFx (vs {RT_TOKENS_PER_SEC:.0f} tok/s) : {rtfx:.2f}x  {'PASS >1' if rtfx > 1 else 'FAIL <1'}")
    print(f"\nsample: {tok.decode(generated[:24])!r}")


if __name__ == "__main__":
    main()
