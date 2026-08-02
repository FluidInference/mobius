"""Run the converted Riva-4B CoreML pipeline and compare against reference.npz.

Pipeline: host-side embedding lookup (numpy) -> stateful decoder (prefill with
Q=prompt_len, then Q=1 decode steps) -> lm_head -> greedy argmax.

Also reports prefill latency and per-token decode latency.

Usage:
    uv run run_coreml.py --model-dir ./out --reference reference.npz
"""

# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "coremltools>=8.0",
#     "numpy<2",
# ]
# ///

import argparse
import time
from pathlib import Path

import numpy as np

HIDDEN_SIZE = 3072
HEAD_DIM = 128
ROPE_THETA = 1_000_000.0


def rope_tables(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cos/sin tables in concatenated-halves layout, [1, len(positions), 128]."""
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float64) / HEAD_DIM))
    freqs = np.outer(positions.astype(np.float64), inv_freq)  # [Q, 64]
    emb = np.concatenate([freqs, freqs], axis=-1)  # [Q, 128]
    return (
        np.cos(emb)[None].astype(np.float16),
        np.sin(emb)[None].astype(np.float16),
    )


def causal_mask(q_len: int, end_step: int) -> np.ndarray:
    past = end_step - q_len
    mask = np.zeros((1, 1, q_len, end_step), dtype=np.float16)
    for r in range(q_len):
        mask[0, 0, r, past + r + 1 :] = -30_000.0
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=".")
    parser.add_argument("--reference", default="reference.npz")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--compute-units", default="CPU_AND_GPU",
                        choices=["ALL", "CPU_AND_GPU", "CPU_ONLY", "CPU_AND_NE"])
    parser.add_argument("--suffix", default="", help="Model filename suffix, e.g. _int4")
    args = parser.parse_args()

    import coremltools as ct

    model_dir = Path(args.model_dir)
    ref = np.load(args.reference)
    prompt_ids = ref["prompt_ids"]
    ref_gen = ref["gen_ids"]
    eos_id = int(ref["eos_token_id"])

    cu = getattr(ct.ComputeUnit, args.compute_units)
    print(f"Loading models (compute_units={args.compute_units})...")
    t0 = time.time()
    decoder = ct.models.MLModel(
        str(model_dir / f"riva4b_decoder_stateful{args.suffix}.mlpackage"), compute_units=cu
    )
    lm_head = ct.models.MLModel(str(model_dir / f"riva4b_lm_head{args.suffix}.mlpackage"), compute_units=cu)
    embed = np.load(model_dir / "embed_tokens_fp16.npy", mmap_mode="r")
    print(f"Loaded in {time.time() - t0:.1f}s")

    state = decoder.make_state()

    def run_decoder(token_ids: np.ndarray, past_len: int) -> np.ndarray:
        q = len(token_ids)
        positions = np.arange(past_len, past_len + q)
        cos, sin = rope_tables(positions)
        hidden = embed[token_ids][None].astype(np.float16)  # [1, Q, 3072]
        out = decoder.predict(
            {
                "hidden_states": hidden,
                "position_cos": cos,
                "position_sin": sin,
                "attention_mask": causal_mask(q, past_len + q),
            },
            state=state,
        )
        return out["output_hidden"]  # [1, Q, 3072]

    def logits_for(hidden_last: np.ndarray) -> np.ndarray:
        out = lm_head.predict({"hidden_states": hidden_last.reshape(1, 1, HIDDEN_SIZE)})
        return out["logits"].reshape(-1)

    # ---- Prefill ----
    t0 = time.time()
    hidden = run_decoder(prompt_ids, past_len=0)
    prefill_s = time.time() - t0
    print(f"Prefill: {len(prompt_ids)} tokens in {prefill_s:.2f}s ({len(prompt_ids)/prefill_s:.1f} tok/s)")

    logits = logits_for(hidden[0, -1])

    # First-step logits parity
    ref_logits = ref["first_logits"]
    top1_ref = int(np.argmax(ref_logits))
    top1_cml = int(np.argmax(logits))
    corr = np.corrcoef(ref_logits.astype(np.float64), logits.astype(np.float64))[0, 1]
    print(f"First-step logits: corr={corr:.6f}, argmax ref={top1_ref} coreml={top1_cml} "
          f"{'MATCH' if top1_ref == top1_cml else 'MISMATCH'}")

    # ---- Greedy decode ----
    gen = []
    past = len(prompt_ids)
    tok = top1_cml
    decode_times = []
    while len(gen) < args.max_new_tokens:
        gen.append(tok)
        if tok == eos_id:
            break
        t0 = time.time()
        hidden = run_decoder(np.array([tok]), past_len=past)
        logits = logits_for(hidden[0, -1])
        decode_times.append(time.time() - t0)
        tok = int(np.argmax(logits))
        past += 1

    dt = np.array(decode_times)
    print(f"Decode: {len(dt)} steps, mean {dt.mean()*1000:.1f}ms/tok ({1.0/dt.mean():.1f} tok/s), "
          f"p50 {np.percentile(dt,50)*1000:.1f}ms")

    gen = np.array(gen)
    n = min(len(gen), len(ref_gen))
    match = int((gen[:n] == ref_gen[:n]).sum())
    print(f"\nToken parity vs reference: {match}/{n} match")
    print(f"  ref:    {ref_gen[:n].tolist()}")
    print(f"  coreml: {gen[:n].tolist()}")


if __name__ == "__main__":
    main()
