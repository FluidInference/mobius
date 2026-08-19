"""Profile where decode time goes in the Riva-4B CoreML pipeline.

Hypotheses tested:
  A. growing mask shape per step (real decode) -> per-step shape re-specialization
  B. constant mask shape per step -> steady-state kernel reuse
  C. lm_head call cost
  D. embedding lookup cost (host-side numpy)

Usage:
    uv run profile_decode.py --model-dir ./out --suffix _int4
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
STEPS = 24


def bench(fn, n=STEPS, warmup=2):
    for _ in range(warmup):
        fn(0)
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        fn(i)
        times.append(time.perf_counter() - t0)
    t = np.array(times) * 1000
    return f"mean {t.mean():6.1f}ms  p50 {np.percentile(t,50):6.1f}ms  min {t.min():6.1f}ms  max {t.max():6.1f}ms"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="./out")
    parser.add_argument("--suffix", default="_int4")
    parser.add_argument("--compute-units", default="CPU_AND_GPU")
    args = parser.parse_args()

    import coremltools as ct

    model_dir = Path(args.model_dir)
    cu = getattr(ct.ComputeUnit, args.compute_units)
    print(f"Loading (suffix={args.suffix!r}, cu={args.compute_units})...")
    t0 = time.time()
    decoder = ct.models.MLModel(
        str(model_dir / f"riva4b_decoder_stateful{args.suffix}.mlpackage"), compute_units=cu
    )
    head_path = model_dir / f"riva4b_lm_head{args.suffix}.mlpackage"
    lm_head = ct.models.MLModel(str(head_path), compute_units=cu) if head_path.exists() else None
    embed = np.load(model_dir / "embed_tokens_fp16.npy", mmap_mode="r")
    print(f"Loaded in {time.time() - t0:.1f}s\n")

    hidden1 = np.random.randn(1, 1, HIDDEN_SIZE).astype(np.float16)
    cos1 = np.random.randn(1, 1, HEAD_DIM).astype(np.float16)
    sin1 = np.random.randn(1, 1, HEAD_DIM).astype(np.float16)

    state = decoder.make_state()

    def dec_step(end_step):
        mask = np.zeros((1, 1, 1, end_step), dtype=np.float16)
        return decoder.predict(
            {"hidden_states": hidden1, "position_cos": cos1, "position_sin": sin1,
             "attention_mask": mask},
            state=state,
        )

    # A: growing shape (real decode pattern), starting at end_step=40
    print("A. decoder, GROWING end_step (40..):   ", bench(lambda i: dec_step(40 + i)))

    # B: constant shapes
    for es in (64, 512, 1024):
        print(f"B. decoder, CONSTANT end_step={es:4d}:    ", bench(lambda i, es=es: dec_step(es)))

    # C: lm_head alone
    if lm_head is not None:
        def head_step(i):
            return lm_head.predict({"hidden_states": hidden1})
        print("C. lm_head alone:                      ", bench(head_step))

    # D: embedding lookup alone
    def emb_step(i):
        return embed[[i % 1000]][None].astype(np.float16)
    print("D. embed lookup (numpy mmap):          ", bench(emb_step))

    # E: interleaved decoder + head (real decode pattern with two models)
    if lm_head is not None:
        def interleaved(i):
            out = dec_step(64)
            h = out["output_hidden"].astype(np.float16).reshape(1, 1, HIDDEN_SIZE)
            return lm_head.predict({"hidden_states": h})
        print("E. interleaved decoder+head:           ", bench(interleaved))

    # F: decoder loop with deliberate host-side gap between predicts.
    # Quantifies the GPU idle/power-state penalty as a function of gap length.
    for gap_ms in (0.5, 2, 5, 10):
        def gapped(i, g=gap_ms):
            t_end = time.perf_counter() + g / 1000.0
            while time.perf_counter() < t_end:  # busy-wait, no sleep syscall
                pass
            return dec_step(64)
        t = bench(gapped)
        print(f"F. decoder w/ {gap_ms:4.1f}ms host gap:        ", t)


if __name__ == "__main__":
    main()
