"""Benchmark the CoreML LuxTTS pipeline vs the PyTorch reference.

Times each component (text encoder, per-step fm decoder, torch vocoder) and
reports end-to-end synthesis latency + RTFx for the oracle utterance, across
compute units. Uses the same oracle inputs as coreml/parity.py.

Usage:
    .venv/bin/python -m coreml.benchmark --compute-units ALL --runs 10
"""

import argparse
import json
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from coreml.convert_coreml import MAX_FRAMES, MAX_TOKENS, load_model
from zipvoice.models.modules.solver import get_time_steps


def timeit(fn, runs, warmup=2):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return ts.mean(), ts.std(), ts.min()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-dir", default="build/oracle")
    parser.add_argument("--coreml-dir", default="build/coreml")
    parser.add_argument("--compute-units", default="ALL", choices=["ALL", "CPU_ONLY", "CPU_AND_NE", "CPU_AND_GPU"])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--torch-baseline", action="store_true", help="also time the pure-torch pipeline")
    args = parser.parse_args()

    oracle = Path(args.oracle_dir)
    meta = json.loads((oracle / "meta.json").read_text())
    prompt_features = torch.from_numpy(np.load(oracle / "prompt_features.npy"))
    prompt_tokens = np.load(oracle / "prompt_tokens.npy").tolist()
    text_tokens = np.load(oracle / "text_tokens.npy").tolist()
    prompt_len = meta["prompt_features_lens"]
    num_steps = meta["num_steps"]
    gen_seconds = meta["wav_seconds_48k"]

    cu = getattr(ct.ComputeUnit, args.compute_units)
    t0 = time.perf_counter()
    te = ct.models.MLModel(str(Path(args.coreml_dir) / "TextEncoder.mlpackage"), compute_units=cu)
    fm = ct.models.MLModel(str(Path(args.coreml_dir) / "FmDecoder.mlpackage"), compute_units=cu)
    load_ms = (time.perf_counter() - t0) * 1000

    cat = prompt_tokens + text_tokens
    S = len(cat)
    tok_in = np.zeros((1, MAX_TOKENS), dtype=np.int32)
    tok_in[0, :S] = cat
    tmask = np.zeros((1, MAX_TOKENS), dtype=np.float32)
    tmask[0, S:] = 1.0

    speed = 1.3
    features_len = prompt_len + int(np.ceil(prompt_len / len(prompt_tokens) * len(text_tokens) / speed))
    fmask = np.zeros((1, MAX_FRAMES), dtype=np.float32)
    fmask[0, features_len:] = 1.0
    x = np.random.default_rng(0).standard_normal((1, MAX_FRAMES, 100)).astype(np.float32)
    cond = np.random.default_rng(1).standard_normal((1, MAX_FRAMES, 100)).astype(np.float32)
    timesteps = get_time_steps(num_step=num_steps, t_shift=0.5)

    print(f"compute_units={args.compute_units} runs={args.runs}")
    print(f"utterance: prompt {prompt_len} + gen {features_len - prompt_len} frames "
          f"(bucket {MAX_FRAMES}), gen audio {gen_seconds:.3f}s @48k")
    print(f"model load: {load_ms:.0f} ms\n")

    m, s, lo = timeit(lambda: te.predict({"tokens": tok_in, "padding_mask": tmask}), args.runs)
    print(f"text_encoder   : {m:7.2f} ± {s:.2f} ms (min {lo:.2f})")
    te_ms = m

    def one_step(t):
        return fm.predict({
            "t": np.array([t], dtype=np.float32), "x": x,
            "text_condition": cond, "speech_condition": cond,
            "guidance_scale": np.array([3.0], dtype=np.float32), "padding_mask": fmask,
        })
    m, s, lo = timeit(lambda: one_step(0.5), args.runs)
    print(f"fm_decoder/step: {m:7.2f} ± {s:.2f} ms (min {lo:.2f})")
    step_ms = m

    def full_core():
        te.predict({"tokens": tok_in, "padding_mask": tmask})
        for i in range(num_steps):
            one_step(float(timesteps[i]))
    m, s, lo = timeit(full_core, args.runs)
    print(f"core pipeline  : {m:7.2f} ± {s:.2f} ms (min {lo:.2f})  "
          f"[te + {num_steps} steps; predicted {te_ms + num_steps * step_ms:.1f}]")
    core_ms = m

    # torch vocoder on the generated mel region
    from scripts.reference_infer import load_models_cpu_torch
    _, _, vocos, _, _ = load_models_cpu_torch()
    vocos.freq_range = 12000
    vocos.return_48k = True
    mel = torch.randn(1, 100, features_len - prompt_len) / 0.1 * 0.05
    with torch.no_grad():
        m, s, lo = timeit(lambda: vocos.decode(mel), args.runs)
    print(f"vocoder (torch): {m:7.2f} ± {s:.2f} ms (min {lo:.2f})")
    voc_ms = m

    total = core_ms + voc_ms
    print(f"\nend-to-end     : {total:7.2f} ms -> RTFx {gen_seconds * 1000 / total:.1f}x "
          f"(core only: {gen_seconds * 1000 / core_ms:.1f}x)")

    if args.torch_baseline:
        model, _ = load_model()
        xt = torch.from_numpy(x[:, :features_len, :])
        ct_t = torch.from_numpy(cond[:, :features_len, :])
        pm = torch.zeros(1, features_len, dtype=torch.bool)
        with torch.no_grad():
            m, s, lo = timeit(
                lambda: model.forward_fm_decoder(
                    t=torch.tensor(0.5), xt=xt, text_condition=ct_t,
                    speech_condition=ct_t, padding_mask=pm, guidance_scale=torch.tensor(3.0)),
                args.runs)
        print(f"\n[torch cpu baseline] fm_decoder/step: {m:7.2f} ± {s:.2f} ms (min {lo:.2f})")


if __name__ == "__main__":
    main()
