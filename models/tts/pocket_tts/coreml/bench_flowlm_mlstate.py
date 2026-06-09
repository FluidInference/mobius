#!/usr/bin/env python3
"""Trial 22 Part B: MLState-resident KV caches vs cache-as-IO for flowlm_step_ane.

Converts a STATEFUL variant of the real `TraceableFlowLMStepANE` graph: same
weights, same math, but the 12 cache tensors ([1, L, 16, 64] x k/v x 6 layers)
become `ct.StateType` buffers that live inside the model (iOS18+). The host
then sends only `sequence [1,1,32]` + `position [1]` per call and reads
`transformer_out` + `is_eos` — the ~50 MB/call of fp32 cache marshalling
(25 MB in + 25 MB out at L=512) never crosses the MLModel boundary.

Position is kept as a normal input (4 bytes; the host already tracks it), so
the state is exactly the 12 cache tensors the task names.

Stateful-write pattern: the scatter-free one-hot write from the traceable
produces `new_k`/`new_v`, which are slice-assigned (`buf[:] = new`) into
registered buffers — the coremltools jit.trace state pattern (read buffer ->
compute -> slice + copy_ -> coreml_update_state).

Measures (10 warmup + 200 timed, median/p95) state-resident vs the existing
cache-as-IO `flowlm_step_ane.mlpackage`, at CPU_AND_NE and ALL, plus a
3-step autoregressive parity check (CPU_ONLY) to prove the state graph
actually accumulates the cache across calls.

Requires coremltools >= 8 (`MLModel.make_state`); prints a limitation notice
and exits if the runtime can't drive MLState predict.

Usage (from models/tts/pocket_tts):
    uv run python coreml/bench_flowlm_mlstate.py [--language english]
        [--max-seq-len 512] [--warmup 10] [--iters 200] [--skip-convert]
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import coremltools as ct
import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../coreml
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)  # for: from pocket_tts import ...
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "convert_models", "traceable"))

from traceable_flowlm_step_ane import TraceableFlowLMStepANE

H, D = 16, 64
NUM_LAYERS = 6
PREFIX_LEN = 136


class StatefulFlowLMStepANE(TraceableFlowLMStepANE):
    """TraceableFlowLMStepANE with the 12 k/v cache tensors as mutable state.

    Same weights/graph; the per-layer attention math is the parent's
    `_streaming_attention_t1` verbatim — only the cache plumbing changes
    (registered buffers + in-place slice assignment instead of I/O tensors).
    """

    def __init__(self, num_layers: int = 6, max_seq_len: int = 512):
        super().__init__(num_layers=num_layers, max_seq_len=max_seq_len)
        for i in range(num_layers):
            self.register_buffer(
                f"k_cache{i}", torch.zeros(1, max_seq_len, H, D), persistent=False
            )
            self.register_buffer(
                f"v_cache{i}", torch.zeros(1, max_seq_len, H, D), persistent=False
            )

    def forward(self, sequence: torch.Tensor, position: torch.Tensor):  # type: ignore[override]
        x = self.input_linear(sequence)  # [1, 1, 1024]

        for i in range(self.num_layers):
            residual = x
            x_norm = getattr(self, f"norm{i}_1")(x)
            attn_out, new_k, new_v, _ = self._streaming_attention_t1(
                x_norm,
                getattr(self, f"attn{i}_in_proj"),
                getattr(self, f"attn{i}_out_proj"),
                getattr(self, f"k_cache{i}"),
                getattr(self, f"v_cache{i}"),
                position,
            )
            # Slice-assignment (not whole-buffer copy_): the coremltools
            # jit frontend only lowers select/slice + copy_ into
            # coreml_update_state ("No matching select or slice" otherwise).
            getattr(self, f"k_cache{i}")[:] = new_k
            getattr(self, f"v_cache{i}")[:] = new_v
            x = residual + attn_out

            residual = x
            x_norm = getattr(self, f"norm{i}_2")(x)
            ffn_out = getattr(self, f"linear{i}_2")(
                torch.nn.functional.gelu(getattr(self, f"linear{i}_1")(x_norm))
            )
            x = residual + ffn_out

        x = self.out_norm(x)
        is_eos = self.out_eos(x)
        return x, is_eos


def convert_stateful(language: str, max_seq_len: int, output_path: str) -> None:
    print(f"Loading model (language={language})...")
    from pocket_tts import TTSModel

    model = TTSModel.load_model(language=language, lsd_decode_steps=8)
    model.eval()

    print("Building stateful step model...")
    base = TraceableFlowLMStepANE.from_flowlm(model.flow_lm, max_seq_len=max_seq_len)
    stateful = StatefulFlowLMStepANE(num_layers=base.num_layers, max_seq_len=max_seq_len)
    stateful.load_state_dict(base.state_dict(), strict=False)
    stateful.eval()

    print("Tracing...")
    sequence = torch.randn(1, 1, 32)
    position = torch.tensor([0.0])
    with torch.no_grad():
        traced = torch.jit.trace(stateful, (sequence, position))

    print("Converting (fp16, iOS18, 12x StateType caches)...")
    states = []
    for i in range(NUM_LAYERS):
        for kind in ("k", "v"):
            states.append(
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(1, max_seq_len, H, D), dtype=np.float16
                    ),
                    name=f"{kind}_cache{i}",
                )
            )
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="sequence", shape=(1, 1, 32), dtype=np.float32),
            ct.TensorType(name="position", shape=(1,), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="transformer_out", dtype=np.float32),
            ct.TensorType(name="is_eos", dtype=np.float32),
        ],
        states=states,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    print(f"Saving {output_path}...")
    mlmodel.save(output_path)


def make_io_feed(max_seq_len: int, rng: np.random.Generator) -> dict:
    feed = {"sequence": rng.standard_normal((1, 1, 32)).astype(np.float32)}
    for i in range(NUM_LAYERS):
        k = np.zeros((1, max_seq_len, H, D), dtype=np.float32)
        v = np.zeros((1, max_seq_len, H, D), dtype=np.float32)
        k[:, :PREFIX_LEN] = rng.standard_normal((1, PREFIX_LEN, H, D)).astype(np.float32)
        v[:, :PREFIX_LEN] = rng.standard_normal((1, PREFIX_LEN, H, D)).astype(np.float32)
        feed[f"k_cache{i}"] = k
        feed[f"v_cache{i}"] = v
        feed[f"position{i}"] = np.array([float(PREFIX_LEN)], dtype=np.float32)
    return feed


def bench_stateful(path: str, compute_units, warmup: int, iters: int):
    model = ct.models.MLModel(path, compute_units=compute_units)
    st = model.make_state()
    feed = {
        "sequence": np.random.default_rng(0)
        .standard_normal((1, 1, 32))
        .astype(np.float32),
        "position": np.array([float(PREFIX_LEN)], dtype=np.float32),
    }
    for _ in range(warmup):
        model.predict(feed, state=st)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.predict(feed, state=st)
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times), statistics.quantiles(times, n=20)[18]


def bench_io(path: str, max_seq_len: int, compute_units, warmup: int, iters: int):
    model = ct.models.MLModel(path, compute_units=compute_units)
    feed = make_io_feed(max_seq_len, np.random.default_rng(0))
    for _ in range(warmup):
        model.predict(feed)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.predict(feed)
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times), statistics.quantiles(times, n=20)[18]


def parity_check(state_path: str, io_path: str, max_seq_len: int) -> float:
    """3 autoregressive steps from empty caches: state-resident vs host-carried."""
    sm = ct.models.MLModel(state_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    im = ct.models.MLModel(io_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    st = sm.make_state()

    rng = np.random.default_rng(7)
    io_caches = {}
    for i in range(NUM_LAYERS):
        io_caches[f"k_cache{i}"] = np.zeros((1, max_seq_len, H, D), dtype=np.float32)
        io_caches[f"v_cache{i}"] = np.zeros((1, max_seq_len, H, D), dtype=np.float32)

    worst = 0.0
    for step in range(3):
        seq = rng.standard_normal((1, 1, 32)).astype(np.float32)
        pos = np.array([float(step)], dtype=np.float32)

        got_s = sm.predict({"sequence": seq, "position": pos}, state=st)

        io_feed = {"sequence": seq}
        io_feed.update(io_caches)
        for i in range(NUM_LAYERS):
            io_feed[f"position{i}"] = pos
        got_i = im.predict(io_feed)
        for i in range(NUM_LAYERS):
            io_caches[f"k_cache{i}"] = got_i[f"new_k_cache{i}"]
            io_caches[f"v_cache{i}"] = got_i[f"new_v_cache{i}"]

        d = float(
            np.abs(got_s["transformer_out"] - got_i["transformer_out"]).max()
        )
        worst = max(worst, d)
        print(f"  parity step {step}: d_transformer_out={d:.3e}")
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", default="english")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--skip-convert", action="store_true", help="Reuse an existing stateful mlpackage"
    )
    args = parser.parse_args()

    ct_ver = tuple(int(p) for p in ct.__version__.split(".")[:2])
    if ct_ver < (8, 0) or not hasattr(ct.models.MLModel, "make_state"):
        print(
            f"LIMITATION: coremltools {ct.__version__} cannot drive MLState predict "
            "(needs >= 8.0 with MLModel.make_state). Falling back is not implemented "
            "here — use the L-bucket sweep (bench_flowlm_lbucket.py) to extrapolate "
            "the per-MB marshalling cost instead."
        )
        return 1

    build_dir = os.path.join(_SCRIPT_DIR, "build", args.language)
    suffix = "" if args.max_seq_len == 512 else f"_l{args.max_seq_len}"
    state_path = os.path.join(build_dir, f"flowlm_step_ane_state{suffix}.mlpackage")
    io_path = os.path.join(build_dir, f"flowlm_step_ane{suffix}.mlpackage")

    if not args.skip_convert or not os.path.exists(state_path):
        convert_stateful(args.language, args.max_seq_len, state_path)
    if not os.path.exists(io_path):
        print(f"cache-as-IO baseline missing: {io_path} (run convert_flowlm_step_ane.py)")
        return 1

    print("\nParity (state-resident vs host-carried caches, CPU_ONLY, 3 AR steps):")
    worst = parity_check(state_path, io_path, args.max_seq_len)
    print(f"  worst: {worst:.3e} (fp16-state vs fp32-IO band)")

    units = [("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE), ("ALL", ct.ComputeUnit.ALL)]
    io_mb = 2 * NUM_LAYERS * 2 * args.max_seq_len * H * D * 4 / 1e6

    print(f"\nwarmup={args.warmup}, iters={args.iters}, L={args.max_seq_len}")
    print(f"{'variant':<26s} {'cache IO/call':>14s} {'units':<12s} {'median':>9s} {'p95':>9s}")
    print("-" * 76)
    rows = {}
    for uname, cu in units:
        med, p95 = bench_io(io_path, args.max_seq_len, cu, args.warmup, args.iters)
        rows[("io", uname)] = med
        print(f"{'cache-as-IO (fp32)':<26s} {io_mb:>11.1f} MB {uname:<12s} {med:>7.2f}ms {p95:>7.2f}ms")
    for uname, cu in units:
        med, p95 = bench_stateful(state_path, cu, args.warmup, args.iters)
        rows[("state", uname)] = med
        print(f"{'MLState-resident (fp16)':<26s} {0.0:>11.1f} MB {uname:<12s} {med:>7.2f}ms {p95:>7.2f}ms")

    for uname, _ in units:
        a, b = rows.get(("io", uname)), rows.get(("state", uname))
        if a and b:
            print(
                f"\nMLState saves {a - b:+.2f} ms/call median @ {uname} "
                f"({(1 - b / a) * 100:.1f}%) by keeping {io_mb:.1f} MB/call inside the model"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
