"""Convert CosyVoice3 CausalMaskedDiffWithDiT to CoreML mlpackage.

Pipeline mirrors convert-coreml.py (HiFT):
  1. Load flow + state_dict via hyperpyyaml
  2. Wrap with src.flow_coreml.FlowCoreML (static shapes, traceable euler+CFG)
  3. Sanity-check PyTorch wrapper output
  4. torch.jit.trace
  5. coremltools convert → mlpackage (FP32 default, FP16 optional)

Bucket sizing
-------------
The wrapper uses a single static N_total = N_prompt + N_new tokens, producing
M = N_total * token_mel_ratio mel frames. Default is N=125 → 250 mel frames,
which matches the existing HiFT mlpackage (T=250).

Usage:
    uv run python convert-flow.py --output-dir ./build/flow-fp32
    uv run python convert-flow.py --output-dir ./build/flow-fp16 --fp16
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

from src.flow_coreml import FlowCoreML  # noqa: E402


def _make_precision(fp16: bool):
    """FP16 everywhere except numerically sensitive ops.

    Flow has RMSNorm / LayerNorm + softmax attention inside the DiT blocks.
    Pinning {pow, reduce_mean, rsqrt, softmax, layer_norm, gelu} to fp32
    is necessary but **not sufficient** — the QK^T matmul output saturates
    in fp16 in 9 of 22 DiT blocks (peak ~1.6M at block 17), producing NaN
    even with `softmax` pinned. See TRIALS_AND_ERRORS.md Phase 3 + the
    "Findings preserved from removed exploratory scripts" section.
    Shipping config is `Flow-N250-fp32`.
    """
    if not fp16:
        return ct.precision.FLOAT32
    FP32_OPS = {"pow", "reduce_mean", "rsqrt", "softmax", "layer_norm", "gelu"}
    return ct.transform.FP16ComputePrecision(
        op_selector=lambda op: op.op_type not in FP32_OPS
    )


def load_flow() -> torch.nn.Module:
    from hyperpyyaml import load_hyperpyyaml

    yaml = HERE / "cosyvoice3_dl" / "cosyvoice3.yaml"
    pt = HERE / "cosyvoice3_dl" / "flow.pt"
    with open(yaml) as f:
        cfg = load_hyperpyyaml(f, overrides={"llm": None, "hift": None})
    flow = cfg["flow"]
    sd = torch.load(str(pt), map_location="cpu", weights_only=False)
    missing, unexpected = flow.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"[warn] unexpected: {len(unexpected)} (first 3: {unexpected[:3]})")
    flow.eval()
    return flow


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-tokens", type=int, default=125,
                   help="Total token slots = N_prompt + N_new (mel frames = 2x)")
    p.add_argument("--n-timesteps", type=int, default=10)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--min-deployment", default="macOS14",
                   choices=["macOS13", "macOS14", "macOS15"])
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    N = args.n_tokens
    M = N * 2  # token_mel_ratio=2

    print("[1/5] Loading flow...")
    flow = load_flow()
    n_params = sum(p.numel() for p in flow.parameters())
    print(f"      params: {n_params:,}")

    print(f"[2/5] Wrapping with FlowCoreML (N_total={N}, M={M})...")
    wrapper = FlowCoreML(flow, n_total_tokens=N, n_timesteps=args.n_timesteps).eval()

    # Dummy inputs
    torch.manual_seed(0)
    N_prompt = N // 2
    token_total = torch.randint(0, 6561, (1, N), dtype=torch.int64)
    num_prompt_tokens = torch.tensor([N_prompt], dtype=torch.int32)
    prompt_feat = torch.randn(1, M, 80)
    embedding = torch.randn(1, 192)

    print("[3/5] Verifying PyTorch wrapper output...")
    with torch.no_grad():
        full_mel, num_prompt_mel = wrapper(token_total, num_prompt_tokens, prompt_feat, embedding)
    print(f"      mel shape: {tuple(full_mel.shape)}  range=[{full_mel.min().item():.3f}, {full_mel.max().item():.3f}]")
    print(f"      num_prompt_mel: {int(num_prompt_mel.item())}")

    print("[4/5] Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper,
            (token_total, num_prompt_tokens, prompt_feat, embedding),
            strict=False,
        )

    print("[5/5] Converting to CoreML mlpackage...")
    precision = _make_precision(args.fp16)
    min_dep = {
        "macOS13": ct.target.macOS13,
        "macOS14": ct.target.macOS14,
        "macOS15": ct.target.macOS15,
    }[args.min_deployment]

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="token_total", shape=(1, N), dtype=np.int32),
            ct.TensorType(name="num_prompt_tokens", shape=(1,), dtype=np.int32),
            ct.TensorType(name="prompt_feat", shape=(1, M, 80), dtype=np.float32),
            ct.TensorType(name="embedding", shape=(1, 192), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="mel", dtype=np.float32),
            ct.TensorType(name="num_prompt_mel", dtype=np.int32),
        ],
        compute_precision=precision,
        minimum_deployment_target=min_dep,
        convert_to="mlprogram",
        # Skip the post-convert MLModel load; for large fp16 Flow this triggers
        # anecompilerservice which can hang for >1 h on the DiT graph.
        skip_model_load=True,
    )

    tag = "fp16" if args.fp16 else "fp32"
    mlp = out_dir / f"Flow-N{N}-{tag}.mlpackage"
    mlmodel.save(str(mlp))
    print(f"      saved: {mlp}")

    # Save reference IO for parity
    ref_pt = out_dir / f"ref-N{N}.pt"
    torch.save(
        {
            "token_total": token_total,
            "num_prompt_tokens": num_prompt_tokens,
            "prompt_feat": prompt_feat,
            "embedding": embedding,
            "mel": full_mel,
            "num_prompt_mel": num_prompt_mel,
        },
        str(ref_pt),
    )
    print(f"      ref  : {ref_pt}")


if __name__ == "__main__":
    main()
