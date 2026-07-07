"""fp32 eager parity: AneZipformerLayer vs original Zipformer2EncoderLayer.

Gates (hard-fail):
  - per-submodule max_abs_diff < 1e-3
  - whole-layer max_abs_diff < 1e-2 and cosine > 0.9999

Run: .venv/bin/python -m coreml.ane.parity
"""

import sys

sys.path.insert(0, ".")

import torch

from coreml.ane.layer import AneZipformerLayer, ane_to_tbc, tbc_to_ane
from coreml.convert_coreml import (
    load_model,
    patch_coremltools_int,
    patch_simple_downsample,
)

SEQ_LEN = 1024
EMBED_DIM = 512

SUBMODULE_GATE = 1e-3
LAYER_GATE = 1e-2
LAYER_COS_GATE = 0.9999


def stats(a: torch.Tensor, b: torch.Tensor):
    a = a.detach().flatten().double()
    b = b.detach().flatten().double()
    max_abs = (a - b).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    return max_abs, cos


def main():
    torch.manual_seed(0)
    patch_coremltools_int()
    patch_simple_downsample()
    model, _ = load_model()
    model.eval()

    enc0 = model.fm_decoder.encoders[0]
    layer = enc0.layers[0] if hasattr(enc0, "layers") else enc0.encoder.layers[0]

    # Real positional embedding, captured once from the jit-scripted encoder_pos.
    with torch.no_grad():
        pos_emb = enc0.encoder_pos(torch.randn(SEQ_LEN, 1, EMBED_DIM)).detach()

    src = torch.randn(SEQ_LEN, 1, EMBED_DIM)
    time_emb = torch.randn(1, EMBED_DIM)  # already projected, as the layer sees it

    # Capture original submodule inputs/outputs via hooks.
    captured = {}
    names = [
        "self_attn_weights",
        "feed_forward1",
        "feed_forward2",
        "feed_forward3",
        "nonlin_attention",
        "self_attn1",
        "self_attn2",
        "conv_module1",
        "conv_module2",
        "norm",
        "bypass_mid",
        "bypass",
    ]
    hooks = []
    for name in names:
        mod = getattr(layer, name)

        def hook(m, args, output, name=name):
            captured[name] = (
                tuple(a.detach().clone() if isinstance(a, torch.Tensor) else a for a in args),
                output.detach().clone(),
            )

        hooks.append(mod.register_forward_hook(hook))

    with torch.no_grad():
        ref = layer(src, pos_emb, time_emb=time_emb)
    for h in hooks:
        h.remove()

    ane = AneZipformerLayer(layer, pos_emb, SEQ_LEN).eval()

    rows = []
    failures = []

    def check(name, orig_out, ane_out, gate=SUBMODULE_GATE):
        max_abs, cos = stats(orig_out, ane_out)
        ok = max_abs < gate
        rows.append((name, max_abs, cos, gate, ok))
        if not ok:
            failures.append(name)

    def w_to_ane(w):  # (H, 1, Sq, Sk) -> (H, Sq, 1, Sk)
        return w.permute(0, 2, 1, 3).contiguous()

    with torch.no_grad():
        # Attention weights.
        (aw_args, aw_out) = captured["self_attn_weights"]
        ane_aw = ane.self_attn_weights(tbc_to_ane(aw_args[0]))
        check("self_attn_weights", aw_out, ane_aw.permute(0, 2, 1, 3))

        # Feedforwards.
        for name in ("feed_forward1", "feed_forward2", "feed_forward3"):
            args, out = captured[name]
            check(name, out, ane_to_tbc(getattr(ane, name)(tbc_to_ane(args[0]))))

        # Nonlin attention (head-0 weights).
        args, out = captured["nonlin_attention"]
        check(
            "nonlin_attention",
            out,
            ane_to_tbc(ane.nonlin_attention(tbc_to_ane(args[0]), w_to_ane(args[1]))),
        )

        # Self attentions (all heads).
        for name in ("self_attn1", "self_attn2"):
            args, out = captured[name]
            check(
                name,
                out,
                ane_to_tbc(getattr(ane, name)(tbc_to_ane(args[0]), w_to_ane(args[1]))),
            )

        # Convolution modules.
        for name in ("conv_module1", "conv_module2"):
            args, out = captured[name]
            check(name, out, ane_to_tbc(getattr(ane, name)(tbc_to_ane(args[0]))))

        # Norm and bypasses.
        args, out = captured["norm"]
        check("norm", out, ane_to_tbc(ane.norm(tbc_to_ane(args[0]))))
        for name in ("bypass_mid", "bypass"):
            args, out = captured[name]
            check(
                name,
                out,
                ane_to_tbc(getattr(ane, name)(tbc_to_ane(args[0]), tbc_to_ane(args[1]))),
            )

        # Whole layer.
        ane_out = ane_to_tbc(
            ane(tbc_to_ane(src), time_emb.reshape(1, EMBED_DIM, 1, 1))
        )
        layer_max_abs, layer_cos = stats(ref, ane_out)

    print(f"{'submodule':<20} {'max_abs_diff':>14} {'cosine':>12} {'gate':>8}  ok")
    for name, max_abs, cos, gate, ok in rows:
        print(f"{name:<20} {max_abs:>14.3e} {cos:>12.8f} {gate:>8.0e}  {'PASS' if ok else 'FAIL'}")
    layer_ok = layer_max_abs < LAYER_GATE and layer_cos > LAYER_COS_GATE
    print(
        f"{'WHOLE LAYER':<20} {layer_max_abs:>14.3e} {layer_cos:>12.8f} "
        f"{LAYER_GATE:>8.0e}  {'PASS' if layer_ok else 'FAIL'} (cos gate {LAYER_COS_GATE})"
    )
    if not layer_ok:
        failures.append("WHOLE LAYER")

    if failures:
        print(f"FAILED gates: {failures}")
        sys.exit(1)
    print("all parity gates passed")


if __name__ == "__main__":
    main()
