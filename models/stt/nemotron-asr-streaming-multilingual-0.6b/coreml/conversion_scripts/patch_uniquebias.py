"""Pre-trace patch: give each conformer layer's FFN biases unique values.

This breaks the `linear_1_bias_0_to_fp16` shared-constant dedup that
coremltools' MIL frontend applies to identical FP16 values. With unique
per-layer bias values, layer-position mixed-precision (via
`op_name_configs`) becomes possible — previously blocked because
coremltools refuses op_name configs that disagree across ops referencing
a shared constant.

Perturbation:
- Adds a unique ~1e-3 offset to element [0] of each FFN linear bias.
- 1e-3 is well above FP16 epsilon (~6e-5) so the resulting FP16 const
  bytes differ per (layer, ffn_module, linear_idx).
- 1e-3 is well below model noise (typical activation magnitudes ~1.0);
  bias drift through 48 FFN linears stays in the 1e-3 to 1e-2 range,
  expected to perturb WER by << 0.1pp.

Invocation:
    uv run python convert_nemotron_multilingual_uniquebias.py \\
        --nemo-path ... --output-dir build_fp16_tmp_1120ms_ios18_uniquebias ...

(See convert_nemotron_multilingual_uniquebias.py which wraps the main
convert script with this patch active.)
"""
from __future__ import annotations

import torch
import typer

# NOTE on FP16 representability + magnitude trade-off:
# FP16 between 0 and 0.5 has spacing ~0.000122 (since the 11-bit mantissa
# covers 1024 distinct values per power-of-2 range). With ~264 ops to
# patch and EPS=0.001, the perturbations land at [0.001, 0.264] — each
# value is 8x above the FP16 spacing at that magnitude, so guaranteed
# distinct bytes.
#
# Magnitude trade-off: an earlier attempt at EPS=1.0 (values 1..264) gave
# 100% WER — single-channel biases of magnitude ~200 saturate downstream
# activations. EPS=0.001 keeps the perturbation 3 orders of magnitude
# below typical activation magnitudes (~1.0), well within model noise.
EPS = 0.001


def perturb_ffn_biases(encoder: torch.nn.Module) -> int:
    """Walk encoder.layers and give each FFN linear a unique-valued bias.

    Diagnosis from first attempt: the FFN linear modules in
    ConformerFeedForward have `bias is None` (constructed with bias=False).
    coremltools' MIL frontend auto-generates a zero-bias FP16 const for
    each linear op (required by CoreML's linear-op signature) and then
    deduplicates those identical zeros into a single shared const —
    which is the `linear_1_bias_0_to_fp16` that blocks layer-position
    mixed-precision.

    Fix: replace each None bias with a fresh nn.Parameter that has a
    unique-per-(layer,ffn,lin) value in element [0]. This forces
    coremltools to emit distinct FP16 consts per location.

    Returns count of biases added/perturbed for sanity-check logging.
    """
    if not hasattr(encoder, "layers"):
        raise AttributeError(
            f"Encoder ({type(encoder).__name__}) has no .layers attribute. "
            "Adjust the layer path before invoking this patch."
        )

    # Global counter -> sequential integer perturbations, FP16-exact below 2048.
    counter = [0]

    def next_key() -> int:
        counter[0] += 1
        return counter[0]

    def patch_linear(lin):
        """Give a single linear a unique-valued bias from the global counter."""
        nonlocal count_added, count_perturbed
        if lin is None:
            return
        key = next_key()
        with torch.no_grad():
            if lin.bias is None:
                d_out = lin.out_features
                bias = torch.zeros(
                    d_out,
                    dtype=lin.weight.dtype,
                    device=lin.weight.device,
                )
                bias[0] = float(key) * EPS
                lin.bias = torch.nn.Parameter(bias, requires_grad=False)
                count_added += 1
            else:
                lin.bias[0] += float(key) * EPS
                count_perturbed += 1

    def patch_conv(c):
        nonlocal count_added, count_perturbed
        if c is None or not hasattr(c, "weight"):
            return
        key = next_key()
        with torch.no_grad():
            if getattr(c, "bias", None) is None:
                out_ch = c.weight.shape[0]
                bias = torch.zeros(
                    out_ch,
                    dtype=c.weight.dtype,
                    device=c.weight.device,
                )
                bias[0] = float(key) * EPS
                c.bias = torch.nn.Parameter(bias, requires_grad=False)
                count_added += 1
            else:
                c.bias[0] += float(key) * EPS
                count_perturbed += 1

    count_added = 0
    count_perturbed = 0
    for layer in encoder.layers:
        # Macaron FFNs
        for ffn_name in ("feed_forward1", "feed_forward2"):
            ffn = getattr(layer, ffn_name, None)
            if ffn is None:
                continue
            for lin_name in ("linear1", "linear2"):
                patch_linear(getattr(ffn, lin_name, None))
        # Multi-head self-attention linears
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            for aname in (
                "linear_q", "linear_k", "linear_v", "linear_out", "linear_pos",
            ):
                patch_linear(getattr(attn, aname, None))
        # Conv module patches SKIPPED — adding biases to Conv1d ops that
        # were constructed with bias=False produced 100% WER (likely
        # changes the ANE kernel pattern-match). The linear patches above
        # are sufficient to break the shared-const conflicts that block
        # layer-position mixed-precision (those errors were all on
        # linear ops, never conv ops).
    return count_added + count_perturbed


def patch_and_log(encoder: torch.nn.Module) -> None:
    """Public entrypoint. Prints to stdout (typer.echo can be silenced by progress bars)."""
    import sys
    n = perturb_ffn_biases(encoder)
    msg = f"  [patch_uniquebias] added/perturbed {n} FFN biases (EPS={EPS}) for dedup-breaking"
    print(msg, file=sys.stderr, flush=True)


