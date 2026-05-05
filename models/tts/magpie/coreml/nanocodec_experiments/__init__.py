"""Per-module CoreML conversion for ANE diagnostics.

Goal: convert individual nn.Modules in isolation, run coreml-cli --fallback on
each, and emit a ledger of (module, shape) -> (ane_percent, rejection_reasons,
op_counts). Used to find the specific submodules that block ANE in the fused
Magpie graphs (decoder_step, nanocodec_decoder).

This is a diagnostic tool. The deliverable from Phase A-D is bigger fused
mlmodelcs that all land 100% ANE — not these standalone files.
"""
