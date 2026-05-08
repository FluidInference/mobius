"""Decomposed StyleTTS2 inference pipeline.

The pipeline factors `run_inference.make_inference_fn` into discrete stages
without forking the model-loading or helper code paths. All shared
utilities (model load, style computation, mel preprocessing, masking,
seed, vendor sys.path setup) come from `run_inference` so there is one
canonical source of truth.

Modules:
    stages.py        — pure functions for each logical inference stage
    orchestrator.py  — wires stages into an end-to-end inference call
    ref_s_guard.py   — defensive clone/snapshot for the voice reference vector

Use `scripts/parity_check.py` to validate orchestrator output against
`run_inference.make_inference_fn` via audio-array MSE/RMSE/correlation.
"""

from .ref_s_guard import RefSGuard, freeze_ref_s
from .stages import (
    StageInputs,
    StageOutputs,
    decode_audio,
    phonemize_and_tokenize,
    predict_duration_and_alignment,
    predict_f0_and_noise,
    sample_and_blend_style,
    text_encode_and_bert,
)

__all__ = [
    "RefSGuard",
    "StageInputs",
    "StageOutputs",
    "decode_audio",
    "freeze_ref_s",
    "phonemize_and_tokenize",
    "predict_duration_and_alignment",
    "predict_f0_and_noise",
    "sample_and_blend_style",
    "text_encode_and_bert",
]
