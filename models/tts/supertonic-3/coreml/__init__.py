"""Hand-port of Supertonic-3 sub-networks from ONNX → PyTorch.

The four ONNX graphs (text_encoder, duration_predictor, vector_estimator,
vocoder) are reimplemented module-by-module against the published `tts.json`
hyperparameters, with weights loaded directly from the ONNX initializers by
name. Goal: trace each module with `torch.jit.trace` and emit a CoreML
mlpackage via `coremltools.convert(..., source="pytorch")`.
"""
