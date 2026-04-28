"""Convert FlowLM step model to CoreML."""
import argparse
import torch
import numpy as np
import coremltools as ct
import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONVERT_MODELS_DIR = os.path.dirname(_SCRIPT_DIR)
_COREML_DIR = os.path.dirname(_CONVERT_MODELS_DIR)
_PROJECT_DIR = os.path.dirname(_COREML_DIR)
sys.path.insert(0, _PROJECT_DIR)  # for: from pocket_tts import ...
sys.path.insert(0, _SCRIPT_DIR)  # for: from _language_arg import ...
sys.path.insert(0, os.path.join(_CONVERT_MODELS_DIR, "traceable"))  # for: from traceable_* import ...

from _language_arg import (
    add_compute_args,
    add_language_arg,
    build_output_dir,
    resolve_compute_precision,
    resolve_compute_units,
)
from traceable_flowlm_step import TraceableFlowLMStep


def _apply_int8_to_wrapper(step_model, num_layers: int):
    """Apply Kyutai-recipe-equivalent int8 weight quantization to the traceable wrapper.

    Mirrors the *layer selection* of `pocket_tts.quantization.apply_dynamic_int8(
    flow_lm, {"attention", "ffn"})` (upstream PR kyutai-labs/pocket-tts#147),
    but uses `coremltools.optimize.torch.quantization.PostTrainingQuantizer`
    instead of `torch.ao.quantize_dynamic`. CoreML cannot consume
    `quantized::linear_dynamic` ops (ct.convert raises NotImplementedError),
    so we fall back to weight-only int8 PTQ (per-channel symmetric) on the
    same set of linears the upstream recipe quantizes.

    Why operate on the wrapper, not the upstream flow_lm: `TraceableFlowLMStep
    .from_flowlm()` rebuilds fresh `nn.Linear` modules and copies weights via
    `.weight.data.copy_(...)`. Quantizing the upstream `flow_lm` would not
    propagate to the traced graph.

    Quantized (4 linears per transformer layer, weight-only int8, per-channel):
        attn{i}_in_proj  : 3072x1024
        attn{i}_out_proj : 1024x1024
        linear{i}_1      : 4096x1024 (FFN expand)
        linear{i}_2      : 1024x4096 (FFN contract)

    Left at fp32 (CRITICAL -- matches upstream "attention" + "ffn" config):
        input_linear : 32x1024
        out_eos      : 1024x1   -- EOS regression head; quantizing this
                                   collapses the EOS dynamic range and
                                   breaks termination (the v1 v2 bug).
        out_norm, norm{i}_1, norm{i}_2 : LayerNorms (untouched by recipe)
    """
    from coremltools.optimize.torch.quantization import (
        PostTrainingQuantizer,
        PostTrainingQuantizerConfig,
        ModulePostTrainingQuantizerConfig,
    )

    body_linear_names = []
    for i in range(num_layers):
        body_linear_names.extend([
            f"attn{i}_in_proj",
            f"attn{i}_out_proj",
            f"linear{i}_1",
            f"linear{i}_2",
        ])

    body_cfg = ModulePostTrainingQuantizerConfig(
        weight_dtype="int8",
        granularity="per_channel",
        quantization_scheme="symmetric",
    )
    # global_config=None means "do not quantize anything by default" -- only
    # modules explicitly named in module_name_configs get touched. This keeps
    # input_linear, out_eos, and all norms at fp32.
    ptq_config = PostTrainingQuantizerConfig(
        global_config=None,
        module_name_configs={name: body_cfg for name in body_linear_names},
    )
    quantizer = PostTrainingQuantizer(step_model, ptq_config)
    quantized_model = quantizer.compress(inplace=False)
    return quantized_model


def convert_flowlm_step(
    language: str,
    compute_precision: str = "fp16",
    compute_units: str = "ALL",
    int8: bool = False,
):
    print(f"Loading model (language={language})...")
    from pocket_tts import TTSModel
    model = TTSModel.load_model(language=language, lsd_decode_steps=8)
    model.eval()

    print("Creating traceable step model...")
    max_seq_len = 512
    step_model = TraceableFlowLMStep.from_flowlm(model.flow_lm, max_seq_len=max_seq_len)
    step_model.eval()
    num_layers = step_model.num_layers
    print(f"num_layers={num_layers}")

    if int8:
        print(
            f"Applying coremltools PTQ int8 (per-channel symmetric, weight-only) "
            f"to attn+FFN linears ({num_layers} layers x 4 linears) -- pre-trace, "
            f"matches upstream pocket_tts.quantization.apply_dynamic_int8 "
            f"selection but skips out_eos/input_linear..."
        )
        step_model = _apply_int8_to_wrapper(step_model, num_layers)
        step_model.eval()

    print("Creating example inputs...")
    B = 1
    T = 1
    H = 16
    D = 64

    sequence = torch.randn(B, T, 32)
    bos_emb = model.flow_lm.bos_emb.data

    # Create example caches and positions
    caches = []
    positions = []
    for i in range(num_layers):
        cache = torch.zeros(2, B, max_seq_len, H, D)
        # Fill some positions with data (simulating voice/text conditioning)
        cache[:, :, :136, :, :] = torch.randn(2, B, 136, H, D)
        caches.append(cache)
        positions.append(torch.tensor([136.0]))

    print("Tracing model...")
    trace_inputs = [sequence, bos_emb]
    for cache, pos in zip(caches, positions):
        trace_inputs.append(cache)
        trace_inputs.append(pos)
    with torch.no_grad():
        traced = torch.jit.trace(step_model, tuple(trace_inputs))

    # NOTE: Force fp32 IO contract via `dtype=np.float32` on every TensorType
    # plus anonymous fp32 outputs. With `compute_precision=fp16`, internal
    # ops still run in fp16 (the perf win we want) but coremltools inserts
    # fp16↔fp32 cast ops at the IO boundary so Swift can drive the model
    # with `MLMultiArrayDataType.float32` buffers. Without this, the macOS
    # MLE5 binder rejects fp16 MLMultiArrays ("Invalid heap allocated handle").
    print("Converting to CoreML (precision={}, IO=fp32)...".format(compute_precision))
    inputs = [
        ct.TensorType(name="sequence", shape=(1, 1, 32), dtype=np.float32),
        ct.TensorType(name="bos_emb", shape=(32,), dtype=np.float32),
    ]
    for i in range(num_layers):
        inputs.append(ct.TensorType(
            name=f"cache{i}", shape=(2, 1, max_seq_len, H, D), dtype=np.float32))
        inputs.append(ct.TensorType(
            name=f"position{i}", shape=(1,), dtype=np.float32))

    # Outputs: (x, is_eos, cache0_out, position0_out, ..., cacheN_out, positionN_out)
    n_outputs = 2 + 2 * num_layers
    ct_outputs = [ct.TensorType(dtype=np.float32) for _ in range(n_outputs)]

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=ct_outputs,
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=resolve_compute_precision(compute_precision),
    )

    output_dir = build_output_dir(_COREML_DIR, language)
    os.makedirs(output_dir, exist_ok=True)
    # int8 builds are saved as `flowlm_stepv2.mlpackage` to match the HF
    # naming under `FluidInference/pocket-tts-coreml/v2/<lang>/`.
    out_basename = "flowlm_stepv2.mlpackage" if int8 else "flowlm_step.mlpackage"
    output_path = os.path.join(output_dir, out_basename)
    print(f"Saving to {output_path} (precision={compute_precision}, int8={int8})...")
    mlmodel.save(output_path)

    # Test
    print(f"\nTesting CoreML model (compute_units={compute_units})...")
    coreml_model = ct.models.MLModel(output_path, compute_units=resolve_compute_units(compute_units))

    # Create test inputs
    test_seq = np.random.randn(1, 1, 32).astype(np.float32)
    test_bos = bos_emb.numpy().astype(np.float32)

    test_caches = {}
    test_positions = {}
    for i in range(num_layers):
        cache = np.zeros((2, 1, max_seq_len, H, D), dtype=np.float32)
        cache[:, :, :136, :, :] = np.random.randn(2, 1, 136, H, D).astype(np.float32)
        test_caches[f'cache{i}'] = cache
        test_positions[f'position{i}'] = np.array([136.0], dtype=np.float32)

    outputs = coreml_model.predict({
        'sequence': test_seq,
        'bos_emb': test_bos,
        **test_caches,
        **test_positions,
    })

    print(f"Output keys: {list(outputs.keys())}")
    for k, v in outputs.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, range=[{v.min():.4f}, {v.max():.4f}]")

    print("\nDone!")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_language_arg(parser)
    add_compute_args(parser)
    parser.add_argument(
        "--int8",
        action="store_true",
        help=(
            "Apply torch.ao dynamic int8 quantization to attention + FFN "
            "linears in the FlowLM transformer (per kyutai-labs/pocket-tts#147 "
            "RECOMMENDED_CONFIG). Quantization is applied PRE-trace so the "
            "scale/zero-point dequant happens dynamically at runtime; "
            "input_linear and out_eos stay at fp32. Saves to "
            "flowlm_stepv2.mlpackage."
        ),
    )
    args = parser.parse_args()
    convert_flowlm_step(
        args.language, args.compute_precision, args.compute_units, int8=args.int8
    )
