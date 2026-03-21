#!/usr/bin/env python3
"""CLI for exporting Qwen3-ASR-0.6B components to CoreML.

Architecture:
  Qwen3ASRForConditionalGeneration
    └── thinker
          ├── audio_tower  → AudioEncoderFullWrapper  → qwen3_asr_audio_encoder.mlpackage
          ├── model         → (per-layer or chunked)   → qwen3_asr_decoder_*.mlpackage
          │   ├── embed_tokens → TextEmbeddingWrapper  → qwen3_asr_embedding.mlpackage
          │   └── norm         (fused into lm_head)
          └── lm_head       → LMHeadWrapper            → qwen3_asr_lm_head.mlpackage

Usage:
  uv run python convert-qwen3-asr.py
  uv run python convert-qwen3-asr.py --output-dir ./build/qwen3-asr
  uv run python convert-qwen3-asr.py --components audio_encoder  # audio encoder only
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import coremltools as ct
import numpy as np
import torch
import typer

from individual_components import (
    AudioEncoderFullWrapper,
    DecoderLayerWrapper,
    DecoderPrefillWrapper,
    DecoderStackWrapper,
    ExportSettings,
    LMHeadWrapper,
    TextEmbeddingWrapper,
    _coreml_convert,
)

DEFAULT_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
AUTHOR = "Fluid Inference"
SAMPLE_RATE = 16000
NUM_MEL_BINS = 128
MAX_AUDIO_SECONDS = 30.0  # max audio duration for tracing

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def _quantize_weights(model: ct.models.MLModel, dtype: str) -> ct.models.MLModel:
    """Apply post-training weight quantization/palettization to a CoreML model."""
    if dtype.startswith("palettize"):
        # palettize8, palettize6, palettize4
        from coremltools.optimize.coreml import (
            OpPalettizerConfig,
            OptimizationConfig,
            palettize_weights,
        )

        bits_map = {
            "palettize8": 8,
            "palettize6": 6,
            "palettize4": 4,
        }
        if dtype not in bits_map:
            raise ValueError(f"Unsupported palettization: {dtype}. Use 'palettize8', 'palettize6', or 'palettize4'.")

        config = OptimizationConfig(
            global_config=OpPalettizerConfig(nbits=bits_map[dtype])
        )
        return palettize_weights(model, config=config)

    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
    )

    dtype_map = {
        "int8": "int8",
        "int4": "int4",
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported quantization dtype: {dtype}. Use 'int8', 'int4', 'palettize8', 'palettize6', or 'palettize4'.")

    config = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(dtype=dtype_map[dtype])
    )
    return linear_quantize_weights(model, config=config)


def _save_mlpackage(model: ct.models.MLModel, path: Path, description: str) -> None:
    try:
        model.minimum_deployment_target = ct.target.iOS17
    except Exception:
        pass
    model.short_description = description
    model.author = AUTHOR
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    typer.echo(f"  Saved: {path}")


def _load_qwen3_asr_modules():
    """Load Qwen3-ASR configuration and modeling modules directly via importlib.

    This bypasses the qwen_asr/__init__.py which imports heavy inference
    dependencies (nagisa, soynlp, qwen-omni-utils, etc.) that we don't need
    for CoreML conversion.
    """
    qwen_asr_path = Path(__file__).resolve().parents[5] / "qwen3-asr"
    if not qwen_asr_path.exists():
        raise FileNotFoundError(
            f"qwen3-asr source not found at {qwen_asr_path}\n"
            "Clone it: git clone https://github.com/QwenLM/Qwen3-ASR.git qwen3-asr"
        )

    tb_dir = qwen_asr_path / "qwen_asr" / "core" / "transformers_backend"

    # Create stub package entries so relative imports within the modules work
    for pkg_name, pkg_path in [
        ("qwen_asr", qwen_asr_path / "qwen_asr"),
        ("qwen_asr.core", qwen_asr_path / "qwen_asr" / "core"),
        ("qwen_asr.core.transformers_backend", tb_dir),
    ]:
        if pkg_name not in sys.modules:
            mod = types.ModuleType(pkg_name)
            mod.__path__ = [str(pkg_path)]
            mod.__package__ = pkg_name
            sys.modules[pkg_name] = mod

    # Load configuration module first (no internal deps)
    config_fqn = "qwen_asr.core.transformers_backend.configuration_qwen3_asr"
    spec = importlib.util.spec_from_file_location(
        config_fqn, tb_dir / "configuration_qwen3_asr.py"
    )
    config_mod = importlib.util.module_from_spec(spec)
    sys.modules[config_fqn] = config_mod
    spec.loader.exec_module(config_mod)

    # Load modeling module (has relative import from .configuration_qwen3_asr)
    model_fqn = "qwen_asr.core.transformers_backend.modeling_qwen3_asr"
    spec2 = importlib.util.spec_from_file_location(
        model_fqn, tb_dir / "modeling_qwen3_asr.py"
    )
    model_mod = importlib.util.module_from_spec(spec2)
    sys.modules[model_fqn] = model_mod
    spec2.loader.exec_module(model_mod)

    return config_mod, model_mod


def _load_model(model_id: str):
    """Load Qwen3-ASR model via transformers with custom class registration."""
    typer.echo(f"Loading model: {model_id}")

    config_mod, model_mod = _load_qwen3_asr_modules()
    typer.echo("  Loaded Qwen3-ASR source modules (bypassed heavy deps)")

    from transformers import AutoConfig, AutoModel

    # Patch ROPE_INIT_FUNCTIONS for transformers 5.x compatibility.
    # Qwen3-ASR uses rope_type='default' which was removed in transformers 5.0.
    # We recreate the original default rope init: standard RoPE with no scaling.
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    if "default" not in ROPE_INIT_FUNCTIONS:
        def _default_rope_init(config, device=None, **kwargs):
            base = config.rope_theta
            dim = config.head_dim
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim)
            )
            return inv_freq, 1.0  # attention_scaling = 1.0

        ROPE_INIT_FUNCTIONS["default"] = _default_rope_init
        typer.echo("  Patched ROPE_INIT_FUNCTIONS: added 'default' rope type")

    # Register custom classes
    AutoConfig.register("qwen3_asr", config_mod.Qwen3ASRConfig)
    AutoConfig.register("qwen3_asr_audio_encoder", config_mod.Qwen3ASRAudioEncoderConfig)
    AutoModel.register(config_mod.Qwen3ASRConfig, model_mod.Qwen3ASRForConditionalGeneration)
    typer.echo("  Registered Qwen3ASR custom classes with AutoConfig/AutoModel")

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # Patch missing attributes for transformers 5.x compatibility
    # Qwen3-ASR config doesn't define pad_token_id but model code accesses it
    def _ensure_attr(cfg, attr, default):
        if not hasattr(cfg, attr):
            setattr(cfg, attr, default)

    _ensure_attr(config, "pad_token_id", None)
    if hasattr(config, "thinker_config"):
        _ensure_attr(config.thinker_config, "pad_token_id", None)

    model = AutoModel.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    typer.echo(f"  Model loaded. Type: {type(model).__name__}")
    return model


def _get_audio_encoder(model):
    """Extract the audio encoder from the model hierarchy."""
    if hasattr(model, "thinker"):
        return model.thinker.audio_tower
    if hasattr(model, "audio_tower"):
        return model.audio_tower
    raise AttributeError("Cannot find audio_tower in model")


def _get_text_model(model):
    """Extract the text model from the hierarchy."""
    if hasattr(model, "thinker"):
        return model.thinker.model
    if hasattr(model, "model"):
        return model.model
    raise AttributeError("Cannot find text model in model")


def _get_lm_head(model):
    """Extract the LM head."""
    if hasattr(model, "thinker"):
        return model.thinker.lm_head
    if hasattr(model, "lm_head"):
        return model.lm_head
    raise AttributeError("Cannot find lm_head in model")


def _get_text_norm(model):
    """Extract the final norm layer from text model."""
    text_model = _get_text_model(model)
    return text_model.norm


# ---------------------------------------------------------------------------
# Audio Encoder Conversion
# ---------------------------------------------------------------------------

def convert_audio_encoder(
    model,
    settings: ExportSettings,
) -> Path:
    """Convert the audio encoder to CoreML.

    Strategy: Export the full encoder (conv + transformer + projection) as a
    single CoreML model. The input is a fixed-size mel spectrogram.
    Chunking/windowing must be done in Swift before calling this model.
    """
    typer.echo("\n=== Converting Audio Encoder ===")

    audio_encoder = _get_audio_encoder(model)
    audio_encoder.eval()

    wrapper = AudioEncoderFullWrapper(audio_encoder)
    wrapper.eval()

    # Create trace input: fixed-size mel spectrogram
    # For 30s audio at 16kHz: 30 * 16000 = 480000 samples
    # Mel frames at 10ms hop: 480000 / 160 = 3000 frames
    # But audio encoder chunks into windows of 100 frames (n_window*2)
    # So we trace with a single window: 100 frames
    max_mel_frames = 100  # single window (n_window * 2)
    mel_input = torch.randn(1, NUM_MEL_BINS, max_mel_frames, dtype=torch.float32)

    typer.echo(f"  Trace input shape: {mel_input.shape}")

    with torch.inference_mode():
        ref_output = wrapper(mel_input)
        typer.echo(f"  Reference output shape: {ref_output.shape}")

    mel_input = mel_input.clone()

    typer.echo("  Tracing audio encoder...")
    traced = torch.jit.trace(wrapper, (mel_input,), strict=False)
    traced.eval()

    # Output frames after 3x stride-2 conv: ceil(100/8) = 13
    output_frames = ref_output.shape[1]

    inputs = [
        ct.TensorType(
            name="mel_input",
            shape=(1, NUM_MEL_BINS, max_mel_frames),
            dtype=np.float32,
        ),
    ]
    outputs = [
        ct.TensorType(name="audio_features", dtype=np.float32),
    ]

    typer.echo("  Converting to CoreML...")
    coreml_model = _coreml_convert(
        traced, inputs, outputs, settings,
        compute_units_override=settings.compute_units,
        compute_precision_override=None,  # default FP16
    )

    path = settings.output_dir / "qwen3_asr_audio_encoder.mlpackage"
    _save_mlpackage(coreml_model, path, "Qwen3-ASR audio encoder (single window)")

    return path


# ---------------------------------------------------------------------------
# Text Embedding Conversion
# ---------------------------------------------------------------------------

def convert_embedding(
    model,
    settings: ExportSettings,
) -> Path:
    """Convert the token embedding layer to CoreML."""
    typer.echo("\n=== Converting Token Embedding ===")

    text_model = _get_text_model(model)
    wrapper = TextEmbeddingWrapper(text_model)
    wrapper.eval()

    # Trace with a sample token sequence
    seq_len = 32
    input_ids = torch.zeros(1, seq_len, dtype=torch.int32)

    with torch.inference_mode():
        ref_output = wrapper(input_ids)
        typer.echo(f"  Reference output shape: {ref_output.shape}")

    input_ids = input_ids.clone()

    typer.echo("  Tracing embedding layer...")
    traced = torch.jit.trace(wrapper, (input_ids,), strict=False)
    traced.eval()

    inputs = [
        ct.TensorType(
            name="input_ids",
            shape=(1, ct.RangeDim(1, settings.max_seq_length)),
            dtype=np.int32,
        ),
    ]
    outputs = [
        ct.TensorType(name="embeddings", dtype=np.float32),
    ]

    typer.echo("  Converting to CoreML...")
    coreml_model = _coreml_convert(
        traced, inputs, outputs, settings,
        compute_units_override=settings.compute_units,
    )

    path = settings.output_dir / "qwen3_asr_embedding.mlpackage"
    _save_mlpackage(coreml_model, path, "Qwen3-ASR token embedding")

    return path


# ---------------------------------------------------------------------------
# LM Head Conversion
# ---------------------------------------------------------------------------

def convert_lm_head(
    model,
    settings: ExportSettings,
) -> Path:
    """Convert the LM head (norm + linear) to CoreML."""
    typer.echo("\n=== Converting LM Head ===")

    lm_head = _get_lm_head(model)
    norm = _get_text_norm(model)
    wrapper = LMHeadWrapper(lm_head, norm)
    wrapper.eval()

    hidden_size = 1024
    # Use realistic magnitude for tracing: after 28 decoder layers, hidden
    # states have values in ~[-300, 300], not the [-3, 3] range of randn.
    # Tracing with realistic magnitudes prevents CoreML numerical issues.
    hidden_states = torch.randn(1, 1, hidden_size, dtype=torch.float32) * 200.0

    with torch.inference_mode():
        ref_output = wrapper(hidden_states)
        typer.echo(f"  Reference output shape: {ref_output.shape}")
        typer.echo(f"  Trace input range: [{hidden_states.min():.1f}, {hidden_states.max():.1f}]")

    hidden_states = hidden_states.clone()

    typer.echo("  Tracing LM head...")
    traced = torch.jit.trace(wrapper, (hidden_states,), strict=False)
    traced.eval()

    inputs = [
        ct.TensorType(
            name="hidden_states",
            shape=(1, 1, hidden_size),
            dtype=np.float32,
        ),
    ]
    outputs = [
        ct.TensorType(name="logits", dtype=np.float32),
    ]

    typer.echo("  Converting to CoreML...")
    coreml_model = _coreml_convert(
        traced, inputs, outputs, settings,
        compute_units_override=settings.compute_units,
        compute_precision_override=ct.precision.FLOAT32,
    )

    path = settings.output_dir / "qwen3_asr_lm_head.mlpackage"
    _save_mlpackage(coreml_model, path, "Qwen3-ASR LM head (norm + linear)")

    return path


# ---------------------------------------------------------------------------
# Decoder Layer Conversion
# ---------------------------------------------------------------------------

def convert_decoder_layers(
    model,
    settings: ExportSettings,
) -> List[Path]:
    """Convert each decoder layer individually to CoreML.

    Each layer is a self-contained unit with explicit KV-cache I/O,
    enabling stateful inference in Swift.
    """
    typer.echo("\n=== Converting Decoder Layers ===")

    text_model = _get_text_model(model)
    num_layers = len(text_model.layers)
    hidden_size = 1024
    num_kv_heads = 8
    head_dim = 128

    typer.echo(f"  {num_layers} layers to convert")

    paths = []

    for layer_idx in range(num_layers):
        typer.echo(f"\n  Layer {layer_idx}/{num_layers - 1}...")
        layer = text_model.layers[layer_idx]
        wrapper = DecoderLayerWrapper(layer, layer_idx)
        wrapper.eval()

        # Trace inputs for single-token generation step
        cache_len = 32  # trace with a short cache
        hidden_states = torch.randn(1, 1, hidden_size, dtype=torch.float32)
        k_cache = torch.randn(1, num_kv_heads, cache_len, head_dim, dtype=torch.float32)
        v_cache = torch.randn(1, num_kv_heads, cache_len, head_dim, dtype=torch.float32)
        position_cos = torch.randn(1, 1, head_dim, dtype=torch.float32)
        position_sin = torch.randn(1, 1, head_dim, dtype=torch.float32)
        # Causal mask: attend to all previous + current token
        attention_mask = torch.zeros(1, 1, 1, cache_len + 1, dtype=torch.float32)

        with torch.inference_mode():
            ref_hs, ref_k, ref_v = wrapper(
                hidden_states, k_cache, v_cache, position_cos, position_sin, attention_mask
            )
            if layer_idx == 0:
                typer.echo(f"    Output hidden: {ref_hs.shape}")
                typer.echo(f"    Output K cache: {ref_k.shape}")
                typer.echo(f"    Output V cache: {ref_v.shape}")

        # Clone for tracing
        trace_inputs = (
            hidden_states.clone(),
            k_cache.clone(),
            v_cache.clone(),
            position_cos.clone(),
            position_sin.clone(),
            attention_mask.clone(),
        )

        typer.echo(f"    Tracing layer {layer_idx}...")
        traced = torch.jit.trace(wrapper, trace_inputs, strict=False)
        traced.eval()

        max_cache = settings.max_seq_length
        inputs = [
            ct.TensorType(name="hidden_states", shape=(1, 1, hidden_size), dtype=np.float32),
            ct.TensorType(
                name="k_cache",
                shape=(1, num_kv_heads, ct.RangeDim(0, max_cache), head_dim),
                dtype=np.float32,
            ),
            ct.TensorType(
                name="v_cache",
                shape=(1, num_kv_heads, ct.RangeDim(0, max_cache), head_dim),
                dtype=np.float32,
            ),
            ct.TensorType(name="position_cos", shape=(1, 1, head_dim), dtype=np.float32),
            ct.TensorType(name="position_sin", shape=(1, 1, head_dim), dtype=np.float32),
            ct.TensorType(
                name="attention_mask",
                shape=(1, 1, 1, ct.RangeDim(1, max_cache + 1)),
                dtype=np.float32,
            ),
        ]
        outputs = [
            ct.TensorType(name="output_hidden", dtype=np.float32),
            ct.TensorType(name="k_cache_out", dtype=np.float32),
            ct.TensorType(name="v_cache_out", dtype=np.float32),
        ]

        typer.echo(f"    Converting layer {layer_idx} to CoreML...")
        coreml_model = _coreml_convert(
            traced, inputs, outputs, settings,
            compute_units_override=settings.compute_units,
        )

        path = settings.output_dir / f"qwen3_asr_decoder_layer_{layer_idx:02d}.mlpackage"
        _save_mlpackage(coreml_model, path, f"Qwen3-ASR decoder layer {layer_idx}")
        paths.append(path)

    return paths


# ---------------------------------------------------------------------------
# Consolidated Decoder Stack Conversion
# ---------------------------------------------------------------------------

def convert_decoder_stack(
    model,
    settings: ExportSettings,
    no_optimize: bool = False,
) -> Path:
    """Convert all 28 decoder layers as a single consolidated CoreML model.

    Single-token decode model: seq_len=1 fixed, RangeDim on cache length.
    Uses stacked KV caches [28, 8, seq_len, 128] instead of per-layer tensors.
    """
    typer.echo("\n=== Converting Decoder Stack (all layers) ===")

    text_model = _get_text_model(model)
    num_layers = len(text_model.layers)
    hidden_size = 1024
    num_kv_heads = 8
    head_dim = 128

    typer.echo(f"  {num_layers} layers in single model")

    wrapper = DecoderStackWrapper(text_model)
    wrapper.eval()

    # Trace inputs — single token (seq_len=1) with attention_mask
    # Use a realistic cache length for tracing: typical prompt is ~100-200 tokens.
    # Tracing with a short cache (e.g. 32) can cause coremltools MIL optimization
    # passes to produce graphs that are numerically unstable at longer cache lengths.
    cache_len = 256
    hidden_states = torch.randn(1, 1, hidden_size, dtype=torch.float32)
    k_caches = torch.randn(num_layers, num_kv_heads, cache_len, head_dim, dtype=torch.float32)
    v_caches = torch.randn(num_layers, num_kv_heads, cache_len, head_dim, dtype=torch.float32)
    position_cos = torch.randn(1, 1, head_dim, dtype=torch.float32)
    position_sin = torch.randn(1, 1, head_dim, dtype=torch.float32)
    attention_mask = torch.zeros(1, 1, 1, cache_len + 1, dtype=torch.float32)

    with torch.inference_mode():
        ref_hs, ref_k, ref_v = wrapper(
            hidden_states, k_caches, v_caches, position_cos, position_sin, attention_mask
        )
        typer.echo(f"  Output hidden: {ref_hs.shape}")
        typer.echo(f"  Output K caches: {ref_k.shape}")
        typer.echo(f"  Output V caches: {ref_v.shape}")

    trace_inputs = (
        hidden_states.clone(),
        k_caches.clone(),
        v_caches.clone(),
        position_cos.clone(),
        position_sin.clone(),
        attention_mask.clone(),
    )

    typer.echo("  Tracing decoder stack...")
    traced = torch.jit.trace(wrapper, trace_inputs, strict=False)
    traced.eval()

    max_cache = settings.max_seq_length
    inputs = [
        ct.TensorType(
            name="hidden_states",
            shape=(1, 1, hidden_size),
            dtype=np.float32,
        ),
        ct.TensorType(
            name="k_caches",
            shape=(num_layers, num_kv_heads, ct.RangeDim(0, max_cache), head_dim),
            dtype=np.float32,
        ),
        ct.TensorType(
            name="v_caches",
            shape=(num_layers, num_kv_heads, ct.RangeDim(0, max_cache), head_dim),
            dtype=np.float32,
        ),
        ct.TensorType(
            name="position_cos",
            shape=(1, 1, head_dim),
            dtype=np.float32,
        ),
        ct.TensorType(
            name="position_sin",
            shape=(1, 1, head_dim),
            dtype=np.float32,
        ),
        ct.TensorType(
            name="attention_mask",
            shape=(1, 1, 1, ct.RangeDim(1, max_cache + 1)),
            dtype=np.float32,
        ),
    ]
    outputs = [
        ct.TensorType(name="output_hidden", dtype=np.float32),
        ct.TensorType(name="k_caches_out", dtype=np.float32),
        ct.TensorType(name="v_caches_out", dtype=np.float32),
    ]

    typer.echo("  Converting to CoreML...")
    coreml_model = _coreml_convert(
        traced, inputs, outputs, settings,
        compute_units_override=settings.compute_units,
        compute_precision_override=ct.precision.FLOAT32,
        no_optimize=no_optimize,
    )

    path = settings.output_dir / "qwen3_asr_decoder_stack.mlpackage"
    _save_mlpackage(coreml_model, path, f"Qwen3-ASR decoder stack ({num_layers} layers)")

    return path


def convert_decoder_prefill(
    model,
    settings: ExportSettings,
) -> Path:
    """Convert decoder stack for batched prefill with ALL FIXED shapes.

    This model processes the entire prompt in one call. All dimensions are
    fixed to ensure CoreML's GPU/ANE execution planner can compile it.
    The causal mask is baked into the model as a constant buffer.

    Prompts shorter than PREFILL_SEQ_LEN are padded; the caller trims
    the KV cache after inference.
    """
    typer.echo("\n=== Converting Decoder Prefill (fixed shapes) ===")

    text_model = _get_text_model(model)
    num_layers = len(text_model.layers)
    hidden_size = 1024
    num_kv_heads = 8
    head_dim = 128

    wrapper = DecoderPrefillWrapper(text_model)
    wrapper.eval()

    PREFILL_SEQ = DecoderPrefillWrapper.PREFILL_SEQ_LEN  # 512
    typer.echo(f"  {num_layers} layers, prefill seq_len={PREFILL_SEQ}")

    # All fixed shapes — no RangeDim anywhere
    hidden_states = torch.randn(1, PREFILL_SEQ, hidden_size, dtype=torch.float32)
    k_caches = torch.randn(num_layers, num_kv_heads, 1, head_dim, dtype=torch.float32)
    v_caches = torch.randn(num_layers, num_kv_heads, 1, head_dim, dtype=torch.float32)
    position_cos = torch.randn(1, PREFILL_SEQ, head_dim, dtype=torch.float32)
    position_sin = torch.randn(1, PREFILL_SEQ, head_dim, dtype=torch.float32)

    with torch.inference_mode():
        ref_hs, ref_k, ref_v = wrapper(
            hidden_states, k_caches, v_caches, position_cos, position_sin
        )
        typer.echo(f"  Output hidden: {ref_hs.shape}")
        typer.echo(f"  Output K caches: {ref_k.shape}")
        typer.echo(f"  Output V caches: {ref_v.shape}")

    trace_inputs = (
        hidden_states.clone(),
        k_caches.clone(),
        v_caches.clone(),
        position_cos.clone(),
        position_sin.clone(),
    )

    typer.echo("  Tracing decoder prefill...")
    traced = torch.jit.trace(wrapper, trace_inputs, strict=False)
    traced.eval()

    inputs = [
        ct.TensorType(
            name="hidden_states",
            shape=(1, PREFILL_SEQ, hidden_size),
            dtype=np.float32,
        ),
        ct.TensorType(
            name="k_caches",
            shape=(num_layers, num_kv_heads, 1, head_dim),
            dtype=np.float32,
        ),
        ct.TensorType(
            name="v_caches",
            shape=(num_layers, num_kv_heads, 1, head_dim),
            dtype=np.float32,
        ),
        ct.TensorType(
            name="position_cos",
            shape=(1, PREFILL_SEQ, head_dim),
            dtype=np.float32,
        ),
        ct.TensorType(
            name="position_sin",
            shape=(1, PREFILL_SEQ, head_dim),
            dtype=np.float32,
        ),
    ]
    outputs = [
        ct.TensorType(name="output_hidden", dtype=np.float32),
        ct.TensorType(name="k_caches_out", dtype=np.float32),
        ct.TensorType(name="v_caches_out", dtype=np.float32),
    ]

    typer.echo("  Converting to CoreML...")
    coreml_model = _coreml_convert(
        traced, inputs, outputs, settings,
        compute_units_override=settings.compute_units,
        compute_precision_override=ct.precision.FLOAT32,
    )

    path = settings.output_dir / "qwen3_asr_decoder_prefill.mlpackage"
    _save_mlpackage(coreml_model, path, f"Qwen3-ASR decoder prefill ({PREFILL_SEQ} tokens)")

    return path


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def write_metadata(
    settings: ExportSettings,
    component_paths: Dict[str, object],
    model_id: str,
) -> Path:
    """Write metadata.json documenting all exported components."""
    metadata = {
        "model_id": model_id,
        "architecture": "Qwen3ASRForConditionalGeneration",
        "sample_rate": SAMPLE_RATE,
        "num_mel_bins": NUM_MEL_BINS,
        "max_audio_seconds": settings.max_audio_seconds,
        "max_seq_length": settings.max_seq_length,
        "audio_encoder": {
            "n_window": 50,
            "n_window_infer": 800,
            "mel_window_size": 100,
            "conv_downsample_factor": 8,
            "d_model": 896,
            "output_dim": 1024,
            "num_layers": 18,
            "num_heads": 14,
        },
        "text_decoder": {
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_layers": 28,
            "num_attention_heads": 16,
            "num_kv_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
            "rope_theta": 1000000,
            "mrope_section": [24, 20, 20],
        },
        "special_tokens": {
            "audio_start_token_id": 151669,
            "audio_end_token_id": 151670,
            "audio_token_id": 151676,
            "eos_token_ids": [151645, 151643],
        },
        "components": component_paths,
        "export_settings": {
            "compute_units": settings.compute_units.name,
            "compute_precision": (
                settings.compute_precision.name
                if settings.compute_precision is not None
                else "FLOAT32"
            ),
        },
    }

    path = settings.output_dir / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2, default=str))
    typer.echo(f"\nMetadata written to {path}")
    return path


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

@app.command()
def convert(
    model_id: str = typer.Option(DEFAULT_MODEL_ID, "--model-id", help="HuggingFace model ID"),
    output_dir: Path = typer.Option(
        Path("build/qwen3-asr"),
        "--output-dir",
        help="Output directory for CoreML packages",
    ),
    components: Optional[str] = typer.Option(
        None,
        "--components",
        help="Comma-separated list: audio_encoder,embedding,lm_head,decoder_stack (or decoder_layers for per-layer). Default: all with decoder_stack.",
    ),
    max_seq_length: int = typer.Option(4096, "--max-seq-length", help="Max sequence length for decoder"),
    max_audio_seconds: float = typer.Option(MAX_AUDIO_SECONDS, "--max-audio-seconds", help="Max audio duration"),
    quantize: Optional[str] = typer.Option(
        None,
        "--quantize",
        help="Post-training weight quantization: 'int8' or 'int4'. Applied to decoder_stack and lm_head.",
    ),
    no_ane: bool = typer.Option(
        False,
        "--no-ane",
        help="Target CPU+GPU only (exclude ANE). Useful for debugging ANE-specific issues.",
    ),
    no_optimize: bool = typer.Option(
        False,
        "--no-optimize",
        help="Skip MIL optimization passes. Use minimal pipeline for debugging conversion errors.",
    ),
) -> None:
    """Export Qwen3-ASR-0.6B components to CoreML."""

    target_units = ct.ComputeUnit.CPU_AND_GPU if no_ane else ct.ComputeUnit.ALL
    settings = ExportSettings(
        output_dir=output_dir,
        compute_units=target_units,
        deployment_target=ct.target.iOS17,
        compute_precision=None,  # default (float16); overridden per-component where needed
        max_audio_seconds=max_audio_seconds,
        max_seq_length=max_seq_length,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    typer.echo("Export configuration:")
    typer.echo(f"  Model: {model_id}")
    typer.echo(f"  Output: {output_dir}")
    typer.echo(f"  Max seq length: {max_seq_length}")
    typer.echo(f"  Max audio seconds: {max_audio_seconds}")
    if no_ane:
        typer.echo(f"  Compute units: CPU_AND_GPU (no ANE)")
    if quantize:
        typer.echo(f"  Quantize: {quantize} (decoder_stack + lm_head)")

    # Parse components
    if components is not None:
        convert_list = [c.strip() for c in components.split(",")]
    else:
        convert_list = ["audio_encoder", "embedding", "lm_head", "decoder_stack", "decoder_prefill"]

    typer.echo(f"  Components: {convert_list}")

    # Load model
    model = _load_model(model_id)

    component_paths: Dict[str, object] = {}

    # Convert components
    if "audio_encoder" in convert_list:
        path = convert_audio_encoder(model, settings)
        component_paths["audio_encoder"] = {"path": path.name}

    if "embedding" in convert_list:
        path = convert_embedding(model, settings)
        component_paths["embedding"] = {"path": path.name}

    if "lm_head" in convert_list:
        path = convert_lm_head(model, settings)
        if quantize:
            typer.echo(f"\n  Quantizing lm_head to {quantize}...")
            lm_model = ct.models.MLModel(str(path))
            lm_model = _quantize_weights(lm_model, quantize)
            lm_model.save(str(path))
            typer.echo(f"  Saved quantized lm_head")
        component_paths["lm_head"] = {"path": path.name}

    if "decoder_stack" in convert_list:
        path = convert_decoder_stack(model, settings, no_optimize=no_optimize)
        if quantize:
            typer.echo(f"\n  Quantizing decoder_stack to {quantize}...")
            ds_model = ct.models.MLModel(str(path))
            ds_model = _quantize_weights(ds_model, quantize)
            ds_model.save(str(path))
            typer.echo(f"  Saved quantized decoder_stack")
        component_paths["decoder_stack"] = {"path": path.name, "num_layers": 28}

    if "decoder_prefill" in convert_list:
        path = convert_decoder_prefill(model, settings)
        if quantize:
            typer.echo(f"\n  Quantizing decoder_prefill to {quantize}...")
            dp_model = ct.models.MLModel(str(path))
            dp_model = _quantize_weights(dp_model, quantize)
            dp_model.save(str(path))
            typer.echo(f"  Saved quantized decoder_prefill")
        component_paths["decoder_prefill"] = {"path": path.name, "num_layers": 28}

    if "decoder_layers" in convert_list:
        paths = convert_decoder_layers(model, settings)
        component_paths["decoder_layers"] = {
            "num_layers": len(paths),
            "paths": [p.name for p in paths],
        }

    # Write metadata
    write_metadata(settings, component_paths, model_id)

    typer.echo("\n=== Conversion complete ===")


if __name__ == "__main__":
    app()
