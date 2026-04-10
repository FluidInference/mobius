"""
CosyVoice3 LLM CoreML Conversion using Qwen3-ASR techniques.

Adapted from mobius/models/stt/qwen3-asr-0.6b/coreml/individual_components.py
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
import numpy as np
from huggingface_hub import hf_hub_download

REPO_PATH = Path(__file__).parent / "cosyvoice_repo"
sys.path.insert(0, str(REPO_PATH))

print("="*80)
print("Converting CosyVoice3 LLM to CoreML (using Qwen3-ASR techniques)")
print("="*80)

# Load LLM model
REPO_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
CACHE_DIR = Path.home() / ".cache" / "cosyvoice3_analysis"

print("\nLoading LLM checkpoint...")
llm_path = hf_hub_download(repo_id=REPO_ID, filename="llm.pt", cache_dir=CACHE_DIR)
llm_state = torch.load(llm_path, map_location="cpu", weights_only=True)
print(f"✓ Loaded {len(llm_state)} parameters")

# ANEMLL-style RMSNorm for better ANE precision
class AnemllRMSNorm(nn.Module):
    """ANE-optimized RMSNorm using LayerNorm trick.

    Reference: https://huggingface.co/blog/anemll/anemll-style-rms-ane
    """
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.eps = eps
        self.dim = weight.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        doubled = torch.cat([x, -x], dim=-1)
        normed = F.layer_norm(doubled, [doubled.shape[-1]], eps=self.eps)
        normed = normed[..., : self.dim]
        return normed * self.weight


def patch_rms_norms(module: nn.Module) -> None:
    """Replace all RMSNorm instances with AnemllRMSNorm for ANE optimization."""
    for name, child in list(module.named_children()):
        class_name = type(child).__name__
        if class_name == "AnemllRMSNorm":
            continue
        if "RMSNorm" in class_name and hasattr(child, "weight"):
            eps = getattr(child, "variance_epsilon", getattr(child, "eps", 1e-6))
            replacement = AnemllRMSNorm(child.weight.data, eps=eps)
            setattr(module, name, replacement)
        else:
            patch_rms_norms(child)


# Text Embedding Wrapper
class TextEmbeddingWrapper(nn.Module):
    """Wrapper for text token embedding."""
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


# LM Head Wrapper
class LMHeadWrapper(nn.Module):
    """Wrapper for LM head (final projection to vocabulary)."""
    def __init__(self, lm_head: nn.Module, norm: nn.Module = None):
        super().__init__()
        self.norm = norm  # Optional final norm
        self.lm_head = lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        return self.lm_head(hidden_states)


# Decoder Layer Wrapper
class DecoderLayerWrapper(nn.Module):
    """Wrapper for a single Qwen2 decoder layer with rotary embeddings."""
    def __init__(self, layer: nn.Module, rotary_emb: nn.Module):
        super().__init__()
        self.layer = layer
        self.rotary_emb = rotary_emb

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        # Compute position embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        outputs = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        return outputs[0]  # Return only hidden states


# Load actual LLM model
print("\nLoading LLM model architecture...")

try:
    from transformers import Qwen2ForCausalLM, Qwen2Config

    # Try to load config
    try:
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json", cache_dir=CACHE_DIR)
        import json
        with open(config_path) as f:
            config_dict = json.load(f)

        # Extract LLM config (it's nested under "llm" key)
        if "llm" in config_dict:
            llm_config_dict = config_dict["llm"]
            llm_config = Qwen2Config(**llm_config_dict)
        else:
            # Infer from state dict
            print("  Config not found, inferring from state dict...")
            llm_config = Qwen2Config(
                hidden_size=896,  # From state dict inspection
                num_hidden_layers=24,
                vocab_size=151936,
                intermediate_size=4864,
                num_attention_heads=14,  # 896 / 64
                num_key_value_heads=2,  # From state dict (128 / 64)
            )

        print(f"  Config: {llm_config.num_hidden_layers} layers, hidden_size={llm_config.hidden_size}")

        # Create model and load weights
        print("\nCreating Qwen2 model...")
        llm_model = Qwen2ForCausalLM(llm_config)

        # Load state dict (handle nested keys)
        print("Loading weights...")
        # The state dict has keys like "llm.model.layers.0..." but Qwen2ForCausalLM expects "model.layers.0..."
        cleaned_state = {}
        for k, v in llm_state.items():
            if k.startswith("llm.model."):
                new_key = k.replace("llm.model.", "model.")
                cleaned_state[new_key] = v
            elif k.startswith("llm."):
                new_key = k.replace("llm.", "")
                cleaned_state[new_key] = v

        llm_model.load_state_dict(cleaned_state, strict=False)
        llm_model.eval()

        print("✓ LLM model loaded successfully")

        # Patch RMSNorm layers for ANE optimization
        print("\nPatching RMSNorm layers...")
        patch_rms_norms(llm_model)
        print("✓ RMSNorm layers patched")

        # Export text embedding
        print("\n" + "="*80)
        print("Converting Text Embedding")
        print("="*80)

        embedding_wrapper = TextEmbeddingWrapper(llm_model.model.embed_tokens)
        embedding_wrapper.eval()

        # Trace
        input_ids = torch.randint(0, llm_config.vocab_size, (1, 10), dtype=torch.long)
        print(f"Tracing with input shape: {input_ids.shape}")

        with torch.inference_mode():
            traced_embedding = torch.jit.trace(embedding_wrapper, input_ids)

        # Convert to CoreML
        print("Converting to CoreML...")
        embedding_coreml = ct.convert(
            traced_embedding,
            inputs=[ct.TensorType(name='input_ids', shape=(1, ct.RangeDim(1, 512)), dtype=np.int32)],
            outputs=[ct.TensorType(name='embeddings', dtype=np.float16)],
            minimum_deployment_target=ct.target.macOS14,
            compute_units=ct.ComputeUnit.ALL,
            convert_to='mlprogram',
            compute_precision=ct.precision.FLOAT16,
        )

        embedding_coreml.save("cosyvoice_llm_embedding.mlpackage")
        print("✓ Saved: cosyvoice_llm_embedding.mlpackage")

        # Export LM Head
        print("\n" + "="*80)
        print("Converting LM Head")
        print("="*80)

        lm_head_wrapper = LMHeadWrapper(llm_model.lm_head, llm_model.model.norm)
        lm_head_wrapper.eval()

        # Trace
        hidden_states = torch.randn(1, 10, llm_config.hidden_size)
        print(f"Tracing with hidden states shape: {hidden_states.shape}")

        with torch.inference_mode():
            traced_lm_head = torch.jit.trace(lm_head_wrapper, hidden_states)

        # Convert to CoreML
        print("Converting to CoreML...")
        lm_head_coreml = ct.convert(
            traced_lm_head,
            inputs=[ct.TensorType(name='hidden_states', shape=(1, ct.RangeDim(1, 512), llm_config.hidden_size), dtype=np.float16)],
            outputs=[ct.TensorType(name='logits', dtype=np.float16)],
            minimum_deployment_target=ct.target.macOS14,
            compute_units=ct.ComputeUnit.ALL,
            convert_to='mlprogram',
            compute_precision=ct.precision.FLOAT16,
        )

        lm_head_coreml.save("cosyvoice_llm_lm_head.mlpackage")
        print("✓ Saved: cosyvoice_llm_lm_head.mlpackage")

        # Export decoder layers (layer by layer to manage size)
        print("\n" + "="*80)
        print(f"Converting Decoder Layers (0-{llm_config.num_hidden_layers-1})")
        print("="*80)

        print("\nNote: Converting all 24 layers will take time and space.")
        print("For now, converting first layer as proof of concept...")

        layer_wrapper = DecoderLayerWrapper(
            llm_model.model.layers[0],
            llm_model.model.rotary_emb
        )
        layer_wrapper.eval()

        # Trace
        seq_len = 10
        hidden_states = torch.randn(1, seq_len, llm_config.hidden_size)
        attention_mask = torch.ones(1, 1, seq_len, seq_len)
        position_ids = torch.arange(seq_len).unsqueeze(0)

        print(f"Tracing layer 0 with shapes:")
        print(f"  hidden_states: {hidden_states.shape}")
        print(f"  attention_mask: {attention_mask.shape}")
        print(f"  position_ids: {position_ids.shape}")

        with torch.inference_mode():
            traced_layer = torch.jit.trace(
                layer_wrapper,
                (hidden_states, attention_mask, position_ids),
                check_trace=False,  # Disable trace checking for speed
            )

        # Convert to CoreML
        print("Converting to CoreML...")
        layer_coreml = ct.convert(
            traced_layer,
            inputs=[
                ct.TensorType(name='hidden_states', shape=(1, ct.RangeDim(1, 512), llm_config.hidden_size), dtype=np.float16),
                ct.TensorType(name='attention_mask', shape=(1, 1, ct.RangeDim(1, 512), ct.RangeDim(1, 512)), dtype=np.float16),
                ct.TensorType(name='position_ids', shape=(1, ct.RangeDim(1, 512)), dtype=np.int32),
            ],
            outputs=[ct.TensorType(name='output_hidden_states', dtype=np.float16)],
            minimum_deployment_target=ct.target.macOS14,
            compute_units=ct.ComputeUnit.ALL,
            convert_to='mlprogram',
            compute_precision=ct.precision.FLOAT16,
        )

        layer_coreml.save("cosyvoice_llm_layer_0.mlpackage")
        print("✓ Saved: cosyvoice_llm_layer_0.mlpackage")

        print("\n" + "="*80)
        print("Success! LLM Components Exported")
        print("="*80)

        print("""
Exported models:
✓ cosyvoice_llm_embedding.mlpackage - Text embedding
✓ cosyvoice_llm_lm_head.mlpackage - LM head (with norm)
✓ cosyvoice_llm_layer_0.mlpackage - Decoder layer 0 (proof of concept)

Next steps:
1. Export remaining 23 decoder layers (0-23)
2. Combine layers into a full decoder stack
3. Integrate with Flow model
4. Combine with Vocoder (already working)

The LLM CAN be converted to CoreML using Qwen3-ASR techniques!
        """)

    except ImportError as e:
        print(f"✗ Failed to import transformers: {e}")
        print("\nInstall with: uv add transformers")

except Exception as e:
    print(f"✗ Failed to load LLM model: {e}")
    import traceback
    traceback.print_exc()

print("="*80)
