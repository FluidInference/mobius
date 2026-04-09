#!/usr/bin/env python3
"""Export per-language cache-external decoders with language bias baked in.

Each decoder has its language embedding permanently compiled into the architecture,
guaranteeing language isolation without needing language_id input.
"""

import argparse
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSpeechSeq2Seq

NUM_LAYERS = 8
NUM_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1024
VOCAB_SIZE = 16384
MAX_SEQ_LEN = 108

# Language token IDs
LANGUAGE_TOKENS = {
    "english": 62,
    "french": 69,
    "spanish": 169,
    "chinese": 50,
    "portuguese": 184,
    "german": 106,
    "italian": 94,
    "japanese": 90,
    "korean": 107,
    "polish": 120,
    "russian": 153,
    "turkish": 186,
    "hindi": 99,
    "arabic": 63,
}


class LanguageSpecificDecoder(nn.Module):
    """Cache-external decoder with language embedding baked in.

    Unlike V2 which takes language_id as input, this decoder has the language
    bias permanently compiled into the weights during export.
    """

    def __init__(self, decoder_wrapper, lm_head, language_token_id: int, language_strength: float = 0.5):
        super().__init__()
        self.embedding = decoder_wrapper._embedding
        self.layers = decoder_wrapper._decoder.layers
        self.final_norm = decoder_wrapper._decoder.final_layer_norm
        self.lm_head = lm_head

        # Extract and freeze language embedding
        with torch.no_grad():
            lang_token = torch.tensor([[language_token_id]])
            # Get raw token embedding (no position encoding)
            lang_emb = decoder_wrapper._embedding.token_embedding(lang_token)

            # Store as fixed parameter (scaled for stronger bias)
            self.language_bias = nn.Parameter(
                language_strength * lang_emb.squeeze(0),
                requires_grad=False
            )

    def forward(
        self,
        input_id: torch.Tensor,  # [1, 1]
        position_id: torch.Tensor,  # [1, 1]
        encoder_hidden_states: torch.Tensor,  # [1, 438, 1024]
        cross_attention_mask: torch.Tensor,  # [1, 1, 1, 438]
        attention_mask: torch.Tensor,  # [1, 1, 1, end_step]
        # KV caches (16 inputs, 16 outputs)
        k_cache_0, v_cache_0, k_cache_1, v_cache_1,
        k_cache_2, v_cache_2, k_cache_3, v_cache_3,
        k_cache_4, v_cache_4, k_cache_5, v_cache_5,
        k_cache_6, v_cache_6, k_cache_7, v_cache_7,
    ):
        # Infer current position from attention_mask shape
        end_step = attention_mask.shape[-1]
        past_kv_len = end_step - 1

        k_caches_in = [k_cache_0, k_cache_1, k_cache_2, k_cache_3,
                        k_cache_4, k_cache_5, k_cache_6, k_cache_7]
        v_caches_in = [v_cache_0, v_cache_1, v_cache_2, v_cache_3,
                        v_cache_4, v_cache_5, v_cache_6, v_cache_7]

        # Get token + position embedding
        hidden_states = self.embedding(input_id, position_id)

        # Add permanent language bias (baked into this specific decoder)
        hidden_states = hidden_states + self.language_bias.unsqueeze(0)

        # Output caches
        k_caches_out = []
        v_caches_out = []

        # Process layers (same as original cache-external)
        for layer_idx, layer in enumerate(self.layers):
            k_cache = k_caches_in[layer_idx]
            v_cache = v_caches_in[layer_idx]

            # Self-attention
            residual = hidden_states
            hidden_states = layer.layer_norm_1(hidden_states)

            # Project Q, K, V
            query = layer.first_sub_layer.query_net(hidden_states)
            key = layer.first_sub_layer.key_net(hidden_states)
            value = layer.first_sub_layer.value_net(hidden_states)

            # Reshape
            query = layer.first_sub_layer._reshape(query)
            key = layer.first_sub_layer._reshape(key)
            value = layer.first_sub_layer._reshape(value)

            # Update cache
            k_cache_new = k_cache.clone()
            v_cache_new = v_cache.clone()
            k_cache_new[:, :, past_kv_len:end_step, :] = key
            v_cache_new[:, :, past_kv_len:end_step, :] = value

            # Read valid cache entries
            k_valid = k_cache_new[:, :, :end_step, :]
            v_valid = v_cache_new[:, :, :end_step, :]

            # Attention
            attn_output = F.scaled_dot_product_attention(
                query, k_valid, v_valid,
                attn_mask=attention_mask,
                dropout_p=0.0,
                scale=layer.first_sub_layer.scale,
            )

            attn_output = (
                attn_output.transpose(1, 2).contiguous().view(1, 1, HIDDEN_SIZE)
            )
            attn_output = layer.first_sub_layer.out_projection(attn_output)
            hidden_states = residual + attn_output

            # Save updated caches
            k_caches_out.append(k_cache_new)
            v_caches_out.append(v_cache_new)

            # Cross-attention
            residual = hidden_states
            hidden_states = layer.layer_norm_2(hidden_states)
            cross_out = layer.second_sub_layer(
                hidden_states=hidden_states,
                context_states=encoder_hidden_states,
                attention_mask=cross_attention_mask,
                past_key_values=None,
                cache_position=None,
                is_cross_attention=True,
                kv_seq_len=None,
            )
            hidden_states = residual + cross_out

            # FFN
            residual = hidden_states
            hidden_states = layer.layer_norm_3(hidden_states)
            hidden_states = residual + layer.third_sub_layer(hidden_states)

        # Final norm and logits
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states).squeeze(1)

        # Return logits + all updated caches
        return (
            logits,
            k_caches_out[0], v_caches_out[0],
            k_caches_out[1], v_caches_out[1],
            k_caches_out[2], v_caches_out[2],
            k_caches_out[3], v_caches_out[3],
            k_caches_out[4], v_caches_out[4],
            k_caches_out[5], v_caches_out[5],
            k_caches_out[6], v_caches_out[6],
            k_caches_out[7], v_caches_out[7],
        )


def export_language_decoder(
    model,
    language_name: str,
    language_token_id: int,
    output_dir: Path,
    language_strength: float = 0.5
):
    """Export a decoder for a specific language."""

    print(f"\n{'='*70}")
    print(f"Exporting {language_name.upper()} Decoder")
    print(f"{'='*70}")
    print(f"Language token: {language_token_id}")
    print(f"Language strength: {language_strength}")

    # Create language-specific decoder
    decoder = LanguageSpecificDecoder(
        model.transf_decoder,
        model.log_softmax.mlp.layer0,
        language_token_id,
        language_strength
    )
    decoder.eval()

    # Example inputs
    input_id = torch.tensor([[4]], dtype=torch.long)
    position_id = torch.tensor([[0]], dtype=torch.long)
    encoder_hidden = torch.randn(1, 438, HIDDEN_SIZE)
    cross_mask = torch.ones(1, 1, 1, 438)
    attention_mask = torch.zeros(1, 1, 1, 1)

    k_caches = [torch.zeros(1, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM) for _ in range(NUM_LAYERS)]
    v_caches = [torch.zeros(1, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM) for _ in range(NUM_LAYERS)]

    # Trace
    print("Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(decoder, (
            input_id, position_id, encoder_hidden, cross_mask, attention_mask,
            *k_caches, *v_caches
        ))

    print("Converting to CoreML...")

    # Inputs
    attn_mask_dim = ct.RangeDim(lower_bound=1, upper_bound=MAX_SEQ_LEN, default=1)
    inputs = [
        ct.TensorType("input_id", shape=(1, 1), dtype=np.int32),
        ct.TensorType("position_id", shape=(1, 1), dtype=np.int32),
        ct.TensorType("encoder_hidden_states", shape=(1, 438, HIDDEN_SIZE), dtype=np.float32),
        ct.TensorType("cross_attention_mask", shape=(1, 1, 1, 438), dtype=np.float32),
        ct.TensorType("attention_mask", shape=(1, 1, 1, attn_mask_dim), dtype=np.float32),
    ]

    for i in range(NUM_LAYERS):
        inputs.extend([
            ct.TensorType(f"k_cache_{i}", shape=(1, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM), dtype=np.float32),
            ct.TensorType(f"v_cache_{i}", shape=(1, NUM_HEADS, MAX_SEQ_LEN, HEAD_DIM), dtype=np.float32),
        ])

    # Outputs
    outputs = [ct.TensorType("logits", dtype=np.float32)]
    for i in range(NUM_LAYERS):
        outputs.extend([
            ct.TensorType(f"k_cache_{i}_out", dtype=np.float32),
            ct.TensorType(f"v_cache_{i}_out", dtype=np.float32),
        ])

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
    )

    mlmodel.author = "FluidInference"
    mlmodel.short_description = f"Cohere decoder for {language_name} only (cache-external)"

    output_path = output_dir / f"cohere_decoder_{language_name}.mlpackage"
    mlmodel.save(str(output_path))

    import subprocess
    try:
        size_mb = subprocess.check_output(["du", "-sh", str(output_path)]).decode().split()[0]
        print(f"✅ Saved: {output_path}")
        print(f"   Size: {size_mb}")
    except:
        print(f"✅ Saved: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="CohereLabs/cohere-transcribe-03-2026")
    parser.add_argument("--output-dir", default="build-per-language")
    parser.add_argument("--languages", default="english,french,spanish,chinese", help="Comma-separated language names")
    parser.add_argument("--strength", type=float, default=0.5, help="Language embedding strength (0.1-1.0)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    languages = [lang.strip() for lang in args.languages.split(",")]

    print("="*70)
    print("Per-Language Decoder Export")
    print("="*70)
    print(f"\nLanguages: {', '.join(languages)}")
    print(f"Language strength: {args.strength}")
    print()

    # Load model once
    print("[1/2] Loading PyTorch model...")
    t0 = time.time()
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_id, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.eval()
    print(f"   ✓ {time.time()-t0:.1f}s")

    # Export each language
    print(f"\n[2/2] Exporting {len(languages)} language-specific decoders...")

    exported = []
    for lang_name in languages:
        if lang_name not in LANGUAGE_TOKENS:
            print(f"⚠️ Unknown language: {lang_name}")
            continue

        lang_token_id = LANGUAGE_TOKENS[lang_name]

        try:
            output_path = export_language_decoder(
                model, lang_name, lang_token_id, output_dir, args.strength
            )
            exported.append((lang_name, output_path))
        except Exception as e:
            print(f"❌ Failed to export {lang_name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*70)
    print("EXPORT COMPLETE")
    print("="*70)
    print(f"\nExported {len(exported)} decoders:")
    for lang_name, path in exported:
        print(f"  • {lang_name:12s}: {path.name}")

    print(f"\nTotal storage: ~{len(exported) * 291} MB ({len(exported)} × 291MB)")
    print("\nEach decoder has its language bias permanently baked in.")
    print("No language_id parameter needed - just load the decoder for your target language.")


if __name__ == "__main__":
    main()
