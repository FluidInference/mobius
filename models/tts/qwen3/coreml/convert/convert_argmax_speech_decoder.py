#!/usr/bin/env python3
"""
Argmax-style SpeechDecoder CoreML Conversion

Reverse-engineered from Argmax TTSKit's SpeechDecoder MIL program (2108 lines):

SpeechDecoder (streaming audio codec decoder with KV-cached attention):
  Inputs:
    - audio_codes [1, 16, 1] (int32): 16 codebook tokens for one frame
    - cache_length [1] (int32): current position
    - hidden_context [1, 1024, 1, 4] (fp16): sliding window context buffer
    - key_cache [1, 8192, 1, 256] (fp16): attention KV cache
    - key_padding_mask [1, 256] (fp16): mask for valid cache positions
    - kv_cache_update_mask [1, 256] (fp16): one-hot write mask
    - value_cache [1, 8192, 1, 256] (fp16): same as key_cache

  Outputs:
    - audio [1, 1, 1, 1920] (fp16): one frame of audio (1920 samples at 24kHz = 80ms)
    - key_cache_updates [1, 8192, 1, 1] (fp16): new KV for cache write
    - value_cache_updates [1, 8192, 1, 1] (fp16): new KV for cache write
    - hidden_context_update [1, 1024, 1, 1] (fp16): new context frame

Architecture from MIL:
  - RVQ first quantizer output projection: [2048, 256] → [512] via conv
  - RVQ rest quantizer projections: 15 × [2048, 256] → [512] each
  - Sum all 16 projected quantizer outputs
  - Self-attention decoder blocks (8 layers with KV cache, 8192 = 8×1024)
  - hidden_context: sliding window of 4 frames for causal conv context
  - ConvTranspose1d upsampling: 512→512→512→512→512→256 (6 stages)
  - Final: gelu activations, 1x1 conv to 1 channel → 1920 audio samples

Streaming approach: processes one codec frame at a time with KV cache,
produces exactly 1920 audio samples per call (80ms at 24kHz).

Usage:
    python convert_argmax_speech_decoder.py [--model-path ./model_0.6b]
"""

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
import argparse

MAX_SEQ_LEN = 256
HIDDEN_CONTEXT_LEN = 4
UPSAMPLE_RATE = 1920


def patch_rmsnorm_for_trace():
    """Monkey-patch RMSNorm classes to avoid dynamic dtype casts in JIT trace."""
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2DecoderRMSNorm,
    )

    def rmsnorm_forward(self, hidden_states):
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.float() * hidden_states

    Qwen3TTSTokenizerV2DecoderRMSNorm.forward = rmsnorm_forward


def patch_coremltools_int_cast():
    """Fix coremltools _cast to handle multi-dimensional constant arrays."""
    import coremltools.converters.mil.frontend.torch.ops as ct_ops
    import numpy as np
    from coremltools.converters.mil import Builder as mb

    def patched_cast(context, node, dtype, dtype_name):
        inputs = ct_ops._get_inputs(context, node, expected=1)
        x = inputs[0]
        if x.can_be_folded_to_const():
            val = x.val
            if isinstance(val, np.ndarray) and val.size == 1:
                val = val.item()
            if not isinstance(val, dtype):
                val = dtype(val)
            res = mb.const(val=val, name=node.name)
            context.add(res, node.name)
        elif len(x.shape) > 0:
            x = mb.squeeze(x=x, name=node.name + "_item")
            res = mb.cast(x=x, dtype=dtype_name, name=node.name)
            context.add(res, node.name)
        else:
            res = mb.cast(x=x, dtype=dtype_name, name=node.name)
            context.add(res, node.name)

    ct_ops._cast = patched_cast


class ArgmaxSpeechDecoder(nn.Module):
    """
    Streaming speech decoder that processes one codec frame at a time.

    Takes 16 codebook token IDs, looks up and projects each through RVQ output
    projections, sums them, processes through transformer layers with KV cache,
    then upsamples through convTranspose layers to produce audio.

    The hidden_context maintains a sliding window of past decoder hidden states
    for causal convolution context in the upsampling stages.
    """

    def __init__(self, decoder, speech_tokenizer):
        super().__init__()
        self.decoder = decoder
        # We need access to the tokenizer's internal quantizer for embeddings
        self.speech_tokenizer = speech_tokenizer

    def forward(
        self,
        audio_codes: torch.Tensor,        # [1, 16, 1]
        cache_length: torch.Tensor,       # [1]
        hidden_context: torch.Tensor,     # [1, 1024, 1, 4]
        key_cache: torch.Tensor,          # [1, 8192, 1, 256]
        key_padding_mask: torch.Tensor,   # [1, 256]
        kv_cache_update_mask: torch.Tensor,  # [1, 256]
        value_cache: torch.Tensor,        # [1, 8192, 1, 256]
    ):
        """
        Process one codec frame and produce one audio chunk.

        Returns:
            audio: [1, 1, 1, 1920] — 80ms of audio at 24kHz
            key_cache_updates: [1, 8192, 1, 1]
            value_cache_updates: [1, 8192, 1, 1]
            hidden_context_update: [1, 1024, 1, 1]
        """
        # This is a placeholder for the actual speech decoder forward pass.
        # The real implementation requires access to the tokenizer's internal
        # decoder structure, which varies by model version.
        #
        # The conversion approach is:
        # 1. Extract the decoder from speech_tokenizer.model.decoder
        # 2. Wrap it with the streaming logic (KV cache + hidden context)
        # 3. Trace and convert
        #
        # For now, we provide the conversion framework with correct I/O specs.
        raise NotImplementedError(
            "SpeechDecoder conversion requires access to the internal decoder structure. "
            "See convert_decoder.py for the non-streaming version."
        )


class TraceableVQLayer(nn.Module):
    """Pre-computed codebook embedding for a single VQ layer.

    Replaces VectorQuantization.decode() with a trace-friendly version
    that pre-normalizes the codebook and uses plain F.embedding.
    """

    def __init__(self, vq_layer):
        super().__init__()
        # Pre-compute normalized embedding: embedding_sum / cluster_usage
        cb = vq_layer._codebook
        embedding = cb.embedding_sum / cb.cluster_usage.clamp(min=cb.epsilon)[:, None]
        self.embedding = nn.Parameter(embedding, requires_grad=False)  # [2048, 256]

        # project_out may be identity or a linear layer
        self.project_out = vq_layer.project_out

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        codes: [B, T] — token indices
        Returns: [B, D, T] — projected quantized output
        """
        quantized = torch.nn.functional.embedding(codes, self.embedding)  # [B, T, 256]
        quantized = self.project_out(quantized)  # [B, T, D]
        quantized = quantized.transpose(1, 2)  # [B, D, T]
        return quantized


class SpeechDecoderWrapper(nn.Module):
    """
    Wrapper for the full speech decoder (non-streaming batch mode).

    Bypasses the problematic VQ quantizer trace path by manually unrolling
    the codebook embedding lookups. The RVQ decode is just:
      1. Look up each codebook's codes in its pre-normalized embedding table
      2. Project through output projection
      3. Sum all quantized outputs (first via output_proj, rest via output_proj)
      4. Pass through transformer + upsampler + decoder

    This avoids torch.jit.trace failures from autograd.Function in the VQ path.
    """

    def __init__(self, tokenizer_model):
        super().__init__()
        decoder = tokenizer_model.decoder

        # Extract the post-quantizer parts
        self.pre_conv = decoder.pre_conv
        self.pre_transformer = decoder.pre_transformer
        self.upsample = decoder.upsample
        self.audio_decoder = decoder.decoder

        # Extract quantizer components
        quantizer = decoder.quantizer
        self.n_q_semantic = quantizer.n_q_semantic  # 1

        # Build trace-friendly VQ layers for rvq_first
        self.first_vq_layers = nn.ModuleList()
        for layer in quantizer.rvq_first.vq.layers:
            self.first_vq_layers.append(TraceableVQLayer(layer))
        self.first_output_proj = quantizer.rvq_first.output_proj  # Conv1d(256, 512)

        # Build trace-friendly VQ layers for rvq_rest
        self.rest_vq_layers = nn.ModuleList()
        for layer in quantizer.rvq_rest.vq.layers:
            self.rest_vq_layers.append(TraceableVQLayer(layer))
        self.rest_output_proj = quantizer.rvq_rest.output_proj  # Conv1d(256, 512)

    def _manual_transformer(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Trace-friendly re-implementation of the pre_transformer forward pass.

        The HuggingFace model's forward uses decorators (@check_model_inputs,
        @dynamic_rope_update) and dispatchers that fail with torch.jit.trace
        on torch 2.10.0. This manually calls each layer's components.

        The full path is:
          input_proj(1024→512) → layers → norm → output_proj(512→1024)

        Args:
            hidden_states: [B, T, 1024] — input from pre_conv
        Returns:
            [B, T, 1024] — output matching pre_transformer
        """
        pre_t = self.pre_transformer

        # input_proj: 1024 → 512
        hidden_states = pre_t.input_proj(hidden_states)

        # RoPE: compute cos/sin for positions 0..T-1
        batch_size, seq_len, _ = hidden_states.shape
        position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)

        # Manual RoPE computation (avoiding @dynamic_rope_update decorator)
        rotary = pre_t.rotary_emb
        inv_freq = rotary.inv_freq
        inv_freq_expanded = inv_freq[None, :, None].float().expand(batch_size, -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * rotary.attention_scaling
        sin = emb.sin() * rotary.attention_scaling
        cos = cos.to(dtype=hidden_states.dtype)
        sin = sin.to(dtype=hidden_states.dtype)

        for layer in pre_t.layers:
            residual = hidden_states

            # Input layernorm
            h = layer.input_layernorm(hidden_states)

            # Self-attention (manual, avoiding HF dispatch)
            attn = layer.self_attn
            input_shape = h.shape[:-1]
            hidden_shape = (*input_shape, -1, attn.head_dim)

            query_states = attn.q_norm(attn.q_proj(h).view(hidden_shape)).transpose(1, 2)
            key_states = attn.k_norm(attn.k_proj(h).view(hidden_shape)).transpose(1, 2)
            value_states = attn.v_proj(h).view(hidden_shape).transpose(1, 2)

            # Apply RoPE
            q_embed = (query_states * cos.unsqueeze(1)) + (self._rotate_half(query_states) * sin.unsqueeze(1))
            k_embed = (key_states * cos.unsqueeze(1)) + (self._rotate_half(key_states) * sin.unsqueeze(1))

            # Attention computation (causal, with sliding window if needed)
            scale = attn.head_dim ** -0.5
            attn_weights = torch.matmul(q_embed, k_embed.transpose(-1, -2)) * scale

            # Causal mask: always apply (works for T=1 too since it's empty)
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=hidden_states.device),
                diagonal=1
            )
            attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

            attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(h.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

            attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
            attn_output = attn.o_proj(attn_output)

            # Self-attention layer scale + residual
            hidden_states = residual + layer.self_attn_layer_scale(attn_output)

            # MLP with residual
            residual = hidden_states
            h = layer.post_attention_layernorm(hidden_states)
            mlp_output = layer.mlp(h)
            hidden_states = residual + layer.mlp_layer_scale(mlp_output)

        # Final norm + output_proj: 512 → 1024
        hidden_states = pre_t.norm(hidden_states)
        hidden_states = pre_t.output_proj(hidden_states)
        return hidden_states

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            codes: [B, 16, T] — T frames, 16 codebooks

        Returns:
            audio: [B, 1, T*1920]
        """
        # 1. Manual RVQ decode (replacing quantizer.decode)
        # rvq_first: codes[:, :1, :] through 1 VQ layer + output_proj
        codes_first = codes[:, :self.n_q_semantic, :]  # [B, 1, T]
        quantized_first = self.first_vq_layers[0](codes_first[:, 0, :])  # [B, 256, T]
        quantized_first = self.first_output_proj(quantized_first)  # [B, 512, T]

        # rvq_rest: codes[:, 1:, :] through 15 VQ layers + output_proj
        codes_rest = codes[:, self.n_q_semantic:, :]  # [B, 15, T]
        quantized_rest_sum = self.rest_vq_layers[0](codes_rest[:, 0, :])  # [B, 256, T]
        for i in range(1, len(self.rest_vq_layers)):
            quantized_rest_sum = quantized_rest_sum + self.rest_vq_layers[i](codes_rest[:, i, :])
        quantized_rest = self.rest_output_proj(quantized_rest_sum)  # [B, 512, T]

        hidden = quantized_first + quantized_rest  # [B, 512, T]

        # 2. Pre-conv + manual transformer (trace-friendly)
        hidden = self.pre_conv(hidden).transpose(1, 2)  # [B, T, D]
        hidden = self._manual_transformer(hidden)
        hidden = hidden.permute(0, 2, 1)  # [B, D, T]

        # 3. Upsample
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)

        # 4. Final decoder
        wav = hidden
        for block in self.audio_decoder:
            wav = block(wav)

        return wav.clamp(min=-1, max=1)


def main():
    parser = argparse.ArgumentParser(description="Convert Argmax-style SpeechDecoder")
    parser.add_argument("--model-path", default="./model_0.6b", help="Path to Qwen3-TTS model")
    parser.add_argument("--tokenizer-path", default="./tokenizer_12hz", help="Path to tokenizer")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--quantize-w8", action="store_true", help="Apply W8A16 palettization")
    parser.add_argument("--streaming", action="store_true",
                        help="Convert streaming version with KV cache (experimental)")
    args = parser.parse_args()

    from qwen_tts import Qwen3TTSTokenizer

    patch_rmsnorm_for_trace()
    patch_coremltools_int_cast()

    print("=" * 60)
    print("Argmax-style SpeechDecoder Conversion")
    print("=" * 60)

    # 1. Load tokenizer
    print("\n1. Loading speech tokenizer...")
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_path, device_map="cpu"
    )
    decoder = tokenizer.model.decoder
    decoder.eval()

    param_count = sum(p.numel() for p in decoder.parameters())
    print(f"   Decoder parameters: {param_count:,}")

    # Inspect structure
    print("\n   Decoder structure:")
    for name, module in decoder.named_children():
        if hasattr(module, 'weight'):
            print(f"     {name}: {type(module).__name__} {tuple(module.weight.shape)}")
        else:
            sub_count = sum(p.numel() for p in module.parameters())
            print(f"     {name}: {type(module).__name__} ({sub_count:,} params)")

    if not args.streaming:
        # Non-streaming batch decoder: accepts [1, 16, T] for T frames
        print("\n2. Converting batch decoder (variable-length T)...")

        wrapper = SpeechDecoderWrapper(tokenizer.model)
        wrapper.eval()

        # Verify output matches original decoder at multiple lengths
        for test_T in [1, 10, 50]:
            test_codes = torch.randint(0, 2048, (1, 16, test_T))
            with torch.no_grad():
                test_audio = wrapper(test_codes)
                orig_audio = decoder(test_codes)
            diff = (test_audio - orig_audio).abs().max().item()
            print(f"   T={test_T}: output={test_audio.shape}, diff={diff:.6f}")
            if diff > 0.01:
                print("   WARNING: Large difference! Check wrapper implementation.")

        # Trace with a fixed multi-frame size
        # The decoder has causal convolutions with dynamic padding that fails with
        # variable shapes in CoreML. Use a fixed batch size instead.
        # 125 frames = 10s at 12Hz = maximum generation length.
        BATCH_T = 125
        trace_codes = torch.randint(0, 2048, (1, 16, BATCH_T))
        print(f"\n3. Tracing with T={BATCH_T}...")
        with torch.no_grad():
            trace_ref = wrapper(trace_codes)
        traced = torch.jit.trace(wrapper, (trace_codes,), strict=False)

        # Verify trace
        with torch.no_grad():
            traced_out = traced(trace_codes)
        diff = (traced_out - trace_ref).abs().max().item()
        print(f"   Trace verify T={BATCH_T}: output={traced_out.shape}, diff={diff:.6f}")

        # Convert to CoreML with fixed shape [1, 16, BATCH_T]
        print(f"\n4. Converting to CoreML (T={BATCH_T} fixed)...")
        ml_model = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="audio_codes", shape=(1, 16, BATCH_T), dtype=np.int32),
            ],
            outputs=[
                ct.TensorType(name="audio", dtype=np.float16),
            ],
            minimum_deployment_target=ct.target.iOS18,
            compute_precision=ct.precision.FLOAT16,
        )

        # W8A16 quantization
        if args.quantize_w8:
            print("\n5. Applying W8A16 palettized quantization...")
            from coremltools.optimize.coreml import (
                OpPalettizerConfig,
                OptimizationConfig,
                palettize_weights,
            )
            op_config = OpPalettizerConfig(mode="kmeans", nbits=8, weight_threshold=512)
            opt_config = OptimizationConfig(global_config=op_config)
            ml_model = palettize_weights(ml_model, config=opt_config)
            print("   W8A16 applied")

        sd_path = f"{args.output_dir}/SpeechDecoder.mlpackage"
        ml_model.save(sd_path)
        print(f"   Saved: {sd_path}")

        # Verify CoreML
        print("\n6. Verifying CoreML model...")
        loaded = ct.models.MLModel(sd_path)
        vC = torch.randint(0, 2048, (1, 16, BATCH_T))
        result = loaded.predict({
            "audio_codes": vC.numpy().astype(np.int32),
        })
        with torch.no_grad():
            pt_ref = wrapper(vC)
        cml_audio = result['audio'].astype(np.float32).flatten()
        pt_audio = pt_ref.detach().numpy().astype(np.float32).flatten()
        n = min(len(cml_audio), len(pt_audio))
        diff = np.abs(cml_audio[:n] - pt_audio[:n]).max()
        expected_samples = BATCH_T * 1920
        print(f"   T={BATCH_T}: CoreML shape={result['audio'].shape}, "
              f"expected={expected_samples} samples, diff={diff:.6f}")

    else:
        print("\n[STREAMING MODE - Experimental]")
        print("Streaming SpeechDecoder with KV cache requires restructuring")
        print("the internal decoder to process one frame at a time with:")
        print("  - KV cache [1, 8192, 1, 256] for self-attention layers")
        print("  - hidden_context [1, 1024, 1, 4] sliding window")
        print("")
        print("This requires deep introspection of the decoder's internal")
        print("attention layers. The Argmax model uses 8 transformer layers")
        print("(8192 = 8 × 1024 KV-heads) with the same pattern as CodeDecoder.")
        print("")
        print("For now, use the non-streaming single-frame version.")
        print("The streaming version can be built incrementally.")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
