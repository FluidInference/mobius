# Convert Code Predictor to CoreML - V3 (Non-autoregressive)
# Each codebook is predicted INDEPENDENTLY from codebook0

import torch
import torch.nn as nn
import numpy as np
import coremltools as ct

MAX_CODEC_TOKENS = 125  # ~10 seconds at 12Hz


def apply_rotary_pos_emb_simple(q, k, cos, sin):
    """Apply rotary position embeddings."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotate half the hidden dims."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


class TracableCodePredictorV3(nn.Module):
    """
    Traceable Code Predictor - Non-autoregressive version.

    Key insight: Each codebook is predicted INDEPENDENTLY from codebook0!
    - gen_steps=N: embed[N-1](codebook0) → transformer → lm_head[N] → codebook N

    All 14 codebooks (1-14) are predicted in parallel from codebook0.

    Input:
    - codebook0: [B, seq_len] - First codebook tokens from main LM

    Output:
    - all_codebooks: [B, 14, seq_len] - Predicted tokens for codebooks 1-14
    """

    def __init__(self, code_predictor, config):
        super().__init__()
        self.config = config
        self.num_predictions = 14  # Codebooks 1-14

        # Extract components from code_predictor
        self.layers = code_predictor.model.layers
        self.norm = code_predictor.model.norm

        # 15 embeddings - embed[N-1] for gen_steps=N
        self.embeddings = code_predictor.model.codec_embedding

        # Projection layer (if exists)
        self.small_to_mtp_projection = code_predictor.small_to_mtp_projection

        # 15 lm_heads - lm_head[N] for gen_steps=N
        self.lm_heads = code_predictor.lm_head

        # Config values
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rms_norm_eps = config.rms_norm_eps

        # Pre-compute RoPE
        self._precompute_rope(MAX_CODEC_TOKENS)

    def _precompute_rope(self, max_seq_len):
        """Pre-compute rotary position embeddings."""
        dim = self.head_dim
        base = self.config.rope_theta

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer('cos_cached', emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0).unsqueeze(0))

    def _run_transformer(self, hidden_states):
        """Run the 5-layer transformer."""
        bsz, seq_len, _ = hidden_states.shape

        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=hidden_states.device, dtype=hidden_states.dtype),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            hidden_states = self._run_layer(layer, hidden_states, causal_mask, cos, sin)

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def _run_layer(self, layer, hidden_states, causal_mask, cos, sin):
        """Run a single transformer layer."""
        residual = hidden_states

        hidden_states = layer.input_layernorm(hidden_states)
        attn_output = self._run_attention(layer.self_attn, hidden_states, causal_mask, cos, sin)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def _run_attention(self, attn, hidden_states, causal_mask, cos, sin):
        """Run self-attention with QK-norm and RoPE."""
        bsz, q_len, _ = hidden_states.shape

        query_states = attn.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = attn.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        value_states = attn.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)

        query_states = attn.q_norm(query_states).transpose(1, 2)
        key_states = attn.k_norm(key_states).transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        query_states, key_states = apply_rotary_pos_emb_simple(query_states, key_states, cos, sin)

        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            key_states = key_states.repeat_interleave(n_rep, dim=1)
            value_states = value_states.repeat_interleave(n_rep, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn_weights = attn_weights + causal_mask
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        attn_output = attn.o_proj(attn_output)

        return attn_output

    def forward(self, codebook0):
        """
        Predict all 14 codebooks independently from codebook0.

        For gen_steps=N (1-14):
        - embed[N-1](codebook0) → projection → transformer → lm_head[N] → codebook N

        Args:
            codebook0: [B, seq_len] - First codebook tokens

        Returns:
            all_codebooks: [B, 14, seq_len] - Predicted codebook tokens 1-14
        """
        bsz, seq_len = codebook0.shape

        all_codebooks = []

        # Predict each codebook independently from codebook0
        for gen_steps in range(1, 15):  # gen_steps 1-14
            # embed[gen_steps-1] embeds codebook0 for this step
            inputs_embeds = self.embeddings[gen_steps - 1](codebook0)

            # Project to model dimension
            inputs_embeds = self.small_to_mtp_projection(inputs_embeds)

            # Run transformer
            hidden_states = self._run_transformer(inputs_embeds)

            # Get logits from lm_head[gen_steps]
            logits = self.lm_heads[gen_steps](hidden_states)
            predicted = torch.argmax(logits, dim=-1)
            all_codebooks.append(predicted)

        return torch.stack(all_codebooks, dim=1)


def verify_wrapper(original_cp, wrapper, codebook0):
    """Verify wrapper matches original."""
    print("\n=== Verification ===")

    with torch.no_grad():
        wrapper_output = wrapper(codebook0)
        print(f"Wrapper output shape: {wrapper_output.shape}")

        # Compare with original for each gen_steps
        matches = 0
        total = 0

        for gen_steps in range(1, 15):
            output = original_cp(input_ids=codebook0, generation_steps=gen_steps)
            original_tokens = torch.argmax(output.logits, dim=-1)
            wrapper_tokens = wrapper_output[:, gen_steps - 1]

            step_match = (original_tokens == wrapper_tokens).all().item()
            matches += step_match
            total += 1

            if not step_match:
                print(f"  gen_steps={gen_steps}: MISMATCH")
                print(f"    Original: {original_tokens[0, :5].tolist()}")
                print(f"    Wrapper:  {wrapper_tokens[0, :5].tolist()}")
            else:
                print(f"  gen_steps={gen_steps}: MATCH")

        print(f"\nTotal: {matches}/{total} steps match")

    return matches == total


def convert_to_coreml(wrapper):
    """Convert to CoreML."""
    print("\n=== Converting to CoreML ===")

    wrapper.eval()

    example_codebook0 = torch.randint(0, 2048, (1, MAX_CODEC_TOKENS))

    print("Tracing model...")
    traced = torch.jit.trace(wrapper, (example_codebook0,))

    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="codebook0", shape=(1, MAX_CODEC_TOKENS), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="all_codebooks", dtype=np.int32),
        ],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
    )

    output_path = "qwen3_tts_code_predictor_v3.mlpackage"
    print(f"Saving to {output_path}...")
    mlmodel.save(output_path)

    import subprocess
    result = subprocess.run(["du", "-sh", output_path], capture_output=True, text=True)
    print(f"Model size: {result.stdout.strip()}")

    return mlmodel


def main():
    print("=" * 60)
    print("Qwen3-TTS Code Predictor V3 - Non-Autoregressive")
    print("=" * 60)

    print("\n1. Loading PyTorch model...")
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        "./model_0.6b",
        device_map="cpu",
        torch_dtype=torch.float32,
    )

    talker = model.model.talker
    code_predictor = talker.code_predictor
    config = code_predictor.config

    print(f"   Code predictor hidden size: {config.hidden_size}")
    print(f"   Number of embeddings: {len(code_predictor.model.codec_embedding)}")
    print(f"   Number of lm_heads: {len(code_predictor.lm_head)}")

    print("\n2. Creating wrapper...")
    wrapper = TracableCodePredictorV3(code_predictor, config)
    wrapper.eval()

    print("\n3. Verifying wrapper...")
    test_codebook0 = torch.randint(0, 2048, (1, 10))
    is_valid = verify_wrapper(code_predictor, wrapper, test_codebook0)

    if is_valid:
        print("\n4. Converting to CoreML...")
        mlmodel = convert_to_coreml(wrapper)

        print("\n" + "=" * 60)
        print("Conversion complete!")
        print("=" * 60)
    else:
        print("\nVerification failed! Not converting.")


if __name__ == "__main__":
    main()
