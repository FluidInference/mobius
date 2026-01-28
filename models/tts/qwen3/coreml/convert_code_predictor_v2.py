# Convert Code Predictor to CoreML - V2 with correct embeddings
# Uses talker's codec_embedding for codebook 0, code_predictor's for codebooks 1-14

import torch
import torch.nn as nn
import numpy as np
import coremltools as ct
from pathlib import Path

MAX_CODEC_TOKENS = 125  # ~10 seconds at 12Hz
NUM_CODEBOOKS = 16
NUM_PREDICTIONS = 15


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


class TracableCodePredictorV2(nn.Module):
    """
    Traceable Code Predictor with correct embedding handling.

    Key insight:
    - Step 0: Use talker's codec_embedding for codebook 0 → lm_head[0] → codebook 1
    - Step N (1-14): Use code_predictor's embed[N-1] for codebook N → lm_head[N] → codebook N+1

    Input:
    - codebook0: [B, seq_len] - First codebook tokens from main LM

    Output:
    - all_codebooks: [B, 15, seq_len] - Predicted tokens for codebooks 1-15
    """

    def __init__(self, code_predictor, talker, config):
        super().__init__()
        self.config = config
        self.num_predictions = NUM_PREDICTIONS

        # Extract components from code_predictor
        self.layers = code_predictor.model.layers
        self.norm = code_predictor.model.norm

        # Code predictor's embeddings (for codebooks 1-14 in decode mode)
        # embed[N-1] embeds codebook N tokens
        self.cp_embeddings = code_predictor.model.codec_embedding

        # Talker's codec_embedding (for codebook 0)
        self.talker_codec_embedding = talker.model.codec_embedding

        # 15 lm_heads (for codebooks 1-15)
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
        Predict all 15 codebooks autoregressively.

        Flow:
        - Step 0: talker_embed(codebook0) → transformer → lm_head[0] → codebook 1
        - Step N (1-14): cp_embed[N-1](codebook_N) → transformer → lm_head[N] → codebook N+1

        Args:
            codebook0: [B, seq_len] - First codebook tokens

        Returns:
            all_codebooks: [B, 15, seq_len] - Predicted codebook tokens 1-15
        """
        bsz, seq_len = codebook0.shape

        all_codebooks = []

        # Step 0: Use talker's embedding for codebook 0
        current_input = self.talker_codec_embedding(codebook0)
        hidden_states = self._run_transformer(current_input)
        logits = self.lm_heads[0](hidden_states)
        codebook1 = torch.argmax(logits, dim=-1)
        all_codebooks.append(codebook1)

        # Steps 1-14: Use code predictor's embeddings
        current_tokens = codebook1
        for step in range(1, self.num_predictions):
            # embed[step-1] embeds codebook_{step} tokens
            current_input = self.cp_embeddings[step - 1](current_tokens)
            hidden_states = self._run_transformer(current_input)
            logits = self.lm_heads[step](hidden_states)
            next_tokens = torch.argmax(logits, dim=-1)
            all_codebooks.append(next_tokens)
            current_tokens = next_tokens

        return torch.stack(all_codebooks, dim=1)


def verify_wrapper(original_cp, wrapper, codebook0):
    """Verify wrapper matches original."""
    print("\n=== Verification ===")

    with torch.no_grad():
        wrapper_output = wrapper(codebook0)
        print(f"Wrapper output shape: {wrapper_output.shape}")

        # Compare step by step with original
        # Original gen_steps=N uses embed[N-1] + lm_head[N]
        # My step N uses the same for N >= 1

        original_tokens = []
        current_tokens = codebook0

        # Step 0: Use prefill-like approach
        # We can't easily replicate prefill, so skip comparison for step 0

        # Steps 1-14: Should match exactly
        for step in range(1, NUM_PREDICTIONS):
            # gen_steps = step uses embed[step-1] + lm_head[step]
            output = original_cp(input_ids=current_tokens, generation_steps=step)
            next_tokens = torch.argmax(output.logits, dim=-1)
            original_tokens.append(next_tokens)
            current_tokens = next_tokens

        # For comparison, use wrapper's predicted tokens as conditioning
        wrapper_tokens = []
        current_tokens = wrapper_output[:, 0]  # codebook 1 from wrapper

        for step in range(1, NUM_PREDICTIONS):
            output = original_cp(input_ids=current_tokens, generation_steps=step)
            next_tokens = torch.argmax(output.logits, dim=-1)
            wrapper_tokens.append(next_tokens)
            current_tokens = next_tokens

        # Compare steps 1-14 (codebooks 2-15)
        wrapper_subset = wrapper_output[:, 1:]  # codebooks 2-15
        wrapper_from_orig = torch.stack(wrapper_tokens, dim=1)

        print(f"\nComparing codebooks 2-15 (steps 1-14):")
        print(f"  Wrapper shape: {wrapper_subset.shape}")
        print(f"  From original: {wrapper_from_orig.shape}")

        matches = (wrapper_subset == wrapper_from_orig).sum().item()
        total = wrapper_subset.numel()
        print(f"  Matches: {matches}/{total} ({100 * matches / total:.1f}%)")

        if matches == total:
            print("  PERFECT MATCH for steps 1-14!")
        else:
            for i in range(wrapper_subset.shape[1]):
                cb_matches = (wrapper_subset[:, i] == wrapper_from_orig[:, i]).sum().item()
                cb_total = wrapper_subset[:, i].numel()
                if cb_matches != cb_total:
                    print(f"    Codebook {i + 2}: {cb_matches}/{cb_total} matches")

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

    output_path = "qwen3_tts_code_predictor_v2.mlpackage"
    print(f"Saving to {output_path}...")
    mlmodel.save(output_path)

    import subprocess
    result = subprocess.run(["du", "-sh", output_path], capture_output=True, text=True)
    print(f"Model size: {result.stdout.strip()}")

    return mlmodel


def main():
    print("=" * 60)
    print("Qwen3-TTS Code Predictor V2 - Correct Embeddings")
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
    print(f"   Talker codec_embedding: {talker.model.codec_embedding}")

    print("\n2. Creating wrapper...")
    wrapper = TracableCodePredictorV2(code_predictor, talker, config)
    wrapper.eval()

    print("\n3. Verifying wrapper...")
    test_codebook0 = torch.randint(0, 2048, (1, 10))
    verify_wrapper(code_predictor, wrapper, test_codebook0)

    print("\n4. Converting to CoreML...")
    mlmodel = convert_to_coreml(wrapper)

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
