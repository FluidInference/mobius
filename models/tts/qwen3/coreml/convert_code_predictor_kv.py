#!/usr/bin/env python3
"""
Qwen3-TTS Code Predictor with KV Cache - CoreML Conversion

Fixes the tracing issue by splitting into two models with manual attention:
1. CP Prefill: processes [past_hidden, cb0_embed] (2 tokens) → kv_cache + all_logits
2. CP Decode: processes 1 token + kv_cache → new_kv_cache + all_logits

Both use manual attention (no SDPA) so causal masking traces correctly.
Embedding tables exported as numpy for external lookup.

Architecture (from config.json):
- 5 layers, 16 heads, 8 KV heads, head_dim=128, hidden=1024
- intermediate_size=3072 (SwiGLU MLP)
- q_norm + k_norm (RMSNorm on head_dim)
- Standard 1D RoPE (not multimodal)
- 15 lm_heads (CB1-CB15), 15 embedding tables
"""

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np

MAX_CP_KV_LEN = 20  # max 17 positions (2 initial + 15 generated)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_cp(q, k, cos, sin):
    """Apply rotary embeddings. Handles various cos/sin shapes."""
    # cos/sin from Qwen2/3 rotary_emb: [batch, seq_len, head_dim]
    # Need: [batch, 1, seq_len, head_dim] to broadcast with [batch, heads, seq_len, head_dim]
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    elif cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CPPrefill(nn.Module):
    """Code predictor prefill - processes 2 initial tokens with manual causal attention."""

    def __init__(self, code_predictor, codec_embedding):
        super().__init__()
        self.codec_embedding = codec_embedding
        self.layers = code_predictor.model.layers
        self.norm = code_predictor.model.norm
        self.rotary_emb = code_predictor.model.rotary_emb
        self.lm_heads = code_predictor.lm_head

        attn0 = self.layers[0].self_attn
        cfg = attn0.config
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = attn0.head_dim
        self.hidden_size = self.num_heads * self.head_dim
        self.num_layers = len(self.layers)
        self.has_qk_norm = hasattr(attn0, "q_norm")

    def forward(self, past_hidden, cb0_token):
        """
        Args:
            past_hidden: [1, 1, 1024] - hidden state from LM decoder
            cb0_token: [1, 1] - CB0 token ID

        Returns:
            all_logits: [15, 1, 2048]
            kv_cache: [10, 1, 8, 2, 128]
        """
        cb0_embed = self.codec_embedding(cb0_token)  # [1, 1, 1024]
        hidden_states = torch.cat([past_hidden, cb0_embed], dim=1)  # [1, 2, 1024]

        # Position IDs for 2 tokens
        position_ids = torch.arange(2, device=hidden_states.device).unsqueeze(0)
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        # Causal mask for 2x2 attention
        causal_mask = torch.tensor([[0.0, -1e9], [0.0, 0.0]])

        # Process through layers
        kv_list = []
        for layer in self.layers:
            hidden_states, new_key, new_value = self._run_prefill_layer(
                layer, hidden_states, cos, sin, causal_mask
            )
            kv_list.append(new_key)
            kv_list.append(new_value)

        # Final norm
        hidden_states = self.norm(hidden_states)

        # All 15 lm_heads on last position
        last_hidden = hidden_states[:, -1:, :]  # [1, 1, 1024]
        all_logits = []
        for head in self.lm_heads:
            logits = head(last_hidden).squeeze(1)  # [1, 2048]
            all_logits.append(logits)
        all_logits = torch.stack(all_logits, dim=0)  # [15, 1, 2048]

        kv_cache = torch.stack(kv_list, dim=0)

        return all_logits, kv_cache

    def _run_prefill_layer(self, layer, hidden_states, cos, sin, causal_mask):
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        bsz, seq_len, _ = hidden_states.shape

        q = layer.self_attn.q_proj(hidden_states).view(
            bsz, seq_len, self.num_heads, self.head_dim
        )
        k = layer.self_attn.k_proj(hidden_states).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim
        )
        v = layer.self_attn.v_proj(hidden_states).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim
        )

        if self.has_qk_norm:
            q = layer.self_attn.q_norm(q)
            k = layer.self_attn.k_norm(k)

        q = q.transpose(1, 2)  # [bsz, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rotary_pos_emb_cp(q, k, cos, sin)

        new_key = k
        new_value = v

        # GQA expansion
        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Manual causal attention
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim**0.5)
        attn_weights = attn_weights + causal_mask
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            q.dtype
        )
        attn_out = torch.matmul(attn_weights, v)

        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        attn_out = layer.self_attn.o_proj(attn_out)

        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_key, new_value


class CPDecode(nn.Module):
    """Code predictor decode step - single token with KV cache and manual attention."""

    def __init__(self, code_predictor):
        super().__init__()
        self.layers = code_predictor.model.layers
        self.norm = code_predictor.model.norm
        self.rotary_emb = code_predictor.model.rotary_emb
        self.lm_heads = code_predictor.lm_head

        attn0 = self.layers[0].self_attn
        cfg = attn0.config
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = attn0.head_dim
        self.hidden_size = self.num_heads * self.head_dim
        self.num_layers = len(self.layers)
        self.has_qk_norm = hasattr(attn0, "q_norm")

    def forward(self, input_embed, kv_cache, position):
        """
        Args:
            input_embed: [1, 1, 1024]
            kv_cache: [10, 1, 8, kv_len, 128]
            position: [1] - current position

        Returns:
            all_logits: [15, 1, 2048]
            new_kv_cache: [10, 1, 8, kv_len+1, 128]
        """
        hidden_states = input_embed

        # Position embedding for single token
        position_ids = position.unsqueeze(0)  # [1, 1]
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        # Process through layers with KV cache
        new_keys = []
        new_values = []
        cache_idx = 0

        for layer in self.layers:
            hidden_states, new_key, new_value = self._run_decode_layer(
                layer,
                hidden_states,
                kv_cache[cache_idx],
                kv_cache[cache_idx + 1],
                cos,
                sin,
            )
            new_keys.append(new_key)
            new_values.append(new_value)
            cache_idx += 2

        # Final norm
        hidden_states = self.norm(hidden_states)

        # All 15 lm_heads
        all_logits = []
        for head in self.lm_heads:
            logits = head(hidden_states).squeeze(1)  # [1, 2048]
            all_logits.append(logits)
        all_logits = torch.stack(all_logits, dim=0)  # [15, 1, 2048]

        # Build new KV cache
        new_kv_list = []
        for i in range(self.num_layers):
            old_key = kv_cache[i * 2]
            old_value = kv_cache[i * 2 + 1]
            new_kv_list.append(torch.cat([old_key, new_keys[i]], dim=2))
            new_kv_list.append(torch.cat([old_value, new_values[i]], dim=2))
        new_kv_cache = torch.stack(new_kv_list, dim=0)

        return all_logits, new_kv_cache

    def _run_decode_layer(
        self, layer, hidden_states, key_cache, value_cache, cos, sin
    ):
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        bsz, q_len, _ = hidden_states.shape

        q = layer.self_attn.q_proj(hidden_states).view(
            bsz, q_len, self.num_heads, self.head_dim
        )
        k = layer.self_attn.k_proj(hidden_states).view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        )
        v = layer.self_attn.v_proj(hidden_states).view(
            bsz, q_len, self.num_kv_heads, self.head_dim
        )

        if self.has_qk_norm:
            q = layer.self_attn.q_norm(q)
            k = layer.self_attn.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rotary_pos_emb_cp(q, k, cos, sin)

        new_key = k
        new_value = v

        # Concatenate with cached keys/values
        full_key = torch.cat([key_cache, k], dim=2)
        full_value = torch.cat([value_cache, v], dim=2)

        # GQA expansion
        n_rep = self.num_heads // self.num_kv_heads
        if n_rep > 1:
            full_key = full_key.repeat_interleave(n_rep, dim=1)
            full_value = full_value.repeat_interleave(n_rep, dim=1)

        # Attention (no causal mask needed for single query token)
        attn_weights = torch.matmul(q, full_key.transpose(-1, -2)) / (
            self.head_dim**0.5
        )
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            q.dtype
        )
        attn_out = torch.matmul(attn_weights, full_value)

        attn_out = attn_out.transpose(1, 2).reshape(bsz, q_len, -1)
        attn_out = layer.self_attn.o_proj(attn_out)

        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_key, new_value


def run_kv_code_predictor(prefill_wrapper, decode_wrapper, cp_embeddings, past_hidden, cb0_token):
    """Run full 15-step code predictor using KV-cached models (PyTorch)."""
    with torch.no_grad():
        all_logits, kv_cache = prefill_wrapper(past_hidden, cb0_token)

    tokens = []
    # Step 0: CB1 from prefill
    cb1 = torch.argmax(all_logits[0], dim=-1).item()
    tokens.append(cb1)

    # Steps 1-14: CB2-CB15 from decode
    with torch.no_grad():
        for step in range(1, 15):
            embed = cp_embeddings[step - 1](
                torch.tensor([[tokens[-1]]])
            )  # [1, 1, 1024]
            all_logits, kv_cache = decode_wrapper(
                embed, kv_cache, torch.tensor([step + 1])
            )
            token = torch.argmax(all_logits[step], dim=-1).item()
            tokens.append(token)

    return tokens


def run_reference_code_predictor(cp, cp_embeddings, codec_embedding, past_hidden, cb0_token):
    """Run original code predictor (no KV cache) for reference comparison."""
    with torch.no_grad():
        cb0_embed = codec_embedding(cb0_token)
        hidden = torch.cat([past_hidden, cb0_embed], dim=1)

        ref_tokens = []
        for i in range(15):
            outputs = cp.model(inputs_embeds=hidden, use_cache=False)
            hs = outputs.last_hidden_state
            logits = cp.lm_head[i](hs[:, -1:, :])
            token = torch.argmax(logits, dim=-1)
            ref_tokens.append(token.item())
            embed = cp_embeddings[i](token)
            hidden = torch.cat([hidden, embed], dim=1)

    return ref_tokens


def main():
    print("=" * 60)
    print("Qwen3-TTS Code Predictor KV - Prefill + Decode")
    print("=" * 60)

    from qwen_tts import Qwen3TTSModel

    print("\n1. Loading model...")
    model = Qwen3TTSModel.from_pretrained(
        "./model_0.6b", device_map="cpu", torch_dtype=torch.float32
    )
    talker = model.model.talker
    cp = talker.code_predictor
    codec_embedding = talker.model.codec_embedding
    cp_embeddings = cp.get_input_embeddings()

    # Inspect structure
    print("\n2. Inspecting code predictor structure...")
    attn0 = cp.model.layers[0].self_attn
    cfg = attn0.config
    print(f"   Layers: {len(cp.model.layers)}")
    print(f"   num_heads: {cfg.num_attention_heads}")
    print(f"   num_kv_heads: {cfg.num_key_value_heads}")
    print(f"   head_dim: {attn0.head_dim}")
    print(f"   has q_norm: {hasattr(attn0, 'q_norm')}")
    print(f"   lm_heads: {len(cp.lm_head)}")
    print(f"   embeddings: {len(cp_embeddings)}")

    # Test RoPE output shape
    test_input = torch.randn(1, 2, 1024)
    test_pos = torch.arange(2).unsqueeze(0)
    cos, sin = cp.model.rotary_emb(test_input, test_pos)
    print(f"   RoPE cos shape: {cos.shape}")
    print(f"   RoPE sin shape: {sin.shape}")

    num_kv_heads = cfg.num_key_value_heads
    head_dim = attn0.head_dim
    num_layers = len(cp.model.layers)

    print("\n3. Creating wrappers...")
    prefill_wrapper = CPPrefill(cp, codec_embedding)
    prefill_wrapper.eval()
    decode_wrapper = CPDecode(cp)
    decode_wrapper.eval()

    # Compare against reference
    print("\n4. Comparing KV-cached vs reference...")
    past_hidden = torch.randn(1, 1, 1024)
    cb0_token = torch.tensor([[1995]])

    ref_tokens = run_reference_code_predictor(
        cp, cp_embeddings, codec_embedding, past_hidden, cb0_token
    )
    our_tokens = run_kv_code_predictor(
        prefill_wrapper, decode_wrapper, cp_embeddings, past_hidden, cb0_token
    )

    print(f"   Reference: {ref_tokens}")
    print(f"   KV-cache:  {our_tokens}")
    print(f"   Match: {ref_tokens == our_tokens}")

    if ref_tokens != our_tokens:
        print("   WARNING: Mismatch! Debugging...")
        mismatches = sum(1 for a, b in zip(ref_tokens, our_tokens) if a != b)
        print(f"   {mismatches}/15 tokens differ")

    # Test with a second random input to verify it's not stuck
    print("\n   Testing with different input...")
    past_hidden_2 = torch.randn(1, 1, 1024)
    cb0_token_2 = torch.tensor([[500]])

    ref_tokens_2 = run_reference_code_predictor(
        cp, cp_embeddings, codec_embedding, past_hidden_2, cb0_token_2
    )
    our_tokens_2 = run_kv_code_predictor(
        prefill_wrapper, decode_wrapper, cp_embeddings, past_hidden_2, cb0_token_2
    )

    print(f"   Reference 2: {ref_tokens_2}")
    print(f"   KV-cache 2:  {our_tokens_2}")
    print(f"   Match 2: {ref_tokens_2 == our_tokens_2}")
    print(f"   Different from first: {our_tokens != our_tokens_2}")

    # Trace and convert
    print("\n5. Tracing prefill model...")
    prefill_example = (
        torch.randn(1, 1, 1024),
        torch.tensor([[1000]]),
    )
    traced_prefill = torch.jit.trace(prefill_wrapper, prefill_example)

    # Verify traced prefill
    with torch.no_grad():
        traced_logits, traced_kv = traced_prefill(past_hidden, cb0_token)
        orig_logits, orig_kv = prefill_wrapper(past_hidden, cb0_token)
    print(f"   Traced vs original prefill logits match: {torch.allclose(traced_logits, orig_logits, atol=1e-5)}")

    print("\n6. Converting prefill to CoreML...")
    prefill_inputs = [
        ct.TensorType(name="past_hidden", shape=(1, 1, 1024), dtype=np.float32),
        ct.TensorType(name="cb0_token", shape=(1, 1), dtype=np.int32),
    ]
    ml_prefill = ct.convert(
        traced_prefill,
        inputs=prefill_inputs,
        outputs=[
            ct.TensorType(name="all_logits", dtype=np.float32),
            ct.TensorType(name="kv_cache", dtype=np.float32),
        ],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT32,
    )
    prefill_path = "qwen3_tts_cp_prefill.mlpackage"
    ml_prefill.save(prefill_path)
    print(f"   Saved to {prefill_path}")

    print("\n7. Tracing decode model...")
    kv_len = 2
    decode_example = (
        torch.randn(1, 1, 1024),
        torch.randn(num_layers * 2, 1, num_kv_heads, kv_len, head_dim),
        torch.tensor([kv_len]),
    )
    traced_decode = torch.jit.trace(decode_wrapper, decode_example)

    # Verify traced decode
    with torch.no_grad():
        test_embed = torch.randn(1, 1, 1024)
        test_kv = torch.randn(num_layers * 2, 1, num_kv_heads, kv_len, head_dim)
        traced_d_logits, traced_d_kv = traced_decode(test_embed, test_kv, torch.tensor([kv_len]))
        orig_d_logits, orig_d_kv = decode_wrapper(test_embed, test_kv, torch.tensor([kv_len]))
    print(f"   Traced vs original decode logits match: {torch.allclose(traced_d_logits, orig_d_logits, atol=1e-5)}")

    print("\n8. Converting decode to CoreML...")
    decode_inputs = [
        ct.TensorType(name="input_embed", shape=(1, 1, 1024), dtype=np.float32),
        ct.TensorType(
            name="kv_cache",
            shape=(
                num_layers * 2,
                1,
                num_kv_heads,
                ct.RangeDim(lower_bound=2, upper_bound=MAX_CP_KV_LEN),
                head_dim,
            ),
            dtype=np.float32,
        ),
        ct.TensorType(name="position", shape=(1,), dtype=np.int32),
    ]
    ml_decode = ct.convert(
        traced_decode,
        inputs=decode_inputs,
        outputs=[
            ct.TensorType(name="all_logits", dtype=np.float32),
            ct.TensorType(name="new_kv_cache", dtype=np.float32),
        ],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT32,
    )
    decode_path = "qwen3_tts_cp_decode.mlpackage"
    ml_decode.save(decode_path)
    print(f"   Saved to {decode_path}")

    # Export embedding tables
    print("\n9. Exporting code predictor embeddings...")
    all_cp_embeds = []
    for i in range(15):
        weight = cp_embeddings[i].weight.detach().numpy()
        all_cp_embeds.append(weight)
    all_cp_embeds = np.stack(all_cp_embeds, axis=0)  # [15, 2048, 1024]
    np.save("cp_embeddings.npy", all_cp_embeds)
    print(f"   Saved cp_embeddings.npy: {all_cp_embeds.shape}")
    print(f"   Size: {all_cp_embeds.nbytes / 1024 / 1024:.1f} MB")

    # Verify CoreML models
    print("\n10. Verifying CoreML models...")
    ml_pf = ct.models.MLModel(prefill_path)
    ml_dc = ct.models.MLModel(decode_path)
    loaded_embeds = np.load("cp_embeddings.npy")

    # Run full pipeline with CoreML
    pf_out = ml_pf.predict({
        "past_hidden": past_hidden.numpy().astype(np.float32),
        "cb0_token": cb0_token.numpy().astype(np.int32),
    })

    coreml_kv = pf_out["kv_cache"]
    coreml_logits = pf_out["all_logits"]
    print(f"   CoreML prefill logits shape: {coreml_logits.shape}")
    print(f"   CoreML prefill kv shape: {coreml_kv.shape}")

    coreml_cb1 = int(np.argmax(coreml_logits[0], axis=-1).item())
    coreml_tokens = [coreml_cb1]

    for step in range(1, 15):
        embed = loaded_embeds[step - 1][coreml_tokens[-1]]  # [1024]
        embed = embed.reshape(1, 1, 1024)

        dc_out = ml_dc.predict({
            "input_embed": embed.astype(np.float32),
            "kv_cache": coreml_kv.astype(np.float32),
            "position": np.array([step + 1], dtype=np.int32),
        })

        coreml_kv = dc_out["new_kv_cache"]
        logits = dc_out["all_logits"]
        token = int(np.argmax(logits[step], axis=-1).item())
        coreml_tokens.append(token)

    print(f"   CoreML tokens:  {coreml_tokens}")
    print(f"   PyTorch tokens: {our_tokens}")
    print(f"   Match: {coreml_tokens == our_tokens}")

    if coreml_tokens != our_tokens:
        mismatches = sum(1 for a, b in zip(coreml_tokens, our_tokens) if a != b)
        print(f"   {mismatches}/15 tokens differ")

    # Test with second input
    print("\n   Testing CoreML with different input...")
    pf_out_2 = ml_pf.predict({
        "past_hidden": past_hidden_2.numpy().astype(np.float32),
        "cb0_token": cb0_token_2.numpy().astype(np.int32),
    })
    coreml_kv_2 = pf_out_2["kv_cache"]
    coreml_logits_2 = pf_out_2["all_logits"]

    coreml_tokens_2 = [int(np.argmax(coreml_logits_2[0], axis=-1).item())]
    for step in range(1, 15):
        embed = loaded_embeds[step - 1][coreml_tokens_2[-1]].reshape(1, 1, 1024)
        dc_out = ml_dc.predict({
            "input_embed": embed.astype(np.float32),
            "kv_cache": coreml_kv_2.astype(np.float32),
            "position": np.array([step + 1], dtype=np.int32),
        })
        coreml_kv_2 = dc_out["new_kv_cache"]
        token = int(np.argmax(dc_out["all_logits"][step], axis=-1).item())
        coreml_tokens_2.append(token)

    print(f"   CoreML tokens 2:  {coreml_tokens_2}")
    print(f"   PyTorch tokens 2: {our_tokens_2}")
    print(f"   Match 2: {coreml_tokens_2 == our_tokens_2}")
    print(f"   Different from first: {coreml_tokens != coreml_tokens_2}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
