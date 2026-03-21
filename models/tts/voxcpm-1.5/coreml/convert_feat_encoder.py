"""Convert the VoxCPM 1.5 feat_encoder (VoxCPMLocEnc) to CoreML.

The feat_encoder converts audio latent patches into LM-compatible embeddings.
Input: [B*T, patch_size+1, 1024] (after projection and special token prepend)
Output: [B*T, 1024] (CLS token from position 0)

Architecture: Linear projection + 8-layer MiniCPM4 transformer (non-causal, GQA).

We trace the full VoxCPMLocEnc module which handles:
1. in_proj: [B, T, P, 64] -> [B, T, P, 1024]
2. Prepend special_token: [B, T, P+1, 1024]
3. Reshape to [B*T, P+1, 1024]
4. 8-layer transformer (non-causal)
5. Extract CLS token (position 0): [B*T, 1024]
6. Reshape to [B, T, 1024]
"""

import time
from typing import Tuple

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def patch_gqa_attention(model: nn.Module):
    """Replace scaled_dot_product_attention with GQA-expanded version.

    CoreML's coremltools doesn't support enable_gqa=True in SDPA,
    so we manually repeat KV heads to match query heads.
    """
    from voxcpm.modules.minicpm4.model import MiniCPMAttention, apply_rotary_pos_emb

    def patched_forward(
        self,
        hidden_states: torch.Tensor,
        position_emb: Tuple[torch.Tensor, torch.Tensor],
        is_causal: bool,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Expand KV heads to match query heads (GQA -> MHA)
        num_groups = self.num_heads // self.num_key_value_heads
        key_states = key_states.repeat_interleave(num_groups, dim=1)
        value_states = value_states.repeat_interleave(num_groups, dim=1)

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        past_key_value = (key_states, value_states)
        return attn_output, past_key_value

    for module in model.modules():
        if isinstance(module, MiniCPMAttention):
            import types
            module.forward = types.MethodType(patched_forward, module)


class TraceableLocEnc(nn.Module):
    """Wraps VoxCPMLocEnc for tracing with fixed batch*time dimension."""

    def __init__(self, loc_enc):
        super().__init__()
        self.in_proj = loc_enc.in_proj
        self.special_token = loc_enc.special_token
        self.encoder = loc_enc.encoder

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: [B, T, P, D] where P=4 (patch_size), D=64 (latent_dim)
        Returns:
            embeddings: [B, T, 1024]
        """
        b, t, p, d = feat.shape

        # Project to hidden dim
        x = self.in_proj(feat)  # [B, T, P, 1024]

        # Prepend special token
        special = self.special_token.expand(b, t, -1, -1)  # [B, T, 1, 1024]
        x = torch.cat([special, x], dim=2)  # [B, T, P+1, 1024]

        # Reshape for encoder
        x = x.reshape(b * t, p + 1, -1)  # [B*T, P+1, 1024]

        # Run encoder (non-causal attention) - returns (hidden_states, layer_caches)
        hidden, _ = self.encoder(inputs_embeds=x, is_causal=False)

        # Extract CLS token
        cls = hidden[:, 0, :]  # [B*T, 1024]

        # Reshape back
        return cls.reshape(b, t, -1)  # [B, T, 1024]


def main():
    print("=== Converting feat_encoder to CoreML ===\n")

    # Load model
    print("[1/5] Loading VoxCPM 1.5...")
    from voxcpm import VoxCPM
    model = VoxCPM.from_pretrained("openbmb/VoxCPM1.5", load_denoiser=False, optimize=False)
    tts = model.tts_model
    fe = tts.feat_encoder.float().cpu().eval()

    # Patch GQA attention (CoreML doesn't support enable_gqa=True in SDPA)
    patch_gqa_attention(fe)

    encoder = TraceableLocEnc(fe)
    encoder.eval()

    # Test with T=1 (single step, used during generation)
    # During prompt encoding T can be larger, but for CoreML we'll use T=1
    # and call it in a loop
    test_feat = torch.randn(1, 1, 4, 64)

    with torch.no_grad():
        pt_output = encoder(test_feat)
    print(f"  PyTorch: input {test_feat.shape} -> output {pt_output.shape}")

    # Trace
    print("\n[2/5] Tracing with torch.jit.trace...")
    with torch.no_grad():
        traced = torch.jit.trace(encoder, test_feat)

    # Verify trace
    with torch.no_grad():
        traced_output = traced(test_feat)
    max_diff = (pt_output - traced_output).abs().max().item()
    print(f"  Trace parity: max diff = {max_diff:.2e}")

    # Convert to CoreML
    print("\n[3/5] Converting to CoreML...")
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="feat", shape=(1, 1, 4, 64)),
        ],
        outputs=[
            ct.TensorType(name="embedding"),
        ],
        minimum_deployment_target=ct.target.macOS14,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        compute_precision=ct.precision.FLOAT16,
    )
    print(f"  Conversion took {time.time() - t0:.1f}s")

    # Save
    out_path = "feat_encoder.mlpackage"
    mlmodel.save(out_path)
    print(f"  Saved to {out_path}")

    # Validate CoreML output
    print("\n[4/5] Validating CoreML output...")
    coreml_pred = mlmodel.predict({"feat": test_feat.numpy()})
    coreml_output = coreml_pred["embedding"]
    diff = np.abs(pt_output.numpy() - coreml_output)
    corr = np.corrcoef(pt_output.numpy().flatten(), coreml_output.flatten())[0, 1]
    print(f"  Absolute: max={diff.max():.4e}, mean={diff.mean():.4e}")
    print(f"  Correlation: {corr:.6f}")

    if corr > 0.999:
        print("  PASS: Excellent correlation")
    elif corr > 0.99:
        print("  WARN: Good correlation")
    else:
        print("  FAIL: Poor correlation")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
