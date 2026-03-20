"""Convert the VoxCPM 1.5 residual LM batch prefill to CoreML.

Processes an entire sequence through the 8-layer residual LM at once,
matching PyTorch's batch forward behavior.

Input:
  embeds: [1, 512, 1024] - FSQ-masked base LM output + audio feature embeddings (zero-padded)

Output:
  all_hidden: [1, 512, 1024] - normalized hidden states for all positions (Swift picks last real)
  k0..k7, v0..v7: [1, 2, 512, 64] - KV caches (fixed size)

Memory note: Runs in subprocess to avoid OOM on 16GB machines.
"""

import os
import sys
import time

MAX_SEQ_LEN = 512


def step1_trace_and_save():
    """Step 1: Load model, trace, save TorchScript. Exits to free memory."""
    from typing import Tuple

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def patch_gqa_attention_batch(model):
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

            # Explicit causal mask for CoreML compatibility
            seq_range = torch.arange(q_len, device=hidden_states.device)
            causal_mask = seq_range.unsqueeze(0) <= seq_range.unsqueeze(1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

            # Expand KV for GQA
            num_groups = self.num_heads // self.num_key_value_heads
            expanded_k = key_states.repeat_interleave(num_groups, dim=1)
            expanded_v = value_states.repeat_interleave(num_groups, dim=1)

            query_states = query_states.contiguous()
            expanded_k = expanded_k.contiguous()
            expanded_v = expanded_v.contiguous()

            attn_output = F.scaled_dot_product_attention(
                query_states, expanded_k, expanded_v,
                attn_mask=causal_mask.expand(bsz, self.num_heads, q_len, q_len),
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

    class TraceableResidualLMPrefill(nn.Module):
        """Batch prefill for residual LM — processes entire sequence at once.

        Input embeds are always zero-padded to MAX_SEQ_LEN in Swift.
        This ensures all outputs have fixed shapes.
        Outputs all_hidden for all positions — Swift picks the right one.
        """

        def __init__(self, tts_model):
            super().__init__()
            self.residual_lm = tts_model.residual_lm
            self.num_layers = len(self.residual_lm.layers)

        def forward(self, embeds: torch.Tensor):
            # embeds: [1, MAX_SEQ_LEN, 1024] (zero-padded beyond real tokens)

            position_ids = torch.arange(MAX_SEQ_LEN, dtype=torch.long, device=embeds.device)
            position_emb = self.residual_lm.rope_emb(position_ids)

            hidden = embeds
            kv_caches = []

            for layer in self.residual_lm.layers:
                hidden, kv = layer(hidden, position_emb, True)
                kv_caches.append(kv)

            all_hidden = self.residual_lm.norm(hidden)  # [1, 512, 1024]

            # Output all positions — Swift picks last real position
            # KV caches are already [1, 2, MAX_SEQ_LEN, 64] — fixed size
            outputs = [all_hidden]
            for k, v in kv_caches:
                outputs.extend([k, v])

            return tuple(outputs)

    print("[1/4] Loading VoxCPM 1.5...")
    from voxcpm import VoxCPM
    model = VoxCPM.from_pretrained("openbmb/VoxCPM1.5", load_denoiser=False, optimize=False)
    tts = model.tts_model.float().cpu().eval()
    del model

    patch_gqa_attention_batch(tts.residual_lm)

    prefill = TraceableResidualLMPrefill(tts)
    prefill.eval()
    del tts

    # Always use MAX_SEQ_LEN — input is zero-padded in Swift
    embeds = torch.randn(1, MAX_SEQ_LEN, 1024)

    print(f"[2/4] Testing PyTorch forward (S={MAX_SEQ_LEN})...")
    with torch.no_grad():
        outputs = prefill(embeds)
    all_hidden = outputs[0]
    print(f"  all_hidden: {all_hidden.shape}")
    print(f"  KV cache 0 shape: {outputs[1].shape}")
    print(f"  Total output tensors: {len(outputs)} (1 + 16 caches)")
    assert all_hidden.shape == (1, MAX_SEQ_LEN, 1024), f"all_hidden shape: {all_hidden.shape}"
    assert outputs[1].shape == (1, 2, MAX_SEQ_LEN, 64), f"KV shape: {outputs[1].shape}"

    print(f"\n[3/4] Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(prefill, [embeds])

    with torch.no_grad():
        traced_outputs = traced(embeds)
    max_diff = (outputs[0] - traced_outputs[0]).abs().max().item()
    print(f"  Trace parity: max diff = {max_diff:.2e}")

    ts_path = "residual_lm_prefill_traced.pt"
    traced.save(ts_path)
    print(f"\n[4/4] Saved TorchScript to {ts_path}")


def step2_convert_coreml():
    """Step 2: Load TorchScript, convert to CoreML, save."""
    import gc
    import coremltools as ct

    ts_path = "residual_lm_prefill_traced.pt"
    print(f"\n[1/3] Converting TorchScript to CoreML...")
    t0 = time.time()

    ct_inputs = [
        ct.TensorType(name="embeds", shape=(1, MAX_SEQ_LEN, 1024)),
    ]

    ct_outputs = [ct.TensorType(name="all_hidden")]
    for i in range(8):
        ct_outputs.append(ct.TensorType(name=f"k{i}"))
        ct_outputs.append(ct.TensorType(name=f"v{i}"))

    mlmodel = ct.convert(
        ts_path,
        inputs=ct_inputs,
        outputs=ct_outputs,
        minimum_deployment_target=ct.target.macOS14,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
    )
    elapsed = time.time() - t0
    print(f"  Conversion took {elapsed:.1f}s")

    gc.collect()

    print(f"[2/3] Saving...")
    out_path = "residual_lm_prefill.mlpackage"
    mlmodel.save(out_path)
    print(f"  Saved to {out_path}")

    del mlmodel
    gc.collect()

    print(f"[3/3] Cleaning up {ts_path}")
    os.remove(ts_path)


def step3_quantize_int8():
    """Step 3: Load FP16 CoreML, quantize to INT8, save."""
    import gc
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
    )

    fp16_path = "residual_lm_prefill.mlpackage"
    print(f"\n[1/2] Loading FP16 model from {fp16_path}...")
    mlmodel = ct.models.MLModel(fp16_path, skip_model_load=True)

    print("[2/2] INT8 quantization...")
    op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
    config = OptimizationConfig(global_config=op_config)
    mlmodel_int8 = linear_quantize_weights(mlmodel, config=config)

    del mlmodel
    gc.collect()

    int8_path = "residual_lm_prefill_int8.mlpackage"
    mlmodel_int8.save(int8_path)
    print(f"  Saved INT8 to {int8_path}")


def main():
    import subprocess

    print("=== Converting Residual LM Prefill to CoreML ===")
    print("(Running in separate subprocesses to manage memory)\n")

    script = os.path.abspath(__file__)

    print("--- Step 1: Trace PyTorch Model ---")
    r = subprocess.run([sys.executable, script, "--step1"], check=False)
    if r.returncode != 0:
        print(f"Step 1 failed with code {r.returncode}")
        sys.exit(1)

    print("\n--- Step 2: Convert to CoreML ---")
    r = subprocess.run([sys.executable, script, "--step2"], check=False)
    if r.returncode != 0:
        print(f"Step 2 failed with code {r.returncode}")
        sys.exit(1)

    print("\n--- Step 3: INT8 Quantize ---")
    r = subprocess.run([sys.executable, script, "--step3"], check=False)
    if r.returncode != 0:
        print(f"Step 3 failed with code {r.returncode}")
        sys.exit(1)

    print("\n=== Done ===")


if __name__ == "__main__":
    if "--step1" in sys.argv:
        step1_trace_and_save()
    elif "--step2" in sys.argv:
        step2_convert_coreml()
    elif "--step3" in sys.argv:
        step3_quantize_int8()
    else:
        main()
