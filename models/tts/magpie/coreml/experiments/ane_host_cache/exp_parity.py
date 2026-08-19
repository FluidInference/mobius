"""Prove the §6.3 host-owned-cache rewrite is mathematically equivalent to the current graph.

Same random weights copied into both formulations; drive an identical multi-step decode and
compare logits at each step. If they match, the rewrite is a pure restructuring (only CoreML
fp16 drift remains), so the 40% ANE speedup is free of correctness cost.
"""

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
COREML_DIR = HERE.parents[2]
sys.path.insert(0, str(COREML_DIR))
sys.path.insert(0, str(HERE.parent))
from traceable.traceable_decoder_step import TraceableDecoderStep  # noqa: E402
from exp_convert import (  # noqa: E402
    HostCacheDecoderStep, N_LAYERS, D_MODEL, D_FFN, SA_HEADS, D_HEAD,
    XA_HEADS, XA_D_MEM, MAX_SEQ, T_ENC, MASK_NEG,
)


def copy_weights(old: TraceableDecoderStep, new: HostCacheDecoderStep):
    for lo, ln in zip(old.layers, new.layers):
        ln.self_attn.qkv_proj.load_state_dict(lo.self_attn.qkv_proj.state_dict())
        ln.self_attn.o_proj.load_state_dict(lo.self_attn.o_proj.state_dict())
        ln.norm_sa.load_state_dict(lo.norm_sa.state_dict())
        ln.cross_attn.q_proj.load_state_dict(lo.cross_attn.q_proj.state_dict())
        ln.cross_attn.kv_proj.load_state_dict(lo.cross_attn.kv_proj.state_dict())
        ln.cross_attn.o_proj.load_state_dict(lo.cross_attn.o_proj.state_dict())
        ln.norm_xa_q.load_state_dict(lo.norm_xa_query.state_dict())
        ln.norm_xa_m.load_state_dict(lo.norm_xa_memory.state_dict())
        ln.norm_ff.load_state_dict(lo.norm_ff.state_dict())
        ln.ffn.conv1.load_state_dict(lo.ffn.conv1.state_dict())
        ln.ffn.conv2.load_state_dict(lo.ffn.conv2.state_dict())
    # both norm_out set to Identity in main() -> nothing to copy there
    new.final_proj.load_state_dict(old.final_proj.state_dict())


def main():
    torch.manual_seed(0)
    old = TraceableDecoderStep(
        n_layers=N_LAYERS, d_model=D_MODEL, d_ffn=D_FFN, sa_n_heads=SA_HEADS,
        xa_n_heads=XA_HEADS, xa_d_memory=XA_D_MEM, max_seq_len=MAX_SEQ,
    ).eval()
    old.norm_out = torch.nn.Identity()  # match default production path
    new = HostCacheDecoderStep().eval()
    new.norm_out = torch.nn.Identity()  # equivalence: both skip output norm
    copy_weights(old, new)

    enc = torch.randn(1, T_ENC, D_MODEL)
    enc_mask = torch.ones(1, T_ENC, dtype=torch.bool)
    mem_add = torch.zeros(1, 1, 1, T_ENC)

    # OLD state: per-layer full caches + positions
    ck = [torch.zeros(1, MAX_SEQ, SA_HEADS, D_HEAD) for _ in range(N_LAYERS)]
    cv = [torch.zeros(1, MAX_SEQ, SA_HEADS, D_HEAD) for _ in range(N_LAYERS)]
    # NEW state: host cache
    pk = [torch.zeros(1, MAX_SEQ, SA_HEADS, D_HEAD) for _ in range(N_LAYERS)]
    pv = [torch.zeros(1, MAX_SEQ, SA_HEADS, D_HEAD) for _ in range(N_LAYERS)]

    max_diff = 0.0
    with torch.no_grad():
        for pos in range(8):
            audio = torch.randn(1, 1, D_MODEL)

            # OLD
            old_args = (audio, enc, enc_mask)
            for i in range(N_LAYERS):
                old_args += (ck[i], cv[i], torch.tensor([float(pos)]))
            old_out = old(*old_args)
            old_logits = old_out[0]
            # feed back new caches (outs: logits, hidden, then ck,cv,p per layer)
            for i in range(N_LAYERS):
                ck[i] = old_out[2 + 3 * i]
                cv[i] = old_out[2 + 3 * i + 1]

            # NEW
            am = torch.full((1, 1, 1, MAX_SEQ + 1), MASK_NEG)
            am[..., :pos] = 0.0
            am[..., MAX_SEQ] = 0.0
            new_out = new(audio, enc, mem_add, am, *sum([[pk[i], pv[i]] for i in range(N_LAYERS)], []))
            new_logits = new_out[0]
            slices = new_out[1:]
            for i in range(N_LAYERS):
                pk[i][:, pos] = slices[2 * i][:, 0]
                pv[i][:, pos] = slices[2 * i + 1][:, 0]

            d = (old_logits - new_logits).abs().max().item()
            max_diff = max(max_diff, d)
            print(f"step {pos}: max|old-new| logits = {d:.3e}")

    print(f"\nMAX diff across steps: {max_diff:.3e}  ->  {'EQUIVALENT' if max_diff < 1e-3 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
