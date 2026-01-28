"""One-time export of model constants as .npy files.

Exports:
- bos_emb.npy         [32]        BOS embedding
- emb_mean.npy        [32]        Latent normalization mean
- emb_std.npy         [32]        Latent normalization std
- quantizer_weight.npy [512, 32]  Quantizer output projection
- text_embed_table.npy [4001, 1024] Text token embedding table

Run once, then the generation script can load these without PyTorch.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "constants")


def export():
    import torch
    from pocket_tts import TTSModel

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading model...")
    model = TTSModel.load_model(lsd_decode_steps=8)
    model.eval()

    # FlowLM constants
    flow_lm = model.flow_lm

    bos_emb = flow_lm.bos_emb.data.numpy().astype(np.float32)
    print(f"bos_emb: {bos_emb.shape}")
    np.save(os.path.join(OUTPUT_DIR, "bos_emb.npy"), bos_emb)

    emb_mean = flow_lm.emb_mean.numpy().astype(np.float32)
    print(f"emb_mean: {emb_mean.shape}")
    np.save(os.path.join(OUTPUT_DIR, "emb_mean.npy"), emb_mean)

    emb_std = flow_lm.emb_std.numpy().astype(np.float32)
    print(f"emb_std: {emb_std.shape}")
    np.save(os.path.join(OUTPUT_DIR, "emb_std.npy"), emb_std)

    # Quantizer weight
    q_weight = model.mimi.quantizer.output_proj.weight.detach().numpy().astype(np.float32)
    print(f"quantizer_weight: {q_weight.shape}")
    np.save(os.path.join(OUTPUT_DIR, "quantizer_weight.npy"), q_weight)

    # Text embedding table
    embed_table = flow_lm.conditioner.embed.weight.detach().numpy().astype(np.float32)
    print(f"text_embed_table: {embed_table.shape}")
    np.save(os.path.join(OUTPUT_DIR, "text_embed_table.npy"), embed_table)

    # Also export the Mimi decoder init state shapes (for reference)
    from traceable_decoder import TraceableMimiDecoder
    decoder = TraceableMimiDecoder.from_mimi(model.mimi)
    mimi_state = decoder.init_state(batch_size=1)
    mimi_state_np = {}
    for k, v in mimi_state.items():
        arr = v.numpy().astype(np.float32)
        mimi_state_np[k] = arr
    np.savez(os.path.join(OUTPUT_DIR, "mimi_init_state.npz"), **mimi_state_np)
    print(f"mimi_init_state: {len(mimi_state_np)} tensors")

    print(f"\nAll constants saved to {OUTPUT_DIR}/")

    # Print sizes
    total = 0
    for f in os.listdir(OUTPUT_DIR):
        path = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(path)
        total += size
        print(f"  {f}: {size / 1024:.1f} KB")
    print(f"  TOTAL: {total / 1024:.1f} KB")


if __name__ == "__main__":
    export()
