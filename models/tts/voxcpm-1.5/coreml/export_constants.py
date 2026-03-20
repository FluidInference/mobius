"""Export VoxCPM 1.5 constants for the CoreML generation pipeline.

Exports projection weights, embedding tables, and configuration values
that are needed by the Swift inference code but don't warrant their own
CoreML model.

Output files:
  constants/embed_tokens.npy         - [vocab_size, 1024] text embedding table
  constants/enc_to_lm_proj_w.npy     - [1024, 1024] weight
  constants/enc_to_lm_proj_b.npy     - [1024] bias
  constants/lm_to_dit_proj_w.npy     - [1024, 1024] weight
  constants/lm_to_dit_proj_b.npy     - [1024] bias
  constants/res_to_dit_proj_w.npy    - [1024, 1024] weight
  constants/res_to_dit_proj_b.npy    - [1024] bias
  constants/config.json              - generation config values
"""

import json
import os

import numpy as np
import torch


def main():
    print("=== Exporting VoxCPM 1.5 Constants ===\n")

    print("[1/3] Loading VoxCPM 1.5...")
    from voxcpm import VoxCPM
    model = VoxCPM.from_pretrained("openbmb/VoxCPM1.5", load_denoiser=False, optimize=False)
    tts = model.tts_model.float().cpu().eval()

    os.makedirs("constants", exist_ok=True)

    # Text embedding table
    print("\n[2/3] Exporting weights...")
    embed_tokens = tts.base_lm.embed_tokens.weight.detach().numpy()
    np.save("constants/embed_tokens.npy", embed_tokens)
    print(f"  embed_tokens: {embed_tokens.shape} ({embed_tokens.nbytes / 1024 / 1024:.1f} MB)")

    # Projection layers
    for name in ["enc_to_lm_proj", "lm_to_dit_proj", "res_to_dit_proj"]:
        proj = getattr(tts, name)
        w = proj.weight.detach().numpy()
        np.save(f"constants/{name}_w.npy", w)
        print(f"  {name}_w: {w.shape}")
        if proj.bias is not None:
            b = proj.bias.detach().numpy()
            np.save(f"constants/{name}_b.npy", b)
            print(f"  {name}_b: {b.shape}")

    # Config values
    print("\n[3/3] Exporting config...")
    config = tts.config
    lm_config = config.lm_config

    scale_emb = getattr(lm_config, "scale_emb", 1.0)
    if scale_emb is None:
        scale_emb = 1.0

    gen_config = {
        "hidden_size": lm_config.hidden_size,
        "vocab_size": lm_config.vocab_size,
        "scale_emb": float(scale_emb),
        "patch_size": config.patch_size,
        "feat_dim": config.feat_dim,
        "sample_rate": 44100,
        "hop_length": 1764,
        "token_rate_hz": 6.25,
        "max_seq_len": 512,
        "base_lm_layers": lm_config.num_hidden_layers,
        "residual_lm_layers": config.residual_lm_num_layers,
        "num_kv_heads": lm_config.num_key_value_heads,
        "head_dim": lm_config.hidden_size // lm_config.num_attention_heads,
        "dit_hidden_dim": config.dit_config.hidden_dim,
        "default_inference_timesteps": 10,
        "default_cfg_value": float(config.dit_config.cfm_config.inference_cfg_rate),
    }

    with open("constants/config.json", "w") as f:
        json.dump(gen_config, f, indent=2)
    print(f"  Config: {json.dumps(gen_config, indent=2)}")

    # Tokenizer info
    print("\n  Tokenizer info:")
    if hasattr(model, 'tokenizer') and model.tokenizer is not None:
        tokenizer = model.tokenizer
        print(f"    Vocab size: {tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 'unknown'}")
        print(f"    Type: {type(tokenizer).__name__}")
        gen_config["tokenizer_type"] = type(tokenizer).__name__
        if hasattr(tokenizer, "name_or_path"):
            gen_config["tokenizer_name"] = tokenizer.name_or_path
    else:
        print("    Tokenizer not loaded (MiniCPM4 tokenizer from HuggingFace)")
        gen_config["tokenizer_name"] = "openbmb/MiniCPM-1B-sft-bf16"

    with open("constants/config.json", "w") as f:
        json.dump(gen_config, f, indent=2)

    total_size = sum(
        os.path.getsize(os.path.join("constants", f))
        for f in os.listdir("constants")
    ) / 1024 / 1024
    print(f"\n  Total constants size: {total_size:.1f} MB")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
