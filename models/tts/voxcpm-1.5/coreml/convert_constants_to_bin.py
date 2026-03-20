"""Convert VoxCPM 1.5 numpy constants to raw Float16 binary files for Swift.

Reads .npy files from constants/ and writes raw .bin files to constants_bin/.
Also downloads the tokenizer from openbmb/MiniCPM-1B-sft-bf16.

Output layout (for HuggingFace upload):
    constants_bin/
        config.json
        embed_tokens.bin          # [73448, 1024] Float16 (~143 MB)
        enc_to_lm_proj_w.bin      # [1024, 1024] Float16
        enc_to_lm_proj_b.bin      # [1024] Float16
        lm_to_dit_proj_w.bin      # [1024, 1024] Float16
        lm_to_dit_proj_b.bin      # [1024] Float16
        res_to_dit_proj_w.bin     # [1024, 1024] Float16
        res_to_dit_proj_b.bin     # [1024] Float16
        tokenizer.json
        tokenizer_config.json
"""

import json
import os
import shutil

import numpy as np


def main():
    src_dir = "constants"
    dst_dir = "constants_bin"
    os.makedirs(dst_dir, exist_ok=True)

    npy_files = [
        "embed_tokens",
        "enc_to_lm_proj_w",
        "enc_to_lm_proj_b",
        "lm_to_dit_proj_w",
        "lm_to_dit_proj_b",
        "res_to_dit_proj_w",
        "res_to_dit_proj_b",
    ]

    print("Converting .npy → .bin (Float16)...")
    for name in npy_files:
        src = os.path.join(src_dir, f"{name}.npy")
        dst = os.path.join(dst_dir, f"{name}.bin")
        arr = np.load(src)
        print(f"  {name}: {arr.shape} {arr.dtype} → float16", end="")
        arr_f16 = arr.astype(np.float16)
        arr_f16.tofile(dst)
        src_size = os.path.getsize(src)
        dst_size = os.path.getsize(dst)
        print(f"  ({src_size / 1e6:.1f} MB → {dst_size / 1e6:.1f} MB)")

    # Copy config.json
    shutil.copy2(os.path.join(src_dir, "config.json"), os.path.join(dst_dir, "config.json"))
    print("  Copied config.json")

    # Download tokenizer
    print("\nDownloading tokenizer from openbmb/MiniCPM-1B-sft-bf16...")
    from huggingface_hub import hf_hub_download

    for fname in ["tokenizer.json", "tokenizer_config.json"]:
        path = hf_hub_download("openbmb/MiniCPM-1B-sft-bf16", fname)
        shutil.copy2(path, os.path.join(dst_dir, fname))
        print(f"  Downloaded {fname}")

    # Summary
    print(f"\nOutput directory: {dst_dir}/")
    total = 0
    for f in sorted(os.listdir(dst_dir)):
        size = os.path.getsize(os.path.join(dst_dir, f))
        total += size
        print(f"  {f:40s} {size / 1e6:8.1f} MB")
    print(f"  {'TOTAL':40s} {total / 1e6:8.1f} MB")


if __name__ == "__main__":
    main()
