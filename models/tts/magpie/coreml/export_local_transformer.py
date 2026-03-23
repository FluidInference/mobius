"""Export local transformer weights as individual .npy files.

Extracts weights from the NeMo model's local transformer and saves them
as individual numpy arrays in constants/local_transformer/.

Can also re-export from a previously extracted local_transformer.pt checkpoint.

Usage:
    # From NeMo model (requires NeMo)
    python export_local_transformer.py

    # From extracted .pt checkpoint
    python export_local_transformer.py --from-pt build/extracted/local_transformer.pt

    # Custom output directory
    python export_local_transformer.py --output-dir constants/local_transformer
"""
import argparse
import os

import numpy as np
import torch


def export_from_nemo(nemo_path=None):
    """Load NeMo model and return the three local transformer state dicts."""
    print("Loading MagpieTTS model...")
    from nemo.collections.tts.models import MagpieTTSModel
    if nemo_path:
        model = MagpieTTSModel.restore_from(nemo_path, map_location="cpu")
    else:
        model = MagpieTTSModel.from_pretrained("nvidia/magpie_tts_multilingual_357m")
    model.eval()

    transformer_sd = model.local_transformer.state_dict()
    in_proj_sd = model.local_transformer_in_projection.state_dict()
    out_proj_sds = [proj.state_dict() for proj in model.local_transformer_out_projections]

    return transformer_sd, in_proj_sd, out_proj_sds


def export_from_pt(pt_path):
    """Load from a previously extracted local_transformer.pt checkpoint."""
    print(f"Loading from {pt_path}...")
    data = torch.load(pt_path, map_location="cpu", weights_only=True)

    return (
        data["transformer_state_dict"],
        data["in_projection_state_dict"],
        data["out_projections_state_dict"],
    )


def save_weights(transformer_sd, in_proj_sd, out_proj_sds, output_dir):
    """Save individual weight tensors as .npy files."""
    os.makedirs(output_dir, exist_ok=True)

    def save(name, tensor):
        arr = tensor.detach().cpu().numpy()
        path = os.path.join(output_dir, f"{name}.npy")
        np.save(path, arr)
        print(f"  {name}: {arr.shape} ({arr.dtype})")

    # In-projection: decoder d_model (768) → local transformer d (256)
    save("in_proj_weight", in_proj_sd["weight"])
    save("in_proj_bias", in_proj_sd["bias"])

    # Transformer layer weights
    # The NeMo local transformer is a small 1-layer transformer with:
    #   - Learnable positional embedding
    #   - Pre-norm self-attention (single head)
    #   - Pre-norm FFN (Conv1d with kernel_size=1)
    #
    # State dict keys vary by NeMo version. We detect and map them.
    print("\nTransformer state dict keys:")
    for k, v in transformer_sd.items():
        print(f"  {k}: {v.shape}")

    # Build a mapping from state dict keys to output names.
    # Try common NeMo key patterns.
    key_map = find_key_mapping(transformer_sd)

    print("\nExporting transformer weights:")
    for out_name, sd_key in key_map.items():
        save(out_name, transformer_sd[sd_key])

    # Per-codebook output projections (8 codebooks)
    print(f"\nExporting {len(out_proj_sds)} output projections:")
    for i, sd in enumerate(out_proj_sds):
        save(f"out_proj_{i}_weight", sd["weight"])
        save(f"out_proj_{i}_bias", sd["bias"])


def find_key_mapping(sd):
    """Map NeMo state dict keys to our output file names.

    Uses explicit key mapping for known NeMo naming conventions.
    The NeMo MagpieTTS local transformer state dict has these keys:

        layers.0.norm_self.weight          → norm1_weight   (pre-SA LayerNorm)
        layers.0.self_attention.qkv_net.weight → sa_qkv_weight
        layers.0.self_attention.causal_mask    → (skipped, not a learned weight)
        layers.0.self_attention.o_net.weight   → sa_o_weight
        layers.0.norm_pos_ff.weight        → norm2_weight   (pre-FFN LayerNorm)
        layers.0.pos_ff.proj.conv.weight   → ffn_conv1_weight (up-projection)
        layers.0.pos_ff.o_net.conv.weight  → ffn_conv2_weight (down-projection)
        position_embeddings.weight         → pos_emb
    """
    # Explicit mapping table: (substring to match in key) → output name
    # Order matters: more specific substrings first to avoid ambiguity.
    explicit_rules = [
        ("position_embeddings",    "pos_emb"),
        ("norm_self",              "norm1_weight"),
        ("norm_pos_ff",            "norm2_weight"),
        ("qkv_net",               "sa_qkv_weight"),
        ("self_attention.o_net",   "sa_o_weight"),
        ("pos_ff.proj",           "ffn_conv1_weight"),
        ("pos_ff.o_net",          "ffn_conv2_weight"),
    ]

    mapping = {}
    skip_fragments = ["causal_mask"]  # Non-learned buffers

    for sd_key in sd.keys():
        # Skip non-weight buffers
        if any(frag in sd_key for frag in skip_fragments):
            continue

        matched = False
        for substring, out_name in explicit_rules:
            if substring in sd_key:
                if out_name in mapping:
                    raise RuntimeError(
                        f"Duplicate match: both '{mapping[out_name]}' and "
                        f"'{sd_key}' match rule '{substring}' → '{out_name}'"
                    )
                mapping[out_name] = sd_key
                matched = True
                break

        if not matched:
            print(f"  WARNING: No rule for key '{sd_key}' ({sd[sd_key].shape})")

    expected = {
        "pos_emb", "norm1_weight", "norm2_weight",
        "sa_qkv_weight", "sa_o_weight",
        "ffn_conv1_weight", "ffn_conv2_weight",
    }
    missing = expected - set(mapping.keys())
    if missing:
        print(f"\nState dict keys found:")
        for k, v in sd.items():
            print(f"  {k}: {v.shape}")
        raise RuntimeError(
            f"Could not map all weights. Missing: {missing}. "
            f"The NeMo key naming may have changed — update explicit_rules "
            f"in find_key_mapping()."
        )

    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Export local transformer weights as individual .npy files"
    )
    parser.add_argument("--nemo-path", type=str, default=None,
                        help="Path to .nemo checkpoint (default: download from HF)")
    parser.add_argument("--from-pt", type=str, default=None,
                        help="Load from extracted local_transformer.pt instead of NeMo")
    parser.add_argument("--output-dir", type=str, default="constants/local_transformer",
                        help="Output directory (default: constants/local_transformer)")
    args = parser.parse_args()

    if args.from_pt:
        transformer_sd, in_proj_sd, out_proj_sds = export_from_pt(args.from_pt)
    else:
        transformer_sd, in_proj_sd, out_proj_sds = export_from_nemo(args.nemo_path)

    save_weights(transformer_sd, in_proj_sd, out_proj_sds, args.output_dir)

    print(f"\nDone. Files saved to {args.output_dir}/")
    total = sum(
        os.path.getsize(os.path.join(args.output_dir, f))
        for f in os.listdir(args.output_dir) if f.endswith(".npy")
    )
    print(f"Total size: {total / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
