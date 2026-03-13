"""Export constants from MagpieTTS model for CoreML inference.

Extracts and saves:
- Audio embedding tables (numpy arrays)
- Speaker context embeddings (numpy arrays)
- Text embedding table (numpy array)
- Model configuration (JSON)
- Tokenizer files

Usage:
    python export_constants.py [--nemo-path /path/to/model.nemo]
"""
import argparse
import json
import os
import shutil

import numpy as np
import torch


def export_constants(nemo_path=None, output_dir="constants"):
    # Load model
    print("Loading MagpieTTS model...")
    from nemo.collections.tts.models import MagpieTTSModel
    if nemo_path:
        model = MagpieTTSModel.restore_from(nemo_path, map_location="cpu")
    else:
        model = MagpieTTSModel.from_pretrained("nvidia/magpie_tts_multilingual_357m")
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    cfg = model.cfg

    # 1. Audio embedding tables
    print("Exporting audio embedding tables...")
    for i, emb in enumerate(model.audio_embeddings):
        path = os.path.join(output_dir, f"audio_embedding_{i}.npy")
        np.save(path, emb.weight.detach().cpu().numpy())
        print(f"  audio_embedding_{i}: {emb.weight.shape}")

    # 2. Text embedding table
    print("Exporting text embedding table...")
    if hasattr(model, "text_embedding"):
        path = os.path.join(output_dir, "text_embedding.npy")
        np.save(path, model.text_embedding.weight.detach().cpu().numpy())
        print(f"  text_embedding: {model.text_embedding.weight.shape}")

    # 3. Speaker context embeddings
    if model.has_baked_context_embedding and model.baked_context_embedding is not None:
        print("Exporting speaker embeddings...")
        emb_weight = model.baked_context_embedding.weight.detach().cpu().numpy()
        T = int(model._baked_embedding_T.item()) if model._baked_embedding_T is not None else None
        D = int(model._baked_embedding_D.item()) if model._baked_embedding_D is not None else None
        lens = model.baked_context_embedding_len.detach().cpu().numpy() if model.baked_context_embedding_len is not None else None

        np.save(os.path.join(output_dir, "speaker_embeddings_raw.npy"), emb_weight)
        print(f"  speaker_embeddings_raw: {emb_weight.shape}")

        # Reshape each speaker's embedding to (T, D) and save individually
        num_speakers = model.num_baked_speakers
        for spk_idx in range(num_speakers):
            spk_emb = emb_weight[spk_idx]  # Flat (T*D,)
            if T and D:
                spk_emb = spk_emb.reshape(T, D)
            spk_len = int(lens[spk_idx]) if lens is not None else T
            spk_emb = spk_emb[:spk_len]
            np.save(os.path.join(output_dir, f"speaker_{spk_idx}.npy"), spk_emb)
            print(f"  speaker_{spk_idx}: {spk_emb.shape}")

        speaker_info = {
            "num_speakers": num_speakers,
            "T": T,
            "D": D,
            "lens": lens.tolist() if lens is not None else None,
            "names": {
                "0": "John",
                "1": "Sofia",
                "2": "Aria",
                "3": "Jason",
                "4": "Leo",
            },
        }
        with open(os.path.join(output_dir, "speaker_info.json"), "w") as f:
            json.dump(speaker_info, f, indent=2)

    # 4. Model constants
    print("Exporting model constants...")
    constants = {
        "model_type": model.model_type,
        "embedding_dim": int(cfg.embedding_dim),
        "frame_stacking_factor": model.frame_stacking_factor,
        "num_audio_codebooks": model.num_audio_codebooks,
        "codebook_size": model.codebook_size,
        "num_all_tokens_per_codebook": model.num_all_tokens_per_codebook,
        "sample_rate": model.sample_rate,
        "output_sample_rate": model.output_sample_rate,
        "codec_samples_per_frame": model.codec_model_samples_per_frame,
        "special_tokens": {
            "audio_bos_id": model.audio_bos_id,
            "audio_eos_id": model.audio_eos_id,
            "context_audio_bos_id": model.context_audio_bos_id,
            "context_audio_eos_id": model.context_audio_eos_id,
            "mask_token_id": model.mask_token_id,
            "text_bos_id": model.bos_id,
            "text_eos_id": model.eos_id,
        },
        "encoder": {
            "n_layers": int(dict(cfg.encoder)["n_layers"]),
            "d_model": int(dict(cfg.encoder)["d_model"]),
            "d_ffn": int(dict(cfg.encoder)["d_ffn"]),
            "sa_n_heads": int(dict(cfg.encoder)["sa_n_heads"]),
        },
        "decoder": {
            "n_layers": int(dict(cfg.decoder)["n_layers"]),
            "d_model": int(dict(cfg.decoder)["d_model"]),
            "d_ffn": int(dict(cfg.decoder)["d_ffn"]),
            "sa_n_heads": int(dict(cfg.decoder)["sa_n_heads"]),
        },
    }

    # Add inference parameters
    if hasattr(model, "inference_parameters") and model.inference_parameters is not None:
        ip = model.inference_parameters
        constants["inference"] = {
            "temperature": float(ip.temperature),
            "topk": int(ip.topk),
            "cfg_scale": float(ip.cfg_scale),
            "max_decoder_steps": int(ip.max_decoder_steps),
            "min_generated_frames": int(ip.min_generated_frames),
        }

    with open(os.path.join(output_dir, "constants.json"), "w") as f:
        json.dump(constants, f, indent=2, default=str)

    # 5. Export tokenizer info
    print("Exporting tokenizer info...")
    tok = model.tokenizer
    tokenizer_info = {
        "type": type(tok).__name__,
    }
    if hasattr(tok, "num_tokens"):
        tokenizer_info["num_tokens"] = tok.num_tokens
    if hasattr(tok, "tokenizer_names"):
        tokenizer_info["tokenizer_names"] = tok.tokenizer_names
    if hasattr(tok, "num_tokens_per_tokenizer"):
        tokenizer_info["num_tokens_per_tokenizer"] = {
            k: v for k, v in tok.num_tokens_per_tokenizer.items()
        }
    if hasattr(tok, "tokenizers"):
        tokenizer_info["available_tokenizers"] = list(tok.tokenizers.keys())
    with open(os.path.join(output_dir, "tokenizer_info.json"), "w") as f:
        json.dump(tokenizer_info, f, indent=2, default=str)

    print(f"\nAll constants saved to {output_dir}/")
    for f_name in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f_name))
        unit = "KB" if size < 1024 * 1024 else "MB"
        size_val = size / 1024 if unit == "KB" else size / 1024 / 1024
        print(f"  {f_name}: {size_val:.1f} {unit}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="constants")
    args = parser.parse_args()
    export_constants(args.nemo_path, args.output_dir)
