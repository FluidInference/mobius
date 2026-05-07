"""Truncate Magpie speaker embeddings T_ctx=110 -> T_ctx=N (default 64).

Reads `speaker_embeddings_raw.npy` (shape (num_speakers, T_ctx*D_model) flat),
reshapes to (num_speakers, T_ctx, D_model), takes the LAST N frames of each
speaker (most recent prosodic state — bias toward the speaker's voice as
they head into the new synthesis), then re-flattens and saves.

Also writes updated `speaker_info.json` and patches `constants.json` to
include `speaker_context_length` so the Swift loader (which falls back to
110 when missing) picks up the new T.

Usage:
    python truncate_speaker_embeddings.py \
        --constants-dir ~/.cache/fluidaudio/Models/magpie-tts/constants \
        --t-ctx 64
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def truncate(constants_dir: Path, t_ctx_new: int) -> None:
    info_path = constants_dir / "speaker_info.json"
    raw_path = constants_dir / "speaker_embeddings_raw.npy"
    cfg_path = constants_dir / "constants.json"

    info = json.loads(info_path.read_text())
    t_old = info["T"]
    d_model = info["D"]
    num_speakers = info["num_speakers"]

    if t_ctx_new >= t_old:
        raise ValueError(
            f"requested t_ctx={t_ctx_new} >= current T={t_old}; nothing to do"
        )

    raw = np.load(raw_path)  # (num_speakers, T_old * D)
    assert raw.shape == (num_speakers, t_old * d_model), \
        f"unexpected raw shape {raw.shape}, expected {(num_speakers, t_old * d_model)}"

    # Reshape -> (num_speakers, T_old, D)
    emb = raw.reshape(num_speakers, t_old, d_model)
    # Take LAST t_ctx_new frames
    emb_trunc = emb[:, -t_ctx_new:, :]
    raw_trunc = emb_trunc.reshape(num_speakers, t_ctx_new * d_model).astype(raw.dtype)
    np.save(raw_path, raw_trunc)
    print(f"speaker_embeddings_raw.npy: {raw.shape} -> {raw_trunc.shape}")

    # Update speaker_info
    info["T"] = t_ctx_new
    info["lens"] = [t_ctx_new] * num_speakers
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    print(f"speaker_info.json: T 110 -> {t_ctx_new}")

    # Patch constants.json with speaker_context_length
    cfg = json.loads(cfg_path.read_text())
    cfg["speaker_context_length"] = t_ctx_new
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"constants.json: speaker_context_length={t_ctx_new} added")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--constants-dir", required=True, type=Path)
    p.add_argument("--t-ctx", type=int, default=64)
    args = p.parse_args()
    truncate(args.constants_dir, args.t_ctx)


if __name__ == "__main__":
    main()
