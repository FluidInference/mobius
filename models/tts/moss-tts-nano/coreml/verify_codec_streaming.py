"""Check that per-frame streaming codec decode matches full-utterance decode in PyTorch.

Streaming decode with chunk_duration = one 12.5 Hz frame (0.08 s) is the mode the
CoreML step decoder will replicate, so it has to match the non-streaming path first.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default=str(HERE / "build" / "ref_pytorch_audio_token_ids.npy"))
    parser.add_argument("--chunk-frames", type=int, default=1)
    args = parser.parse_args()

    codec = AutoModel.from_pretrained(CODEC_REPO, trust_remote_code=True).eval()
    tokens = torch.from_numpy(np.load(args.tokens)).long()  # (T, nq)
    codes = tokens.T.unsqueeze(1)  # (nq, 1, T)
    frame_sec = codec.downsample_rate / codec.sampling_rate
    print(f"frames={codes.shape[-1]} nq={codes.shape[0]} frame={frame_sec*1000:.0f}ms downsample={codec.downsample_rate}")

    with torch.no_grad():
        t0 = time.perf_counter()
        full = codec.decode(codes, return_dict=True).audio
        t_full = time.perf_counter() - t0
        t0 = time.perf_counter()
        stream = codec.decode(codes, return_dict=True, chunk_duration=frame_sec * args.chunk_frames).audio
        t_stream = time.perf_counter() - t0

    print(f"full: {tuple(full.shape)} in {t_full*1000:.0f}ms   stream: {tuple(stream.shape)} in {t_stream*1000:.0f}ms")
    n = min(full.shape[-1], stream.shape[-1])
    diff = (full[..., :n] - stream[..., :n]).abs()
    print(f"max|diff|={diff.max().item():.3e}  mean|diff|={diff.mean().item():.3e}  peak={full.abs().max().item():.3f}")
    samples_per_frame = full.shape[-1] // codes.shape[-1]
    print(f"samples/frame/channel={samples_per_frame}  channels={full.shape[1]}")


if __name__ == "__main__":
    main()
