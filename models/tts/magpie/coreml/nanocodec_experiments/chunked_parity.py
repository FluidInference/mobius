"""Chunked-inference audio parity for nanocodec.

Compares three renderings of the same 256-token random codec sequence:
  1. PyTorch sin² Snake reference (fp32, full T=256 forward).
  2. CoreML Taylor5Clipped T=256 single call (current production build).
  3. CoreML Taylor5Clipped T=t_in stitched with stride `stride`.

Reports SNR(2 vs 1), SNR(3 vs 1), SNR(3 vs 2). The third is the one we care
about for chunking correctness — it isolates the chunking artifact from the
Taylor5Clipped Snake approximation error.

Run (default T_in=16, stride=8, overlap=8):
  uv run --extra nemo python nanocodec_experiments/chunked_parity.py

Sweep operating points:
  uv run --extra nemo python nanocodec_experiments/chunked_parity.py --t-in 24 --stride 8

`overlap` is implied as `t_in - stride`. Each call left-pads `overlap`
codec frames with zeros at sequence start, then discards
`overlap * 1024` leading audio samples per call so the kept tails are
back-to-back stride*1024-sample slices.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import coremltools as ct


HERE = Path(__file__).resolve().parent
COREML_DIR = HERE.parent


def _snr(ref: np.ndarray, hyp: np.ndarray) -> tuple[float, float, float]:
    n = min(ref.shape[-1], hyp.shape[-1])
    r = ref[..., :n].reshape(-1).astype(np.float64)
    h = hyp[..., :n].reshape(-1).astype(np.float64)
    err = r - h
    rms_ref = float(np.sqrt(np.mean(r * r)) + 1e-12)
    rms_err = float(np.sqrt(np.mean(err * err)) + 1e-12)
    snr_db = 20.0 * np.log10(rms_ref / rms_err)
    return snr_db, float(np.max(np.abs(err))), float(np.mean(np.abs(err)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-in", type=int, default=16,
                        help="Fixed model input token count (default 16).")
    parser.add_argument("--stride", type=int, default=8,
                        help="Output frames produced per call (default 8). "
                             "Implies left-overlap = t_in - stride.")
    args = parser.parse_args()
    t_in = args.t_in
    stride = args.stride
    overlap = t_in - stride

    t256_path = COREML_DIR / "build" / "nanocodec_decoder_taylor5_clipped.mlpackage"
    t_chunk_path = COREML_DIR / "build" / f"nanocodec_decoder_t{t_in}.mlpackage"
    if not t256_path.exists():
        print(f"ERROR: missing {t256_path}", file=sys.stderr)
        return 1
    if not t_chunk_path.exists():
        print(f"ERROR: missing {t_chunk_path} (run convert_nanocodec.py "
              f"--max-frames {t_in} --output build/nanocodec_decoder_t{t_in}.mlpackage)",
              file=sys.stderr)
        return 1

    sys.path.insert(0, str(COREML_DIR))
    from nemo.collections.tts.models import MagpieTTSModel

    print("Loading nvidia/magpie_tts_multilingual_357m...", file=sys.stderr)
    model = MagpieTTSModel.from_pretrained("nvidia/magpie_tts_multilingual_357m")
    model.eval()
    codec = model._codec_model
    codec.eval()

    num_codebooks = 8
    T = 256
    samples_per_frame = 1024
    codebook_size = 2016

    rng = np.random.default_rng(seed=42)
    tokens_np = rng.integers(0, codebook_size, size=(1, num_codebooks, T)).astype(np.int32)
    tokens_t = torch.from_numpy(tokens_np.astype(np.int64))

    # 1. PyTorch reference (full T=256, sin² Snake)
    print("[1/3] PyTorch sin² reference...", file=sys.stderr)
    with torch.no_grad():
        ref_audio_t = codec.decode(
            tokens=tokens_t, tokens_len=torch.tensor([T], dtype=torch.long)
        )
    if isinstance(ref_audio_t, tuple):
        ref_audio_t = ref_audio_t[0]
    pytorch_audio = ref_audio_t.detach().cpu().numpy()
    if pytorch_audio.ndim == 3:
        pytorch_audio = pytorch_audio[:, 0]

    # 2. CoreML T=256 (Taylor5Clipped, full single call)
    print("[2/3] CoreML T=256 single call...", file=sys.stderr)
    m_t256 = ct.models.MLModel(str(t256_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    out256 = m_t256.predict({"tokens": tokens_np})
    audio_t256 = out256["audio"]

    # 3. CoreML T=t_in stitched (sequential calls; left-context = overlap)
    print(
        f"[3/3] CoreML T={t_in} stitched (stride={stride}, overlap={overlap} "
        f"frame{'s' if overlap != 1 else ''})...",
        file=sys.stderr,
    )
    m_chunk = ct.models.MLModel(str(t_chunk_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    chunks_audio = []
    for start in range(0, T, stride):
        ctx_start = start - overlap
        # Build a (1, C, t_in) input. Left-pad with zero codes if ctx_start<0.
        if ctx_start < 0:
            pad = -ctx_start
            piece = np.concatenate(
                [
                    np.zeros((1, num_codebooks, pad), dtype=np.int32),
                    tokens_np[:, :, 0: start + stride],
                ],
                axis=2,
            )
        else:
            piece = tokens_np[:, :, ctx_start: start + stride]
        # Last window may extend past T; right-pad with zeros if needed.
        if piece.shape[2] < t_in:
            pad = t_in - piece.shape[2]
            piece = np.concatenate(
                [piece, np.zeros((1, num_codebooks, pad), dtype=np.int32)],
                axis=2,
            )
        assert piece.shape[2] == t_in, (
            f"piece must be t_in={t_in} frames wide, got {piece.shape}"
        )
        out = m_chunk.predict({"tokens": piece})["audio"]
        # The model emits t_in*samples_per_frame samples; discard the leading
        # `overlap*samples_per_frame` left-context samples and keep the rest.
        keep_start = overlap * samples_per_frame
        keep_end = keep_start + stride * samples_per_frame
        chunks_audio.append(out[..., keep_start:keep_end])
    audio_t8 = np.concatenate(chunks_audio, axis=-1)
    # Trim to the exact T*samples_per_frame in case the last window was right-padded.
    audio_t8 = audio_t8[..., : T * samples_per_frame]

    # Crop all to the shortest length.
    n = min(pytorch_audio.shape[-1], audio_t256.shape[-1], audio_t8.shape[-1])
    pytorch_audio = pytorch_audio[..., :n]
    audio_t256 = audio_t256[..., :n]
    audio_t8 = audio_t8[..., :n]

    print()
    print(
        f"=== Audio parity (seed=42, T={T}, t_in={t_in}, stride={stride}, "
        f"overlap={overlap}) ==="
    )
    print(f"samples compared : {n}")
    print()

    tag = f"T={t_in:<3d} stitched"
    snr_a, max_a, mean_a = _snr(pytorch_audio, audio_t256)
    print(f"[A] PyTorch sin²        vs CoreML T=256 Taylor5: "
          f"SNR={snr_a:6.2f} dB  max_abs={max_a:.3e}  mean_abs={mean_a:.3e}")

    snr_b, max_b, mean_b = _snr(pytorch_audio, audio_t8)
    print(f"[B] PyTorch sin²        vs CoreML {tag}: "
          f"SNR={snr_b:6.2f} dB  max_abs={max_b:.3e}  mean_abs={mean_b:.3e}")

    snr_c, max_c, mean_c = _snr(audio_t256, audio_t8)
    print(f"[C] CoreML T=256 single vs CoreML {tag}: "
          f"SNR={snr_c:6.2f} dB  max_abs={max_c:.3e}  mean_abs={mean_c:.3e}")

    # Save the three waveforms for inspection.
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    np.save(out_dir / "chunked_pytorch.npy", pytorch_audio[0].astype(np.float32))
    np.save(out_dir / "chunked_t256.npy", audio_t256[0].astype(np.float32))
    np.save(
        out_dir / f"chunked_t{t_in}_s{stride}.npy",
        audio_t8[0].astype(np.float32),
    )
    print(f"\nWaveforms saved under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
