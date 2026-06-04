#!/usr/bin/env python3
"""
Encoder logit parity check: fp32 PyTorch vs fp16 CoreML.

Runs the prompt-aware streaming encoder side-by-side on the same FLEURS
audio chunks and reports per-chunk cosine similarity and max-abs-diff of
the `encoded` output (and of the output caches, since those compound
into the next chunk).

The PyTorch path uses NeMo's checkpoint loaded fp32 on CPU wrapped with
`EncoderStreamingWithPostPrompt` — the same module that was traced for
CoreML export. The CoreML path uses the saved `build_fp16/encoder.mlpackage`.

Mel features come from the CoreML preprocessor on both paths so the
parity check isolates the encoder + prompt_kernel; preprocessor drift
(if any) is not in scope here.

Usage:
    cd conversion_scripts && .venv/bin/python ../parity_encoder.py \
        --nemo-path /path/to/multilingual.nemo \
        --model-dir ../build_fp16 \
        --langs en_us,cmn_hans_cn,ja_jp \
        --chunks 5 \
        --target-lang auto
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR / "conversion_scripts"))

# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    na = np.linalg.norm(af)
    nb = np.linalg.norm(bf)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(af, bf) / (na * nb))


def diffs(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    """Return (max_abs, mean_abs, rel_l2) for a vs b."""
    d = (a.astype(np.float64) - b.astype(np.float64))
    max_abs = float(np.max(np.abs(d)))
    mean_abs = float(np.mean(np.abs(d)))
    denom = float(np.linalg.norm(a.astype(np.float64)))
    rel_l2 = float(np.linalg.norm(d) / denom) if denom > 0 else float("nan")
    return max_abs, mean_abs, rel_l2


# ---------------------------------------------------------------------------
# FLEURS loader (HF cache only — same loader as benchmark_fleurs.py)
# ---------------------------------------------------------------------------


def load_first_fleurs_utt(lang: str) -> Tuple[str, np.ndarray, int, str]:
    import soundfile as sf
    from datasets import load_dataset, Audio

    ds = load_dataset("google/fleurs", lang, split="test", streaming=False)
    ds = ds.cast_column("audio", Audio(decode=False))
    ex = next(iter(ds))
    audio_bytes = ex["audio"]["bytes"]
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return str(ex.get("id", "0")), audio, int(sr), ex["transcription"]


# ---------------------------------------------------------------------------
# Parity driver
# ---------------------------------------------------------------------------


def run_parity(
    nemo_path: str,
    model_dir: str,
    langs: List[str],
    target_lang: str,
    num_chunks: int,
) -> dict:
    import torch
    import coremltools as ct
    import nemo.collections.asr as nemo_asr

    from multilingual_components import (
        EncoderStreamingWithPostPrompt,
        NUM_PROMPTS,
    )

    model_dir_p = Path(model_dir)
    metadata = json.loads((model_dir_p / "metadata.json").read_text())
    chunk_mel_frames = int(metadata["chunk_mel_frames"])
    pre_encode_cache = int(metadata["pre_encode_cache"])
    total_mel_frames = int(metadata["total_mel_frames"])
    sample_rate = int(metadata["sample_rate"])
    mel_features = int(metadata.get("mel_features", 128))
    cache_channel_shape = tuple(metadata["cache_channel_shape"])
    cache_time_shape = tuple(metadata["cache_time_shape"])
    prompt_dictionary = dict(metadata["prompt_dictionary"])
    default_prompt_id = int(metadata.get("default_prompt_id", 101))
    chunk_samples = int(chunk_mel_frames * 0.01 * sample_rate)

    def resolve_prompt_id(tl: str) -> int:
        if tl in prompt_dictionary:
            return int(prompt_dictionary[tl])
        if len(tl) == 2:
            for k, v in prompt_dictionary.items():
                if k.lower().startswith(tl.lower() + "-"):
                    return int(v)
        return default_prompt_id

    print(f"Loading NeMo PyTorch model: {nemo_path}")
    t0 = time.time()
    pt_model = nemo_asr.models.ASRModel.restore_from(
        restore_path=nemo_path, map_location="cpu"
    )
    pt_model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s, class={type(pt_model).__name__}")

    att_context = metadata.get("att_context_size", [56, 0])
    pt_model.encoder.setup_streaming_params(att_context_size=list(att_context))

    pt_wrap = EncoderStreamingWithPostPrompt(
        pt_model.encoder.eval(),
        pt_model.prompt_kernel.eval(),
        num_prompts=NUM_PROMPTS,
    ).eval()

    print("Loading CoreML preprocessor + encoder (fp16)...")
    preproc = ct.models.MLModel(str(model_dir_p / "preprocessor.mlpackage"))
    enc_cml = ct.models.MLModel(str(model_dir_p / "encoder.mlpackage"))

    all_chunk_stats: List[dict] = []

    for lang in langs:
        print("=" * 72)
        print(f"[{lang}] target_lang={target_lang}")
        utt_id, audio, sr, ref = load_first_fleurs_utt(lang)
        assert sr == sample_rate, f"got sr={sr}"
        print(f"  utt={utt_id}  dur={len(audio)/sr:.2f}s")
        print(f"  ref: {ref[:80]}{'...' if len(ref)>80 else ''}")

        prompt_id = resolve_prompt_id(target_lang)
        prompt_id_pt = __import__("torch").tensor([prompt_id], dtype=__import__("torch").int32)
        prompt_id_np = np.array([prompt_id], dtype=np.int32)

        # Independent cache state for each path — drift compounds.
        cc_pt = np.zeros(cache_channel_shape, dtype=np.float32)
        ct_pt = np.zeros(cache_time_shape, dtype=np.float32)
        cl_pt = np.array([0], dtype=np.int32)
        cc_cml = np.zeros(cache_channel_shape, dtype=np.float32)
        ct_cml = np.zeros(cache_time_shape, dtype=np.float32)
        cl_cml = np.array([0], dtype=np.int32)

        mel_cache = None
        offset = 0
        total_samples = len(audio)

        for chunk_idx in range(num_chunks):
            if offset >= total_samples:
                break
            end = min(offset + chunk_samples, total_samples)
            audio_chunk = audio[offset:end]
            if len(audio_chunk) < chunk_samples:
                audio_chunk = np.pad(audio_chunk, (0, chunk_samples - len(audio_chunk)))
            audio_chunk2 = audio_chunk.reshape(1, -1).astype(np.float32)
            audio_len = np.array([audio_chunk2.shape[1]], dtype=np.int32)

            # === preprocessor (CoreML — same mel for both encoders) ===
            pre_out = preproc.predict({
                "audio": audio_chunk2,
                "audio_length": audio_len,
            })
            chunk_mel = pre_out["mel"]  # (1, 128, ~112)

            if mel_cache is not None:
                input_mel = np.concatenate([mel_cache, chunk_mel], axis=2)
            else:
                input_mel = np.pad(
                    chunk_mel, ((0, 0), (0, 0), (pre_encode_cache, 0)), mode="constant"
                )
            cur = input_mel.shape[2]
            if cur < total_mel_frames:
                input_mel = np.pad(
                    input_mel, ((0, 0), (0, 0), (0, total_mel_frames - cur)), mode="constant"
                )
            elif cur > total_mel_frames:
                input_mel = input_mel[:, :, :total_mel_frames]
            mel_cache = (
                chunk_mel[:, :, -pre_encode_cache:]
                if chunk_mel.shape[2] >= pre_encode_cache
                else chunk_mel
            )

            mel_len_np = np.array([total_mel_frames], dtype=np.int32)

            # === CoreML encoder (fp16) ===
            cml_out = enc_cml.predict({
                "mel": input_mel.astype(np.float32),
                "mel_length": mel_len_np,
                "cache_channel": cc_cml,
                "cache_time": ct_cml,
                "cache_len": cl_cml,
                "prompt_id": prompt_id_np,
            })
            enc_cml_out = cml_out["encoded"]
            cc_cml = cml_out["cache_channel_out"]
            ct_cml = cml_out["cache_time_out"]
            cl_cml = cml_out["cache_len_out"]

            # === PyTorch encoder (fp32) ===
            import torch as _torch
            with _torch.inference_mode():
                pt_out = pt_wrap(
                    _torch.from_numpy(input_mel.astype(np.float32)),
                    _torch.from_numpy(mel_len_np),
                    _torch.from_numpy(cc_pt),
                    _torch.from_numpy(ct_pt),
                    _torch.from_numpy(cl_pt),
                    prompt_id_pt,
                )
            enc_pt_out = pt_out[0].cpu().numpy()
            cc_pt = pt_out[2].cpu().numpy()
            ct_pt = pt_out[3].cpu().numpy()
            cl_pt = pt_out[4].cpu().numpy().astype(np.int32)

            cos = cosine_sim(enc_pt_out, enc_cml_out)
            mx, mn, rl2 = diffs(enc_pt_out, enc_cml_out)
            cc_cos = cosine_sim(cc_pt, cc_cml)
            ct_cos = cosine_sim(ct_pt, ct_cml)
            cc_mx, _, cc_rl2 = diffs(cc_pt, cc_cml)
            ct_mx, _, ct_rl2 = diffs(ct_pt, ct_cml)

            stat = {
                "lang": lang,
                "utt_id": utt_id,
                "chunk": chunk_idx,
                "encoded_shape": list(enc_pt_out.shape),
                "encoded_cos": cos,
                "encoded_max_abs": mx,
                "encoded_mean_abs": mn,
                "encoded_rel_l2": rl2,
                "encoded_abs_mean_pt": float(np.mean(np.abs(enc_pt_out))),
                "cache_ch_cos": cc_cos,
                "cache_ch_max_abs": cc_mx,
                "cache_ch_rel_l2": cc_rl2,
                "cache_t_cos": ct_cos,
                "cache_t_max_abs": ct_mx,
                "cache_t_rel_l2": ct_rl2,
            }
            all_chunk_stats.append(stat)
            print(
                f"  chunk {chunk_idx}: "
                f"enc cos={cos:.6f} max|Δ|={mx:.4e} mean|Δ|={mn:.4e} relL2={rl2:.4e} "
                f"| cache_ch cos={cc_cos:.6f} max|Δ|={cc_mx:.4e} relL2={cc_rl2:.4e} "
                f"| cache_t cos={ct_cos:.6f} max|Δ|={ct_mx:.4e} relL2={ct_rl2:.4e}"
            )
            offset += chunk_samples

    return {"chunks": all_chunk_stats}


def summarize(stats: dict) -> None:
    chunks = stats["chunks"]
    if not chunks:
        return
    print("=" * 72)
    print("SUMMARY (per-chunk encoded-tensor parity)")
    print(f"{'lang':<12} {'chunk':>5} {'cos':>10} {'max|Δ|':>12} {'relL2':>12} "
          f"{'cc_cos':>10} {'ct_cos':>10}")
    for c in chunks:
        print(
            f"{c['lang']:<12} {c['chunk']:>5} {c['encoded_cos']:>10.6f} "
            f"{c['encoded_max_abs']:>12.4e} {c['encoded_rel_l2']:>12.4e} "
            f"{c['cache_ch_cos']:>10.6f} {c['cache_t_cos']:>10.6f}"
        )
    # Aggregate
    cos = np.array([c["encoded_cos"] for c in chunks])
    mx = np.array([c["encoded_max_abs"] for c in chunks])
    rl2 = np.array([c["encoded_rel_l2"] for c in chunks])
    print()
    print(f"encoded cosine:  min={cos.min():.6f}  median={np.median(cos):.6f}  max={cos.max():.6f}")
    print(f"encoded max|Δ|:  min={mx.min():.4e}  median={np.median(mx):.4e}  max={mx.max():.4e}")
    print(f"encoded relL2 :  min={rl2.min():.4e}  median={np.median(rl2):.4e}  max={rl2.max():.4e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo-path", required=True, help="Path to .nemo checkpoint.")
    ap.add_argument("--model-dir", default="build_fp16", help="CoreML build dir.")
    ap.add_argument(
        "--langs",
        default="en_us,cmn_hans_cn,ja_jp",
        help="Comma-separated FLEURS lang codes.",
    )
    ap.add_argument(
        "--target-lang",
        default="auto",
        help='Prompt: "auto" or "en-US" etc.',
    )
    ap.add_argument(
        "--chunks", type=int, default=5,
        help="Number of streaming chunks per utt (each ~1.12s).",
    )
    ap.add_argument("--out", default=None, help="Optional JSON output path.")
    args = ap.parse_args()

    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    stats = run_parity(
        nemo_path=args.nemo_path,
        model_dir=args.model_dir,
        langs=langs,
        target_lang=args.target_lang,
        num_chunks=args.chunks,
    )
    summarize(stats)
    if args.out:
        Path(args.out).write_text(json.dumps(stats, indent=2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
