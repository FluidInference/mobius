"""End-to-end Supertonic-3 TTS inference using the CoreML .mlpackage files.

Mirrors `infer.py` but every PyTorch port is replaced with a `ct.models.MLModel`
prediction. Shapes that the CoreML conversion pinned (text T=128) or floored
(vector_estimator L>=17, vocoder L_ttl>=4) are handled here by padding.

All four sub-models accept batch=1 only, so multi-utt requests are looped.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import coremltools as ct

from .infer import (
    AVAILABLE_LANGS,
    Style,
    UnicodeProcessor,
    _chunk_text,
    _sanitize,
    load_voice_style,
    sample_noisy_latent,
)


# CoreML-pinned shape constants (must match conversion settings; see trials.md).
TEXT_T_FIXED = 128         # text_encoder / duration_predictor pinned T
VEC_EST_L_MIN = 17         # vector_estimator latent/text RangeDim lower bound
VOCODER_L_MIN = 4          # vocoder latent RangeDim lower bound


def _pad_text(text_ids: np.ndarray, text_mask: np.ndarray,
              target_T: int = TEXT_T_FIXED) -> Tuple[np.ndarray, np.ndarray]:
    """Pad/truncate text_ids [B,T] and text_mask [B,1,T] to the fixed target T."""
    bsz, T = text_ids.shape
    if T == target_T:
        return text_ids, text_mask
    if T > target_T:
        return text_ids[:, :target_T], text_mask[:, :, :target_T]
    pad = target_T - T
    text_ids_p = np.pad(text_ids, ((0, 0), (0, pad)), constant_values=0)
    text_mask_p = np.pad(text_mask, ((0, 0), (0, 0), (0, pad)), constant_values=0.0)
    return text_ids_p, text_mask_p


def _pad_latent_axis(arr: np.ndarray, target_L: int) -> np.ndarray:
    """Right-pad the last axis of arr (zeros) up to target_L."""
    L = arr.shape[-1]
    if L >= target_L:
        return arr
    pad_width = [(0, 0)] * arr.ndim
    pad_width[-1] = (0, target_L - L)
    return np.pad(arr, pad_width, constant_values=0.0)


class TextToSpeechCoreML:
    """CoreML-backed equivalent of `infer.py::TextToSpeech`."""

    def __init__(
        self,
        mlpackage_dir: Path,
        tts_json_path: Path,
        unicode_indexer_path: Path,
        compute_units: ct.ComputeUnit = ct.ComputeUnit.CPU_AND_NE,
        seed: int | None = None,
    ):
        with open(tts_json_path) as f:
            self.cfgs = json.load(f)
        self.sample_rate = int(self.cfgs["ae"]["sample_rate"])
        self.base_chunk_size = int(self.cfgs["ae"]["base_chunk_size"])
        self.chunk_compress_factor = int(self.cfgs["ttl"]["chunk_compress_factor"])
        self.ldim = int(self.cfgs["ttl"]["latent_dim"])

        self.text_processor = UnicodeProcessor(unicode_indexer_path)

        def _load(name: str) -> ct.models.MLModel:
            return ct.models.MLModel(
                str(mlpackage_dir / name),
                compute_units=compute_units,
            )

        print(f"Loading 4 mlpackages from {mlpackage_dir} (compute_units={compute_units.name})...")
        self.dp_ml = _load("DurationPredictor.mlpackage")
        self.text_enc_ml = _load("TextEncoder.mlpackage")
        self.vec_est_ml = _load("VectorEstimator.mlpackage")
        self.vocoder_ml = _load("Vocoder.mlpackage")

        self.rng = np.random.default_rng(seed)

    # --------------------------------------------------- per-sample CoreML run
    def _infer_single(
        self,
        text: str,
        lang: str,
        style_ttl: np.ndarray,   # [1, 50, 256]
        style_dp: np.ndarray,    # [1, 8, 16]
        total_step: int,
        speed: float,
    ) -> Tuple[np.ndarray, float]:
        text_ids_np, text_mask_np = self.text_processor([text], [lang])
        text_ids_p, text_mask_p = _pad_text(text_ids_np, text_mask_np)
        text_ids_in = text_ids_p.astype(np.int32)
        text_mask_in = text_mask_p.astype(np.float32)

        # duration: [1]
        dp_out = self.dp_ml.predict({
            "text_ids": text_ids_in,
            "style_dp": style_dp.astype(np.float32),
            "text_mask": text_mask_in,
        })
        duration = np.asarray(dp_out["duration"], dtype=np.float32) / speed

        # text_emb: [1, 256, 128]
        te_out = self.text_enc_ml.predict({
            "text_ids": text_ids_in,
            "style_ttl": style_ttl.astype(np.float32),
            "text_mask": text_mask_in,
        })
        text_emb = np.asarray(te_out["text_emb"], dtype=np.float32)

        # Sample noisy latent (uses real per-sample duration → real L).
        noisy_np, latent_mask_np = sample_noisy_latent(
            duration,
            sample_rate=self.sample_rate,
            base_chunk_size=self.base_chunk_size,
            chunk_compress_factor=self.chunk_compress_factor,
            latent_dim=self.ldim,
            rng=self.rng,
        )
        L_true = noisy_np.shape[-1]
        # CoreML vector_estimator floor is 17; pad with zeros if shorter.
        L_use = max(L_true, VEC_EST_L_MIN)
        if L_use > L_true:
            noisy_np = _pad_latent_axis(noisy_np, L_use)
            latent_mask_np = _pad_latent_axis(latent_mask_np, L_use)

        xt = noisy_np.astype(np.float32)
        latent_mask = latent_mask_np.astype(np.float32)
        total_step_in = np.array([float(total_step)], dtype=np.float32)
        for step in range(total_step):
            cur_in = np.array([float(step)], dtype=np.float32)
            ve_out = self.vec_est_ml.predict({
                "noisy_latent": xt,
                "text_emb": text_emb,
                "style_ttl": style_ttl.astype(np.float32),
                "latent_mask": latent_mask,
                "text_mask": text_mask_in,
                "current_step": cur_in,
                "total_step": total_step_in,
            })
            xt = np.asarray(ve_out["denoised_latent"], dtype=np.float32)

        # Vocoder needs L_ttl >= 4 — we already enforced L >= 17 above so safe.
        if xt.shape[-1] < VOCODER_L_MIN:
            xt = _pad_latent_axis(xt, VOCODER_L_MIN)
        vc_out = self.vocoder_ml.predict({"latent": xt})
        wav = np.asarray(vc_out["wav"], dtype=np.float32)   # [1, 512*6*L_use]

        # Trim wav back to the actual L_true length, then to per-sample seconds.
        if L_use > L_true:
            wav = wav[:, : (512 * self.chunk_compress_factor) * L_true]
        return wav, float(duration[0])

    # ----------------- chunked single-utt entry (matches infer.py::__call__) -----
    def __call__(
        self,
        text: str,
        lang: str,
        style: Style,
        total_step: int,
        speed: float = 1.05,
        silence_duration: float = 0.3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert style.ttl.shape[0] == 1, "single-text path requires bsz=1 style"
        max_len = 120 if lang in ("ko", "ja") else 300
        chunks = _chunk_text(text, max_len=max_len)
        wav_cat: np.ndarray | None = None
        dur_cat: float = 0.0
        for chunk in chunks:
            wav, dur_s = self._infer_single(chunk, lang, style.ttl, style.dp, total_step, speed)
            if wav_cat is None:
                wav_cat = wav
                dur_cat = dur_s
            else:
                silence = np.zeros((1, int(silence_duration * self.sample_rate)), dtype=np.float32)
                wav_cat = np.concatenate([wav_cat, silence, wav], axis=1)
                dur_cat = dur_cat + dur_s + silence_duration
        return wav_cat, np.array([dur_cat], dtype=np.float32)

    # ----------------- batch entry: loop per sample (CoreML is batch=1 only) -----
    def batch(
        self,
        text_list: List[str],
        lang_list: List[str],
        style: Style,
        total_step: int,
        speed: float = 1.05,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        assert len(text_list) == len(lang_list) == style.ttl.shape[0]
        wavs: List[np.ndarray] = []
        durs: List[float] = []
        for i in range(len(text_list)):
            wav, dur_s = self._infer_single(
                text_list[i], lang_list[i],
                style.ttl[i : i + 1], style.dp[i : i + 1],
                total_step, speed,
            )
            wavs.append(wav)
            durs.append(dur_s)
        return wavs, np.array(durs, dtype=np.float32)


# --------------------------------------------------------------- CLI
def main() -> None:
    p = argparse.ArgumentParser(description="Supertonic-3 CoreML-port TTS")
    p.add_argument("--mlpackage-dir", type=Path,
                   default=Path("build/supertonic-3-coreml/_mlpackage"),
                   help="Directory containing the 4 .mlpackage bundles")
    p.add_argument("--tts-json", type=Path,
                   default=Path("build/supertonic-3-coreml/_onnx/tts.json"))
    p.add_argument("--unicode-indexer", type=Path,
                   default=Path("build/supertonic-3-coreml/_onnx/unicode_indexer.json"))
    p.add_argument("--voice-style", type=Path, nargs="+", required=True)
    p.add_argument("--text", type=str, nargs="+", required=True)
    p.add_argument("--lang", type=str, nargs="+", default=["en"])
    p.add_argument("--total-step", type=int, default=8)
    p.add_argument("--speed", type=float, default=1.05)
    p.add_argument("--batch", action="store_true")
    p.add_argument("--save-dir", type=Path, default=Path("results_coreml"))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--compute-units",
        type=str,
        default="CPU_AND_NE",
        choices=["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"],
    )
    args = p.parse_args()

    if len(args.lang) == 1 and len(args.text) > 1:
        args.lang = args.lang * len(args.text)
    assert len(args.voice_style) == len(args.text) == len(args.lang), \
        "voice-style / text / lang counts must match"

    cu = getattr(ct.ComputeUnit, args.compute_units)

    print("=== Supertonic-3 TTS (CoreML port) ===")
    tts = TextToSpeechCoreML(
        mlpackage_dir=args.mlpackage_dir,
        tts_json_path=args.tts_json,
        unicode_indexer_path=args.unicode_indexer,
        compute_units=cu,
        seed=args.seed,
    )
    style = load_voice_style(args.voice_style)
    print(f"Loaded {style.ttl.shape[0]} voice styles  | sample_rate={tts.sample_rate}")

    t0 = time.time()
    if args.batch:
        wavs, dur = tts.batch(args.text, args.lang, style, args.total_step, args.speed)
    else:
        wav, dur = tts(args.text[0], args.lang[0], style, args.total_step, args.speed)
        wavs = [wav]
    print(f"Generated in {time.time()-t0:.2f}s; per-sample duration(s)={dur.tolist()}")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf
    except ImportError as e:
        raise SystemExit("pip install soundfile to write .wav files") from e
    for b, w in enumerate(wavs):
        fname = f"{_sanitize(args.text[b], 20)}_coreml.wav"
        n = int(tts.sample_rate * dur[b])
        sf.write(args.save_dir / fname, w[0, :n], tts.sample_rate)
        print(f"  wrote {args.save_dir / fname}  ({n / tts.sample_rate:.2f}s)")


if __name__ == "__main__":  # pragma: no cover
    main()
