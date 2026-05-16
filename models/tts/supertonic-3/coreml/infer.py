"""End-to-end Supertonic-3 TTS inference using the PyTorch ports.

Mirrors `supertonic/py/helper.py::TextToSpeech._infer`, but every ONNX session
is replaced with our local PyTorch port:

    text → UnicodeProcessor → text_ids, text_mask
        ↓
        ├── duration_predictor → duration (seconds, per-sample) → /speed
        └── text_encoder       → text_emb (B, 256, T)
        ↓
    sample noisy_latent (B, 144, L) + latent_mask (B, 1, L)
        ↓
    for step in range(total_step):
        vector_estimator(noisy_latent, text_emb, style_ttl,
                         latent_mask, text_mask, current_step, total_step)
        → updated noisy_latent
        ↓
    vocoder(noisy_latent) → wav (B, 512 * 6 * L)
        ↓
    trim per-sample to int(sample_rate * duration[b])
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple
from unicodedata import normalize

import numpy as np
import torch
import torch.nn.functional as F

from .duration_predictor import build_duration_predictor_from_onnx
from .text_encoder import build_text_encoder_from_onnx
from .vector_estimator import build_vector_estimator_from_onnx
from .vocoder import build_vocoder_from_onnx


# Match `supertonic/py/helper.py::AVAILABLE_LANGS`.
AVAILABLE_LANGS = [
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es",
    "et", "fi", "fr", "hi", "hr", "hu", "id", "it", "lt", "lv",
    "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk",
    "vi", "na",
]


# ---------------------------------------------------------------- text processor

class UnicodeProcessor:
    """Port of supertonic/py/helper.py UnicodeProcessor."""

    _emoji_re = re.compile(
        "["
        "\U0001f600-\U0001f64f"
        "\U0001f300-\U0001f5ff"
        "\U0001f680-\U0001f6ff"
        "\U0001f700-\U0001f77f"
        "\U0001f780-\U0001f7ff"
        "\U0001f800-\U0001f8ff"
        "\U0001f900-\U0001f9ff"
        "\U0001fa00-\U0001fa6f"
        "\U0001fa70-\U0001faff"
        "\u2600-\u26ff"
        "\u2700-\u27bf"
        "\U0001f1e6-\U0001f1ff"
        "]+",
        flags=re.UNICODE,
    )

    _char_repl = {
        "–": "-", "‑": "-", "—": "-", "_": " ",
        "\u201c": '"', "\u201d": '"',
        "\u2018": "'", "\u2019": "'",
        "´": "'", "`": "'",
        "[": " ", "]": " ", "|": " ", "/": " ",
        "#": " ", "→": " ", "←": " ",
    }
    _expr_repl = {"@": " at ", "e.g.,": "for example, ", "i.e.,": "that is, "}

    def __init__(self, unicode_indexer_path: Path):
        with open(unicode_indexer_path) as f:
            # 16-bit codepoint → token-id lookup table (list of 65536 ints, -1 if unsupported).
            self.indexer = json.load(f)

    def _preprocess(self, text: str, lang: str) -> str:
        text = normalize("NFKD", text)
        text = self._emoji_re.sub("", text)
        for k, v in self._char_repl.items():
            text = text.replace(k, v)
        text = re.sub(r"[♥☆♡©\\]", "", text)
        for k, v in self._expr_repl.items():
            text = text.replace(k, v)
        for p in (",", ".", "!", "?", ";", ":", "'"):
            text = re.sub(rf" \{p}" if p in r".?!" else f" {re.escape(p)}", p, text)
        while '""' in text:
            text = text.replace('""', '"')
        while "''" in text:
            text = text.replace("''", "'")
        while "``" in text:
            text = text.replace("``", "`")
        text = re.sub(r"\s+", " ", text).strip()
        if not re.search(r"[.!?;:,'\"')\]}…。」』】〉》›»]$", text):
            text += "."
        if lang not in AVAILABLE_LANGS:
            raise ValueError(f"Invalid language: {lang}. Supported: {AVAILABLE_LANGS}")
        return f"<{lang}>" + text + f"</{lang}>"

    def __call__(
        self, text_list: Sequence[str], lang_list: Sequence[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        texts = [self._preprocess(t, lang) for t, lang in zip(text_list, lang_list)]
        lengths = np.array([len(t) for t in texts], dtype=np.int64)
        max_len = int(lengths.max())
        text_ids = np.zeros((len(texts), max_len), dtype=np.int64)
        for i, t in enumerate(texts):
            uv = [ord(c) for c in t]
            text_ids[i, : len(uv)] = np.array(
                [self.indexer[v] for v in uv], dtype=np.int64
            )
        # mask: [B, 1, max_len].
        ids = np.arange(max_len)[None, :]
        mask = (ids < lengths[:, None]).astype(np.float32)
        text_mask = mask.reshape(-1, 1, max_len)
        return text_ids, text_mask


# ---------------------------------------------------------------- latent helpers

def _length_to_mask(lengths: np.ndarray, max_len: int | None = None) -> np.ndarray:
    max_len = int(max_len if max_len is not None else lengths.max())
    ids = np.arange(max_len)[None, :]
    mask = (ids < lengths[:, None]).astype(np.float32)
    return mask.reshape(-1, 1, max_len)


def sample_noisy_latent(
    duration_sec: np.ndarray,
    sample_rate: int,
    base_chunk_size: int,
    chunk_compress_factor: int,
    latent_dim: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Match helper.py::TextToSpeech.sample_noisy_latent.

    Returns:
        noisy_latent: [B, ldim * cf, L]   (cf = chunk_compress_factor)
        latent_mask:  [B, 1, L]
    """
    bsz = duration_sec.shape[0]
    wav_lengths = (duration_sec * sample_rate).astype(np.int64)
    wav_len_max = int(wav_lengths.max())
    chunk_size = base_chunk_size * chunk_compress_factor
    L = (wav_len_max + chunk_size - 1) // chunk_size
    noisy = rng.standard_normal((bsz, latent_dim * chunk_compress_factor, L)).astype(np.float32)
    latent_lengths = (wav_lengths + chunk_size - 1) // chunk_size
    latent_mask = _length_to_mask(latent_lengths, max_len=L)
    noisy = noisy * latent_mask
    return noisy, latent_mask


# ---------------------------------------------------------------- voice style

class Style:
    """Container matching helper.py::Style."""

    def __init__(self, ttl: np.ndarray, dp: np.ndarray):
        self.ttl = ttl  # [B, 50, 256]
        self.dp = dp    # [B, 8, 16]


def load_voice_style(paths: Sequence[Path]) -> Style:
    """Match helper.py::load_voice_style: accepts list of JSON files, stacks to [B, ...]."""
    bsz = len(paths)
    with open(paths[0]) as f:
        first = json.load(f)
    ttl_dims = first["style_ttl"]["dims"]
    dp_dims = first["style_dp"]["dims"]
    ttl = np.zeros([bsz, ttl_dims[1], ttl_dims[2]], dtype=np.float32)
    dp = np.zeros([bsz, dp_dims[1], dp_dims[2]], dtype=np.float32)
    for i, p in enumerate(paths):
        with open(p) as f:
            cfg = json.load(f)
        ttl[i] = np.array(cfg["style_ttl"]["data"], dtype=np.float32).reshape(ttl_dims[1], ttl_dims[2])
        dp[i] = np.array(cfg["style_dp"]["data"], dtype=np.float32).reshape(dp_dims[1], dp_dims[2])
    return Style(ttl, dp)


# ---------------------------------------------------------------- TTS driver

class TextToSpeech:
    """PyTorch-port equivalent of helper.py::TextToSpeech."""

    def __init__(
        self,
        onnx_dir: Path,
        device: str = "cpu",
        seed: int | None = None,
    ):
        self.device = torch.device(device)
        with open(onnx_dir / "tts.json") as f:
            self.cfgs = json.load(f)
        self.sample_rate = int(self.cfgs["ae"]["sample_rate"])
        self.base_chunk_size = int(self.cfgs["ae"]["base_chunk_size"])
        self.chunk_compress_factor = int(self.cfgs["ttl"]["chunk_compress_factor"])
        self.ldim = int(self.cfgs["ttl"]["latent_dim"])

        self.text_processor = UnicodeProcessor(onnx_dir / "unicode_indexer.json")

        # Build all four PyTorch ports.
        self.dp = build_duration_predictor_from_onnx(onnx_dir / "duration_predictor.onnx").to(self.device).eval()
        self.text_enc = build_text_encoder_from_onnx(onnx_dir / "text_encoder.onnx").to(self.device).eval()
        self.vec_est = build_vector_estimator_from_onnx(onnx_dir / "vector_estimator.onnx").to(self.device).eval()
        self.vocoder = build_vocoder_from_onnx(onnx_dir / "vocoder.onnx", tts_json_path=onnx_dir / "tts.json").to(self.device).eval()

        self.rng = np.random.default_rng(seed)

    # ----------------- low-level: one forward pass over a batch -----------------
    @torch.no_grad()
    def _infer(
        self,
        text_list: List[str],
        lang_list: List[str],
        style: Style,
        total_step: int,
        speed: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert len(text_list) == style.ttl.shape[0], "voice styles must match texts"
        bsz = len(text_list)

        # Tokenize.
        text_ids_np, text_mask_np = self.text_processor(text_list, lang_list)
        text_ids = torch.from_numpy(text_ids_np).to(self.device)              # int64
        text_mask = torch.from_numpy(text_mask_np).to(self.device)            # float32

        style_ttl = torch.from_numpy(style.ttl).to(self.device)               # [B, 50, 256]
        style_dp = torch.from_numpy(style.dp).to(self.device)                 # [B, 8, 16]

        # Duration in seconds (per-sample).
        duration = self.dp(text_ids, style_dp, text_mask).cpu().numpy()       # [B]
        duration = duration / speed

        # Text embedding.
        text_emb = self.text_enc(text_ids, style_ttl, text_mask)              # [B, 256, T]

        # Sample noisy latent.
        noisy_np, latent_mask_np = sample_noisy_latent(
            duration,
            sample_rate=self.sample_rate,
            base_chunk_size=self.base_chunk_size,
            chunk_compress_factor=self.chunk_compress_factor,
            latent_dim=self.ldim,
            rng=self.rng,
        )
        xt = torch.from_numpy(noisy_np).to(self.device)
        latent_mask = torch.from_numpy(latent_mask_np).to(self.device)

        total_step_t = torch.full((bsz,), float(total_step), dtype=torch.float32, device=self.device)
        for step in range(total_step):
            current_step_t = torch.full((bsz,), float(step), dtype=torch.float32, device=self.device)
            xt = self.vec_est(xt, text_emb, style_ttl, latent_mask, text_mask, current_step_t, total_step_t)

        wav = self.vocoder(xt).cpu().numpy()                                   # [B, 512 * 6 * L]
        return wav, duration

    # ----------------- batch entry: helper.py::TextToSpeech.batch -----------------
    def batch(
        self, text_list: List[str], lang_list: List[str],
        style: Style, total_step: int, speed: float = 1.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self._infer(text_list, lang_list, style, total_step, speed)

    # ----------------- single-utt entry with chunking -----------------
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
        dur_cat: np.ndarray | None = None
        for chunk in chunks:
            wav, dur = self._infer([chunk], [lang], style, total_step, speed)
            if wav_cat is None:
                wav_cat = wav
                dur_cat = dur
            else:
                silence = np.zeros((1, int(silence_duration * self.sample_rate)), dtype=np.float32)
                wav_cat = np.concatenate([wav_cat, silence, wav], axis=1)
                dur_cat = dur_cat + dur + silence_duration
        return wav_cat, dur_cat


# ---------------------------------------------------------------- text chunker (verbatim from helper.py)

def _chunk_text(text: str, max_len: int = 300) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    chunks: List[str] = []
    for paragraph in paragraphs:
        pattern = r"(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)(?<!Ph\.D\.)(?<!etc\.)(?<!e\.g\.)(?<!i\.e\.)(?<!vs\.)(?<!Inc\.)(?<!Ltd\.)(?<!Co\.)(?<!Corp\.)(?<!St\.)(?<!Ave\.)(?<!Blvd\.)(?<!\b[A-Z]\.)(?<=[.!?])\s+"
        sentences = re.split(pattern, paragraph)
        cur = ""
        for s in sentences:
            if len(cur) + len(s) + 1 <= max_len:
                cur += (" " if cur else "") + s
            else:
                if cur:
                    chunks.append(cur.strip())
                cur = s
        if cur:
            chunks.append(cur.strip())
    return chunks


def _sanitize(text: str, max_len: int) -> str:
    return re.sub(r"[^\w]", "_", text[:max_len], flags=re.UNICODE)


# ---------------------------------------------------------------- CLI

def main() -> None:
    p = argparse.ArgumentParser(description="Supertonic-3 PyTorch-port TTS")
    p.add_argument("--onnx-dir", type=Path, default=Path("build/supertonic-3-coreml/_onnx"),
                   help="Directory containing the 4 .onnx files + tts.json + unicode_indexer.json")
    p.add_argument("--voice-style", type=Path, nargs="+", required=True,
                   help="Voice-style JSON path(s) (e.g. M1.json). Multiple → batch mode.")
    p.add_argument("--text", type=str, nargs="+", required=True,
                   help="Text(s) to synthesize.")
    p.add_argument("--lang", type=str, nargs="+", default=["en"],
                   help="Language code(s) per text (default: en).")
    p.add_argument("--total-step", type=int, default=8)
    p.add_argument("--speed", type=float, default=1.05)
    p.add_argument("--batch", action="store_true",
                   help="Batch mode (disables auto chunking; bsz follows --voice-style count).")
    p.add_argument("--save-dir", type=Path, default=Path("results"))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if len(args.lang) == 1 and len(args.text) > 1:
        args.lang = args.lang * len(args.text)
    assert len(args.voice_style) == len(args.text) == len(args.lang), \
        "voice-style / text / lang counts must match"

    print("=== Supertonic-3 TTS (PyTorch port) ===")
    tts = TextToSpeech(args.onnx_dir, device=args.device, seed=args.seed)
    style = load_voice_style(args.voice_style)
    print(f"Loaded {style.ttl.shape[0]} voice styles  | sample_rate={tts.sample_rate}")

    t0 = time.time()
    if args.batch:
        wav, dur = tts.batch(args.text, args.lang, style, args.total_step, args.speed)
    else:
        wav, dur = tts(args.text[0], args.lang[0], style, args.total_step, args.speed)
    print(f"Generated in {time.time()-t0:.2f}s; per-sample duration(s)={dur.tolist()}")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf
    except ImportError as e:
        raise SystemExit("pip install soundfile to write .wav files") from e
    for b in range(wav.shape[0]):
        fname = f"{_sanitize(args.text[b], 20)}.wav"
        n = int(tts.sample_rate * dur[b].item())
        sf.write(args.save_dir / fname, wav[b, :n], tts.sample_rate)
        print(f"  wrote {args.save_dir / fname}  ({n / tts.sample_rate:.2f}s)")


if __name__ == "__main__":  # pragma: no cover
    main()
