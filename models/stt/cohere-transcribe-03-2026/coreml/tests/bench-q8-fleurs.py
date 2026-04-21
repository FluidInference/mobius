"""FLEURS benchmark against the q8 STATEFUL .mlpackage downloaded from
HuggingFace (FluidInference/cohere-transcribe-03-2026-coreml, q8/ subdir).

Verifies the "same model weights + fixed host-side Python works" claim for
the INT8 decoder+encoder that ships on HF. We load the models from
``hf-upload/q8-download/q8/`` (populated by ``huggingface-cli download``)
and run the FIXED inference path on the same 3-sample-per-language slice
used in bench-fix-vs-broken.py.

This script is FIXED-path only. The broken pipeline is already proven on f16
by bench-fix-vs-broken.py.

Usage:
    uv run python tests/bench-q8-fleurs.py
    uv run python tests/bench-q8-fleurs.py --language en_us --n 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import coremltools as ct
import librosa
import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "tools"))

from cohere_features_v2 import (  # noqa: E402
    CohereMelSpectrogram,
    pad_or_truncate_to_fixed,
)

Q8_DIR = ROOT / "hf-upload/q8-download/q8"
ENCODER_PATH = Q8_DIR / "cohere_encoder.mlpackage"
DECODER_PATH = Q8_DIR / "cohere_decoder_stateful.mlpackage"
VOCAB_PATH = Q8_DIR / "vocab.json"

SAMPLE_RATE = 16000
MEL_FRAMES_FIXED = 3500
ENCODER_FRAMES_FIXED = 438
MAX_SEQ_LEN = 108
EOS_TOKEN = 3

LANGUAGE_PROMPTS = {
    "en_us": [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13],
    "es_419": [13764, 7, 4, 16, 169, 169, 5, 9, 11, 13],
    "fr_fr": [13764, 7, 4, 16, 69, 69, 5, 9, 11, 13],
    "cmn_hans_cn": [13764, 7, 4, 16, 50, 50, 5, 9, 11, 13],
}


# ----------------------------------------------------------------------
# Decode against stateful decoder (mirrors q8/example_inference.py)
# ----------------------------------------------------------------------
def build_cross_mask(enc_seq_len: int, enc_valid: int) -> np.ndarray:
    mask = np.zeros((1, 1, 1, enc_seq_len), dtype=np.float16)
    if enc_valid < enc_seq_len:
        mask[:, :, :, enc_valid:] = -1.0e4
    return mask


def decode_stateful(
    decoder,
    encoder_hidden: np.ndarray,
    enc_valid: int,
    prompt_ids: list[int],
    max_tokens: int = MAX_SEQ_LEN,
    repetition_penalty: float = 1.1,
    no_repeat_ngram: int = 3,
) -> list[int]:
    state = decoder.make_state()
    cross_mask = build_cross_mask(encoder_hidden.shape[1], enc_valid)
    all_tokens: list[int] = []
    output_tokens: list[int] = []
    last_token = None

    for step in range(max_tokens):
        current = prompt_ids[step] if step < len(prompt_ids) else last_token
        input_id = np.array([[current]], dtype=np.int32)
        attn = np.zeros((1, 1, 1, step + 1), dtype=np.float16)
        pos = np.array([[step]], dtype=np.int32)

        out = decoder.predict(
            {
                "input_id": input_id,
                "encoder_hidden_states": encoder_hidden.astype(np.float16),
                "attention_mask": attn,
                "cross_attention_mask": cross_mask,
                "position_ids": pos,
            },
            state=state,
        )
        logits = out["logits"][0].astype(np.float32).copy()

        if repetition_penalty != 1.0 and all_tokens:
            seen = np.array(sorted(set(all_tokens)))
            pos_mask = logits[seen] >= 0
            logits[seen] = np.where(
                pos_mask,
                logits[seen] / repetition_penalty,
                logits[seen] * repetition_penalty,
            )

        if no_repeat_ngram > 0 and len(all_tokens) >= no_repeat_ngram - 1:
            n = no_repeat_ngram
            prefix = tuple(all_tokens[-(n - 1):]) if n > 1 else ()
            forbidden: set[int] = set()
            hist = all_tokens
            for i in range(len(hist) - (n - 1)):
                if tuple(hist[i : i + n - 1]) == prefix:
                    nxt = i + n - 1
                    if nxt < len(hist):
                        forbidden.add(hist[nxt])
            for t in forbidden:
                logits[t] = -1e9

        next_token = int(np.argmax(logits))
        last_token = next_token
        all_tokens.append(next_token)

        if step >= len(prompt_ids) - 1:
            output_tokens.append(next_token)
            if next_token == EOS_TOKEN:
                break

    return output_tokens


# ----------------------------------------------------------------------
# Detokenization with byte-fallback
# ----------------------------------------------------------------------
import re  # noqa: E402

_BYTE_FALLBACK_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def tokens_to_text(tokens: list[int], vocab: dict[int, str]) -> str:
    out: list[str] = []
    byte_buf: list[int] = []

    def flush() -> None:
        if byte_buf:
            out.append(bytes(byte_buf).decode("utf-8", errors="replace"))
            byte_buf.clear()

    for t in tokens:
        if t <= 4 or t == EOS_TOKEN:
            continue
        s = vocab.get(t, "")
        if s.startswith("<|"):
            continue
        m = _BYTE_FALLBACK_RE.match(s)
        if m is not None:
            byte_buf.append(int(m.group(1), 16))
            continue
        flush()
        out.append(s)
    flush()
    return "".join(out).replace("\u2581", " ").strip()


# ----------------------------------------------------------------------
# Metrics (WER for latin, CER for CJK)
# ----------------------------------------------------------------------
CJK_LANGS = {"cmn_hans_cn", "cmn_hant_hk", "ja_jp", "ko_kr"}


def normalize(s: str) -> str:
    s = s.lower()
    for ch in ",.!?;:\"'()[]-":
        s = s.replace(ch, " ")
    return " ".join(s.split())


def tokens_for_metric(s: str, lang: str) -> list[str]:
    n = normalize(s)
    if lang in CJK_LANGS:
        return [c for c in n if not c.isspace()]
    return n.split()


def wer(ref: str, hyp: str, lang: str) -> float:
    r = tokens_for_metric(ref, lang)
    h = tokens_for_metric(hyp, lang)
    if not r:
        return 1.0 if h else 0.0
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[len(r)][len(h)] / len(r)


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
def run(language: str, n: int, encoder, decoder, vocab, mel_proc):
    manifest = ROOT / f"fleurs_samples/{language}/manifest.json"
    if not manifest.exists():
        print(f"  skip (no manifest): {manifest}")
        return None
    samples = json.loads(manifest.read_text())[:n]
    prompt_ids = LANGUAGE_PROMPTS[language]

    wers = []
    records = []
    total_audio = 0.0
    total_time = 0.0
    for s in samples:
        audio, sr = sf.read(str(ROOT / s["audio"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE).astype(np.float32)

        t0 = time.time()
        mel, valid_mel = mel_proc(audio)
        mel_fixed, feat_len = pad_or_truncate_to_fixed(mel, valid_mel)
        enc_out = encoder.predict({
            "input_features": mel_fixed.astype(np.float32),
            "feature_length": np.array([feat_len], dtype=np.int32),
        })["hidden_states"]
        enc_valid = min(enc_out.shape[1],
                        max(1, int(np.ceil(feat_len / (MEL_FRAMES_FIXED / ENCODER_FRAMES_FIXED)))))
        tokens = decode_stateful(decoder, enc_out, enc_valid, prompt_ids)
        hyp = tokens_to_text(tokens, vocab)
        dt = time.time() - t0

        w = wer(s["text"], hyp, language)
        wers.append(w)
        total_audio += s["duration"]
        total_time += dt
        records.append({"id": s["id"], "ref": s["text"], "hyp": hyp, "wer": w, "t": dt})
        metric = "CER" if language in CJK_LANGS else "WER"
        print(f"--- {s['id']} ({s['duration']:.1f}s) ---")
        print(f"  REF: {s['text']}")
        print(f"  HYP: {hyp}")
        print(f"  {metric}={w * 100:.1f}%  t={dt:.1f}s")

    metric = "CER" if language in CJK_LANGS else "WER"
    rtfx = total_audio / total_time if total_time > 0 else 0.0
    print(f"\n=== {language}: mean {metric} = {np.mean(wers) * 100:.1f}%  RTFx={rtfx:.2f}x ===\n")
    return {
        "language": language,
        "n": len(records),
        "mean_wer": float(np.mean(wers)),
        "rtfx": rtfx,
        "records": records,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", default=None)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    print(f"Loading q8 encoder: {ENCODER_PATH}")
    encoder = ct.models.MLModel(str(ENCODER_PATH))
    print(f"Loading q8 stateful decoder: {DECODER_PATH}")
    decoder = ct.models.MLModel(str(DECODER_PATH))

    vocab_raw = json.loads(VOCAB_PATH.read_text())
    vocab = {int(k): v for k, v in vocab_raw.items()}
    mel_proc = CohereMelSpectrogram()

    langs = [args.language] if args.language else list(LANGUAGE_PROMPTS)
    reports = []
    for lang in langs:
        r = run(lang, args.n, encoder, decoder, vocab, mel_proc)
        if r:
            reports.append(r)

    print("=" * 72)
    print("Q8 SUMMARY (stateful decoder, fixed host code)")
    print("=" * 72)
    for r in reports:
        metric = "CER" if r["language"] in CJK_LANGS else "WER"
        print(f"  {r['language']:15s}  n={r['n']:2d}  {metric}={r['mean_wer'] * 100:6.1f}%  RTFx={r['rtfx']:.2f}x")

    if args.output:
        Path(args.output).write_text(json.dumps(reports, indent=2, ensure_ascii=False))
        print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
