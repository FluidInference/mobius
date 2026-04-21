"""Benchmark the stateless decoder at build-stateless/ on the same 12-sample
FLEURS slice we've been using, with the fixed pipeline (v2 mel + masked
cross-attention + CJK byte-fallback detok + rep penalty + no-repeat-ngram).

Compared against:
  stateless_f16   — build-stateless/cohere_decoder_stateless.mlpackage
                    paired with f16 encoder (hf-upload/f16-download)
  stateless_q8enc — same stateless decoder paired with q8 encoder
                    (hf-upload/q8-download)

If the stateless export truly does not over-generate, both should land
close to the "cache-external f16 reference" of EN 10.6% / ES 4.9% /
FR 16.8% / ZH 14.1%, which is what `tests/bench-fix-vs-broken.py`
measured on the cache-external decoder.
"""
from __future__ import annotations

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
F16_DIR = ROOT / "hf-upload/f16-download/f16"
STATELESS_DECODER = ROOT / "build-stateless/cohere_decoder_stateless.mlpackage"
VOCAB_PATH = Q8_DIR / "vocab.json"

ENCODERS = {
    "f16": F16_DIR / "cohere_encoder.mlpackage",
    "q8": Q8_DIR / "cohere_encoder.mlpackage",
}

SAMPLE_RATE = 16000
MEL_FRAMES_FIXED = 3500
ENCODER_FRAMES_FIXED = 438
EOS = 3
MAX_TOKENS = 108

PROMPTS = {
    "en_us": [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13],
    "es_419": [13764, 7, 4, 16, 169, 169, 5, 9, 11, 13],
    "fr_fr": [13764, 7, 4, 16, 69, 69, 5, 9, 11, 13],
    "cmn_hans_cn": [13764, 7, 4, 16, 50, 50, 5, 9, 11, 13],
}


def build_cross_mask(enc_seq_len: int, enc_valid: int) -> np.ndarray:
    mask = np.zeros((1, 1, 1, enc_seq_len), dtype=np.float32)
    if enc_valid < enc_seq_len:
        mask[:, :, :, enc_valid:] = -1.0e4
    return mask


def decode_stateless(
    decoder,
    enc_out: np.ndarray,
    enc_valid: int,
    prompt_ids: list[int],
    repetition_penalty: float = 1.1,
    no_repeat_ngram: int = 3,
) -> list[int]:
    """Stateless greedy decode — feed the full prefix every step and take
    the last-position logits. No KV cache, no state."""
    cross_mask = build_cross_mask(enc_out.shape[1], enc_valid)
    # Seed the sequence with the language-prompt tokens.
    tokens: list[int] = list(prompt_ids)
    output_tokens: list[int] = []

    for step in range(MAX_TOKENS - len(prompt_ids)):
        input_ids = np.array([tokens], dtype=np.int32)
        out = decoder.predict({
            "input_ids": input_ids,
            "encoder_hidden_states": enc_out.astype(np.float32),
            "cross_attention_mask": cross_mask,
        })
        logits = np.asarray(out["logits"])
        # (1, seq_len, vocab) → take the last position
        last = logits[0, -1, :].astype(np.float32).copy()

        if repetition_penalty != 1.0 and output_tokens:
            seen = np.array(sorted(set(output_tokens)))
            pos_m = last[seen] >= 0
            last[seen] = np.where(
                pos_m, last[seen] / repetition_penalty, last[seen] * repetition_penalty
            )

        if no_repeat_ngram > 0 and len(output_tokens) >= no_repeat_ngram - 1:
            n = no_repeat_ngram
            prefix = tuple(output_tokens[-(n - 1):]) if n > 1 else ()
            forbidden: set[int] = set()
            hist = output_tokens
            for i in range(len(hist) - (n - 1)):
                if tuple(hist[i : i + n - 1]) == prefix:
                    nxt = i + n - 1
                    if nxt < len(hist):
                        forbidden.add(hist[nxt])
            for t in forbidden:
                last[t] = -1e9

        next_tok = int(np.argmax(last))
        tokens.append(next_tok)
        output_tokens.append(next_tok)
        if next_tok == EOS:
            break

    return output_tokens


import re
_BFR = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def detok(tokens, vocab):
    out = []
    bb = []
    def flush():
        if bb:
            out.append(bytes(bb).decode("utf-8", errors="replace"))
            bb.clear()
    for t in tokens:
        if t <= 4 or t == EOS:
            continue
        s = vocab.get(t, "")
        if s.startswith("<|"):
            continue
        m = _BFR.match(s)
        if m is not None:
            bb.append(int(m.group(1), 16))
            continue
        flush()
        out.append(s)
    flush()
    return "".join(out).replace("\u2581", " ").strip()


CJK = {"cmn_hans_cn", "cmn_hant_hk", "ja_jp", "ko_kr"}


def norm(s):
    s = s.lower()
    for ch in ",.!?;:\"'()[]-":
        s = s.replace(ch, " ")
    return " ".join(s.split())


def toks(s, lang):
    n = norm(s)
    if lang in CJK:
        return [c for c in n if not c.isspace()]
    return n.split()


def wer(ref, hyp, lang):
    r = toks(ref, lang)
    h = toks(hyp, lang)
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


def main():
    vocab = {int(k): v for k, v in json.loads(VOCAB_PATH.read_text()).items()}
    mel_proc = CohereMelSpectrogram()

    langs = ["en_us", "es_419", "fr_fr", "cmn_hans_cn"]
    N = 3

    print(f"Pre-computing mel for {len(langs) * N} samples...")
    samples = {}
    for lang in langs:
        manifest = json.loads((ROOT / f"fleurs_samples/{lang}/manifest.json").read_text())[:N]
        for s in manifest:
            audio, sr = sf.read(str(ROOT / s["audio"]), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE).astype(np.float32)
            mel, valid = mel_proc(audio)
            mel_fixed, feat_len = pad_or_truncate_to_fixed(mel, valid)
            samples[(lang, s["id"])] = {
                "mel_fixed": mel_fixed,
                "feat_len": feat_len,
                "ref": s["text"],
            }

    enc_cache: dict[str, dict] = {}
    for ename, epath in ENCODERS.items():
        print(f"\nEncoding via {ename} encoder...")
        t0 = time.time()
        enc = ct.models.MLModel(str(epath))
        print(f"  loaded in {time.time() - t0:.1f}s")
        enc_cache[ename] = {}
        t0 = time.time()
        for key, data in samples.items():
            enc_out = enc.predict({
                "input_features": data["mel_fixed"].astype(np.float32),
                "feature_length": np.array([data["feat_len"]], dtype=np.int32),
            })["hidden_states"]
            enc_valid = min(
                enc_out.shape[1],
                max(1, int(np.ceil(data["feat_len"] / (MEL_FRAMES_FIXED / ENCODER_FRAMES_FIXED)))),
            )
            enc_cache[ename][key] = {"enc_out": enc_out, "enc_valid": enc_valid}
        print(f"  encoded {len(samples)} samples in {time.time() - t0:.1f}s")
        del enc

    print(f"\nLoading stateless decoder: {STATELESS_DECODER.name}...")
    t0 = time.time()
    decoder = ct.models.MLModel(str(STATELESS_DECODER))
    print(f"  loaded in {time.time() - t0:.1f}s")

    CONFIGS = [("stateless_f16enc", "f16"), ("stateless_q8enc", "q8")]
    results: dict = {name: {l: [] for l in langs} for name, _ in CONFIGS}

    for config_name, ename in CONFIGS:
        print(f"\n=== {config_name} ===")
        for lang in langs:
            prompt = PROMPTS[lang]
            manifest = json.loads((ROOT / f"fleurs_samples/{lang}/manifest.json").read_text())[:N]
            for s in manifest:
                key = (lang, s["id"])
                c = enc_cache[ename][key]
                t0 = time.time()
                tokens = decode_stateless(decoder, c["enc_out"], c["enc_valid"], prompt)
                dt = time.time() - t0
                hyp = detok(tokens, vocab)
                w = wer(samples[key]["ref"], hyp, lang)
                results[config_name][lang].append({
                    "id": s["id"],
                    "ref": samples[key]["ref"],
                    "hyp": hyp,
                    "wer": w,
                    "n_tok": len(tokens),
                    "dt_s": dt,
                })
        for lang in langs:
            rs = results[config_name][lang]
            metric = "CER" if lang in CJK else "WER"
            mean_w = np.mean([r["wer"] for r in rs])
            mean_tok = np.mean([r["n_tok"] for r in rs])
            mean_t = np.mean([r["dt_s"] for r in rs])
            print(f"  {lang:15s} mean {metric}={mean_w * 100:6.1f}%   avg tokens={mean_tok:5.1f}   decode={mean_t:.1f}s")

    print("\n" + "=" * 80)
    print("STATELESS DECODER vs stateful baselines (12 FLEURS samples)")
    print("=" * 80)
    # Reference numbers from earlier benchmarks:
    ref_stateful = {"en_us": 73.4, "es_419": 23.3, "fr_fr": 45.2, "cmn_hans_cn": 48.3}
    ref_cacheext = {"en_us": 10.6, "es_419": 4.9, "fr_fr": 16.8, "cmn_hans_cn": 14.1}
    header = f"{'lang':15s}  {'stateful_q8':>12s}  {'cache_ext_f16':>14s}  {'stateless_f16enc':>17s}  {'stateless_q8enc':>17s}"
    print(header)
    for lang in langs:
        metric = "CER" if lang in CJK else "WER"
        sl_f16 = np.mean([r["wer"] for r in results["stateless_f16enc"][lang]]) * 100
        sl_q8 = np.mean([r["wer"] for r in results["stateless_q8enc"][lang]]) * 100
        print(f"{lang:15s}  {ref_stateful[lang]:>10.1f}%  {ref_cacheext[lang]:>12.1f}%  {sl_f16:>15.1f}%  {sl_q8:>15.1f}%  ({metric})")

    out_json = ROOT / "stateless_fleurs_results.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved raw results to {out_json}")


if __name__ == "__main__":
    main()
