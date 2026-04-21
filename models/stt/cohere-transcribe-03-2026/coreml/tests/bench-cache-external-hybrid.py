"""Benchmark cache-external decoder in f16 and q8, paired with f16/q8 encoders,
on the same 12-sample FLEURS slice with the fixed host pipeline.

This is the cache-external analogue of tests/bench-hybrid-configs.py (which
tested stateful combos). The cache-external decoder is the one that
actually produces clean transcripts — the stateful decoder over-generates
past EOS regardless of quantization.

Configs:
  ce_f16_f16 — f16 encoder + f16 cache-external decoder  (reference)
  ce_q8_f16  — q8  encoder + f16 cache-external decoder
  ce_f16_q8  — f16 encoder + q8  cache-external decoder
  ce_q8_q8   — q8  encoder + q8  cache-external decoder

Expected reference numbers (from tests/bench-fix-vs-broken.py on f16/f16):
  en_us 10.6%   es_419 4.9%   fr_fr 16.8%   cmn_hans_cn 14.1%

If q8 survives on the cache-external decoder (unlike the stateful one),
ce_q8_q8 should land near the reference. If not, the hybrid ce_q8_f16
isolates where the quality drops.
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

# Encoders
# IMPORTANT: the cache-external decoder was exported together with its
# *companion* encoder at hf-upload/cohere-transcribe-cache-external-coreml/.
# Pairing the cache-external decoder with the f16/q8 encoder from the
# stateful export path produces very different hidden-state statistics and
# breaks quality (measured ~58% WER on EN vs 10.6% with the companion encoder).
CE_COMPANION_ENC = ROOT / "hf-upload/cohere-transcribe-cache-external-coreml/cohere_encoder.mlpackage"

# Cache-external decoders (f16 baseline + fresh q8 from tests/quantize-cache-external.py)
F16_CE_DECODER = ROOT / "hf-upload/cohere-transcribe-cache-external-coreml/cohere_decoder_cache_external.mlpackage"
Q8_CE_DECODER = ROOT / "build-cache-external-q8/cohere_decoder_cache_external_q8.mlpackage"

VOCAB_PATH = ROOT / "hf-upload/q8-download/q8/vocab.json"

# We use a single (companion) encoder in FP16 across both configs so any
# quality delta is attributable to decoder quantization only. Quantizing the
# 7 GB companion encoder to q8 is a separate follow-up.
ENCODERS = {"companion": CE_COMPANION_ENC}
DECODERS = {"f16": F16_CE_DECODER, "q8": Q8_CE_DECODER}

CONFIGS = [
    ("ce_f16", "companion", "f16"),
    ("ce_q8", "companion", "q8"),
]

SAMPLE_RATE = 16000
MEL_FRAMES_FIXED = 3500
ENCODER_FRAMES_FIXED = 438
MAX_SEQ_LEN = 108
START_TOKEN = 4
EOS = 3

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


def decode_cache_external(
    decoder,
    enc_out: np.ndarray,
    cross_mask: np.ndarray,
    prompt_ids: list[int],
    repetition_penalty: float = 1.1,
    no_repeat_ngram: int = 3,
) -> list[int]:
    """Greedy decode using host-threaded KV cache. 8 layers, 8 heads, head_dim 128."""
    k_caches = [np.zeros((1, 8, MAX_SEQ_LEN, 128), dtype=np.float32) for _ in range(8)]
    v_caches = [np.zeros((1, 8, MAX_SEQ_LEN, 128), dtype=np.float32) for _ in range(8)]

    all_tokens: list[int] = []  # every predicted token (for rep penalty / no-repeat)
    output_tokens: list[int] = []  # tokens emitted after the prompt
    current = prompt_ids[0]

    for step in range(MAX_SEQ_LEN):
        if step < len(prompt_ids):
            current = prompt_ids[step]

        inp = {
            "input_id": np.array([[current]], dtype=np.int32),
            "position_id": np.array([[step]], dtype=np.int32),
            "encoder_hidden_states": enc_out.astype(np.float32),
            "cross_attention_mask": cross_mask.astype(np.float32),
            "attention_mask": np.zeros((1, 1, 1, step + 1), dtype=np.float32),
        }
        for i in range(8):
            inp[f"k_cache_{i}"] = k_caches[i]
            inp[f"v_cache_{i}"] = v_caches[i]
        out = decoder.predict(inp)
        for i in range(8):
            k_caches[i] = out[f"k_cache_{i}_out"]
            v_caches[i] = out[f"v_cache_{i}_out"]

        logits = out["logits"][0].astype(np.float32).copy()

        if repetition_penalty != 1.0 and all_tokens:
            seen = np.array(sorted(set(all_tokens)))
            pos = logits[seen] >= 0
            logits[seen] = np.where(
                pos, logits[seen] / repetition_penalty, logits[seen] * repetition_penalty
            )

        if no_repeat_ngram > 0 and len(all_tokens) >= no_repeat_ngram - 1:
            n = no_repeat_ngram
            prefix = tuple(all_tokens[-(n - 1):]) if n > 1 else ()
            forbidden: set[int] = set()
            for i in range(len(all_tokens) - (n - 1)):
                if tuple(all_tokens[i : i + n - 1]) == prefix:
                    nxt = i + n - 1
                    if nxt < len(all_tokens):
                        forbidden.add(all_tokens[nxt])
            for t in forbidden:
                logits[t] = -1e9

        nxt = int(np.argmax(logits))
        all_tokens.append(nxt)
        if step >= len(prompt_ids) - 1:
            if nxt == EOS:
                break
            output_tokens.append(nxt)
        current = nxt

    return output_tokens


import re
_BFR = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def detok(tokens, vocab):
    out, bb = [], []
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
        print(f"\nEncoding via {ename} encoder ({epath.name})...")
        t0 = time.time()
        enc = ct.models.MLModel(str(epath), compute_units=ct.ComputeUnit.CPU_ONLY)
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

    results: dict = {}
    for config_name, ename, dname in CONFIGS:
        print(f"\n=== {config_name} (enc={ename}, dec={dname}) ===")
        dec_path = DECODERS[dname]
        t0 = time.time()
        dec = ct.models.MLModel(str(dec_path), compute_units=ct.ComputeUnit.CPU_ONLY)
        print(f"  decoder loaded in {time.time() - t0:.1f}s ({dec_path.name})")
        results[config_name] = {l: [] for l in langs}
        for lang in langs:
            prompt = PROMPTS[lang]
            manifest = json.loads((ROOT / f"fleurs_samples/{lang}/manifest.json").read_text())[:N]
            for s in manifest:
                key = (lang, s["id"])
                c = enc_cache[ename][key]
                cross_mask = build_cross_mask(c["enc_out"].shape[1], c["enc_valid"])
                t0 = time.time()
                tokens = decode_cache_external(dec, c["enc_out"], cross_mask, prompt)
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
        del dec

    print("\n" + "=" * 90)
    print("CACHE-EXTERNAL DECODER: f16/q8 hybrid configs (12 FLEURS samples)")
    print("=" * 90)
    ref_cacheext = {"en_us": 10.6, "es_419": 4.9, "fr_fr": 16.8, "cmn_hans_cn": 14.1}
    header = f"{'lang':15s}  {'ref_f16':>8s}  " + "  ".join(f"{cn:>12s}" for cn, _, _ in CONFIGS)
    print(header)
    for lang in langs:
        metric = "CER" if lang in CJK else "WER"
        cells = []
        for cn, _, _ in CONFIGS:
            v = np.mean([r["wer"] for r in results[cn][lang]]) * 100
            cells.append(f"{v:>11.1f}%")
        print(f"{lang:15s}  {ref_cacheext[lang]:>7.1f}%  " + "  ".join(cells) + f"  ({metric})")

    out_json = ROOT / "cache_external_hybrid_results.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved raw results to {out_json}")


if __name__ == "__main__":
    main()
