"""Verify the "INT8 encoder + FP16 decoder" recommendation.

Runs four encoder/decoder configurations on the same 12-sample FLEURS slice,
with no EOS bias, to isolate where the q8 quality loss actually lives.

Configs:
  f16_f16  — full FP16 reference (what production quality looks like)
  q8_f16   — q8 encoder + f16 decoder (the recommended hybrid)
  f16_q8   — f16 encoder + q8 decoder (isolates decoder noise)
  q8_q8    — full q8 (shipped baseline, shows worst case)

If the encoder survives INT8 well, q8_f16 should land close to f16_f16,
and f16_q8 should land close to q8_q8. That would confirm the decoder is
the weak link.
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
VOCAB_PATH = Q8_DIR / "vocab.json"

ENCODERS = {
    "f16": F16_DIR / "cohere_encoder.mlpackage",
    "q8": Q8_DIR / "cohere_encoder.mlpackage",
}
DECODERS = {
    "f16": F16_DIR / "cohere_decoder_stateful.mlpackage",
    "q8": Q8_DIR / "cohere_decoder_stateful.mlpackage",
}

CONFIGS = [
    ("f16_f16", "f16", "f16"),
    ("q8_f16", "q8", "f16"),
    ("f16_q8", "f16", "q8"),
    ("q8_q8", "q8", "q8"),
]

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
    mask = np.zeros((1, 1, 1, enc_seq_len), dtype=np.float16)
    if enc_valid < enc_seq_len:
        mask[:, :, :, enc_valid:] = -1.0e4
    return mask


def decode(
    decoder,
    enc_out: np.ndarray,
    enc_valid: int,
    prompt_ids: list[int],
    repetition_penalty: float = 1.1,
    no_repeat_ngram: int = 3,
) -> list[int]:
    state = decoder.make_state()
    cross_mask = build_cross_mask(enc_out.shape[1], enc_valid)
    all_tokens: list[int] = []
    output_tokens: list[int] = []
    last = None
    for step in range(MAX_TOKENS):
        current = prompt_ids[step] if step < len(prompt_ids) else last
        input_id = np.array([[current]], dtype=np.int32)
        attn = np.zeros((1, 1, 1, step + 1), dtype=np.float16)
        pos = np.array([[step]], dtype=np.int32)
        out = decoder.predict(
            {
                "input_id": input_id,
                "encoder_hidden_states": enc_out.astype(np.float16),
                "attention_mask": attn,
                "cross_attention_mask": cross_mask,
                "position_ids": pos,
            },
            state=state,
        )
        logits = out["logits"][0].astype(np.float32).copy()

        if repetition_penalty != 1.0 and all_tokens:
            seen = np.array(sorted(set(all_tokens)))
            pos_m = logits[seen] >= 0
            logits[seen] = np.where(
                pos_m, logits[seen] / repetition_penalty, logits[seen] * repetition_penalty
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

        next_tok = int(np.argmax(logits))
        last = next_tok
        all_tokens.append(next_tok)
        if step >= len(prompt_ids) - 1:
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

    # Pre-compute mel features (shared across all configs)
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

    # Cache encoder outputs per encoder kind
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

    # Per-config results
    results: dict = {}
    for config_name, ename, dname in CONFIGS:
        print(f"\n=== {config_name} (enc={ename}, dec={dname}) ===")
        t0 = time.time()
        dec = ct.models.MLModel(str(DECODERS[dname]))
        print(f"  decoder loaded in {time.time() - t0:.1f}s")
        results[config_name] = {l: [] for l in langs}
        for lang in langs:
            prompt = PROMPTS[lang]
            manifest = json.loads((ROOT / f"fleurs_samples/{lang}/manifest.json").read_text())[:N]
            for s in manifest:
                key = (lang, s["id"])
                c = enc_cache[ename][key]
                tokens = decode(dec, c["enc_out"], c["enc_valid"], prompt)
                hyp = detok(tokens, vocab)
                w = wer(samples[key]["ref"], hyp, lang)
                results[config_name][lang].append({
                    "id": s["id"],
                    "ref": samples[key]["ref"],
                    "hyp": hyp,
                    "wer": w,
                    "n_tok": len(tokens),
                })
        for lang in langs:
            rs = results[config_name][lang]
            metric = "CER" if lang in CJK else "WER"
            mean_w = np.mean([r["wer"] for r in rs])
            mean_tok = np.mean([r["n_tok"] for r in rs])
            print(f"  {lang:15s} mean {metric}={mean_w * 100:6.1f}%   avg tokens={mean_tok:5.1f}")
        del dec

    # Final comparison
    print("\n" + "=" * 90)
    print("HYBRID ENCODER/DECODER COMPARISON (no EOS bias, 3 samples per language)")
    print("=" * 90)
    header = f"{'lang':15s}  " + "  ".join(f"{cn:>10s}" for cn, _, _ in CONFIGS)
    print(header)
    for lang in langs:
        metric = "CER" if lang in CJK else "WER"
        cells = []
        for cn, _, _ in CONFIGS:
            v = np.mean([r["wer"] for r in results[cn][lang]]) * 100
            cells.append(f"{v:>9.1f}%")
        print(f"{lang:15s}  " + "  ".join(cells) + f"  ({metric})")

    out_json = ROOT / "hybrid_config_results.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved raw results to {out_json}")


if __name__ == "__main__":
    main()
