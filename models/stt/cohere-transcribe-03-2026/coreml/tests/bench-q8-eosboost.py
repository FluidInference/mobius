"""Confirm the diagnosis: q8 over-generation is caused by EOS logit sitting
2-3 units below the top token at the true sentence boundary — well within
INT8 weight-quantization noise. Applying a small constant bias to the EOS
logit (logit_bias_eos) should recover most of the quality without retraining.

Runs the q8 fixed pipeline three times:
    no boost (0.0)  — baseline, same as bench-q8-fleurs.py
    boost +2.0      — just enough to flip the common close cases
    boost +4.0      — aggressive, to check that we don't prematurely EOS

Only EN/FR/ES samples (the ones where we saw the over-generation pattern).
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
ENCODER_PATH = Q8_DIR / "cohere_encoder.mlpackage"
DECODER_PATH = Q8_DIR / "cohere_decoder_stateful.mlpackage"
VOCAB_PATH = Q8_DIR / "vocab.json"

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
    eos_bias: float = 0.0,
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

        # EOS boost (the diagnostic knob)
        if eos_bias != 0.0 and step >= len(prompt_ids) - 1:
            logits[EOS] += eos_bias

        # Repetition penalty
        if repetition_penalty != 1.0 and all_tokens:
            seen = np.array(sorted(set(all_tokens)))
            pos_m = logits[seen] >= 0
            logits[seen] = np.where(pos_m, logits[seen] / repetition_penalty, logits[seen] * repetition_penalty)

        # No-repeat n-gram
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


def detok(tokens: list[int], vocab: dict[int, str]) -> str:
    out: list[str] = []
    bb: list[int] = []
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


def norm(s: str) -> str:
    s = s.lower()
    for ch in ",.!?;:\"'()[]-":
        s = s.replace(ch, " ")
    return " ".join(s.split())


def toks(s: str, lang: str):
    n = norm(s)
    if lang in CJK:
        return [c for c in n if not c.isspace()]
    return n.split()


def wer(ref: str, hyp: str, lang: str) -> float:
    r = toks(ref, lang)
    h = toks(hyp, lang)
    if not r:
        return 1.0 if h else 0.0
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1): dp[i][0] = i
    for j in range(len(h) + 1): dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i-1] == h[j-1]: dp[i][j] = dp[i-1][j-1]
            else: dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[len(r)][len(h)] / len(r)


def main():
    encoder = ct.models.MLModel(str(ENCODER_PATH))
    decoder = ct.models.MLModel(str(DECODER_PATH))
    vocab = {int(k): v for k, v in json.loads(VOCAB_PATH.read_text()).items()}
    mel_proc = CohereMelSpectrogram()

    biases = [0.0, 2.0, 4.0]
    langs = ["en_us", "es_419", "fr_fr", "cmn_hans_cn"]
    N = 3
    results = {b: {l: [] for l in langs} for b in biases}

    for lang in langs:
        manifest = json.loads((ROOT / f"fleurs_samples/{lang}/manifest.json").read_text())[:N]
        prompt_ids = PROMPTS[lang]
        for s in manifest:
            audio, sr = sf.read(str(ROOT / s["audio"]), dtype="float32")
            if audio.ndim > 1: audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE).astype(np.float32)
            mel, valid = mel_proc(audio)
            mel_fixed, feat_len = pad_or_truncate_to_fixed(mel, valid)
            enc_out = encoder.predict({
                "input_features": mel_fixed.astype(np.float32),
                "feature_length": np.array([feat_len], dtype=np.int32),
            })["hidden_states"]
            enc_valid = min(enc_out.shape[1],
                            max(1, int(np.ceil(feat_len / (MEL_FRAMES_FIXED / ENCODER_FRAMES_FIXED)))))

            for b in biases:
                tokens = decode(decoder, enc_out, enc_valid, prompt_ids, eos_bias=b)
                hyp = detok(tokens, vocab)
                w = wer(s["text"], hyp, lang)
                results[b][lang].append({"id": s["id"], "ref": s["text"], "hyp": hyp, "wer": w, "n_tok": len(tokens)})

        # print per-language
        print(f"\n{lang}")
        metric = "CER" if lang in CJK else "WER"
        for b in biases:
            rs = results[b][lang]
            mean_w = np.mean([r["wer"] for r in rs])
            mean_tok = np.mean([r["n_tok"] for r in rs])
            print(f"  eos_bias={b:>4.1f}   mean {metric}={mean_w*100:6.1f}%   avg tokens={mean_tok:5.1f}")

    # final table
    print("\n" + "=" * 72)
    print("EOS BIAS SWEEP SUMMARY")
    print("=" * 72)
    print(f"{'lang':15s}  {'+0.0':>8s}  {'+2.0':>8s}  {'+4.0':>8s}")
    for lang in langs:
        metric = "CER" if lang in CJK else "WER"
        row = [np.mean([r["wer"] for r in results[b][lang]]) * 100 for b in biases]
        print(f"{lang:15s}  {row[0]:>6.1f}%  {row[1]:>6.1f}%  {row[2]:>6.1f}%   ({metric})")

    # Save
    Path(ROOT / "q8_eosbias_results.json").write_text(
        json.dumps({str(b): results[b] for b in biases}, indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
