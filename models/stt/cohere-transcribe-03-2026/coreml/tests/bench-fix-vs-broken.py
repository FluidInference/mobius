"""End-to-end benchmark: BROKEN pipeline vs FIXED pipeline on FLEURS.

Uses the locally available cache-external encoder + decoder from
`hf-upload/cohere-transcribe-cache-external-coreml/`. Runs two inference
paths on the same audio samples:

    OLD: shipped mel preprocessing (from hf-upload/example.py) +
         cross_attention_mask all-ones + greedy argmax.

    NEW: v2 mel preprocessing (CohereMelSpectrogram, matches HF
         CohereAsrFeatureExtractor) + cross_attention_mask that masks
         padded encoder frames + greedy argmax with repetition penalty
         and no-repeat-3gram.

Reports transcription + WER per-sample per-path. This should show
quantitatively that the feature extraction change is the dominant fix.

Usage:
    uv run python tests/bench-fix-vs-broken.py
    uv run python tests/bench-fix-vs-broken.py --language en_us --n 5
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
import sentencepiece as spm
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "tools"))

from cohere_features_v2 import (  # noqa: E402
    CohereMelSpectrogram,
    pad_or_truncate_to_fixed,
)

HF_UPLOAD = ROOT / "hf-upload/cohere-transcribe-cache-external-coreml"
ENCODER_PATH = HF_UPLOAD / "cohere_encoder.mlpackage"
DECODER_PATH = HF_UPLOAD / "cohere_decoder_cache_external.mlpackage"
TOKENIZER_PATH = HF_UPLOAD / "tokenizer.model"

SAMPLE_RATE = 16000
MEL_FRAMES_FIXED = 3500
ENCODER_FRAMES_FIXED = 438
MAX_SEQ_LEN = 108
START_TOKEN = 4
EOS_TOKEN = 3

LANGUAGE_PROMPTS = {
    "en_us": [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13],
    "es_419": [13764, 7, 4, 16, 169, 169, 5, 9, 11, 13],
    "fr_fr": [13764, 7, 4, 16, 69, 69, 5, 9, 11, 13],
    "cmn_hans_cn": [13764, 7, 4, 16, 50, 50, 5, 9, 11, 13],
}


# --------------------------------------------------------------------------
# OLD, broken preprocessing (copied verbatim from hf-upload/example.py)
# --------------------------------------------------------------------------
def old_mel(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=400, hop_length=160,
        n_mels=128, fmin=0, fmax=8000,
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel + 80) / 80
    mel = np.clip(mel, -1, 1)
    return mel.astype(np.float32)


def old_pad_mel(mel: np.ndarray, target: int = MEL_FRAMES_FIXED):
    _, n_frames = mel.shape
    if n_frames >= target:
        return mel[:, :target], n_frames
    padded = np.zeros((mel.shape[0], target), dtype=np.float32)
    padded[:, :n_frames] = mel
    return padded, n_frames


# --------------------------------------------------------------------------
# Decode helpers
# --------------------------------------------------------------------------
def decode_cache_external(
    decoder,
    encoder_hidden: np.ndarray,
    cross_mask: np.ndarray,
    prompt_ids: list[int] | None = None,
    repetition_penalty: float = 1.0,
    no_repeat_ngram: int = 0,
    max_tokens: int = MAX_SEQ_LEN,
) -> list[int]:
    k_caches = [np.zeros((1, 8, MAX_SEQ_LEN, 128), dtype=np.float32) for _ in range(8)]
    v_caches = [np.zeros((1, 8, MAX_SEQ_LEN, 128), dtype=np.float32) for _ in range(8)]

    tokens: list[int] = []  # tokens emitted after the prompt
    all_tokens: list[int] = []  # every predicted token (for rep penalty)
    current = prompt_ids[0] if prompt_ids else START_TOKEN

    for step in range(max_tokens):
        if prompt_ids and step < len(prompt_ids):
            current = prompt_ids[step]

        inp = {
            "input_id": np.array([[current]], dtype=np.int32),
            "position_id": np.array([[step]], dtype=np.int32),
            "encoder_hidden_states": encoder_hidden.astype(np.float32),
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
                pos,
                logits[seen] / repetition_penalty,
                logits[seen] * repetition_penalty,
            )

        if no_repeat_ngram > 0 and len(all_tokens) >= no_repeat_ngram - 1:
            n = no_repeat_ngram
            prefix = tuple(all_tokens[-(n - 1):]) if n > 1 else ()
            for i in range(len(all_tokens) - (n - 1)):
                if tuple(all_tokens[i : i + n - 1]) == prefix:
                    nxt = i + n - 1
                    if nxt < len(all_tokens):
                        logits[all_tokens[nxt]] = -1e9

        nxt = int(np.argmax(logits))
        all_tokens.append(nxt)

        # Emit only tokens after the prompt has been consumed.
        if prompt_ids is None or step >= len(prompt_ids) - 1:
            if nxt == EOS_TOKEN:
                break
            tokens.append(nxt)

        current = nxt

    return tokens


def tokens_to_text(tokens: list[int], sp: "spm.SentencePieceProcessor") -> str:
    """Decode token IDs via SentencePiece so CJK byte-fallback pieces
    (``<0xE7><0xAF><0x87>`` etc.) are reassembled into real UTF-8.

    We strip special/EOS tokens before decoding because decode_ids would
    otherwise render them as literal placeholder strings.
    """
    vocab_size = sp.get_piece_size()
    clean: list[int] = []
    for t in tokens:
        if t <= 4 or t == EOS_TOKEN or t >= vocab_size:
            continue
        piece = sp.id_to_piece(t)
        if piece.startswith("<|"):
            continue
        clean.append(t)
    return sp.decode_ids(clean).strip()


# --------------------------------------------------------------------------
# WER
# --------------------------------------------------------------------------
def normalize_for_wer(s: str) -> str:
    s = s.lower()
    for ch in ",.!?;:\"'()[]-":
        s = s.replace(ch, " ")
    return " ".join(s.split())


CJK_LANGS = {"cmn_hans_cn", "cmn_hant_hk", "ja_jp", "ko_kr"}


def _tokens_for_metric(s: str, language: str) -> list[str]:
    """FLEURS ships CJK refs as space-separated characters, and the model
    emits fluent text without spaces. Use character-level tokens for CJK so
    the metric is a true CER; use word-level tokens otherwise."""
    norm = normalize_for_wer(s)
    if language in CJK_LANGS:
        return [c for c in norm if not c.isspace()]
    return norm.split()


def wer(ref: str, hyp: str, language: str = "") -> float:
    r = _tokens_for_metric(ref, language)
    h = _tokens_for_metric(hyp, language)
    if not r:
        return 1.0 if h else 0.0
    # Levenshtein on words
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


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------
def run(language: str, n: int, verbose: bool = True):
    print(f"Language: {language}")
    manifest_path = ROOT / f"fleurs_samples/{language}/manifest.json"
    if not manifest_path.exists():
        print(f"  skipping — no manifest at {manifest_path}")
        return None
    manifest = json.loads(manifest_path.read_text())
    samples = manifest[:n]

    prompt_ids = LANGUAGE_PROMPTS.get(language)

    print(f"Loading encoder... ({ENCODER_PATH})")
    encoder = ct.models.MLModel(str(ENCODER_PATH))
    print(f"Loading decoder... ({DECODER_PATH})")
    decoder = ct.models.MLModel(str(DECODER_PATH))

    print("Loading tokenizer...")
    sp = spm.SentencePieceProcessor()
    sp.load(str(TOKENIZER_PATH))
    vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]

    mel_v2 = CohereMelSpectrogram()

    old_wers = []
    new_wers = []
    results = []

    for s in samples:
        audio_path = ROOT / s["audio"]
        ref = s["text"]
        audio, sr = sf.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE).astype(np.float32)

        # -------- OLD path --------
        t0 = time.time()
        mel = old_mel(audio)
        padded_mel, actual_frames = old_pad_mel(mel)
        enc_out = encoder.predict({
            "input_features": np.expand_dims(padded_mel, 0).astype(np.float32),
            "feature_length": np.array([actual_frames], dtype=np.int32),
        })["hidden_states"]
        cross_mask_old = np.ones((1, 1, 1, enc_out.shape[1]), dtype=np.float32)
        old_tokens = decode_cache_external(decoder, enc_out, cross_mask_old, prompt_ids=None)
        old_text = tokens_to_text(old_tokens, sp)
        t_old = time.time() - t0

        # -------- NEW path --------
        t0 = time.time()
        mel_new, valid_mel = mel_v2(audio)
        mel_fixed, feat_len = pad_or_truncate_to_fixed(mel_new, valid_mel)
        enc_out_new = encoder.predict({
            "input_features": mel_fixed.astype(np.float32),
            "feature_length": np.array([feat_len], dtype=np.int32),
        })["hidden_states"]
        enc_valid = min(enc_out_new.shape[1],
                        max(1, int(np.ceil(feat_len / (MEL_FRAMES_FIXED / ENCODER_FRAMES_FIXED)))))
        cross_mask_new = np.zeros((1, 1, 1, enc_out_new.shape[1]), dtype=np.float32)
        if enc_valid < enc_out_new.shape[1]:
            cross_mask_new[:, :, :, enc_valid:] = -1.0e4
        new_tokens = decode_cache_external(
            decoder, enc_out_new, cross_mask_new,
            prompt_ids=prompt_ids,
            repetition_penalty=1.1,
            no_repeat_ngram=3,
        )
        new_text = tokens_to_text(new_tokens, sp)
        t_new = time.time() - t0

        w_old = wer(ref, old_text, language)
        w_new = wer(ref, new_text, language)
        old_wers.append(w_old)
        new_wers.append(w_new)
        results.append({
            "id": s["id"],
            "duration": s["duration"],
            "ref": ref,
            "old_hyp": old_text,
            "new_hyp": new_text,
            "wer_old": w_old,
            "wer_new": w_new,
            "time_old_s": t_old,
            "time_new_s": t_new,
        })

        metric_label = "CER" if language in CJK_LANGS else "WER"
        if verbose:
            print(f"\n--- sample {s['id']} ({s['duration']:.1f}s) ---")
            print(f"  REF: {ref}")
            print(f"  OLD: {old_text}")
            print(f"       {metric_label}={w_old * 100:.1f}%  t={t_old:.1f}s")
            print(f"  NEW: {new_text}")
            print(f"       {metric_label}={w_new * 100:.1f}%  t={t_new:.1f}s")

    metric_label = "CER" if language in CJK_LANGS else "WER"
    print(f"\n=== {language} summary over {len(results)} samples ===")
    if old_wers:
        print(f"  OLD mean {metric_label}: {np.mean(old_wers) * 100:.1f}%")
        print(f"  NEW mean {metric_label}: {np.mean(new_wers) * 100:.1f}%")
    return {
        "language": language,
        "n": len(results),
        "old_mean_wer": float(np.mean(old_wers)) if old_wers else None,
        "new_mean_wer": float(np.mean(new_wers)) if new_wers else None,
        "results": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", default=None,
                    help="FLEURS language dir (en_us, fr_fr, es_419, cmn_hans_cn). Default: run all.")
    ap.add_argument("--n", type=int, default=3,
                    help="Samples per language (default 3)")
    ap.add_argument("--output", default=None, help="Write JSON report here")
    args = ap.parse_args()

    langs = [args.language] if args.language else list(LANGUAGE_PROMPTS)
    all_reports = []
    for lang in langs:
        rep = run(lang, args.n)
        if rep:
            all_reports.append(rep)

    print("\n" + "=" * 72)
    print("OVERALL SUMMARY")
    print("=" * 72)
    for r in all_reports:
        print(f"  {r['language']:15s}  n={r['n']:2d}  "
              f"OLD={r['old_mean_wer'] * 100:6.1f}%  NEW={r['new_mean_wer'] * 100:6.1f}%  "
              f"delta={(r['new_mean_wer'] - r['old_mean_wer']) * 100:+.1f}pp")

    if args.output:
        Path(args.output).write_text(json.dumps(all_reports, indent=2, ensure_ascii=False))
        print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()
