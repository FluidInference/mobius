"""Diagnostic: instrument the q8 stateful decoder and dump, per step:
  - top-5 tokens + their logits
  - EOS (id=3) logit and its rank
  - the cumulative detokenized hypothesis so far

Goal: understand WHY q8 over-generates. Three candidate causes:
  (a) EOS logit is quantized so low it never wins (systematic under-estimate)
  (b) EOS is near-top but lexical tokens get a random quantization boost that
      keeps them above EOS
  (c) Model is semantically uncertain about EOS at the true boundary and
      quantization just tips a close decision

We'll know from the data which one it is.
"""
from __future__ import annotations

import json
import sys
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
    "fr_fr": [13764, 7, 4, 16, 69, 69, 5, 9, 11, 13],
}


def build_cross_mask(enc_seq_len: int, enc_valid: int) -> np.ndarray:
    mask = np.zeros((1, 1, 1, enc_seq_len), dtype=np.float16)
    if enc_valid < enc_seq_len:
        mask[:, :, :, enc_valid:] = -1.0e4
    return mask


def piece_to_str(s: str) -> str:
    # Render SentencePiece-ish piece for readable logs
    return s.replace("\u2581", "_")


def probe_sample(sample_path: Path, language: str, encoder, decoder, vocab, mel_proc):
    audio, sr = sf.read(str(sample_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
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
    cross_mask = build_cross_mask(enc_out.shape[1], enc_valid)

    prompt = PROMPTS[language]
    state = decoder.make_state()
    all_tokens: list[int] = []
    last_token = None

    print(f"\n===== {sample_path.name} ({language}) =====")
    print(f"encoder_hidden={enc_out.shape}  valid_frames={enc_valid}")
    print(f"{'step':>4} {'tok':>6} {'piece':<20} {'top1':>6} {'top1_lg':>8} "
          f"{'eos_lg':>8} {'eos_rnk':>7} {'eos_gap':>9}  hyp_so_far")

    for step in range(MAX_TOKENS):
        current = prompt[step] if step < len(prompt) else last_token
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
        logits = out["logits"][0].astype(np.float32)

        # rank of EOS
        top_indices = np.argsort(-logits)
        eos_rank = int(np.where(top_indices == EOS)[0][0])
        eos_logit = float(logits[EOS])
        top_logit = float(logits[top_indices[0]])
        top_token = int(top_indices[0])

        next_token = top_token  # raw greedy (no rep penalty, no n-gram)
        last_token = next_token
        all_tokens.append(next_token)

        # Render everything so far (post-prompt)
        post_prompt = all_tokens[max(0, len(prompt) - 1):]
        buf: list[str] = []
        byte_buf: list[int] = []
        import re
        bfr = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")
        def flush():
            if byte_buf:
                buf.append(bytes(byte_buf).decode("utf-8", errors="replace"))
                byte_buf.clear()
        for t in post_prompt:
            if t <= 4 or t == EOS:
                continue
            s = vocab.get(t, "")
            if s.startswith("<|"):
                continue
            m = bfr.match(s)
            if m is not None:
                byte_buf.append(int(m.group(1), 16))
                continue
            flush()
            buf.append(s)
        flush()
        hyp = "".join(buf).replace("\u2581", " ").strip()

        piece = piece_to_str(vocab.get(next_token, f"?{next_token}"))[:20]
        print(f"{step:>4} {next_token:>6} {piece:<20} {top_token:>6} "
              f"{top_logit:>8.3f} {eos_logit:>8.3f} {eos_rank:>7} "
              f"{top_logit - eos_logit:>9.3f}  {hyp}")

        if next_token == EOS and step >= len(prompt) - 1:
            print(f"  -> EOS at step {step}")
            break


def main():
    print(f"Loading encoder {ENCODER_PATH}")
    encoder = ct.models.MLModel(str(ENCODER_PATH))
    print(f"Loading decoder {DECODER_PATH}")
    decoder = ct.models.MLModel(str(DECODER_PATH))
    vocab_raw = json.loads(VOCAB_PATH.read_text())
    vocab = {int(k): v for k, v in vocab_raw.items()}
    mel_proc = CohereMelSpectrogram()

    # Probe 1: EN sample 0 (correct + trailing "(Thanks for the lack of a better word)")
    # Probe 2: FR sample 0 (correct + trailing gibberish)
    en_manifest = json.loads((ROOT / "fleurs_samples/en_us/manifest.json").read_text())
    fr_manifest = json.loads((ROOT / "fleurs_samples/fr_fr/manifest.json").read_text())
    probe_sample(ROOT / en_manifest[0]["audio"], "en_us", encoder, decoder, vocab, mel_proc)
    probe_sample(ROOT / fr_manifest[0]["audio"], "fr_fr", encoder, decoder, vocab, mel_proc)


if __name__ == "__main__":
    main()
