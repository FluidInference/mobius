#!/usr/bin/env python3
"""
Argmax-style CoreML TTS Inference

Runs the 6-model Argmax-style CoreML pipeline end-to-end to synthesize speech:

  TextProjector  — text token → embedding [1, 1024, 1, 1]
  CodeEmbedder   — codec token → embedding [1, 1024, 1, 1]
  CodeDecoder    — 28-layer transformer, KV-cached, generates CB0 tokens
  MultiCodeDecoder — 5-layer transformer, generates CB1-CB15 in one pass
  MultiCodeEmbedder — CB1-15 token → embedding (used by CodeDecoder for context)
  SpeechDecoder  — 16 codebook tokens → 1920 audio samples (80ms @ 24kHz)

Pipeline flow:
  1. Build dual-embedding prefill sequence:
     - Each position = TextProjector(text_token) + CodeEmbedder(codec_token)
     - Sequence: [role(3)] [control(5)] [text+eos(N+1)] [final_bos(1)]
  2. CodeDecoder prefills on combined embeddings, then generates CB0 tokens
  3. For each CB0 frame, MultiCodeDecoder predicts CB1-CB15 in 2 steps:
     - Feed hidden_states from CodeDecoder
     - Feed CB0 embedding → read all 15 codebook logits at once
  4. SpeechDecoder converts each [16, 1] code frame to 1920 audio samples

Usage:
    python inference.py "Hello world, this is a test."
    python inference.py "Hello world" --greedy
    python inference.py "Hello world" --output hello.wav
"""

import numpy as np
import struct
import time
import os
import sys
import argparse


SAMPLE_RATE = 24000
MAX_CODEC_TOKENS = 125  # 10s at 12Hz
EOS_TOKEN = 2150
CODEC_VOCAB_SIZE = 2048

# Codec control token IDs (from 0.6B talker config)
CODEC_PAD_ID = 2148
CODEC_BOS_ID = 2149
CODEC_EOS_ID = 2150
CODEC_THINK_ID = 2154
CODEC_NOTHINK_ID = 2155
CODEC_THINK_BOS_ID = 2156
CODEC_THINK_EOS_ID = 2157
CODEC_LANG_IDS = {
    "chinese": 2055, "english": 2050, "german": 2053, "italian": 2070,
    "portuguese": 2071, "spanish": 2054, "japanese": 2058, "korean": 2064,
    "french": 2061, "russian": 2069,
}

# TTS special token IDs (from main config)
TTS_PAD_TOKEN_ID = 151671
TTS_BOS_TOKEN_ID = 151672
TTS_EOS_TOKEN_ID = 151673

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "argmax_models")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_0.6b")


def write_wav(filename, samples, sr=24000):
    """Write float32 samples to 16-bit PCM WAV."""
    samples = np.clip(np.array(samples, dtype=np.float32), -1.0, 1.0)
    n = len(samples)
    with open(filename, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + n * 2))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", n * 2))
        pcm = (samples * 32767).astype(np.int16)
        f.write(pcm.tobytes())


def apply_repetition_penalty(logits, generated_ids, penalty=1.05):
    """Apply repetition penalty to logits for already-generated tokens."""
    if not generated_ids or penalty == 1.0:
        return logits
    for token_id in set(generated_ids):
        if token_id < len(logits):
            if logits[token_id] > 0:
                logits[token_id] /= penalty
            else:
                logits[token_id] *= penalty
    return logits


def sample_top_k(logits, temperature=0.9, top_k=50):
    """Sample from logits with temperature and top-k."""
    logits = logits.copy().flatten().astype(np.float64)
    logits /= temperature
    if 0 < top_k < len(logits):
        threshold = np.sort(logits)[-top_k]
        logits[logits < threshold] = -1e9
    exp_l = np.exp(logits - np.max(logits))
    probs = exp_l / exp_l.sum()
    return int(np.random.choice(len(probs), p=probs))


def tokenize(text):
    """Tokenize text into the full input sequence for Qwen3-TTS.

    The official format is:
        <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n

    The model generates codec tokens after the final "assistant\n" prompt.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)

    IM_START = 151644
    IM_END = 151645

    text_ids = tok(text, add_special_tokens=False).input_ids
    assistant_ids = tok("assistant", add_special_tokens=False).input_ids
    newline_id = tok("\n", add_special_tokens=False).input_ids  # [198]

    # <|im_start|> assistant \n text <|im_end|> \n <|im_start|> assistant \n
    return ([IM_START] + assistant_ids + newline_id + text_ids +
            [IM_END] + newline_id + [IM_START] + assistant_ids + newline_id)


def load_models(cpu_only=False):
    """Load all 6 CoreML models."""
    import coremltools as ct

    compute_units = ct.ComputeUnit.CPU_ONLY if cpu_only else ct.ComputeUnit.ALL
    models = {}
    names = ["TextProjector", "CodeEmbedder", "MultiCodeEmbedder",
             "CodeDecoder", "MultiCodeDecoder", "SpeechDecoder"]

    for name in names:
        path = os.path.join(MODELS_DIR, f"{name}.mlpackage")
        if not os.path.exists(path):
            print(f"ERROR: {path} not found")
            sys.exit(1)
        models[name] = ct.models.MLModel(path, compute_units=compute_units)

    return models


def _text_proj(model, token_id):
    """TextProjector: text_projection(text_embeddings(token)) → [1, 1024, 1, 1]."""
    return model.predict({"input_ids": np.array([token_id], dtype=np.int32)})["input_embeds"]


def _code_emb(model, token_id):
    """CodeEmbedder: get_input_embeddings(token) → [1, 1024, 1, 1]."""
    return model.predict({"input_ids": np.array([token_id], dtype=np.int32)})["input_embeds"]


def build_prefill_embeddings(text, models, speaker_embedding=None, lang="english"):
    """Build the dual-embedding prefill sequence matching PyTorch generate().

    Every position gets: TextProjector(text_token) + CodeEmbedder(codec_token).

    Sequence layout (non-streaming, English, with optional speaker):
      [0:3]       role:    text_proj([im_start, assistant, newline])  (no codec overlay)
      [3:7]       ctrl:    tts_pad + codec_emb([think, think_bos, lang_en, think_eos])
      [7]         speaker: tts_pad + speaker_embedding  (only if speaker_embedding provided)
      [7/8]       ctrl:    tts_bos + codec_emb(codec_pad)
      [8/9:+N]    text:    text_proj(text_tokens) + codec_emb(codec_pad) each
      [+N]        eos:     text_proj(tts_eos) + codec_emb(codec_pad)
      [+N+1]      final:   tts_pad + codec_emb(codec_bos)
    """
    text_proj = models["TextProjector"]
    code_emb = models["CodeEmbedder"]

    token_ids = tokenize(text)

    # Role tokens: first 3 of input_id = [im_start, assistant, newline]
    # Text tokens: input_id[3:-5] = actual text
    # Trailing: input_id[-5:] = [im_end, newline, im_start, assistant, newline]
    role_ids = token_ids[:3]       # [151644, 77091, 198]
    text_ids = token_ids[3:-5]     # actual text tokens
    n_text = len(text_ids)

    has_speaker = speaker_embedding is not None
    print(f"  Tokens: {len(token_ids)} (role=3, text={n_text}, trailing=5, speaker={'yes' if has_speaker else 'no'})")

    embeds = []

    # [0:3] Role: text_proj only, no codec overlay
    for tid in role_ids:
        embeds.append(_text_proj(text_proj, tid))

    # [3:7] Control: tts_pad + codec_emb for think tokens
    tts_pad = _text_proj(text_proj, TTS_PAD_TOKEN_ID)
    tts_bos = _text_proj(text_proj, TTS_BOS_TOKEN_ID)
    tts_eos = _text_proj(text_proj, TTS_EOS_TOKEN_ID)

    lang_id = CODEC_LANG_IDS.get(lang, CODEC_LANG_IDS["english"])
    codec_ctrl_tokens = [CODEC_THINK_ID, CODEC_THINK_BOS_ID, lang_id, CODEC_THINK_EOS_ID]
    for ctok in codec_ctrl_tokens:
        embeds.append(tts_pad + _code_emb(code_emb, ctok))

    # [7] Optional speaker embedding: tts_pad + speaker_embed
    if has_speaker:
        # Speaker embedding is [1024], reshape to [1, 1024, 1, 1] to match codec embedding shape
        spk = speaker_embedding.flatten().astype(np.float32)
        spk_emb = spk.reshape(1, len(spk), 1, 1)
        embeds.append(tts_pad + spk_emb)

    # Control: tts_bos + codec_emb(codec_pad)
    embeds.append(tts_bos + _code_emb(code_emb, CODEC_PAD_ID))

    # Text: text_proj(token) + codec_emb(codec_pad)
    codec_pad_emb = _code_emb(code_emb, CODEC_PAD_ID)
    for tid in text_ids:
        embeds.append(_text_proj(text_proj, tid) + codec_pad_emb)

    # EOS: text_proj(tts_eos) + codec_emb(codec_pad)
    embeds.append(tts_eos + codec_pad_emb)

    # Final: tts_pad + codec_emb(codec_bos)
    embeds.append(tts_pad + _code_emb(code_emb, CODEC_BOS_ID))

    print(f"  Prefill sequence: {len(embeds)} embeddings")
    return embeds


def synthesize(text, models, greedy=False, seed=42, speaker_embedding=None, lang="english"):
    """Run the full TTS pipeline and return audio samples."""
    np.random.seed(seed)

    code_emb = models["CodeEmbedder"]
    code_dec = models["CodeDecoder"]
    multi_dec = models["MultiCodeDecoder"]
    speech_dec = models["SpeechDecoder"]

    sample_fn = (lambda l: int(np.argmax(l.flatten()))) if greedy else sample_top_k

    # ─── 1. Build prefill embeddings ─────────────────────────────
    t0 = time.time()
    prefill_embeds = build_prefill_embeddings(text, models, speaker_embedding=speaker_embedding, lang=lang)
    print(f"  Embedding build: {time.time()-t0:.2f}s")

    # ─── 2. CodeDecoder prefill ──────────────────────────────────
    t0 = time.time()
    KV_LEN = 256
    key_cache = np.zeros((1, 28672, 1, KV_LEN), dtype=np.float16)
    value_cache = np.zeros((1, 28672, 1, KV_LEN), dtype=np.float16)
    pos = 0

    for emb in prefill_embeds:
        key_mask = np.full((1, KV_LEN), -1e4, dtype=np.float16)
        key_mask[0, :pos + 1] = 0.0
        update_mask = np.zeros((1, KV_LEN), dtype=np.float16)
        update_mask[0, pos] = 1.0

        out = code_dec.predict({
            "input_embeds": emb.astype(np.float16),
            "cache_length": np.array([pos], dtype=np.int32),
            "key_padding_mask": key_mask,
            "kv_cache_update_mask": update_mask,
            "key_cache": key_cache,
            "value_cache": value_cache,
        })
        key_cache = out["new_key_cache"]
        value_cache = out["new_value_cache"]
        pos += 1

    hidden = out["hidden_states"]
    logits = out["logits"].flatten().astype(np.float64)
    # Suppress tokens [2048, 3072) EXCEPT EOS (2150) — matches PyTorch's suppress_tokens
    eos_logit = logits[EOS_TOKEN]
    logits[CODEC_VOCAB_SIZE:] = -1e9
    logits[EOS_TOKEN] = eos_logit  # restore EOS
    # But suppress EOS for first token (min_new_tokens=2)
    logits[EOS_TOKEN] = -1e9
    cb0 = sample_fn(logits)
    generated_cb0 = [cb0]  # Track generated tokens for repetition penalty

    prefill_time = time.time() - t0
    print(f"  Prefill: {prefill_time:.2f}s ({len(prefill_embeds)} positions), first CB0={cb0}")

    # ─── 3. Autoregressive decode ────────────────────────────────
    t0 = time.time()
    all_frames = []
    MCD_KV_LEN = 16

    # Cache tts_pad_embed for decode loop (added to every decode step input)
    tts_pad = _text_proj(models["TextProjector"], TTS_PAD_TOKEN_ID)
    multi_emb = models["MultiCodeEmbedder"]

    for step in range(MAX_CODEC_TOKENS):
        # --- MultiCodeDecoder: autoregressive CB1-CB15 prediction ---
        # PyTorch's code_predictor generates CBs autoregressively:
        #   Prefill: [hidden, CB0_embed] → lm_head[0] → CB1
        #   Step 1:  CB1 via embed[0]    → lm_head[1] → CB2
        #   Step 2:  CB2 via embed[1]    → lm_head[2] → CB3
        #   ...
        #   Step 14: CB14 via embed[13]  → lm_head[14] → CB15
        mcd_key = np.zeros((1, 5120, 1, MCD_KV_LEN), dtype=np.float16)
        mcd_val = np.zeros((1, 5120, 1, MCD_KV_LEN), dtype=np.float16)

        # Position 0: feed hidden_states from CodeDecoder
        mask = np.full((1, MCD_KV_LEN), -1e4, dtype=np.float16)
        mask[0, 0] = 0.0
        umask = np.zeros((1, MCD_KV_LEN), dtype=np.float16)
        umask[0, 0] = 1.0

        mcd_out = multi_dec.predict({
            "input_embeds": hidden.astype(np.float16),
            "cache_length": np.array([0], dtype=np.int32),
            "key_cache": mcd_key, "value_cache": mcd_val,
            "key_padding_mask": mask, "kv_cache_update_mask": umask,
        })
        mcd_key = mcd_out["new_key_cache"]
        mcd_val = mcd_out["new_value_cache"]

        # Position 1: feed CB0 embedding → lm_head[0] → CB1
        cb0_emb = _code_emb(code_emb, cb0)
        mask = np.full((1, MCD_KV_LEN), -1e4, dtype=np.float16)
        mask[0, :2] = 0.0
        umask = np.zeros((1, MCD_KV_LEN), dtype=np.float16)
        umask[0, 1] = 1.0

        mcd_out = multi_dec.predict({
            "input_embeds": cb0_emb.astype(np.float16),
            "cache_length": np.array([1], dtype=np.int32),
            "key_cache": mcd_key, "value_cache": mcd_val,
            "key_padding_mask": mask, "kv_cache_update_mask": umask,
        })
        mcd_key = mcd_out["new_key_cache"]
        mcd_val = mcd_out["new_value_cache"]

        # After prefill of [hidden, CB0], generation_steps=0, lm_head[0] → CB1
        cb1_logits = mcd_out["all_logits"][0, 0, :]  # lm_head[0]
        cb_tokens = [sample_fn(cb1_logits)]  # CB1

        # Positions 2-15: autoregressive decode for CB2-CB15
        for cb_step in range(1, 15):
            # Feed previous CB token using its embedding table
            # generation_steps=cb_step: embed with get_input_embeddings()[cb_step-1]
            # MultiCodeEmbedder linearized: (cb_step-1) * 2048 + token_id
            prev_cb = cb_tokens[-1]
            lin_idx = (cb_step - 1) * 2048 + prev_cb
            cb_emb = multi_emb.predict({"input_ids": np.array([lin_idx], dtype=np.int32)})["input_embeds"]

            mcd_pos = cb_step + 1  # position in MCD sequence (0=hidden, 1=CB0, 2=CB1, ...)
            mask = np.full((1, MCD_KV_LEN), -1e4, dtype=np.float16)
            mask[0, :mcd_pos + 1] = 0.0
            umask = np.zeros((1, MCD_KV_LEN), dtype=np.float16)
            umask[0, mcd_pos] = 1.0

            mcd_out = multi_dec.predict({
                "input_embeds": cb_emb.astype(np.float16),
                "cache_length": np.array([mcd_pos], dtype=np.int32),
                "key_cache": mcd_key, "value_cache": mcd_val,
                "key_padding_mask": mask, "kv_cache_update_mask": umask,
            })
            mcd_key = mcd_out["new_key_cache"]
            mcd_val = mcd_out["new_value_cache"]

            # lm_head[cb_step] → CB(cb_step+1)
            cb_logits = mcd_out["all_logits"][0, cb_step, :]
            cb_tokens.append(sample_fn(cb_logits))

        frame = [cb0] + cb_tokens  # [16]
        all_frames.append(frame)

        if step < 3 or step % 20 == 0:
            print(f"    [{step:3d}] CB0={cb0:4d}  CB1={cb_tokens[0]:4d}  CB8={cb_tokens[7]:4d}  CB15={cb_tokens[14]:4d}")

        # --- CodeDecoder: next CB0 ---
        # Input = sum(CodeEmbedder(cb0), MultiCodeEmbedder(cb1..cb15)) + tts_pad_embed
        # This matches PyTorch: codec_hiddens.sum() + trailing_text_hidden
        codec_sum = cb0_emb.copy()  # Start with CB0 embedding
        for cb_idx in range(15):
            lin_idx = cb_idx * 2048 + cb_tokens[cb_idx]  # linearized index
            cb_emb = multi_emb.predict({"input_ids": np.array([lin_idx], dtype=np.int32)})["input_embeds"]
            codec_sum = codec_sum + cb_emb
        decode_input = codec_sum + tts_pad  # Add tts_pad_embed overlay

        key_mask = np.full((1, KV_LEN), -1e4, dtype=np.float16)
        key_mask[0, :pos + 1] = 0.0
        update_mask = np.zeros((1, KV_LEN), dtype=np.float16)
        update_mask[0, pos] = 1.0

        out = code_dec.predict({
            "input_embeds": decode_input.astype(np.float16),
            "cache_length": np.array([pos], dtype=np.int32),
            "key_padding_mask": key_mask,
            "kv_cache_update_mask": update_mask,
            "key_cache": key_cache,
            "value_cache": value_cache,
        })
        key_cache = out["new_key_cache"]
        value_cache = out["new_value_cache"]
        hidden = out["hidden_states"]
        pos += 1

        logits = out["logits"].flatten().astype(np.float64)
        # Suppress [2048, 3072) EXCEPT EOS (2150)
        eos_logit = logits[EOS_TOKEN]
        logits[CODEC_VOCAB_SIZE:] = -1e9
        if step >= 1:  # Allow EOS after min_new_tokens=2
            logits[EOS_TOKEN] = eos_logit
        logits = apply_repetition_penalty(logits, generated_cb0)
        cb0 = sample_fn(logits)
        generated_cb0.append(cb0)

        if cb0 == EOS_TOKEN:
            print(f"  EOS at step {step + 1}")
            break

        if pos >= KV_LEN - 1:
            print(f"  KV cache full at step {step + 1}")
            break

    decode_time = time.time() - t0
    num_frames = len(all_frames)
    fps = num_frames / max(decode_time, 0.001)
    print(f"  Decode: {num_frames} frames in {decode_time:.2f}s ({fps:.1f} frames/s)")

    # ─── 4. SpeechDecoder: codes → audio ─────────────────────────
    # Batch all frames into a single call. The SpeechDecoder accepts [1, 16, 125]
    # (fixed T=125). Pad shorter sequences and trim the output.
    SPEECH_DEC_T = 125
    t0 = time.time()
    codes = np.array(all_frames, dtype=np.int32)  # [N, 16]
    padded = np.zeros((SPEECH_DEC_T, 16), dtype=np.int32)
    padded[:num_frames] = codes
    out = speech_dec.predict({"audio_codes": padded.T.reshape(1, 16, SPEECH_DEC_T)})
    audio = out["audio"].astype(np.float32).flatten()[:num_frames * 1920]
    audio_time = time.time() - t0
    duration = len(audio) / SAMPLE_RATE
    print(f"  Audio: {num_frames} frames → {duration:.2f}s in {audio_time:.2f}s")

    total = prefill_time + decode_time + audio_time
    rtf = total / max(duration, 0.001)
    print(f"  Total: {total:.2f}s (RTF={rtf:.2f}x)")

    return audio.astype(np.float32), all_frames


def transcribe(wav_path, lang="english"):
    """Transcribe a WAV file using Whisper."""
    # Map our language names to Whisper language codes
    whisper_lang = {"chinese": "zh", "english": "en", "german": "de", "italian": "it",
                    "portuguese": "pt", "spanish": "es", "japanese": "ja", "korean": "ko",
                    "french": "fr", "russian": "ru"}.get(lang, "en")
    try:
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(wav_path, language=whisper_lang)
        return result["text"].strip()
    except ImportError:
        pass

    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("base", compute_type="int8")
        segments, _ = model.transcribe(wav_path, language=whisper_lang)
        return " ".join(seg.text.strip() for seg in segments)
    except ImportError:
        pass

    # Try mlx-whisper
    try:
        import mlx_whisper
        result = mlx_whisper.transcribe(wav_path, language=whisper_lang)
        return result["text"].strip()
    except ImportError:
        pass

    return None


def main():
    parser = argparse.ArgumentParser(description="Argmax CoreML TTS Inference")
    parser.add_argument("text", nargs="?", default="Hello world, this is a test of the text to speech system.")
    parser.add_argument("--output", "-o", default=None, help="Output WAV path")
    parser.add_argument("--greedy", action="store_true", help="Greedy decoding (deterministic)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-transcribe", action="store_true", help="Skip Whisper transcription")
    parser.add_argument("--cpu-only", action="store_true", help="Use CPU_ONLY compute (higher precision, slower)")
    parser.add_argument("--speaker", type=str, default=None,
                        help="Path to speaker embedding .npy file (1024-dim x-vector)")
    parser.add_argument("--ref-audio", type=str, default=None,
                        help="Path to reference audio WAV for voice cloning (extracts x-vector automatically)")
    parser.add_argument("--lang", type=str, default="english",
                        choices=list(CODEC_LANG_IDS.keys()),
                        help="Language for synthesis (default: english)")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.expanduser("~/Desktop/argmax_coreml_tts.wav")

    # Load speaker embedding
    speaker_embedding = None
    if args.ref_audio:
        from extract_speaker_embedding import read_wav, load_model, extract_embedding, resample_if_needed
        audio, sr = read_wav(args.ref_audio)
        if sr != 24000:
            audio = resample_if_needed(audio, sr, 24000)
        model = load_model(MODEL_PATH)
        speaker_embedding = extract_embedding(model, audio, 24000)
        print(f"Speaker embedding extracted from: {args.ref_audio} (norm={np.linalg.norm(speaker_embedding):.2f})")
    elif args.speaker:
        speaker_embedding = np.load(args.speaker)
        print(f"Speaker embedding: {args.speaker} (shape={speaker_embedding.shape})")

    print("=" * 60)
    print("Argmax-style CoreML TTS")
    print("=" * 60)
    print(f"Text: \"{args.text}\"")
    print(f"Mode: {'greedy' if args.greedy else f'top-k (seed={args.seed})'}")
    print(f"Compute: {'CPU_ONLY' if args.cpu_only else 'ALL'}")
    print(f"Language: {args.lang}")
    print(f"Speaker: {'x-vector' if speaker_embedding is not None else 'none'}")
    print()

    print("Loading models...")
    t0 = time.time()
    models = load_models(cpu_only=args.cpu_only)
    print(f"Models loaded in {time.time()-t0:.1f}s\n")

    print("Synthesizing...")
    audio, frames = synthesize(args.text, models, greedy=args.greedy, seed=args.seed,
                               speaker_embedding=speaker_embedding, lang=args.lang)

    # Normalize and save
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.9
    write_wav(args.output, audio)

    # Stats
    codes = np.array(frames)
    print(f"\nCodebook stats:")
    print(f"  Frames: {codes.shape[0]}")
    for i in [0, 1, 7, 15]:
        col = codes[:, i]
        print(f"  CB{i:2d}: {len(np.unique(col)):3d} unique, range [{col.min():4d}, {col.max():4d}]")

    print(f"\nSaved: {args.output}")
    print(f"Duration: {len(audio)/SAMPLE_RATE:.2f}s")

    # Transcribe
    if not args.no_transcribe:
        print(f"\nTranscribing with Whisper...")
        transcript = transcribe(args.output, lang=args.lang)
        if transcript:
            print(f"  Input:      \"{args.text}\"")
            print(f"  Transcript: \"{transcript}\"")
        else:
            print("  Whisper not available (install: pip install openai-whisper / faster-whisper / mlx-whisper)")


if __name__ == "__main__":
    main()
