#!/usr/bin/env python3
"""
CoreML inference smoke test for Nemotron Multilingual Streaming 0.6B.

True audio-chunked streaming (1.12s chunks, pad_and_drop_preencoded=True
semantics), exercising the prompt_id input. Mirrors the English variant's
test_coreml_streaming.py with three deltas:

  1. Reads `prompt_dictionary` + `lang_tag_token_ids` from metadata.json.
  2. Resolves `--target-lang` (default "auto") to an int32 prompt_id and
     feeds it as the 6th encoder input every chunk.
  3. Detects the leading <xx-XX> token and reports it as detected_lang.

This is a smoke test, not a benchmark. For WER use benchmark_wer.py
adapted from the English variant.
"""
import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import coremltools as ct
import numpy as np
import soundfile as sf


class NemotronMultilingualCoreML:
    """Streaming CoreML inference for the multilingual Nemotron 0.6B."""

    def __init__(self, model_dir: str):
        model_dir = Path(model_dir)

        with open(model_dir / "metadata.json") as f:
            self.metadata = json.load(f)
        with open(model_dir / "tokenizer.json") as f:
            self.tokenizer = json.load(f)

        print("Loading CoreML models...")
        self.preprocessor = ct.models.MLModel(str(model_dir / "preprocessor.mlpackage"))
        self.encoder = ct.models.MLModel(str(model_dir / "encoder.mlpackage"))
        self.decoder = ct.models.MLModel(str(model_dir / "decoder.mlpackage"))
        self.joint = ct.models.MLModel(str(model_dir / "joint.mlpackage"))
        print("Models loaded.")

        self.sample_rate = self.metadata["sample_rate"]
        self.chunk_mel_frames = self.metadata["chunk_mel_frames"]
        self.pre_encode_cache = self.metadata["pre_encode_cache"]
        self.total_mel_frames = self.metadata["total_mel_frames"]
        self.blank_idx = self.metadata["blank_idx"]
        self.vocab_size = self.metadata["vocab_size"]
        self.decoder_hidden = self.metadata["decoder_hidden"]
        self.decoder_layers = self.metadata["decoder_layers"]
        self.mel_features = self.metadata.get("mel_features", 128)

        self.cache_channel_shape = self.metadata["cache_channel_shape"]
        self.cache_time_shape = self.metadata["cache_time_shape"]

        # Prompt-specific bits
        self.prompt_dictionary: dict = self.metadata.get("prompt_dictionary", {})
        self.lang_tag_token_ids: set = set(self.metadata.get("lang_tag_token_ids", []))
        self.default_prompt_id: int = int(self.metadata.get("default_prompt_id", 101))
        self.num_prompts: int = int(self.metadata.get("num_prompts", 128))

        self.chunk_samples = int(self.chunk_mel_frames * 0.01 * self.sample_rate)

    # ---- resolution helpers ----------------------------------------------

    def resolve_prompt_id(self, target_lang: str) -> int:
        if target_lang in self.prompt_dictionary:
            return int(self.prompt_dictionary[target_lang])
        if len(target_lang) == 2:
            for k, v in self.prompt_dictionary.items():
                if k.lower().startswith(target_lang.lower() + "-"):
                    return int(v)
        return self.default_prompt_id

    def piece_for_id(self, tok_id: int) -> str:
        return self.tokenizer.get(str(tok_id), "")

    # ---- state initializers ---------------------------------------------

    def _get_initial_cache(self):
        return (
            np.zeros(self.cache_channel_shape, dtype=np.float32),
            np.zeros(self.cache_time_shape, dtype=np.float32),
            np.array([0], dtype=np.int32),
        )

    def _get_initial_decoder_state(self):
        h = np.zeros((self.decoder_layers, 1, self.decoder_hidden), dtype=np.float32)
        c = np.zeros((self.decoder_layers, 1, self.decoder_hidden), dtype=np.float32)
        return h, c

    # ---- output decoding -------------------------------------------------

    def _decode_tokens(self, tokens: List[int]) -> Tuple[Optional[str], str]:
        detected_lang: Optional[str] = None
        body: List[int] = []
        for tok in tokens:
            if tok == self.blank_idx or tok >= self.vocab_size:
                continue
            if tok in self.lang_tag_token_ids:
                # First lang tag becomes the detected language. Subsequent
                # ones (rare; shouldn't happen mid-utterance) are stripped.
                if detected_lang is None:
                    detected_lang = self.piece_for_id(tok)
                continue
            body.append(tok)
        text = "".join(self.piece_for_id(t) for t in body)
        text = text.replace("\u2581", " ").strip()
        return detected_lang, text

    # ---- streaming inference --------------------------------------------

    def transcribe_streaming(
        self,
        audio: np.ndarray,
        target_lang: str = "auto",
    ) -> Tuple[Optional[str], str, int]:
        """Returns (detected_lang_tag, text, prompt_id_used)."""
        audio = audio.astype(np.float32)
        total_samples = len(audio)

        prompt_id = self.resolve_prompt_id(target_lang)
        prompt_id_input = np.array([prompt_id], dtype=np.int32)

        cache_channel, cache_time, cache_len = self._get_initial_cache()
        h, c = self._get_initial_decoder_state()
        last_token = self.blank_idx
        all_tokens: List[int] = []
        mel_cache = None
        audio_offset = 0

        while audio_offset < total_samples:
            chunk_end = min(audio_offset + self.chunk_samples, total_samples)
            audio_chunk = audio[audio_offset:chunk_end]
            if len(audio_chunk) < self.chunk_samples:
                audio_chunk = np.pad(audio_chunk, (0, self.chunk_samples - len(audio_chunk)))
            audio_chunk = audio_chunk.reshape(1, -1)
            audio_len = np.array([audio_chunk.shape[1]], dtype=np.int32)

            preproc_out = self.preprocessor.predict({
                "audio": audio_chunk,
                "audio_length": audio_len,
            })
            chunk_mel = preproc_out["mel"]

            if mel_cache is not None:
                input_mel = np.concatenate([mel_cache, chunk_mel], axis=2)
            else:
                input_mel = np.pad(
                    chunk_mel, ((0, 0), (0, 0), (self.pre_encode_cache, 0)), mode="constant"
                )

            cur = input_mel.shape[2]
            if cur < self.total_mel_frames:
                input_mel = np.pad(
                    input_mel, ((0, 0), (0, 0), (0, self.total_mel_frames - cur)),
                    mode="constant",
                )
            elif cur > self.total_mel_frames:
                input_mel = input_mel[:, :, : self.total_mel_frames]

            mel_cache = (
                chunk_mel[:, :, -self.pre_encode_cache:]
                if chunk_mel.shape[2] >= self.pre_encode_cache
                else chunk_mel
            )

            enc_out = self.encoder.predict({
                "mel": input_mel.astype(np.float32),
                "mel_length": np.array([self.total_mel_frames], dtype=np.int32),
                "cache_channel": cache_channel,
                "cache_time": cache_time,
                "cache_len": cache_len,
                "prompt_id": prompt_id_input,
            })

            encoded = enc_out["encoded"]
            cache_channel = enc_out["cache_channel_out"]
            cache_time = enc_out["cache_time_out"]
            cache_len = enc_out["cache_len_out"]

            num_enc_frames = encoded.shape[2]
            for t in range(num_enc_frames):
                enc_step = encoded[:, :, t : t + 1]

                for _ in range(10):  # max symbols per frame
                    dec_out = self.decoder.predict({
                        "token": np.array([[last_token]], dtype=np.int32),
                        "token_length": np.array([1], dtype=np.int32),
                        "h_in": h,
                        "c_in": c,
                    })
                    decoder_out = dec_out["decoder_out"]
                    h_new = dec_out["h_out"]
                    c_new = dec_out["c_out"]

                    joint_out = self.joint.predict({
                        "encoder": enc_step.astype(np.float32),
                        "decoder": decoder_out[:, :, :1].astype(np.float32),
                    })
                    logits = joint_out["logits"]
                    pred_token = int(np.argmax(logits[0, 0, 0, :]))

                    if pred_token == self.blank_idx:
                        break
                    all_tokens.append(pred_token)
                    last_token = pred_token
                    h = h_new
                    c = c_new

            audio_offset += self.chunk_samples

        detected, text = self._decode_tokens(all_tokens)
        return detected, text, prompt_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing the 4 mlpackages, metadata.json, tokenizer.json.",
    )
    parser.add_argument("--audio", type=str, required=True, help="Audio file (will be resampled to 16kHz mono if needed).")
    parser.add_argument(
        "--target-lang",
        type=str,
        default="auto",
        help='Language code, e.g. "en-US", "zh-CN", or "auto" (default).',
    )
    parser.add_argument("--duration", type=float, default=None, help="Trim audio to N seconds.")
    parser.add_argument(
        "--compare-to",
        type=str,
        default=None,
        help="Optional reference text to print alongside the hypothesis.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("NEMOTRON MULTILINGUAL COREML - STREAMING SMOKE TEST")
    print("=" * 70)
    print(f"model_dir:   {args.model_dir}")
    print(f"audio:       {args.audio}")
    print(f"target_lang: {args.target_lang}")

    audio, sr = sf.read(args.audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise SystemExit(
            f"Audio must be 16 kHz; got {sr}. Resample before running this test."
        )
    if args.duration:
        audio = audio[: int(args.duration * sr)]
    print(f"audio_len:   {len(audio)/sr:.2f}s")

    runner = NemotronMultilingualCoreML(args.model_dir)
    print(f"prompt_dict: {len(runner.prompt_dictionary)} entries")
    print(f"lang_tags:   {len(runner.lang_tag_token_ids)} ids")

    detected, text, prompt_id = runner.transcribe_streaming(audio, target_lang=args.target_lang)

    print("\n--- result ---")
    print(f"prompt_id_used: {prompt_id}")
    print(f"detected_lang:  {detected}")
    print(f"text:           {text}")
    if args.compare_to:
        print(f"reference:      {args.compare_to}")


if __name__ == "__main__":
    main()
