#!/usr/bin/env python3
"""Streaming OpenVINO inference for Nemotron-3.5-ASR-Streaming-Multilingual 0.6B.

Direct port of mobius's `test_coreml_multilingual.py` streaming loop, with
`ct.models.MLModel(...).predict(d)` replaced by OpenVINO
`core.compile_model(xml, "CPU")` + `compiled(d)`. Same chunking, same cache
plumbing, same greedy RNNT decode — only the runtime differs. This validates
that the OpenVINO IR produced by export_openvino.py transcribes correctly.
"""
import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import openvino as ov
import soundfile as sf


class NemotronOV:
    def __init__(self, model_dir: str, device: str = "CPU"):
        d = Path(model_dir)
        self.metadata = json.loads((d / "metadata.json").read_text())
        self.tokenizer = json.loads((d / "nemotron_vocab.json").read_text())

        core = ov.Core()
        print(f"Loading OpenVINO IR on {device} ...")
        self.preprocessor = core.compile_model(str(d / "nemotron_preprocessor.xml"), device)
        self.encoder = core.compile_model(str(d / "nemotron_encoder.xml"), device)
        self.decoder = core.compile_model(str(d / "nemotron_decoder.xml"), device)
        self.joint = core.compile_model(str(d / "nemotron_joint.xml"), device)
        print("Models loaded.")

        m = self.metadata
        self.sample_rate = m["sample_rate"]
        self.chunk_mel_frames = m["chunk_mel_frames"]
        self.pre_encode_cache = m["pre_encode_cache"]
        self.total_mel_frames = m["total_mel_frames"]
        self.blank_idx = m["blank_idx"]
        self.vocab_size = m["vocab_size"]
        self.decoder_hidden = m["decoder_hidden"]
        self.decoder_layers = m["decoder_layers"]
        self.cache_channel_shape = m["cache_channel_shape"]
        self.cache_time_shape = m["cache_time_shape"]
        self.prompt_dictionary = m.get("prompt_dictionary", {})
        self.lang_tag_token_ids = set(m.get("lang_tag_token_ids", []))
        self.default_prompt_id = int(m.get("default_prompt_id", 101))
        self.chunk_samples = int(self.chunk_mel_frames * 0.01 * self.sample_rate)

    def resolve_prompt_id(self, target_lang: str) -> int:
        if target_lang in self.prompt_dictionary:
            return int(self.prompt_dictionary[target_lang])
        if len(target_lang) == 2:
            for k, v in self.prompt_dictionary.items():
                if k.lower().startswith(target_lang.lower() + "-"):
                    return int(v)
        return self.default_prompt_id

    def piece_for_id(self, t: int) -> str:
        return self.tokenizer.get(str(t), "")

    def _decode_tokens(self, tokens: List[int]) -> Tuple[Optional[str], str]:
        detected, body = None, []
        for tok in tokens:
            if tok == self.blank_idx or tok >= self.vocab_size:
                continue
            if tok in self.lang_tag_token_ids:
                if detected is None:
                    detected = self.piece_for_id(tok)
                continue
            body.append(tok)
        text = "".join(self.piece_for_id(t) for t in body).replace("▁", " ").strip()
        return detected, text

    def transcribe_streaming(self, audio: np.ndarray, target_lang: str = "auto"):
        audio = audio.astype(np.float32)
        total = len(audio)
        prompt_id = self.resolve_prompt_id(target_lang)
        prompt_in = np.array([prompt_id], dtype=np.int32)

        cache_channel = np.zeros(self.cache_channel_shape, dtype=np.float32)
        cache_time = np.zeros(self.cache_time_shape, dtype=np.float32)
        cache_len = np.array([0], dtype=np.int32)
        h = np.zeros((self.decoder_layers, 1, self.decoder_hidden), dtype=np.float32)
        c = np.zeros((self.decoder_layers, 1, self.decoder_hidden), dtype=np.float32)
        last_token = self.blank_idx
        all_tokens: List[int] = []
        mel_cache = None
        off = 0

        while off < total:
            chunk = audio[off:min(off + self.chunk_samples, total)]
            if len(chunk) < self.chunk_samples:
                chunk = np.pad(chunk, (0, self.chunk_samples - len(chunk)))
            chunk = chunk.reshape(1, -1)

            pre = self.preprocessor({
                "audio": chunk.astype(np.float32),
                "audio_length": np.array([chunk.shape[1]], dtype=np.int32),
            })
            chunk_mel = pre["mel"]

            if mel_cache is not None:
                input_mel = np.concatenate([mel_cache, chunk_mel], axis=2)
            else:
                input_mel = np.pad(chunk_mel, ((0, 0), (0, 0), (self.pre_encode_cache, 0)))
            cur = input_mel.shape[2]
            if cur < self.total_mel_frames:
                input_mel = np.pad(input_mel, ((0, 0), (0, 0), (0, self.total_mel_frames - cur)))
            elif cur > self.total_mel_frames:
                input_mel = input_mel[:, :, : self.total_mel_frames]
            mel_cache = (chunk_mel[:, :, -self.pre_encode_cache:]
                         if chunk_mel.shape[2] >= self.pre_encode_cache else chunk_mel)

            enc = self.encoder({
                "mel": input_mel.astype(np.float32),
                "mel_length": np.array([self.total_mel_frames], dtype=np.int32),
                "cache_channel": cache_channel,
                "cache_time": cache_time,
                "cache_len": cache_len,
                "prompt_id": prompt_in,
            })
            encoded = enc["encoded"]
            cache_channel = enc["cache_channel_out"]
            cache_time = enc["cache_time_out"]
            cache_len = enc["cache_len_out"]

            for t in range(encoded.shape[2]):
                enc_step = encoded[:, :, t : t + 1]
                for _ in range(10):
                    dec = self.decoder({
                        "token": np.array([[last_token]], dtype=np.int32),
                        "token_length": np.array([1], dtype=np.int32),
                        "h_in": h, "c_in": c,
                    })
                    decoder_out = dec["decoder_out"]
                    jo = self.joint({
                        "encoder": enc_step.astype(np.float32),
                        "decoder": decoder_out[:, :, :1].astype(np.float32),
                    })
                    logits = jo["logits"]
                    pred = int(np.argmax(logits[0, 0, 0, :]))
                    if pred == self.blank_idx:
                        break
                    all_tokens.append(pred)
                    last_token = pred
                    h = dec["h_out"]
                    c = dec["c_out"]
            off += self.chunk_samples

        detected, text = self._decode_tokens(all_tokens)
        return detected, text, prompt_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--target-lang", default="auto")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--device", default="CPU")
    args = ap.parse_args()

    audio, sr = sf.read(args.audio, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise SystemExit(f"Audio must be 16 kHz; got {sr}.")
    if args.duration:
        audio = audio[: int(args.duration * sr)]
    print(f"audio: {len(audio)/sr:.2f}s @ {sr}Hz")

    runner = NemotronOV(args.model_dir, device=args.device)
    detected, text, pid = runner.transcribe_streaming(audio, target_lang=args.target_lang)
    print("\n--- OpenVINO result ---")
    print(f"prompt_id_used: {pid}")
    print(f"detected_lang:  {detected}")
    print(f"text:           {text}")


if __name__ == "__main__":
    main()
