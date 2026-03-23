#!/usr/bin/env python3
"""Generate PyTorch reference audio for A/B comparison with CoreML output."""

import torch
import numpy as np
import struct
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_0.6b")
SAMPLE_RATE = 24000


def write_wav(filename, samples, sr=24000):
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


def main():
    text = "Hello world, this is a test."
    if len(sys.argv) > 1:
        text = sys.argv[1]

    print(f"Text: {text}")
    print("Loading Qwen3-TTS model...")

    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map="cpu", dtype=torch.float32)
    m = model.model  # Qwen3TTSForConditionalGeneration

    # Build input the same way the official pipeline does
    assistant_text = model._build_assistant_text(text)
    print(f"Assistant text: {repr(assistant_text[:100])}...")

    # Tokenize - returns list of [1, seq_len] tensors
    input_ids = model._tokenize_texts([assistant_text])
    print(f"Input IDs: list of {len(input_ids)} tensors, first shape: {input_ids[0].shape}")
    print(f"Token IDs: {input_ids[0][0].tolist()}")

    # Get default generation kwargs
    gen_kwargs = model._merge_generate_kwargs()
    print(f"Gen kwargs: temperature={gen_kwargs.get('temperature')}, "
          f"top_k={gen_kwargs.get('top_k')}, "
          f"subtalker_dosample={gen_kwargs.get('subtalker_dosample')}, "
          f"subtalker_temperature={gen_kwargs.get('subtalker_temperature')}, "
          f"subtalker_top_k={gen_kwargs.get('subtalker_top_k')}")

    # Generate
    print("Generating...")
    t0 = time.time()

    with torch.no_grad():
        talker_codes_list, _ = m.generate(
            input_ids=input_ids,
            languages=["english"],
            non_streaming_mode=True,
            **gen_kwargs,
        )

    gen_time = time.time() - t0
    print(f"Generation took {gen_time:.2f}s")

    # talker_codes_list is a list of [N, 16] tensors (one per batch)
    talker_codes = talker_codes_list[0]  # [N, 16]
    print(f"Talker codes shape: {talker_codes.shape}")
    if talker_codes.shape[0] == 16:
        # [16, N] format - transpose to [N, 16]
        talker_codes_t = talker_codes.T
    else:
        talker_codes_t = talker_codes
    print(f"CB0 tokens (first 10): {talker_codes_t[:10, 0].tolist()}")

    # Decode to audio using speech_tokenizer
    print("Decoding to audio...")
    t0 = time.time()
    wavs, fs = m.speech_tokenizer.decode([{"audio_codes": talker_codes_t}])
    decode_time = time.time() - t0
    print(f"Decode took {decode_time:.2f}s, sample_rate={fs}")
    audio = wavs[0]  # first batch item

    # audio might be a tensor
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    audio = audio.flatten().astype(np.float32)

    duration = len(audio) / SAMPLE_RATE
    print(f"Audio: {duration:.2f}s, max amplitude: {np.abs(audio).max():.4f}")

    # Save
    out_path = f"/tmp/pytorch_ref_{text[:20].replace(' ', '_').replace('.','')}.wav"
    write_wav(out_path, audio)
    print(f"Saved: {out_path}")

    # Also save raw codes for comparison
    codes_path = f"/tmp/pytorch_ref_codes.npy"
    np.save(codes_path, talker_codes.cpu().numpy() if isinstance(talker_codes, torch.Tensor) else talker_codes)
    print(f"Saved codes: {codes_path}")


if __name__ == "__main__":
    main()
