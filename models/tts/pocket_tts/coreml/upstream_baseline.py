"""Upstream-PyTorch TTS baseline to compare against our CoreML pipeline.

For a given language, loads the upstream `TTSModel`, seeds voice state from
our packed `build/<lang>/constants_bin/<voice>.safetensors` (same bytes the
CoreML generator uses), generates audio, and writes `build/<lang>/upstream.wav`.

Verify step (not run here):
  uv run python verify_with_whisper.py build/<lang>/upstream.wav <iso>

If upstream audio transcribes to the reference text but our CoreML output
does not, the bug lives inside the CoreML generator (prefill, step loop,
or mimi decode). If both fail the same way, the bug is upstream or in the
voice state packing.
"""

from __future__ import annotations

import argparse
import os
import sys

import scipy.io.wavfile as wavfile
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", required=True)
    parser.add_argument("--voice", default="alba")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Deterministic run so we can compare two invocations
    torch.manual_seed(args.seed)

    from pocket_tts.models.tts_model import TTSModel

    voice_path = os.path.join(
        _SCRIPT_DIR, "build", args.language, "constants_bin",
        f"{args.voice}.safetensors",
    )
    if not os.path.isfile(voice_path):
        raise FileNotFoundError(voice_path)

    print(f"[upstream] language={args.language} voice={args.voice}")
    print(f"[upstream] text: {args.text!r}")
    print(f"[upstream] loading TTSModel(language={args.language})...")
    model = TTSModel.load_model(language=args.language, lsd_decode_steps=8)

    print(f"[upstream] loading voice state from {voice_path}")
    voice_state = model.get_state_for_audio_prompt(voice_path)

    # Report voice state shape for sanity — should mirror what our
    # CoreML generator sees when it loads the same safetensors file.
    for k, v in list(voice_state.items())[:3]:
        if isinstance(v, dict):
            for kk, vv in v.items():
                if torch.is_tensor(vv):
                    print(f"    {k}.{kk}: {tuple(vv.shape)} dtype={vv.dtype}")
                else:
                    print(f"    {k}.{kk}: {type(vv).__name__} = {vv}")

    print(f"[upstream] generating...")
    audio = model.generate_audio(voice_state, args.text, copy_state=True)
    print(f"[upstream] audio shape={tuple(audio.shape)} sr={model.sample_rate}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    audio_np = audio.detach().cpu().numpy()
    if audio_np.ndim == 2:
        audio_np = audio_np[0]
    wavfile.write(args.output, model.sample_rate, audio_np)
    dur = audio_np.shape[-1] / model.sample_rate
    print(f"[upstream] saved {args.output} ({dur:.2f}s)")


if __name__ == "__main__":
    main()
