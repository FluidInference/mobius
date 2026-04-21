"""Extract a CosyVoice3 zero-shot voice prompt bundle from a reference WAV.

Produces the exact file pair consumed by the FluidAudio Swift side via
``CosyVoice3PromptAssets.load(from:)``:

    <voice-id>.safetensors
        llm_prompt_speech_ids  int32   [1, N_speech]
        prompt_mel             float32 [1, 2*N_speech, 80]
        spk_embedding          float32 [1, 192]
    <voice-id>.json
        { "prompt_text": "You are a helpful assistant.<|endofprompt|>…" }

Run once per voice at build time; ship the resulting folder as a HuggingFace
dataset. On-device Swift inference then loads the bundle via
``CosyVoice3PromptAssets.load(from:)`` — no Python runtime required.

Usage:
    uv run python verify/extract_voice_prompt.py \\
        --voice-id   alba-zh \\
        --ref-wav    assets/alba.wav \\
        --prompt-text "You are a helpful assistant.<|endofprompt|>最近天气不错。" \\
        --output-dir build/voices

    # Batch — repeat with different --voice-id / --ref-wav pairs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.numpy import save_file


HERE = Path(__file__).parent
ROOT = HERE.parent


def _prime_sys_path() -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(HERE / "CosyVoice"))
    sys.path.insert(0, str(HERE / "CosyVoice" / "third_party" / "Matcha-TTS"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--voice-id", required=True,
                    help="Slug used for the output filenames (e.g. 'alba-zh').")
    ap.add_argument("--ref-wav", required=True,
                    help="Path to the reference WAV for this voice.")
    ap.add_argument("--prompt-text", required=True,
                    help="Prompt text (MUST contain '<|endofprompt|>' — id 151646).")
    ap.add_argument("--output-dir", default=str(ROOT / "build" / "voices"))
    ap.add_argument("--model-dir", default=str(ROOT / "cosyvoice3_dl"))
    args = ap.parse_args()

    if "<|endofprompt|>" not in args.prompt_text:
        raise SystemExit(
            "prompt-text must contain the literal '<|endofprompt|>' token — "
            "CosyVoice3's LLM asserts on it. Typical prefix: "
            "'You are a helpful assistant.<|endofprompt|>' followed by the "
            "speaker's actual utterance from the reference WAV."
        )

    _prime_sys_path()
    from src.text_frontend import build_frontend_inputs  # noqa: E402

    ref_wav = Path(args.ref_wav).expanduser().resolve()
    if not ref_wav.exists():
        raise SystemExit(f"ref-wav not found: {ref_wav}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # tts_text is irrelevant for the voice bundle — the three voice tensors
    # depend only on the reference WAV + prompt text. Pass a throwaway value.
    placeholder_tts = "你好。"

    print(f"[1/2] Extracting voice prompt for '{args.voice_id}' from {ref_wav}…")
    fr = build_frontend_inputs(
        tts_text=placeholder_tts,
        prompt_text=args.prompt_text,
        prompt_wav=str(ref_wav),
        model_dir=Path(args.model_dir),
    )

    n_speech = int(fr.llm_prompt_speech_ids.shape[1])
    mel_frames = int(fr.prompt_mel.shape[1])
    print(f"      N_speech        : {n_speech}")
    print(f"      prompt_mel frames: {mel_frames}  (expected 2*N_speech = {2*n_speech})")
    print(f"      spk_embedding   : {tuple(fr.spk_embedding.shape)}")
    if mel_frames != 2 * n_speech:
        print(
            f"      warning: mel_frames ({mel_frames}) != 2*N_speech ({2*n_speech}); "
            "Flow may reject this bundle."
        )

    tensors = {
        "llm_prompt_speech_ids": fr.llm_prompt_speech_ids.detach().to(torch.int32).cpu().numpy(),
        "prompt_mel":            fr.prompt_mel.detach().to(torch.float32).cpu().numpy(),
        "spk_embedding":         fr.spk_embedding.detach().to(torch.float32).cpu().numpy(),
    }

    st_path = output_dir / f"{args.voice_id}.safetensors"
    json_path = output_dir / f"{args.voice_id}.json"

    print(f"[2/2] Writing → {st_path}")
    save_file(
        tensors,
        str(st_path),
        metadata={
            "voice_id": args.voice_id,
            "ref_wav": ref_wav.name,
            "n_speech": str(n_speech),
            "mel_frames": str(mel_frames),
            "source": "CosyVoice3 frontend_zero_shot (SpeechTokenizer v3 + 24kHz mel + CAMPPlus)",
        },
    )
    json_path.write_text(
        json.dumps({"prompt_text": args.prompt_text}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    size_mb = st_path.stat().st_size / 1024 / 1024
    print(f"      saved: {st_path}  ({size_mb:.2f} MB)")
    print(f"      saved: {json_path}")


if __name__ == "__main__":
    main()
