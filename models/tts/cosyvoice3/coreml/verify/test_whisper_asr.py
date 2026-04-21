"""Run Whisper ASR on both PyTorch and CoreML HiFT outputs to validate semantic fidelity."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import soundfile as sf
import whisper


def main():
    wavs_dir = Path(__file__).parent.parent / "build" / "wavs"
    torch_wav = wavs_dir / "real_mel_torch.wav"
    coreml_wav = wavs_dir / "real_mel_coreml.wav"

    for p in (torch_wav, coreml_wav):
        assert p.exists(), f"missing {p}"

    # Also transcribe the original prompt for reference.
    prompt_wav = Path(__file__).parent / "CosyVoice" / "asset" / "zero_shot_prompt.wav"

    print("Loading Whisper base model...")
    model = whisper.load_model("base")

    def transcribe(label, path):
        print(f"\n--- {label}: {path.name} ---")
        audio, sr = sf.read(str(path))
        print(f"  sr={sr} duration={len(audio)/sr:.2f}s  range=[{audio.min():.3f}, {audio.max():.3f}]  std={audio.std():.4f}")
        # whisper expects 16kHz float32, mono
        result = model.transcribe(str(path), fp16=False, verbose=False)
        print(f"  language: {result.get('language')}")
        print(f"  text: {result['text']!r}")
        return result["text"].strip()

    t_prompt = transcribe("prompt (original)", prompt_wav)
    t_torch = transcribe("pytorch HiFT", torch_wav)
    t_coreml = transcribe("coreml HiFT", coreml_wav)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"prompt : {t_prompt!r}")
    print(f"torch  : {t_torch!r}")
    print(f"coreml : {t_coreml!r}")

    # character-level equality
    print(f"\ntorch == coreml: {t_torch == t_coreml}")

    # rough char-overlap score
    def char_overlap(a, b):
        if not a or not b:
            return 0.0
        la, lb = len(a), len(b)
        # longest common subsequence length
        dp = [[0] * (lb + 1) for _ in range(la + 1)]
        for i in range(la):
            for j in range(lb):
                if a[i] == b[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
        return dp[la][lb] / max(la, lb)

    print(f"torch<->coreml LCS/max-len similarity: {char_overlap(t_torch, t_coreml):.3f}")
    print(f"prompt<->torch  LCS/max-len similarity: {char_overlap(t_prompt, t_torch):.3f}")
    print(f"prompt<->coreml LCS/max-len similarity: {char_overlap(t_prompt, t_coreml):.3f}")


if __name__ == "__main__":
    main()
