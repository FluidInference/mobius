"""Parity: CoreML Vocoder vs torch vocos.decode on the oracle generated mel.

Regenerates the oracle mel deterministically through the pure-torch reference
path (same as coreml/parity.py's torch branch: forward_text_inference +
4-step anchor-Euler with the oracle seed), then feeds it to torch
vocos.decode and the CoreML Vocoder and compares wavs: sample-domain SNR,
log-mel cos, RMS delta, whisper-base transcripts.

Usage:
    .venv/bin/python -m coreml.vocoder.parity_vocoder --coreml-dir build/coreml-vocoder
"""

import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from coreml.convert_coreml import load_model
from coreml.parity import cos
from coreml.vocoder.convert_vocoder import load_vocos, snr_db
from zipvoice.models.modules.solver import get_time_steps


def regenerate_oracle_mel(oracle: Path):
    """Final mel via the pure-torch reference (deterministic, oracle seed)."""
    meta = json.loads((oracle / "meta.json").read_text())
    prompt_features = torch.from_numpy(np.load(oracle / "prompt_features.npy"))
    prompt_tokens = np.load(oracle / "prompt_tokens.npy").tolist()
    text_tokens = np.load(oracle / "text_tokens.npy").tolist()
    prompt_len = meta["prompt_features_lens"]
    speed = 1.0 * 1.3  # generate() multiplies speed by 1.3

    model, _ = load_model()
    with torch.no_grad():
        text_condition, _ = model.forward_text_inference_ratio_duration(
            tokens=[text_tokens], prompt_tokens=[prompt_tokens],
            prompt_features_lens=torch.tensor([prompt_len]), speed=speed,
        )
    features_len = text_condition.shape[1]
    speech_condition = F.pad(prompt_features, (0, 0, 0, features_len - prompt_features.size(1)))
    pad_mask = torch.zeros(1, features_len, dtype=torch.bool)

    timesteps = get_time_steps(num_step=meta["num_steps"], t_shift=0.5)
    torch.manual_seed(meta["seed"])
    x = torch.randn(1, features_len, 100)
    for step in range(meta["num_steps"]):
        t_cur, t_next = float(timesteps[step]), float(timesteps[step + 1])
        with torch.no_grad():
            v = model.forward_fm_decoder(
                t=torch.tensor(t_cur), xt=x, text_condition=text_condition,
                speech_condition=speech_condition, padding_mask=pad_mask,
                guidance_scale=torch.tensor(meta["guidance_scale"]),
            )
        x1p = x + (1.0 - t_cur) * v
        x0p = x - t_cur * v
        x = (1.0 - t_next) * x0p + t_next * x1p if step < meta["num_steps"] - 1 else x1p

    mel = x[:, prompt_len:features_len, :].permute(0, 2, 1) / 0.1  # (1, 100, gen)
    return mel, meta


def log_mel_cos(ref, test, sr=48000):
    import librosa

    def lm(x):
        m = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        return np.log(np.maximum(m, 1e-10))

    return cos(lm(ref), lm(test))


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-dir", default="build/oracle")
    parser.add_argument("--coreml-dir", default="build/coreml-vocoder")
    parser.add_argument("--compute-units", default="CPU_AND_GPU",
                        choices=["ALL", "CPU_ONLY", "CPU_AND_NE", "CPU_AND_GPU"])
    parser.add_argument("--transcribe", action="store_true", help="whisper-base both wavs")
    args = parser.parse_args()

    mel, meta = regenerate_oracle_mel(Path(args.oracle_dir))
    print(f"[mel] regenerated oracle gen-region mel: {tuple(mel.shape)}")

    vocos = load_vocos()
    with torch.no_grad():
        ref_wav = vocos.decode(mel).squeeze(0).clamp(-1, 1).numpy()

    cu = getattr(ct.ComputeUnit, args.compute_units)
    voc_ml = ct.models.MLModel(str(Path(args.coreml_dir) / "Vocoder.mlpackage"), compute_units=cu)
    cm_wav = voc_ml.predict({"mel": mel.numpy().astype(np.float32)})["audio"]
    cm_wav = np.clip(cm_wav.squeeze(0).astype(np.float32), -1, 1)

    n = min(len(ref_wav), len(cm_wav))
    ref_wav, cm_wav = ref_wav[:n], cm_wav[:n]
    print(f"[wav {args.compute_units}] len={n} "
          f"SNR={snr_db(ref_wav, cm_wav):.1f} dB  "
          f"cos={cos(ref_wav, cm_wav):.6f}  "
          f"log-mel cos={log_mel_cos(ref_wav, cm_wav):.5f}  "
          f"RMS delta={rms_db(cm_wav) - rms_db(ref_wav):+.3f} dB")

    # host-side prompt-level scaling, as the pipeline does
    scale = meta["prompt_rms"] / 0.1 if meta["prompt_rms"] < 0.1 else 1.0
    out = Path(args.coreml_dir)
    sf.write(out / "parity_torch_48k.wav", ref_wav * scale, 48000)
    sf.write(out / "parity_coreml_48k.wav", cm_wav * scale, 48000)
    print(f"wrote {out}/parity_torch_48k.wav, parity_coreml_48k.wav")

    if args.transcribe:
        from transformers import pipeline

        asr = pipeline("automatic-speech-recognition", model="openai/whisper-base", device="cpu")
        t_ref = asr(str(out / "parity_torch_48k.wav"))["text"].strip()
        t_cm = asr(str(out / "parity_coreml_48k.wav"))["text"].strip()
        print(f"[whisper torch ] {t_ref}")
        print(f"[whisper coreml] {t_cm}")
        print(f"[transcript match] {t_ref == t_cm}")


if __name__ == "__main__":
    main()
