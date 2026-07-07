"""Full-pipeline quality + speed harness for the ANE-canonical FmDecoder.

Adapts coreml/parity.py: CoreML TextEncoder + AneFmDecoder (4-step host
solver, torch vocoder), compared against the PyTorch oracle at each stage.
Adds wav metrics (log-mel cos, RMS delta), whisper-base transcription, and
predict latency / core RTFx for the requested compute units.

Run: .venv/bin/python -m coreml.ane.pipeline --compute-units CPU_AND_NE
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import coremltools as ct
import numpy as np
import soundfile as sf
import torch

from coreml.convert_coreml import MAX_TOKENS, load_model
from coreml.parity import cos, expand_text_condition
from zipvoice.models.modules.solver import get_time_steps

SEQ_LEN = 1024
FRAME_RATE = 93.75  # feature frames per second


def timeit(fn, runs=10, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts)
    return ts.mean(), ts.std()


def log_mel_cos(ref, test, sr=48000):
    import librosa

    def lm(x):
        m = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        return np.log(np.maximum(m, 1e-10))

    return cos(lm(ref), lm(test))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-dir", default="build/oracle")
    parser.add_argument("--coreml-dir", default="build/coreml-ane")
    parser.add_argument(
        "--compute-units",
        default="CPU_AND_NE",
        choices=["ALL", "CPU_ONLY", "CPU_AND_NE", "CPU_AND_GPU"],
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--skip-quality", action="store_true", help="latency only")
    parser.add_argument("--transcribe", action="store_true")
    args = parser.parse_args()

    oracle = Path(args.oracle_dir)
    meta = json.loads((oracle / "meta.json").read_text())
    prompt_features = torch.from_numpy(np.load(oracle / "prompt_features.npy"))
    prompt_tokens = np.load(oracle / "prompt_tokens.npy").tolist()
    text_tokens = np.load(oracle / "text_tokens.npy").tolist()
    prompt_len = meta["prompt_features_lens"]
    guidance_scale = meta["guidance_scale"]
    num_steps = meta["num_steps"]
    speed = 1.0 * 1.3

    cu = getattr(ct.ComputeUnit, args.compute_units)
    t0 = time.perf_counter()
    te_ml = ct.models.MLModel(str(Path(args.coreml_dir) / "TextEncoder.mlpackage"), compute_units=cu)
    fm_ml = ct.models.MLModel(str(Path(args.coreml_dir) / "AneFmDecoder.mlpackage"), compute_units=cu)
    print(f"[load] {args.compute_units}: {(time.perf_counter() - t0) * 1e3:.0f} ms")

    model, _ = load_model()

    # ---- conditions (CoreML text encoder + host expansion, as parity.py) ----
    cat_tokens = prompt_tokens + text_tokens
    S = len(cat_tokens)
    tok_in = np.full((1, MAX_TOKENS), model.pad_id, dtype=np.int32)
    tok_in[0, :S] = cat_tokens
    tmask = np.zeros((1, MAX_TOKENS), dtype=np.float32)
    tmask[0, S:] = 1.0
    cm_embed = te_ml.predict({"tokens": tok_in, "padding_mask": tmask})["token_embeds"]
    cm_embed = torch.from_numpy(cm_embed[:, : S + 1, :].astype(np.float32))

    features_len = prompt_len + int(np.ceil(prompt_len / len(prompt_tokens) * len(text_tokens) / speed))
    gen_seconds = (features_len - prompt_len) / FRAME_RATE
    text_condition = expand_text_condition(cm_embed, S, features_len)
    speech_condition = torch.nn.functional.pad(
        prompt_features, (0, 0, 0, features_len - prompt_features.size(1))
    )

    with torch.no_grad():
        ref_text_condition, _ = model.forward_text_inference_ratio_duration(
            tokens=[text_tokens], prompt_tokens=[prompt_tokens],
            prompt_features_lens=torch.tensor([prompt_len]), speed=speed,
        )

    def pad_T(z):
        return torch.nn.functional.pad(z, (0, 0, 0, SEQ_LEN - z.size(1)))

    fmask = np.zeros((1, SEQ_LEN), dtype=np.float32)
    fmask[0, features_len:] = 1.0
    text_np = pad_T(text_condition).numpy().astype(np.float32)
    speech_np = pad_T(speech_condition).numpy().astype(np.float32)

    def predict(t_cur, x_np):
        return fm_ml.predict(
            {
                "t": np.array([t_cur], dtype=np.float32),
                "x": x_np,
                "text_condition": text_np,
                "speech_condition": speech_np,
                "guidance_scale": np.array([guidance_scale], dtype=np.float32),
                "padding_mask": fmask,
            }
        )["v"]

    # ---- solver loop: CoreML ANE decoder vs torch oracle ----
    timesteps = get_time_steps(num_step=num_steps, t_shift=0.5)
    if not args.skip_quality:
        torch.manual_seed(meta["seed"])
        x = torch.randn(1, features_len, 100)
        x_ref = x.clone()
        ref_mask = torch.zeros(1, features_len, dtype=torch.bool)
        for step in range(num_steps):
            t_cur, t_next = float(timesteps[step]), float(timesteps[step + 1])
            v = predict(t_cur, pad_T(x).numpy().astype(np.float32))
            v = torch.from_numpy(v[:, :features_len, :].astype(np.float32))
            with torch.no_grad():
                v_ref = model.forward_fm_decoder(
                    t=torch.tensor(t_cur), xt=x_ref, text_condition=ref_text_condition,
                    speech_condition=speech_condition, padding_mask=ref_mask,
                    guidance_scale=torch.tensor(guidance_scale),
                )
            print(
                f"[fm_decoder step {step} t={t_cur:.3f}] cos={cos(v_ref.numpy(), v.numpy()):.6f} "
                f"max_abs_diff={float((v_ref - v).abs().max()):.4e}"
            )

            def euler(x_s, v_s):
                x1p = x_s + (1.0 - t_cur) * v_s
                x0p = x_s - t_cur * v_s
                return (1.0 - t_next) * x0p + t_next * x1p if step < num_steps - 1 else x1p

            x, x_ref = euler(x, v), euler(x_ref, v_ref)

        print(f"[final mel] cos={cos(x_ref.numpy(), x.numpy()):.6f}")

        # ---- vocoder + wav metrics ----
        from scripts.reference_infer import load_models_cpu_torch

        _, _, vocos, _, _ = load_models_cpu_torch()
        vocos.freq_range = 12000
        vocos.return_48k = True
        with torch.no_grad():
            mel = x[:, prompt_len:features_len, :].permute(0, 2, 1) / 0.1
            wav = vocos.decode(mel).squeeze(1).clamp(-1, 1)
        wav_np = wav.numpy().squeeze()
        if meta["prompt_rms"] < 0.1:
            wav_np = wav_np * (meta["prompt_rms"] / 0.1)

        ref_wav, sr = sf.read(oracle / "reference_48k.wav")
        n = min(len(ref_wav), len(wav_np))
        rms_ref = float(np.sqrt((ref_wav[:n] ** 2).mean()))
        rms_cm = float(np.sqrt((wav_np[:n] ** 2).mean()))
        print(
            f"[wav] waveform cos={cos(ref_wav[:n], wav_np[:n]):.4f} "
            f"log-mel cos={log_mel_cos(ref_wav[:n].astype(np.float32), wav_np[:n].astype(np.float32)):.5f} "
            f"RMS {rms_cm:.4f} vs {rms_ref:.4f} (delta {20 * np.log10(rms_cm / rms_ref):+.3f} dB)"
        )
        out = Path(args.coreml_dir) / "parity_48k.wav"
        sf.write(out, wav_np, 48000)
        print(f"wrote {out}")

        if args.transcribe:
            import librosa
            from transformers import pipeline

            asr = pipeline("automatic-speech-recognition", model="openai/whisper-base", device="cpu")
            wav16 = librosa.resample(wav_np.astype(np.float32), orig_sr=48000, target_sr=16000)
            ref16 = librosa.resample(np.asarray(ref_wav, dtype=np.float32), orig_sr=48000, target_sr=16000)
            print(f"[transcript coreml-ane] {asr(wav16)['text'].strip()}")
            print(f"[transcript oracle]     {asr(ref16)['text'].strip()}")

    # ---- latency + core RTFx ----
    x_rand = np.random.default_rng(0).standard_normal((1, SEQ_LEN, 100)).astype(np.float32)
    te_ms, te_sd = timeit(lambda: te_ml.predict({"tokens": tok_in, "padding_mask": tmask}), args.runs)
    fm_ms, fm_sd = timeit(lambda: predict(0.5, x_rand), args.runs)
    core_ms = te_ms + num_steps * fm_ms
    print(
        f"[latency {args.compute_units}] text_encoder {te_ms:.2f}±{te_sd:.2f} ms | "
        f"fm_decoder/step {fm_ms:.2f}±{fm_sd:.2f} ms | core (te+{num_steps} steps) {core_ms:.1f} ms | "
        f"gen {gen_seconds:.3f}s -> core RTFx {gen_seconds * 1e3 / core_ms:.1f}x"
    )


if __name__ == "__main__":
    main()
