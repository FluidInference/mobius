"""Parity check: CoreML text_encoder + fm_decoder vs PyTorch, end-to-end.

Reproduces the LuxTTS pipeline with CoreML components (host-side duration
expansion + 4-step anchor-Euler solver + torch vocoder) and compares against
the pure-torch path at each stage, using the oracle inputs saved by
scripts/reference_infer.py.

Usage:
    .venv/bin/python coreml/parity.py --oracle-dir build/oracle --coreml-dir build/coreml
"""

import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf
import torch

from coreml.convert_coreml import MAX_FRAMES, MAX_TOKENS, load_model
from zipvoice.models.modules.solver import get_time_steps
from zipvoice.models.zipvoice import get_tokens_index, prepare_avg_tokens_durations


def cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def expand_text_condition(token_embeds: torch.Tensor, tokens_len: int, features_len: int):
    """Host-side duration expansion (mirrors forward_text_condition, B=1)."""
    features_lens = torch.tensor([features_len])
    tokens_lens = torch.tensor([tokens_len])
    durations = prepare_avg_tokens_durations(features_lens, tokens_lens)
    index = get_tokens_index(durations, features_len)  # (1, T)
    return torch.gather(
        token_embeds, dim=1, index=index.unsqueeze(-1).expand(1, features_len, token_embeds.size(-1))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-dir", default="build/oracle")
    parser.add_argument("--coreml-dir", default="build/coreml")
    parser.add_argument("--compute-units", default="CPU_ONLY", choices=["ALL", "CPU_ONLY", "CPU_AND_NE"])
    args = parser.parse_args()

    oracle = Path(args.oracle_dir)
    meta = json.loads((oracle / "meta.json").read_text())
    prompt_features = torch.from_numpy(np.load(oracle / "prompt_features.npy"))
    prompt_tokens = np.load(oracle / "prompt_tokens.npy").tolist()
    text_tokens = np.load(oracle / "text_tokens.npy").tolist()
    prompt_features_len = meta["prompt_features_lens"]
    guidance_scale = meta["guidance_scale"]
    num_steps = meta["num_steps"]
    speed = 1.0 * 1.3  # generate() multiplies speed by 1.3

    model, _ = load_model()  # includes convert_scaled_to_non_scaled, same as conversion

    cu = getattr(ct.ComputeUnit, args.compute_units)
    te_ml = ct.models.MLModel(str(Path(args.coreml_dir) / "TextEncoder.mlpackage"), compute_units=cu)
    fm_ml = ct.models.MLModel(str(Path(args.coreml_dir) / "FmDecoder.mlpackage"), compute_units=cu)

    # ---- 1. text encoder ----
    cat_tokens = prompt_tokens + text_tokens
    S = len(cat_tokens)
    S_pad = S + 1  # upstream pad_labels appends one pad slot; its row fills remainder frames
    assert S_pad <= MAX_TOKENS, f"{S_pad} tokens > bucket {MAX_TOKENS}"

    with torch.no_grad():
        ref_embed, _ = model.forward_text_embed([cat_tokens])  # (1, S+1, D)

    tok_in = np.full((1, MAX_TOKENS), model.pad_id, dtype=np.int32)
    tok_in[0, :S] = cat_tokens
    mask_in = np.zeros((1, MAX_TOKENS), dtype=np.float32)
    mask_in[0, S:] = 1.0  # upstream masks the pad slot too (make_pad_mask over S)
    cm_embed = te_ml.predict({"tokens": tok_in, "padding_mask": mask_in})["token_embeds"]
    cm_embed_valid = torch.from_numpy(cm_embed[:, :S_pad, :].astype(np.float32))

    print(f"[text_encoder] cos={cos(ref_embed.numpy(), cm_embed_valid.numpy()):.6f} "
          f"max_abs_diff={float((ref_embed - cm_embed_valid).abs().max()):.4e}")

    # ---- 2. duration + conditions (host) ----
    features_len = prompt_features_len + int(
        np.ceil(prompt_features_len / len(prompt_tokens) * len(text_tokens) / speed)
    )
    assert features_len <= MAX_FRAMES, f"{features_len} frames > bucket {MAX_FRAMES}"

    text_condition = expand_text_condition(cm_embed_valid, S, features_len)
    with torch.no_grad():
        ref_text_condition, _ = model.forward_text_inference_ratio_duration(
            tokens=[text_tokens], prompt_tokens=[prompt_tokens],
            prompt_features_lens=torch.tensor([prompt_features_len]), speed=speed,
        )
    assert ref_text_condition.shape[1] == features_len, (ref_text_condition.shape, features_len)
    print(f"[text_condition] cos={cos(ref_text_condition.numpy(), text_condition.numpy()):.6f}")

    speech_condition = torch.nn.functional.pad(
        prompt_features, (0, 0, 0, features_len - prompt_features.size(1))
    )

    # ---- 3. solver loop: CoreML decoder vs torch decoder ----
    def pad_T(x):
        return torch.nn.functional.pad(x, (0, 0, 0, MAX_FRAMES - x.size(1)))

    frame_mask = np.zeros((1, MAX_FRAMES), dtype=np.float32)
    frame_mask[0, features_len:] = 1.0
    ref_pad_mask = torch.zeros(1, features_len, dtype=torch.bool)

    timesteps = get_time_steps(num_step=num_steps, t_shift=0.5)
    torch.manual_seed(meta["seed"])
    x = torch.randn(1, features_len, 100)
    x_ref = x.clone()

    text_np, speech_np = pad_T(text_condition).numpy(), pad_T(speech_condition).numpy()
    for step in range(num_steps):
        t_cur, t_next = float(timesteps[step]), float(timesteps[step + 1])

        v = fm_ml.predict({
            "t": np.array([t_cur], dtype=np.float32),
            "x": pad_T(x).numpy().astype(np.float32),
            "text_condition": text_np.astype(np.float32),
            "speech_condition": speech_np.astype(np.float32),
            "guidance_scale": np.array([guidance_scale], dtype=np.float32),
            "padding_mask": frame_mask,
        })["v"]
        v = torch.from_numpy(v[:, :features_len, :].astype(np.float32))

        with torch.no_grad():
            v_ref = model.forward_fm_decoder(
                t=torch.tensor(t_cur), xt=x_ref, text_condition=ref_text_condition,
                speech_condition=speech_condition, padding_mask=ref_pad_mask,
                guidance_scale=torch.tensor(guidance_scale),
            )
        print(f"[fm_decoder step {step} t={t_cur:.3f}] cos={cos(v_ref.numpy(), v.numpy()):.6f} "
              f"max_abs_diff={float((v_ref - v).abs().max()):.4e}")

        def euler_update(x_s, v_s):
            x1p = x_s + (1.0 - t_cur) * v_s
            x0p = x_s - t_cur * v_s
            return (1.0 - t_next) * x0p + t_next * x1p if step < num_steps - 1 else x1p

        x, x_ref = euler_update(x, v), euler_update(x_ref, v_ref)

    print(f"[final mel] cos={cos(x_ref.numpy(), x.numpy()):.6f}")

    # ---- 4. vocoder (torch, shared) + wav comparison ----
    from scripts.reference_infer import load_models_cpu_torch  # reuse loader

    _, _, vocos, _, _ = load_models_cpu_torch()
    vocos.freq_range = 12000
    vocos.return_48k = True
    with torch.no_grad():
        mel = x[:, prompt_features_len:features_len, :].permute(0, 2, 1) / 0.1
        wav = vocos.decode(mel).squeeze(1).clamp(-1, 1)
    wav_np = wav.numpy().squeeze()
    if meta["prompt_rms"] < 0.1:
        wav_np = wav_np * (meta["prompt_rms"] / 0.1)

    ref_wav, sr = sf.read(oracle / "reference_48k.wav")
    n = min(len(ref_wav), len(wav_np))
    print(f"[wav] len_ref={len(ref_wav)} len_cm={len(wav_np)} cos(first {n})={cos(ref_wav[:n], wav_np[:n]):.6f}")

    out = Path(args.coreml_dir) / "parity_48k.wav"
    sf.write(out, wav_np, 48000)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
