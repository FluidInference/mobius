"""fp32 eager parity: AneFmDecoder vs torch fm_decoder on the oracle inputs.

Two comparisons per solver step:
  1. GATE  — Ane @ S=1024 + mask vs torch @ S=1024 + mask (apples-to-apples
     rewrite parity): max_abs_diff < 5e-2 and cos > 0.9995.
  2. REPORT — Ane @ S=1024 + mask vs torch @ exact S=751 (the parity.py
     oracle path). Upstream padding semantics leak one garbage frame through
     SimpleDownsample at the 751 boundary (mask[::ds] keeps it), so
     1024+mask vs 751-exact differs at cos ~0.9991-0.9997 / max_abs ~1-2 in
     PURE TORCH already; that floor is inherent to the shipped FmDecoder
     bucket too, not to this rewrite.

Run: .venv/bin/python -m coreml.ane.decoder_parity
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import torch

from coreml.ane.decoder import AneFmDecoder, AneFmDecoderIO
from coreml.convert_coreml import load_model
from zipvoice.models.modules.solver import get_time_steps

SEQ_LEN = 1024
GATE_MAX_ABS = 5e-2
GATE_COS = 0.9995


def cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    oracle = Path("build/oracle")
    meta = json.loads((oracle / "meta.json").read_text())
    prompt_features = torch.from_numpy(np.load(oracle / "prompt_features.npy"))
    prompt_tokens = np.load(oracle / "prompt_tokens.npy").tolist()
    text_tokens = np.load(oracle / "text_tokens.npy").tolist()
    pfl = meta["prompt_features_lens"]
    guidance_scale = meta["guidance_scale"]
    num_steps = meta["num_steps"]
    speed = 1.0 * 1.3  # generate() multiplies speed by 1.3

    model, _ = load_model()

    with torch.no_grad():
        text_condition, _ = model.forward_text_inference_ratio_duration(
            tokens=[text_tokens],
            prompt_tokens=[prompt_tokens],
            prompt_features_lens=torch.tensor([pfl]),
            speed=speed,
        )
    features_len = text_condition.shape[1]
    speech_condition = torch.nn.functional.pad(
        prompt_features, (0, 0, 0, features_len - prompt_features.size(1))
    )

    ane = AneFmDecoderIO(AneFmDecoder(model.fm_decoder, SEQ_LEN)).eval()
    for sl, err in ane.core.basis_errs.items():
        print(f"pos basis S={sl}: reconstruction rel_max_err={err:.2e}")

    def pad_T(x):
        return torch.nn.functional.pad(x, (0, 0, 0, SEQ_LEN - x.size(1)))

    mask_f = torch.zeros(1, SEQ_LEN)
    mask_f[0, features_len:] = 1.0
    mask_b = mask_f > 0.5
    ref_mask = torch.zeros(1, features_len, dtype=torch.bool)
    text_pad, speech_pad = pad_T(text_condition), pad_T(speech_condition)

    timesteps = get_time_steps(num_step=num_steps, t_shift=0.5)
    torch.manual_seed(meta["seed"])
    x0 = torch.randn(1, features_len, 100)
    x_a, x_p, x_r = x0.clone(), x0.clone(), x0.clone()

    failures = []
    for step in range(num_steps):
        t_cur, t_next = float(timesteps[step]), float(timesteps[step + 1])
        with torch.no_grad():
            v_a = ane(
                torch.tensor([t_cur]),
                pad_T(x_a),
                text_pad,
                speech_pad,
                torch.tensor([guidance_scale]),
                mask_f,
            )[:, :features_len]
            v_p = model.forward_fm_decoder(
                t=torch.tensor(t_cur),
                xt=pad_T(x_p),
                text_condition=text_pad,
                speech_condition=speech_pad,
                padding_mask=mask_b,
                guidance_scale=torch.tensor(guidance_scale),
            )[:, :features_len]
            v_r = model.forward_fm_decoder(
                t=torch.tensor(t_cur),
                xt=x_r,
                text_condition=text_condition,
                speech_condition=speech_condition,
                padding_mask=ref_mask,
                guidance_scale=torch.tensor(guidance_scale),
            )
        gate_max = float((v_a - v_p).abs().max())
        gate_cos = cos(v_a.numpy(), v_p.numpy())
        rep_max = float((v_a - v_r).abs().max())
        rep_cos = cos(v_a.numpy(), v_r.numpy())
        ok = gate_max < GATE_MAX_ABS and gate_cos > GATE_COS
        if not ok:
            failures.append(step)
        print(
            f"[step {step} t={t_cur:.3f}] GATE vs torch@1024+mask: "
            f"max_abs={gate_max:.3e} cos={gate_cos:.8f} {'PASS' if ok else 'FAIL'} | "
            f"vs torch@751: max_abs={rep_max:.3e} cos={rep_cos:.6f}"
        )

        def euler(x_s, v_s):
            x1p = x_s + (1.0 - t_cur) * v_s
            x0p = x_s - t_cur * v_s
            return (1.0 - t_next) * x0p + t_next * x1p if step < num_steps - 1 else x1p

        x_a, x_p, x_r = euler(x_a, v_a), euler(x_p, v_p), euler(x_r, v_r)

    print(
        f"[final mel] ane vs torch@1024+mask cos={cos(x_a.numpy(), x_p.numpy()):.8f} "
        f"| ane vs torch@751 cos={cos(x_a.numpy(), x_r.numpy()):.6f}"
    )
    if failures:
        print(f"FAILED gate at steps {failures}")
        sys.exit(1)
    print("all decoder parity gates passed")


if __name__ == "__main__":
    main()
