"""Convert Chatterbox Nano S3Gen to CoreML: meanflow Flow (2-step, no CFG) + HiFT.

Fixture: built-in voice ref (conds.pt) + speech tokens from a seeded stock
Nano T3 run (inference_turbo), cached to build/fixtures/.

Parity strategy mirrors convert-s3gen.py:
  A. stock (captured z, zeroed SineGen randomness)  vs  wrapper — compute path.
  B. wrapper PyTorch  vs  CoreML                    — conversion parity.

Usage:
    uv run python convert-s3gen-nano.py --output-dir build/s3gen-nano [--fp16]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from src.s3gen_coreml import FlowMeanCoreML, HiFTCoreML  # noqa: E402

N_TOKENS = 500          # default bucket (prompt + generated); see --bucket
N_TIMESTEPS = 2         # shipped meanflow checkpoints use 2 plain Euler steps
TEXT = "The quick brown fox jumps over the lazy dog near the river bank."


def load_model():
    from src.nano_ckpt import load_nano

    return load_nano()


def get_speech_tokens(model, cache: Path) -> torch.Tensor:
    if cache.exists():
        return torch.load(cache, weights_only=True)
    from chatterbox.tts_turbo import punc_norm

    torch.manual_seed(42)
    text_tokens = model.tokenizer(punc_norm(TEXT), return_tensors="pt").input_ids
    with torch.inference_mode():
        speech_tokens = model.t3.inference_turbo(
            t3_cond=model.conds.t3, text_tokens=text_tokens,
            temperature=0.8, top_k=1000, top_p=0.95, repetition_penalty=1.2)
        speech_tokens = speech_tokens[speech_tokens < 6561].view(1, -1)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(speech_tokens, cache)
    return speech_tokens


def patch_rel_shift():
    """coremltools has no `view_as`; swap in an equivalent reshape."""
    from chatterbox.models.s3gen.transformer.attention import (
        RelPositionMultiHeadedAttention as RPA)
    import torch as _t

    def rel_shift(self, x: _t.Tensor) -> _t.Tensor:
        zero_pad = _t.zeros((x.size(0), x.size(1), x.size(2), 1),
                            device=x.device, dtype=x.dtype)
        x_padded = _t.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(x.size(0), x.size(1), x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].reshape(x.size(0), x.size(1), x.size(2), x.size(3))[
            :, :, :, : x.size(-1) // 2 + 1]
        return x

    RPA.rel_shift = rel_shift


def zero_sinegen_randomness(s3gen):
    """Monkeypatch SineGen.forward to phase=0 / noise=0 (deterministic ref)."""
    sg = s3gen.mel2wav.m_source.l_sin_gen

    def det_forward(f0):
        F_mat = torch.zeros((f0.size(0), sg.harmonic_num + 1, f0.size(-1)))
        for i in range(sg.harmonic_num + 1):
            F_mat[:, i: i + 1, :] = f0 * (i + 1) / sg.sampling_rate
        theta_mat = 2 * np.pi * (torch.cumsum(F_mat, dim=-1) % 1)
        sine_waves = sg.sine_amp * torch.sin(theta_mat)      # phase = 0
        uv = sg._f02uv(f0)
        return sine_waves * uv, uv, None                     # noise = 0
    sg.forward = det_forward


def main():
    global N_TOKENS
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("build/s3gen-nano"))
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--skip-convert", action="store_true")
    ap.add_argument("--parity-only", action="store_true",
                    help="reuse existing mlpackages; skip trace/convert")
    ap.add_argument("--bucket", type=int, default=N_TOKENS,
                    help="flow token bucket (prompt + generated); 500 ≈ 10 s "
                         "of generated audio after the built-in voice's 250 "
                         "prompt tokens, 1000 ≈ 30 s (FluidAudio #924)")
    args = ap.parse_args()
    N_TOKENS = args.bucket
    args.output_dir.mkdir(parents=True, exist_ok=True)

    patch_rel_shift()
    print("[1/6] loading model...")
    model = load_model()
    s3gen = model.s3gen.float().eval()
    ref = {k: (v.detach() if torch.is_tensor(v) else v)
           for k, v in model.conds.gen.items()}

    print("[2/6] speech tokens fixture...")
    speech_tokens = get_speech_tokens(
        model, HERE / "build" / "fixtures" / "speech_tokens_nano.pt")
    print(f"      {speech_tokens.shape[-1]} generated tokens")

    prompt_token = ref["prompt_token"]
    P = prompt_token.shape[1]
    tokens_real = torch.cat([prompt_token, speech_tokens.view(1, -1)], dim=1)
    T_real = tokens_real.shape[1]
    assert T_real <= N_TOKENS, f"{T_real} > bucket {N_TOKENS}"
    M = N_TOKENS * 2

    tokens = torch.nn.functional.pad(tokens_real, (0, N_TOKENS - T_real)).to(torch.int32)
    token_len = torch.tensor([T_real], dtype=torch.int32)
    prompt_len = torch.tensor([P], dtype=torch.int32)
    prompt_feat = torch.zeros(1, M, 80)
    prompt_feat[:, :ref["prompt_feat"].shape[1]] = ref["prompt_feat"]
    embedding = ref["embedding"]

    # ---- stock reference mel (capture effective z incl. noised_mels overwrite) ----
    print("[3/6] stock flow reference...")
    captured = {}
    orig_forward = type(s3gen.flow.decoder).forward

    def capture_forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None,
                        cond=None, noised_mels=None, meanflow=False):
        assert meanflow, "nano s3gen must run the meanflow branch"
        torch.manual_seed(1234)
        z = torch.randn_like(mu)
        if noised_mels is not None:
            plen = mu.size(2) - noised_mels.size(2)
            z[..., plen:] = noised_mels
        captured["z"] = z.clone()
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        return self.basic_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks,
                                cond=cond), None

    type(s3gen.flow.decoder).forward = capture_forward
    with torch.inference_mode():
        stock_mel = s3gen.flow_inference(
            speech_tokens.view(1, -1), ref_dict={k: (v.clone() if torch.is_tensor(v) else v)
                                                 for k, v in ref.items()},
            finalize=True)
    type(s3gen.flow.decoder).forward = orig_forward
    print(f"      stock mel {tuple(stock_mel.shape)}")

    # ---- wrapper forward ----
    print("[4/6] flow wrapper parity (pytorch)...")
    flow_wrap = FlowMeanCoreML(s3gen.flow, N_TOKENS, N_TIMESTEPS).eval()
    z_full = torch.zeros(1, 80, M)
    z_full[:, :, :captured["z"].shape[2]] = captured["z"]
    with torch.no_grad():
        mel_full = flow_wrap(tokens, token_len, prompt_len, prompt_feat, embedding, z_full)
    wrap_mel = mel_full[:, :, 2 * P:2 * T_real]
    dmel = (wrap_mel - stock_mel).abs()
    print(f"      mel max|d| = {dmel.max().item():.3e}  mean|d| = {dmel.mean().item():.3e}")

    # ---- HiFT ----
    print("[5/6] hift wrapper parity (pytorch)...")
    zero_sinegen_randomness(s3gen)
    with torch.inference_mode():
        stock_wav, _ = s3gen.hift_inference(stock_mel)
    hift_wrap = HiFTCoreML(s3gen.mel2wav).eval()
    T_mel = stock_mel.shape[2]
    L = T_mel * 480
    phase0 = torch.zeros(1, 9, 1)
    noise0 = torch.zeros(1, 9, L)
    with torch.no_grad():
        wrap_wav = hift_wrap(stock_mel, phase0, noise0)
    dwav = (wrap_wav - stock_wav).abs()
    print(f"      wav max|d| = {dwav.max().item():.3e}  mean|d| = {dwav.mean().item():.3e}")

    if args.skip_convert:
        return

    print("[6/6] converting...")
    precision = ct.precision.FLOAT16 if args.fp16 else ct.precision.FLOAT32
    tag = "fp16" if args.fp16 else "fp32"

    if args.parity_only:
        f_path = args.output_dir / f"FlowMean-N{N_TOKENS}-{tag}.mlpackage"
        h_path = args.output_dir / f"HiFT-T{M}-{tag}.mlpackage"
        mel_pad = torch.zeros(1, 80, M)
        mel_pad[:, :, :T_mel] = stock_mel
        noise_pad = torch.zeros(1, 9, M * 480)
        return coreml_parity(f_path, h_path, tokens, token_len, prompt_len,
                             prompt_feat, embedding, z_full, wrap_mel, wrap_wav,
                             mel_pad, noise_pad, phase0, P, T_real, T_mel)

    with torch.no_grad():
        traced_f = torch.jit.trace(
            flow_wrap, (tokens, token_len, prompt_len, prompt_feat, embedding, z_full),
            strict=False)
    mlf = ct.convert(
        traced_f,
        inputs=[
            ct.TensorType(name="tokens", shape=(1, N_TOKENS), dtype=np.int32),
            ct.TensorType(name="token_len", shape=(1,), dtype=np.int32),
            ct.TensorType(name="prompt_len", shape=(1,), dtype=np.int32),
            ct.TensorType(name="prompt_feat", shape=(1, M, 80), dtype=np.float32),
            ct.TensorType(name="embedding", shape=(1, 192), dtype=np.float32),
            ct.TensorType(name="z", shape=(1, 80, M), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="mel", dtype=np.float32)],
        compute_precision=precision,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    )
    f_path = args.output_dir / f"FlowMean-N{N_TOKENS}-{tag}.mlpackage"
    mlf.save(str(f_path))
    print(f"      saved {f_path}")

    mel_pad = torch.zeros(1, 80, M)
    mel_pad[:, :, :T_mel] = stock_mel
    noise_pad = torch.zeros(1, 9, M * 480)
    with torch.no_grad():
        traced_h = torch.jit.trace(hift_wrap, (mel_pad, phase0, noise_pad), strict=False)
    mlh = ct.convert(
        traced_h,
        inputs=[
            ct.TensorType(name="mel", shape=(1, 80, M), dtype=np.float32),
            ct.TensorType(name="phase_vec", shape=(1, 9, 1), dtype=np.float32),
            ct.TensorType(name="noise", shape=(1, 9, M * 480), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="audio", dtype=np.float32)],
        compute_precision=precision,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    )
    h_path = args.output_dir / f"HiFT-T{M}-{tag}.mlpackage"
    mlh.save(str(h_path))
    print(f"      saved {h_path}")

    coreml_parity(f_path, h_path, tokens, token_len, prompt_len, prompt_feat,
                  embedding, z_full, wrap_mel, wrap_wav, mel_pad, noise_pad,
                  phase0, P, T_real, T_mel)


def coreml_parity(f_path, h_path, tokens, token_len, prompt_len, prompt_feat,
                  embedding, z_full, wrap_mel, wrap_wav, mel_pad, noise_pad,
                  phase0, P, T_real, T_mel):
    import time
    for cu in ("CPU_AND_GPU",):
        mpf = ct.models.MLModel(str(f_path), compute_units=getattr(ct.ComputeUnit, cu))
        t0 = time.perf_counter()
        out = mpf.predict({
            "tokens": tokens.numpy(), "token_len": token_len.numpy(),
            "prompt_len": prompt_len.numpy(), "prompt_feat": prompt_feat.numpy(),
            "embedding": embedding.numpy(), "z": z_full.numpy()})
        t_flow = time.perf_counter() - t0
        cm_mel = torch.from_numpy(out["mel"])[:, :, 2 * P:2 * T_real]
        d = (cm_mel - wrap_mel).abs()
        print(f"[coreml/{cu}] flow mel max|d| = {d.max().item():.3e} "
              f"mean|d| = {d.mean().item():.3e}  ({t_flow:.2f}s)")

        mph = ct.models.MLModel(str(h_path), compute_units=getattr(ct.ComputeUnit, cu))
        t0 = time.perf_counter()
        out = mph.predict({"mel": mel_pad.numpy(), "phase_vec": phase0.numpy(),
                           "noise": noise_pad.numpy()})
        t_hift = time.perf_counter() - t0
        cm_wav = torch.from_numpy(out["audio"])[:, :T_mel * 480]
        d = (cm_wav - wrap_wav).abs()
        print(f"[coreml/{cu}] hift wav max|d| = {d.max().item():.3e} "
              f"mean|d| = {d.mean().item():.3e}  ({t_hift:.2f}s)")


if __name__ == "__main__":
    main()
