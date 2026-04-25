"""Numerical parity: upstream flow_lm.flow_net vs CoreML flow_decoder.

cond_step prefill matches upstream (<=1e-4 per layer) and flowlm_step step 0
transformer_out matches upstream (<=1e-4). This script targets the next
stage: the 8-step LSD decode.

For a given language:
  A) Pull transformer_out from upstream flowlm_step via the same manual
     replication we used in parity_step.py (NaN->bos_emb, input_linear,
     transformer(input_cat, voice_state), out_norm, [:, -1]).
  B) Seed identical latent x0 on both sides (bit-identical numpy bytes).
  C) For i in [0..num_steps):
       s_i = i / N, t_i = (i+1) / N
       v_up     = flow_lm.flow_net(transformer_out, s, t, latent_up)
       v_coreml = coreml_flow.predict(transformer_out, latent_cm, s, t)
     Compare velocities; update latents on both sides (same dt).

Report per-step abs_max/abs_mean + final latent diff. Large divergence =
flow_decoder CoreML conversion is drifting from upstream PyTorch.

Usage:
  uv run python parity_flow.py --language german \
      --text "Hallo, das ist ein Sprachsynthesesystem."
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np
import sentencepiece as sp
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)


def _upstream_transformer_out(
    language: str, voice_path: str, token_ids: list[int]
) -> tuple[np.ndarray, "torch.nn.Module"]:
    """Return `(transformer_out [1, 1024], flow_lm module)`.

    Runs an upstream prefill + backbone pass. Caller reuses `flow_lm.flow_net`
    for step-by-step comparison against CoreML.
    """
    from pocket_tts.models.tts_model import TTSModel

    model = TTSModel.load_model(language=language, lsd_decode_steps=8)
    model.eval()
    voice_state = copy.deepcopy(model.get_state_for_audio_prompt(voice_path))
    start_offset = int(voice_state["transformer.layers.0.self_attn"]["offset"].item())
    tokens = torch.tensor([token_ids], dtype=torch.int64)
    required = start_offset + tokens.shape[1] + 64
    model._expand_kv_cache(voice_state, sequence_length=required)
    model._run_flow_lm_and_increment_step(model_state=voice_state, text_tokens=tokens)

    flow_lm = model.flow_lm
    sequence = torch.full(
        (1, 1, flow_lm.ldim), float("nan"), dtype=flow_lm.dtype
    )
    sequence_in = torch.where(torch.isnan(sequence), flow_lm.bos_emb, sequence)
    input_ = flow_lm.input_linear(sequence_in)
    text_embeddings = torch.empty((1, 0, flow_lm.dim), dtype=flow_lm.dtype)
    input_cat = torch.cat([text_embeddings, input_], dim=1)
    with torch.no_grad():
        transformer_out = flow_lm.transformer(input_cat, voice_state)
    if flow_lm.out_norm is not None:
        transformer_out = flow_lm.out_norm(transformer_out)
    transformer_out = transformer_out[:, -1].float()  # [1, 1024]
    return transformer_out.detach().cpu().numpy(), flow_lm


def _run_upstream_flow(flow_lm, transformer_out: np.ndarray, latent0: np.ndarray,
                       num_steps: int = 8) -> tuple[list[np.ndarray], np.ndarray]:
    t_out = torch.from_numpy(transformer_out).to(torch.float32)
    current = torch.from_numpy(latent0).to(torch.float32)
    velocities = []
    for i in range(num_steps):
        s = torch.tensor([[i / num_steps]], dtype=torch.float32)
        t = torch.tensor([[(i + 1) / num_steps]], dtype=torch.float32)
        with torch.no_grad():
            velocity = flow_lm.flow_net(t_out, s, t, current)
        velocities.append(velocity.detach().cpu().numpy().copy())
        current = current + velocity / num_steps
    return velocities, current.detach().cpu().numpy()


def _run_coreml_flow(coreml_flow, transformer_out: np.ndarray, latent0: np.ndarray,
                     num_steps: int = 8) -> tuple[list[np.ndarray], np.ndarray]:
    current = latent0.copy().astype(np.float32)
    t_out = transformer_out.astype(np.float32)
    velocities = []
    dt = 1.0 / num_steps
    # Discover flow output name (single output).
    spec = coreml_flow.get_spec()
    out_name = spec.description.output[0].name
    for i in range(num_steps):
        s = np.array([[i * dt]], dtype=np.float32)
        t = np.array([[(i + 1) * dt]], dtype=np.float32)
        out = coreml_flow.predict({
            "transformer_out": t_out,
            "latent": current,
            "s": s,
            "t": t,
        })
        velocity = out[out_name]
        velocities.append(velocity.copy())
        current = current + velocity * dt
    return velocities, current


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", required=True)
    parser.add_argument("--voice", default="alba")
    parser.add_argument("--text", required=True)
    parser.add_argument("--num-steps", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    lang_dir = os.path.join(_SCRIPT_DIR, "build", args.language)
    voice_path = os.path.join(lang_dir, "constants_bin", f"{args.voice}.safetensors")
    tokenizer_path = os.path.join(lang_dir, "constants_bin", "tokenizer.model")

    tok = sp.SentencePieceProcessor()
    tok.load(tokenizer_path)
    text = args.text.strip()
    if not text[0].isupper():
        text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    if len(text.split()) < 5:
        text = " " * 8 + text
    token_ids = tok.encode(text)
    print(f"Text: {text!r}")
    print(f"Tokens ({len(token_ids)}): {token_ids}")

    # Deterministic shared noise seed (bit-identical bytes on both sides).
    latent0 = (np.random.randn(1, 32).astype(np.float32) * (0.5 ** 0.5))
    print(f"latent0: shape={latent0.shape} "
          f"range=[{latent0.min():.4f}, {latent0.max():.4f}] "
          f"mean={latent0.mean():.4f} std={latent0.std():.4f}")

    print("\n[upstream] prefill + transformer_out")
    t_out, flow_lm = _upstream_transformer_out(args.language, voice_path, token_ids)
    print(f"  transformer_out shape={t_out.shape} "
          f"range=[{t_out.min():.4f}, {t_out.max():.4f}]")

    print("\n[upstream] running flow_net 8 steps...")
    up_vels, up_final = _run_upstream_flow(flow_lm, t_out, latent0, args.num_steps)

    print("\n[coreml] loading flow_decoder...")
    import coremltools as ct
    coreml_flow = ct.models.MLModel(
        os.path.join(lang_dir, "flow_decoder.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )
    print("[coreml] running flow 8 steps (same transformer_out + latent0)...")
    cm_vels, cm_final = _run_coreml_flow(coreml_flow, t_out, latent0, args.num_steps)

    print("\nPer-step velocity diff (upstream vs CoreML flow_decoder):")
    for i, (u, c) in enumerate(zip(up_vels, cm_vels)):
        diff = np.abs(u - c)
        rel = diff.mean() / (np.abs(u).mean() + 1e-9)
        print(
            f"  step {i}: u|mean={np.abs(u).mean():.5f} "
            f"c|mean={np.abs(c).mean():.5f} "
            f"abs_max={diff.max():.5f} abs_mean={diff.mean():.5f} "
            f"rel={rel:.5f}"
        )

    diff_final = np.abs(up_final - cm_final)
    print(
        f"\nFinal latent diff: "
        f"u|mean={np.abs(up_final).mean():.5f} "
        f"c|mean={np.abs(cm_final).mean():.5f} "
        f"abs_max={diff_final.max():.5f} abs_mean={diff_final.mean():.5f}"
    )


if __name__ == "__main__":
    main()
