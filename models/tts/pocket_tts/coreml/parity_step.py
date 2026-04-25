"""Numerical parity: upstream flow_lm vs CoreML flowlm_step on step 0.

After both pipelines produce a matching post-prefill KV cache (verified by
`parity_prefill.py`), run ONE generation step on each and compare the
transformer output.

  A) Upstream:  sequence=NaN[1,1,32], text_embeddings=empty, model_state=voice_state
     → model.flow_lm.forward(...) returns (output_embeddings, is_eos).
       We grab the transformer_out by stubbing flow_net with the identity.
     → Cleanest: replicate flow_lm.forward up through backbone + out_norm
       manually here so we can read transformer_out directly.
  B) CoreML:    coreml_step.predict(sequence=NaN, bos_emb, cache*, position*)
                returns `x` (transformer_out pre-flow-decoder) and `is_eos`.

Compare shapes + max abs diff on `transformer_out`.

Usage:
  uv run python parity_step.py --language german \
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


def _upstream_step0(language: str, voice_path: str, token_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    from pocket_tts.models.tts_model import TTSModel

    model = TTSModel.load_model(language=language, lsd_decode_steps=8)
    voice_state = copy.deepcopy(model.get_state_for_audio_prompt(voice_path))

    start_offset = int(voice_state["transformer.layers.0.self_attn"]["offset"].item())
    tokens = torch.tensor([token_ids], dtype=torch.int64)
    required = start_offset + tokens.shape[1] + 64
    model._expand_kv_cache(voice_state, sequence_length=required)
    model._run_flow_lm_and_increment_step(model_state=voice_state, text_tokens=tokens)

    # Replicate flow_lm.forward up through transformer_out without the
    # stochastic flow decoder.
    flow_lm = model.flow_lm
    sequence = torch.full(
        (1, 1, flow_lm.ldim), float("nan"),
        dtype=flow_lm.dtype,
    )
    sequence_in = torch.where(torch.isnan(sequence), flow_lm.bos_emb, sequence)
    input_ = flow_lm.input_linear(sequence_in)
    text_embeddings = torch.empty(
        (1, 0, flow_lm.dim), dtype=flow_lm.dtype
    )
    input_cat = torch.cat([text_embeddings, input_], dim=1)
    with torch.no_grad():
        transformer_out = flow_lm.transformer(input_cat, voice_state)
    if flow_lm.out_norm is not None:
        transformer_out = flow_lm.out_norm(transformer_out)
    transformer_out = transformer_out[:, -1:].float()  # [1, 1, 1024]
    is_eos_logit = flow_lm.out_eos(transformer_out).float()
    return transformer_out.detach().cpu().numpy(), is_eos_logit.detach().cpu().numpy()


def _coreml_step0(language: str, voice_path: str, token_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    import coremltools as ct
    from safetensors.numpy import load_file

    model_dir = os.path.join(_SCRIPT_DIR, "build", language)
    const_dir = os.path.join(model_dir, "constants")

    coreml_cond = ct.models.MLModel(
        os.path.join(model_dir, "cond_step.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )
    coreml_step = ct.models.MLModel(
        os.path.join(model_dir, "flowlm_step.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )

    # Discover shapes + output ordering.
    step_spec = coreml_step.get_spec()
    num_layers = 0
    cache_slots = None
    for inp in step_spec.description.input:
        if inp.name.startswith("cache") and inp.name[len("cache"):].isdigit():
            num_layers = max(num_layers, int(inp.name[len("cache"):]) + 1)
            if cache_slots is None:
                cache_slots = int(inp.type.multiArrayType.shape[2])
    rank3, cache_names, pos_names = [], [], []
    for out in step_spec.description.output:
        if not out.type.HasField("multiArrayType"):
            continue
        shape = list(out.type.multiArrayType.shape)
        rank = len(shape)
        if rank == 5:
            cache_names.append(out.name)
        elif rank == 1:
            pos_names.append(out.name)
        elif rank == 3:
            rank3.append((out.name, shape))
    eos_name = next(n for n, s in rank3 if s[-1] == 1)
    xfmr_name = next(n for n, s in rank3 if s[-1] != 1)

    cond_spec = coreml_cond.get_spec()
    cond_cache, cond_pos = [], []
    for out in cond_spec.description.output:
        if not out.type.HasField("multiArrayType"):
            continue
        rank = len(out.type.multiArrayType.shape)
        if rank == 5:
            cond_cache.append(out.name)
        elif rank == 1:
            cond_pos.append(out.name)

    # Seed from our voice safetensors (same bytes upstream uses).
    tensors = load_file(voice_path)
    caches = {}
    positions = {}
    for layer in range(num_layers):
        ck = f"transformer.layers.{layer}.self_attn/cache"
        ok = f"transformer.layers.{layer}.self_attn/offset"
        raw = tensors[ck].astype(np.float32)
        padded = np.zeros(
            (2, 1, cache_slots, raw.shape[3], raw.shape[4]), dtype=np.float32
        )
        padded[:, :, : raw.shape[2], :, :] = raw
        caches[f"cache{layer}"] = padded
        positions[f"position{layer}"] = np.array(
            [float(np.asarray(tensors[ok]).reshape(-1)[0])], dtype=np.float32
        )

    embed_table = np.load(os.path.join(const_dir, "text_embed_table.npy"))
    text_emb = embed_table[np.array(token_ids, dtype=np.int64)][np.newaxis, :, :]
    for t in range(text_emb.shape[1]):
        inp = {
            "conditioning": text_emb[:, t:t + 1, :].astype(np.float32),
            **caches,
            **positions,
        }
        out = coreml_cond.predict(inp)
        for i in range(num_layers):
            caches[f"cache{i}"] = out[cond_cache[i]]
            positions[f"position{i}"] = out[cond_pos[i]]

    # One flowlm_step iteration: sequence=NaN[1,1,32], bos_emb loaded from
    # legacy constants (matches what generate_coreml_v4 does).
    bos_emb = np.load(os.path.join(const_dir, "bos_emb.npy")).astype(np.float32)
    if bos_emb.ndim == 1:
        bos_emb_ = bos_emb
    else:
        bos_emb_ = bos_emb.reshape(-1)
    step_inputs = {
        "sequence": np.full((1, 1, 32), np.nan, dtype=np.float32),
        "bos_emb": bos_emb_.astype(np.float32),
        **caches,
        **positions,
    }
    step_out = coreml_step.predict(step_inputs)
    xfmr = step_out[xfmr_name]   # [1, 1, 1024]
    is_eos = step_out[eos_name]  # [1, 1, 1]
    return xfmr, is_eos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", required=True)
    parser.add_argument("--voice", default="alba")
    parser.add_argument("--text", required=True)
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

    up_xfmr, up_eos = _upstream_step0(args.language, voice_path, token_ids)
    print(f"[upstream] xfmr shape={up_xfmr.shape} range=[{up_xfmr.min():.4f}, {up_xfmr.max():.4f}] "
          f"is_eos_logit={up_eos.reshape(-1)[0]:.4f}")

    cm_xfmr, cm_eos = _coreml_step0(args.language, voice_path, token_ids)
    print(f"[coreml]   xfmr shape={cm_xfmr.shape} range=[{cm_xfmr.min():.4f}, {cm_xfmr.max():.4f}] "
          f"is_eos_logit={cm_eos.reshape(-1)[0]:.4f}")

    if up_xfmr.shape != cm_xfmr.shape:
        print(f"!!! shape mismatch")
        return
    diff = np.abs(up_xfmr - cm_xfmr)
    rel = diff.mean() / (np.abs(up_xfmr).mean() + 1e-9)
    print(f"transformer_out diff: abs_max={diff.max():.5f} abs_mean={diff.mean():.5f} rel={rel:.5f}")
    eos_diff = abs(up_eos.reshape(-1)[0] - cm_eos.reshape(-1)[0])
    print(f"is_eos_logit diff: {eos_diff:.5f}")


if __name__ == "__main__":
    main()
