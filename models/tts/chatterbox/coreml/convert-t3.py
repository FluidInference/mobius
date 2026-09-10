"""Convert Chatterbox multilingual T3 (Llama-520M) to CoreML: Prefill + Decode.

Steps:
  1. Load ChatterboxMultilingualTTS from the local HF snapshot (fp32, CPU).
  2. Build a real CFG batch-2 prefill embed via t3.prepare_input_embeds
     (built-in voice from conds.pt, real tokenized text).
  3. PyTorch parity: run the stock patched_model path (eager attention,
     output_attentions) for N steps, recording per-step logits and the three
     alignment-head rows; run T3Prefill/T3Decode wrappers on the same inputs
     and compare.
  4. torch.jit.trace + ct.convert both wrappers, re-check parity on CoreML.

Usage:
    uv run python convert-t3.py --output-dir build/t3 [--fp16] [--steps 20]
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

from src.t3_coreml import ALIGNED_HEADS, T3Decode, T3Prefill  # noqa: E402

T_PREFILL = 256
MAX_LEN = 1024
TEXT = "The quick brown fox jumps over the lazy dog near the river bank."
LANG = "en"


def load_model():
    from chatterbox.mtl_tts import REPO_ID, ChatterboxMultilingualTTS
    from huggingface_hub import snapshot_download

    ckpt_dir = snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt",
                        "grapheme_mtl_merged_expanded_v1.json", "conds.pt",
                        "Cangjie5_TC.json"],
    )
    return ChatterboxMultilingualTTS.from_local(ckpt_dir, "cpu")


def build_prefill_embeds(model):
    """Replicate mtl_tts.generate() up to prepare_input_embeds + BOS concat."""
    from chatterbox.mtl_tts import punc_norm

    t3 = model.t3
    text = punc_norm(TEXT)
    text_tokens = model.tokenizer.text_to_tokens(text, language_id=LANG)
    text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # CFG batch 2
    sot, eot = t3.hp.start_text_token, t3.hp.stop_text_token
    text_tokens = torch.nn.functional.pad(text_tokens, (1, 0), value=sot)
    text_tokens = torch.nn.functional.pad(text_tokens, (0, 1), value=eot)

    with torch.no_grad():
        embeds, len_cond = t3.prepare_input_embeds(
            t3_cond=model.conds.t3,
            text_tokens=text_tokens,
            speech_tokens=t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1]),
            cfg_weight=0.5,
        )
        # inference() appends a second BOS embed on top of the one already in
        # `embeds` — faithful replication of the stock (double-BOS) context.
        bos = torch.tensor([[t3.hp.start_speech_token]], dtype=torch.long)
        bos_embed = t3.speech_emb(bos) + t3.speech_pos_emb.get_fixed_embedding(0)
        embeds = torch.cat([embeds, torch.cat([bos_embed, bos_embed])], dim=1)
    return embeds, len_cond, text_tokens.shape[1]


def stock_reference(model, embeds, len_cond, text_len, steps, greedy_tokens=None):
    """Run the stock T3HuggingfaceBackend path, capturing logits + align rows."""
    from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend

    t3 = model.t3
    backend = T3HuggingfaceBackend(
        config=t3.cfg, llama=t3.tfmr, speech_enc=t3.speech_emb,
        speech_head=t3.speech_head, alignment_stream_analyzer=None)

    captured = {}

    def make_hook(buffer_idx, head_idx):
        def hook(module, inputs, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                captured[buffer_idx] = output[1][0, head_idx].detach()
        return hook

    handles = []
    for i, (layer_idx, head_idx) in enumerate(sorted(ALIGNED_HEADS)):
        handles.append(t3.tfmr.layers[layer_idx].self_attn.register_forward_hook(
            make_hook(i, head_idx)))
    t3.tfmr.config._attn_implementation = "eager"
    t3.tfmr.config.output_attentions = True

    logits_seq, align_seq = [], []
    with torch.no_grad():
        out = backend(inputs_embeds=embeds, past_key_values=None, use_cache=True,
                      output_attentions=True, return_dict=True)
        past = out.past_key_values
        logits_seq.append(out.logits[:, -1, :].clone())
        # first chunk: both BOS query rows, [3, 2, T0]
        align_seq.append(torch.stack([captured[i][-2:] for i in range(3)]).clone())

        for i in range(steps):
            tok = (greedy_tokens[i] if greedy_tokens is not None
                   else int(torch.argmax(logits_seq[-1][0:1], dim=-1)))
            emb = t3.speech_emb(torch.tensor([[tok]])) \
                + t3.speech_pos_emb.get_fixed_embedding(i + 1)
            emb = torch.cat([emb, emb])
            out = backend(inputs_embeds=emb, past_key_values=past,
                          output_attentions=True, return_dict=True)
            past = out.past_key_values
            logits_seq.append(out.logits[:, -1, :].clone())
            align_seq.append(torch.stack([captured[i2][-1] for i2 in range(3)]).clone())

    for h in handles:
        h.remove()
    return logits_seq, align_seq


def wrapper_reference(model, embeds, len_cond, text_len, steps):
    t3 = model.t3
    prefill = T3Prefill(t3.tfmr, t3.speech_head, MAX_LEN, T_PREFILL).eval()
    decode = T3Decode(t3.tfmr, t3.speech_head, MAX_LEN).eval()

    T0 = embeds.shape[1]
    padded = torch.nn.functional.pad(embeds, (0, 0, 0, T_PREFILL - T0))
    input_len = torch.tensor([T0], dtype=torch.int32)

    logits_seq, align_seq = [], []
    with torch.no_grad():
        logits, align, kv_k, kv_v = prefill(padded, input_len)
        logits_seq.append(logits)
        align_seq.append(align)
        for i in range(steps):
            tok = int(torch.argmax(logits_seq[-1][0:1], dim=-1))
            emb = t3.speech_emb(torch.tensor([[tok]])) \
                + t3.speech_pos_emb.get_fixed_embedding(i + 1)
            emb = torch.cat([emb, emb])
            cur_len = torch.tensor([T0 + i], dtype=torch.int32)
            logits, align, kv_k, kv_v = decode(emb, kv_k, kv_v, cur_len)
            logits_seq.append(logits)
            align_seq.append(align)
    return prefill, decode, padded, input_len, logits_seq, align_seq


def compare(tag, stock_logits, stock_align, wrap_logits, wrap_align, ctx_lens):
    worst_l = worst_a = 0.0
    for sl, wl in zip(stock_logits, wrap_logits):
        worst_l = max(worst_l, (sl - wl).abs().max().item())
    for i, (sa, wa) in enumerate(zip(stock_align, wrap_align)):
        # stock rows span the live context; wrapper rows are padded to MAX_LEN
        ctx = ctx_lens[i]
        worst_a = max(worst_a, (sa - wa[..., :ctx]).abs().max().item())
    print(f"[{tag}] max |dlogits| = {worst_l:.3e}   max |dalign| = {worst_a:.3e}")
    return worst_l, worst_a


def convert(prefill, decode, padded, input_len, kv_shape, out_dir: Path, fp16: bool):
    precision = ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32
    tag = "fp16" if fp16 else "fp32"
    target = ct.target.iOS17

    with torch.no_grad():
        traced_p = torch.jit.trace(prefill, (padded, input_len), strict=False)
    mlp = ct.convert(
        traced_p,
        inputs=[
            ct.TensorType(name="inputs_embeds", shape=(2, T_PREFILL, 1024), dtype=np.float32),
            ct.TensorType(name="input_len", shape=(1,), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float32),
            ct.TensorType(name="align_attn", dtype=np.float32),
            ct.TensorType(name="kv_k", dtype=np.float32),
            ct.TensorType(name="kv_v", dtype=np.float32),
        ],
        compute_precision=precision,
        minimum_deployment_target=target,
        convert_to="mlprogram",
    )
    p_path = out_dir / f"T3-Prefill-T{T_PREFILL}-M{MAX_LEN}-{tag}.mlpackage"
    mlp.save(str(p_path))
    print(f"saved {p_path}")

    kv_k = torch.zeros(kv_shape)
    kv_v = torch.zeros(kv_shape)
    emb1 = torch.zeros(2, 1, 1024)
    cur_len = torch.tensor([64], dtype=torch.int32)
    with torch.no_grad():
        traced_d = torch.jit.trace(decode, (emb1, kv_k, kv_v, cur_len), strict=False)
    mld = ct.convert(
        traced_d,
        inputs=[
            ct.TensorType(name="inputs_embeds", shape=(2, 1, 1024), dtype=np.float32),
            ct.TensorType(name="kv_k", shape=tuple(kv_shape), dtype=np.float32),
            ct.TensorType(name="kv_v", shape=tuple(kv_shape), dtype=np.float32),
            ct.TensorType(name="cur_len", shape=(1,), dtype=np.int32),
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float32),
            ct.TensorType(name="align_attn", dtype=np.float32),
            ct.TensorType(name="kv_k_out", dtype=np.float32),
            ct.TensorType(name="kv_v_out", dtype=np.float32),
        ],
        compute_precision=precision,
        minimum_deployment_target=target,
        convert_to="mlprogram",
    )
    d_path = out_dir / f"T3-Decode-M{MAX_LEN}-{tag}.mlpackage"
    mld.save(str(d_path))
    print(f"saved {d_path}")
    return p_path, d_path


def coreml_parity(p_path, d_path, model, padded, input_len, steps,
                  ref_logits, ref_align, T0, compute_units="CPU_AND_NE"):
    t3 = model.t3
    cu = getattr(ct.ComputeUnit, compute_units)
    mp = ct.models.MLModel(str(p_path), compute_units=cu)
    md = ct.models.MLModel(str(d_path), compute_units=cu)

    out = mp.predict({"inputs_embeds": padded.numpy(), "input_len": input_len.numpy()})
    logits = torch.from_numpy(out["logits"])
    kv_k, kv_v = out["kv_k"], out["kv_v"]
    worst_l = (logits - ref_logits[0]).abs().max().item()
    worst_a = (torch.from_numpy(out["align_attn"])[..., :T0]
               - ref_align[0][..., :T0]).abs().max().item()

    for i in range(steps):
        tok = int(torch.argmax(logits[0:1], dim=-1))
        with torch.no_grad():
            emb = t3.speech_emb(torch.tensor([[tok]])) \
                + t3.speech_pos_emb.get_fixed_embedding(i + 1)
        emb = torch.cat([emb, emb]).numpy()
        out = md.predict({
            "inputs_embeds": emb, "kv_k": kv_k, "kv_v": kv_v,
            "cur_len": np.array([T0 + i], dtype=np.int32)})
        kv_k, kv_v = out["kv_k_out"], out["kv_v_out"]
        logits = torch.from_numpy(out["logits"])
        ctx = T0 + i + 1
        worst_l = max(worst_l, (logits - ref_logits[i + 1]).abs().max().item())
        worst_a = max(worst_a, (torch.from_numpy(out["align_attn"])[:, :ctx]
                                - ref_align[i + 1][:, :ctx]).abs().max().item())
    print(f"[coreml] max |dlogits| = {worst_l:.3e}   max |dalign| = {worst_a:.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("build/t3"))
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--skip-convert", action="store_true")
    ap.add_argument("--parity-only", action="store_true",
                    help="reuse existing mlpackages; skip trace/convert")
    ap.add_argument("--compute-units", default="CPU_AND_NE",
                    choices=["ALL", "CPU_ONLY", "CPU_AND_NE", "CPU_AND_GPU"])
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    print("[1/4] loading model...")
    model = load_model()
    embeds, len_cond, text_len = build_prefill_embeds(model)
    T0 = embeds.shape[1]
    print(f"      embeds {tuple(embeds.shape)}  len_cond={len_cond} text_len={text_len}")

    print("[2/4] wrapper forward...")
    prefill, decode, padded, input_len, wrap_logits, wrap_align = \
        wrapper_reference(model, embeds, len_cond, text_len, args.steps)

    print("[3/4] stock reference...")
    greedy = [int(torch.argmax(l[0:1], dim=-1)) for l in wrap_logits[:-1]]
    stock_logits, stock_align = stock_reference(
        model, embeds, len_cond, text_len, args.steps, greedy_tokens=greedy)
    ctx_lens = [T0 + i for i in range(args.steps + 1)]
    compare("pytorch", stock_logits, stock_align, wrap_logits, wrap_align, ctx_lens)

    if args.skip_convert:
        return
    kv_shape = (30, 2, 16, MAX_LEN, 64)
    tag = "fp16" if args.fp16 else "fp32"
    if args.parity_only:
        p_path = args.output_dir / f"T3-Prefill-T{T_PREFILL}-M{MAX_LEN}-{tag}.mlpackage"
        d_path = args.output_dir / f"T3-Decode-M{MAX_LEN}-{tag}.mlpackage"
    else:
        print("[4/4] converting...")
        p_path, d_path = convert(prefill, decode, padded, input_len, kv_shape,
                                 args.output_dir, args.fp16)
    coreml_parity(p_path, d_path, model, padded, input_len, args.steps,
                  wrap_logits, wrap_align, T0, compute_units=args.compute_units)


if __name__ == "__main__":
    main()
