"""Sanity-check Qwen2Prefill / Qwen2Decode wrappers vs upstream CosyVoice3 LLM.

Loads the real CosyVoice3LM (llm.pt weights + Qwen2-BlankEN), runs a short
synthesis prefix through both:

  (a) upstream: CosyVoice3LM.inference_wrapper via Qwen2Encoder.forward_one_step
      (HF Qwen2ForCausalLM with DynamicCache, full prefill + a few decode steps)

  (b) our wrapper: Qwen2Prefill then repeated Qwen2Decode

…and reports per-step (logits_argmax, logits_top1, MAE) so we can see whether
the static-cache re-implementation matches the upstream loop bit-exactly.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE / "CosyVoice"))
sys.path.insert(0, str(HERE / "CosyVoice" / "third_party" / "Matcha-TTS"))

import argparse
import numpy as np
import torch

from src.llm_coreml import Qwen2Prefill, Qwen2Decode


MODEL_DIR = ROOT / "cosyvoice3_dl"


def _load_llm():
    from cosyvoice.cli.cosyvoice import CosyVoice3
    cv = CosyVoice3(str(MODEL_DIR), load_trt=False, load_vllm=False, fp16=False)
    cv.model.llm.float()  # cast Qwen2 from bf16 → fp32
    return cv


def _build_prefill_embeds(cv, tts_text: str, prompt_text: str, prompt_wav_path: str):
    """Replicate CosyVoice3LM.inference up to the point lm_input is built."""
    fe = cv.frontend
    mi = fe.frontend_zero_shot(tts_text, prompt_text, prompt_wav_path, cv.sample_rate, zero_shot_spk_id="")
    # mi keys: text, text_len, prompt_text, prompt_text_len,
    #          prompt_speech_token, prompt_speech_token_len,
    #          prompt_speech_feat, prompt_speech_feat_len, llm_embedding, flow_embedding

    # Emulate Qwen2LM.inference lines 474..494
    llm_model = cv.model.llm
    text = torch.concat([mi["prompt_text"], mi["text"]], dim=1).to(torch.int64)

    text_emb = llm_model.llm.model.model.embed_tokens(text)       # [1, T_txt, 896]
    sos_emb     = llm_model.speech_embedding.weight[llm_model.sos].reshape(1, 1, -1)
    task_id_emb = llm_model.speech_embedding.weight[llm_model.task_id].reshape(1, 1, -1)
    pst = mi["llm_prompt_speech_token"]
    if pst.shape[1] > 0:
        pst_emb = llm_model.speech_embedding(pst)
    else:
        pst_emb = torch.zeros(1, 0, llm_model.speech_embedding.embedding_dim, dtype=text_emb.dtype)

    lm_input = torch.concat([sos_emb, text_emb, task_id_emb, pst_emb], dim=1)
    return lm_input.to(torch.float32), llm_model


def _upstream_prefill_then_decode(llm_model, lm_input: torch.Tensor, n_decode: int):
    """Run HF's Qwen2Model with a correctly-sized attention_mask each step.

    CosyVoice3 was written against transformers 4.40.1; in 5.x the attention
    mask semantics for ``forward_one_step`` (which passes a length-1 mask)
    changed, producing mathematically incorrect outputs.  This helper goes
    through HF directly with a full-length mask so we have a reliable reference.
    """
    from transformers.cache_utils import DynamicCache

    qwen = llm_model.llm.model
    cfg = qwen.config
    results = []
    cache = DynamicCache(config=cfg)
    x = lm_input

    # Step 0: prefill
    attn_mask = torch.ones(1, x.shape[1], dtype=torch.long)
    outs = qwen(inputs_embeds=x, attention_mask=attn_mask,
                past_key_values=cache, use_cache=True,
                output_hidden_states=True, return_dict=True)
    cache = outs.past_key_values
    last_hidden = outs.hidden_states[-1][:, -1, :]
    logp = llm_model.llm_decoder(last_hidden)
    top = int(logp.argmax(-1))
    results.append({"logits": logp.detach().clone(), "argmax": top,
                    "hidden": last_hidden.detach().clone()})

    # Steps 1..n_decode
    for _ in range(n_decode):
        next_emb = llm_model.speech_embedding.weight[top].reshape(1, 1, -1)
        cur = cache.get_seq_length()
        attn_mask = torch.ones(1, cur + 1, dtype=torch.long)
        outs = qwen(inputs_embeds=next_emb, attention_mask=attn_mask,
                    past_key_values=cache, use_cache=True,
                    output_hidden_states=True, return_dict=True)
        cache = outs.past_key_values
        last_hidden = outs.hidden_states[-1][:, -1, :]
        logp = llm_model.llm_decoder(last_hidden)
        top = int(logp.argmax(-1))
        results.append({"logits": logp.detach().clone(), "argmax": top,
                        "hidden": last_hidden.detach().clone()})

    return results


def _ours_prefill_then_decode(llm_model, lm_input: torch.Tensor, n_decode: int, max_len: int):
    t_prefill = lm_input.shape[1]
    qwen = llm_model.llm.model   # Qwen2ForCausalLM
    speech_head = llm_model.llm_decoder

    prefill = Qwen2Prefill(qwen, speech_head, max_len=max_len, t_prefill=t_prefill).eval()
    decode  = Qwen2Decode (qwen, speech_head, max_len=max_len).eval()

    with torch.no_grad():
        input_len = torch.tensor([t_prefill], dtype=torch.int32)
        last_hidden, all_logits, kv_k, kv_v = prefill(lm_input, input_len)

    results = [{
        "logits": all_logits[0, t_prefill - 1].detach().clone().unsqueeze(0),  # [1, 6761]
        "argmax": int(all_logits[0, t_prefill - 1].argmax(-1).item()),
        "hidden": last_hidden[0, t_prefill - 1].detach().clone().unsqueeze(0),
    }]

    cur = t_prefill
    for step in range(n_decode):
        top_id = results[-1]["argmax"]
        next_emb = llm_model.speech_embedding.weight[top_id].reshape(1, 1, -1)
        with torch.no_grad():
            logits, kv_k, kv_v = decode(next_emb,
                                         kv_k, kv_v,
                                         torch.tensor([cur], dtype=torch.int32))
        results.append({"logits": logits[0, 0].unsqueeze(0),
                        "argmax": int(logits[0, 0].argmax(-1).item()),
                        "hidden": None})
        cur += 1
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts-text", default="希望你以后能够做的比我还好用")
    ap.add_argument("--prompt-text",
                    default="You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。")
    ap.add_argument("--n-decode", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=768)
    args = ap.parse_args()

    prompt_wav = HERE / "CosyVoice" / "asset" / "zero_shot_prompt.wav"

    print(f"[1/4] Loading CosyVoice3 end-to-end (CPU, fp32)…")
    cv = _load_llm()

    print(f"[2/4] Building lm_input from frontend…")
    lm_input, llm_model = _build_prefill_embeds(cv, args.tts_text, args.prompt_text, str(prompt_wav))
    T = lm_input.shape[1]
    print(f"      lm_input shape = {tuple(lm_input.shape)}  (T_prefill={T})")
    if T > args.max_len:
        raise SystemExit(f"lm_input T={T} exceeds --max-len={args.max_len}; raise it")

    print(f"[3/4] Running upstream forward_one_step prefill + {args.n_decode} decode steps…")
    with torch.no_grad():
        up = _upstream_prefill_then_decode(llm_model, lm_input, args.n_decode)

    print(f"[4/4] Running our wrapper (Qwen2Prefill + Qwen2Decode) same steps…")
    ours = _ours_prefill_then_decode(llm_model, lm_input, args.n_decode, max_len=args.max_len)

    print("\nPer-step comparison (upstream vs ours):")
    print("  step | upstream_argmax | ours_argmax | MAE(logits) | max|d| ")
    print("  -----+-----------------+-------------+-------------+--------")
    for i, (u, o) in enumerate(zip(up, ours)):
        d = (u["logits"] - o["logits"]).abs()
        mae = float(d.mean().item())
        mx  = float(d.max().item())
        tag = "prefill" if i == 0 else f"dec{i:02d}"
        print(f"  {tag} | {u['argmax']:>15d} | {o['argmax']:>11d} | "
              f"{mae:.3e} | {mx:.3e}")


if __name__ == "__main__":
    main()
