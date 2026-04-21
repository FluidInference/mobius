"""Deeply compare one decode step: upstream vs our wrapper."""
import sys
from pathlib import Path
HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE / "CosyVoice"))
sys.path.insert(0, str(HERE / "CosyVoice" / "third_party" / "Matcha-TTS"))

import torch
from src.llm_coreml import Qwen2Prefill, Qwen2Decode


def main():
    from cosyvoice.cli.cosyvoice import CosyVoice3
    cv = CosyVoice3(str(ROOT / "cosyvoice3_dl"), load_trt=False, load_vllm=False, fp16=False)
    cv.model.llm.float()
    llm_model = cv.model.llm
    qwen = llm_model.llm.model

    # Build lm_input like the other test
    mi = cv.frontend.frontend_zero_shot("希望你以后能够做的比我还好用",
         "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
         str(HERE / "CosyVoice" / "asset" / "zero_shot_prompt.wav"), cv.sample_rate,
         zero_shot_spk_id="")
    text = torch.concat([mi["prompt_text"], mi["text"]], dim=1).to(torch.int64)
    text_emb = qwen.model.embed_tokens(text)
    sos_emb = llm_model.speech_embedding.weight[llm_model.sos].reshape(1, 1, -1)
    task_id_emb = llm_model.speech_embedding.weight[llm_model.task_id].reshape(1, 1, -1)
    pst = mi["llm_prompt_speech_token"]
    pst_emb = llm_model.speech_embedding(pst)
    lm_input = torch.concat([sos_emb, text_emb, task_id_emb, pst_emb], dim=1).to(torch.float32)

    T = lm_input.shape[1]
    print(f"T_prefill = {T}")

    # ------------- Upstream prefill then 1 decode step -------------
    with torch.no_grad():
        masks = torch.tril(torch.ones((1, T, T), dtype=torch.bool))
        y_pred_pre, cache_pre = llm_model.llm.forward_one_step(lm_input, masks=masks, cache=None)
        hid_pre = y_pred_pre[:, -1, :]
        logp_pre = llm_model.llm_decoder(hid_pre)
        top_pre = int(logp_pre.argmax(-1))
        print(f"upstream prefill: hid[:5]={hid_pre[0, :5].tolist()}  argmax={top_pre}")

        next_emb = llm_model.speech_embedding.weight[top_pre].reshape(1, 1, -1)
        masks_dec = torch.tril(torch.ones((1, 1, 1), dtype=torch.bool))
        y_pred_dec, cache_dec = llm_model.llm.forward_one_step(next_emb, masks=masks_dec, cache=cache_pre)
        hid_up_dec = y_pred_dec[:, -1, :]
        logp_up_dec = llm_model.llm_decoder(hid_up_dec)
        top_up_dec = int(logp_up_dec.argmax(-1))
        print(f"upstream dec01:   hid[:5]={hid_up_dec[0, :5].tolist()}  argmax={top_up_dec}")

        # Also inspect HF cache K/V at layer 0 at position 124 (last prefill pos)
        print(f"HF cache type: {type(cache_pre).__name__}")
        print(f"  layers attr: {hasattr(cache_pre, 'layers')}")
        print(f"  dir keys: {[a for a in dir(cache_pre) if 'key' in a.lower() or 'value' in a.lower() or 'layer' in a.lower()][:20]}")
        # transformers 5.x: DynamicCache has .layers = list, each with .keys / .values
        layers_attr = getattr(cache_pre, "layers", None)
        if layers_attr is not None:
            print(f"  len(layers) = {len(layers_attr)}")
            l0 = layers_attr[0]
            print(f"  layer[0] = {type(l0).__name__}, attrs = {[a for a in dir(l0) if not a.startswith('_')][:20]}")
            hf_k0 = getattr(l0, "keys", None)
            if hf_k0 is None and hasattr(l0, "__getitem__"):
                hf_k0 = l0[0]; hf_v0 = l0[1]
            else:
                hf_v0 = l0.values
            print(f"  layer0 keys shape = {tuple(hf_k0.shape)}")

    # ------------- Our prefill + 1 decode step -------------
    max_len = 768
    prefill = Qwen2Prefill(qwen, llm_model.llm_decoder, max_len=max_len, t_prefill=T).eval()
    decode = Qwen2Decode(qwen, llm_model.llm_decoder, max_len=max_len).eval()
    with torch.no_grad():
        last_hidden, all_logits, kv_k, kv_v = prefill(lm_input, torch.tensor([T], dtype=torch.int32))
        hid_ours_pre = last_hidden[:, T-1, :]
        top_ours_pre = int(all_logits[0, T-1].argmax(-1))
        print(f"\nours prefill: hid[:5]={hid_ours_pre[0, :5].tolist()}  argmax={top_ours_pre}")

        # Compare K, V at layer 0 positions 0..T-1
        print(f"  our kv_k[0] shape = {kv_k[0].shape}")
        # Compare against HF
        d_k = (kv_k[0, 0, :, :T, :] - hf_k0[0, :, :T, :]).abs()
        d_v = (kv_v[0, 0, :, :T, :] - hf_v0[0, :, :T, :]).abs()
        print(f"  K[layer0, :T] MAE={d_k.mean().item():.3e}  max={d_k.max().item():.3e}")
        print(f"  V[layer0, :T] MAE={d_v.mean().item():.3e}  max={d_v.max().item():.3e}")

        # Our decode step
        next_emb = llm_model.speech_embedding.weight[top_ours_pre].reshape(1, 1, -1)
        cur_len_t = torch.tensor([T], dtype=torch.int32)
        speech_logits, kv_k2, kv_v2 = decode(next_emb, kv_k, kv_v, cur_len_t)
        hid_ours_dec_from_logits = speech_logits  # actually this IS the logits not hidden
        top_ours_dec = int(speech_logits[0, 0].argmax(-1))
        print(f"ours dec01 argmax = {top_ours_dec}")
        # Compare cache at position T (new)
        print(f"  our new K[layer0, :, T]   vs  HF K[layer0, :, -1]:")
        d_k = (kv_k2[0, 0, :, T, :] - cache_dec.layers[0].keys[0, :, -1, :]).abs() if hasattr(cache_dec, "layers") else None
        print(f"  {d_k.mean().item() if d_k is not None else '??'}")


if __name__ == "__main__":
    main()
