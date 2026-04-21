"""Deeper debug: run one decode step manually and compare hidden state per layer vs HF."""
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

    mi = cv.frontend.frontend_zero_shot("希望你以后能够做的比我还好用",
         "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
         str(HERE / "CosyVoice" / "asset" / "zero_shot_prompt.wav"), cv.sample_rate,
         zero_shot_spk_id="")
    text = torch.concat([mi["prompt_text"], mi["text"]], dim=1).to(torch.int64)
    text_emb = qwen.model.embed_tokens(text)
    sos_emb = llm_model.speech_embedding.weight[llm_model.sos].reshape(1, 1, -1)
    task_id_emb = llm_model.speech_embedding.weight[llm_model.task_id].reshape(1, 1, -1)
    pst_emb = llm_model.speech_embedding(mi["llm_prompt_speech_token"])
    lm_input = torch.concat([sos_emb, text_emb, task_id_emb, pst_emb], dim=1).to(torch.float32)
    T = lm_input.shape[1]

    # Upstream: prefill + capture per-layer hidden states during dec01
    with torch.no_grad():
        masks = torch.tril(torch.ones((1, T, T), dtype=torch.bool))
        _, cache_pre = llm_model.llm.forward_one_step(lm_input, masks=masks, cache=None)
        top = 29
        next_emb = llm_model.speech_embedding.weight[top].reshape(1, 1, -1)

        # Manually run HF model step-by-step
        hf_model = qwen.model  # Qwen2Model
        hid_hf = next_emb
        # Per HF: we need position_ids for the new token
        pos_ids = torch.tensor([[T]], dtype=torch.long)
        cos, sin = hf_model.rotary_emb(hid_hf, pos_ids)
        # Build attention mask [1, 1, 1, T+1] with all zeros (since all past pos valid)
        # HF's create_causal_mask normally does this; we can hand it a 4D mask directly.
        full_mask = torch.zeros(1, 1, 1, T + 1, dtype=torch.float32)
        pk = None  # use cache_pre
        per_layer_hf = []
        # Pass explicitly: we'd need to call each Qwen2DecoderLayer by hand.
        for i, dec in enumerate(hf_model.layers):
            hid_hf = dec(
                hid_hf,
                attention_mask=full_mask,
                position_ids=pos_ids,
                past_key_values=cache_pre,
                use_cache=True,
                position_embeddings=(cos, sin),
            )
            per_layer_hf.append(hid_hf.clone())
        hid_hf = hf_model.norm(hid_hf)
        logp_hf = llm_model.llm_decoder(hid_hf[0, 0])
        top_hf = int(logp_hf.argmax())
        print(f"manual upstream dec01 argmax={top_hf}  hid[:5]={hid_hf[0, 0, :5].tolist()}")

    # Ours: manual layer-by-layer
    max_len = 768
    prefill = Qwen2Prefill(qwen, llm_model.llm_decoder, max_len=max_len, t_prefill=T).eval()
    decode = Qwen2Decode(qwen, llm_model.llm_decoder, max_len=max_len).eval()
    with torch.no_grad():
        last_hidden, _, kv_k, kv_v = prefill(lm_input, torch.tensor([T], dtype=torch.int32))
        # Step through decode layer by layer
        cur = torch.tensor([T], dtype=torch.int32)
        from src.llm_coreml import _rope_cos_sin
        positions = cur.view(1, 1).to(torch.int32)
        our_cos, our_sin = _rope_cos_sin(positions, decode.inv_freq)
        print(f"our cos[0,0,:5]={our_cos[0,0,:5].tolist()}")
        print(f"HF  cos[0,0,:5]={cos[0,0,:5].tolist()}")
        print(f"cos MAE = {(our_cos - cos).abs().mean().item():.3e}")
        print(f"sin MAE = {(our_sin - sin).abs().mean().item():.3e}")

        # Build masks like decode does
        pj = decode.pos_ids.view(1, 1, max_len, 1)
        update_mask = (pj == cur.view(1, 1, 1, 1)).to(torch.float32)
        attendable = decode.pos_ids.view(1, 1, 1, max_len) <= cur.view(1, 1, 1, 1)
        attn_mask = torch.where(attendable,
                                torch.zeros((), dtype=torch.float32),
                                torch.tensor(-1e4, dtype=torch.float32))

        x = next_emb
        per_layer_ours = []
        for i, lyr in enumerate(decode.layers):
            k_i = kv_k[i]; v_i = kv_v[i]
            x, _, _ = lyr(x, our_cos, our_sin, k_i, v_i, update_mask, attn_mask)
            per_layer_ours.append(x.clone())
            if i < 3 or i == len(decode.layers)-1:
                d = (x - per_layer_hf[i][..., :1, :]).abs() if per_layer_hf[i].shape[1] >= 1 else None
                # per_layer_hf[i] shape is [1, 1, 896]
                d = (x - per_layer_hf[i]).abs()
                print(f"layer {i:2d}: MAE={d.mean().item():.3e}  max={d.max().item():.3e}")
        x = decode.norm(x)
        logp_ours = decode.speech_lm_head(x)[0, 0]
        top_ours = int(logp_ours.argmax())
        print(f"ours dec01 argmax={top_ours}  hid[:5]={x[0, 0, :5].tolist()}")


if __name__ == "__main__":
    main()
