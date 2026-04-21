"""Run a few decode steps manually with a correct full-length mask, and confirm
whether the upstream forward_one_step path corresponds to "only new pos" or
"full past + new"."""
import sys
from pathlib import Path
HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE / "CosyVoice"))
sys.path.insert(0, str(HERE / "CosyVoice" / "third_party" / "Matcha-TTS"))

import torch


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
    sos = llm_model.speech_embedding.weight[llm_model.sos].reshape(1, 1, -1)
    tid = llm_model.speech_embedding.weight[llm_model.task_id].reshape(1, 1, -1)
    pst_emb = llm_model.speech_embedding(mi["llm_prompt_speech_token"])
    lm_input = torch.concat([sos, text_emb, tid, pst_emb], dim=1).to(torch.float32)
    T = lm_input.shape[1]
    print(f"T_prefill = {T}")

    with torch.no_grad():
        # Path A: upstream forward_one_step (suspicious)
        cache_A = None
        x = lm_input
        masks = torch.tril(torch.ones((1, T, T), dtype=torch.bool))
        y, cache_A = llm_model.llm.forward_one_step(x, masks=masks, cache=cache_A)
        top_A = int(llm_model.llm_decoder(y[:, -1]).argmax(-1))
        print(f"A: prefill argmax={top_A}")

        results_A = [top_A]
        for step in range(4):
            next_emb = llm_model.speech_embedding.weight[top_A].reshape(1, 1, -1)
            masks1 = torch.tril(torch.ones((1, 1, 1), dtype=torch.bool))
            y, cache_A = llm_model.llm.forward_one_step(next_emb, masks=masks1, cache=cache_A)
            top_A = int(llm_model.llm_decoder(y[:, -1]).argmax(-1))
            results_A.append(top_A)
        print(f"A (forward_one_step): argmaxes = {results_A}")

        # Path B: upstream HF model directly with all-ones 2D mask of past+new length
        hf_model = qwen.model
        from transformers.cache_utils import DynamicCache
        cache_B = DynamicCache(config=qwen.config)
        # Prefill: full-length mask
        attn_mask_prefill = torch.ones(1, T, dtype=torch.long)
        outs = qwen(inputs_embeds=lm_input,
                    attention_mask=attn_mask_prefill,
                    past_key_values=cache_B,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True)
        cache_B = outs.past_key_values
        top_B = int(llm_model.llm_decoder(outs.hidden_states[-1][:, -1]).argmax(-1))
        results_B = [top_B]
        for step in range(4):
            next_emb = llm_model.speech_embedding.weight[top_B].reshape(1, 1, -1)
            cur_len = cache_B.get_seq_length()
            attn_mask_dec = torch.ones(1, cur_len + 1, dtype=torch.long)
            outs = qwen(inputs_embeds=next_emb,
                        attention_mask=attn_mask_dec,
                        past_key_values=cache_B,
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True)
            cache_B = outs.past_key_values
            top_B = int(llm_model.llm_decoder(outs.hidden_states[-1][:, -1]).argmax(-1))
            results_B.append(top_B)
        print(f"B (full-length attn_mask): argmaxes = {results_B}")


if __name__ == "__main__":
    main()
