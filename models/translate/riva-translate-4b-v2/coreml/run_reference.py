"""Greedy reference generation with HF transformers for parity checking.

Dumps prompt token ids, generated token ids, and first-step logits to
reference.npz for run_coreml.py to compare against.

Usage:
    uv run run_reference.py --max-new-tokens 32
"""

# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#     "torch>=2.4",
#     "transformers>=4.48",
#     "numpy<2",
#     "safetensors",
#     "huggingface_hub",
# ]
# ///

import argparse
from pathlib import Path

import numpy as np
import torch

SOURCE_TEXT = "The weather in San Francisco is unusually warm for this time of year."


def build_prompt(tokenizer, text: str, source_lang: str, target_lang: str) -> str:
    messages = [
        {
            "role": "user",
            "content": f"Translate the following text from {source_lang} to {target_lang}: {text}",
        }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="nvidia/Riva-Translate-4B-Instruct-v2")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--source-lang", default="English")
    parser.add_argument("--target-lang", default="German")
    parser.add_argument("--output", default="reference.npz")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    prompt = build_prompt(tokenizer, SOURCE_TEXT, args.source_lang, args.target_lang)
    print(f"Prompt:\n{prompt!r}\n")

    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    print(f"Prompt tokens: {input_ids.shape[1]}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.eval()

    with torch.no_grad():
        # First-step logits for numeric comparison
        first = model(input_ids)
        first_logits = first.logits[0, -1, :].float().numpy()

        out = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )

    gen_ids = out[0, input_ids.shape[1] :].numpy()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    print(f"Reference translation: {text!r}")
    print(f"Generated ids: {gen_ids.tolist()}")

    np.savez(
        Path(args.output),
        prompt_ids=input_ids[0].numpy(),
        gen_ids=gen_ids,
        first_logits=first_logits,
        eos_token_id=np.int64(tokenizer.eos_token_id),
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
