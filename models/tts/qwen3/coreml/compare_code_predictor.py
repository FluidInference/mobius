# Compare Code Predictor: PyTorch vs CoreML
import torch
import numpy as np
import coremltools as ct
from qwen_tts import Qwen3TTSModel

MAX_CODEC_TOKENS = 125

print("Loading models...")
model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
talker = model.model.talker
processor = model.processor
config = talker.config

# Get the code predictor
code_predictor = talker.code_predictor

# Load CoreML model
coreml_code_predictor = ct.models.MLModel("qwen3_tts_code_predictor_v2.mlpackage")

# Generate some LM tokens using PyTorch
text = "Hello world, this is a test of the text to speech system."
inputs = processor(text=text, return_tensors="pt")
text_ids = inputs.input_ids

EOS_TOKEN = config.codec_eos_token_id

with torch.no_grad():
    text_embed = talker.model.text_embedding(text_ids)
    text_projected = talker.text_projection(text_embed)

    lang_embed = talker.model.codec_embedding(torch.tensor([[config.codec_language_id["english"]]]))
    bos_embed = talker.model.codec_embedding(torch.tensor([[config.codec_bos_id]]))

    combined = torch.cat([lang_embed, text_projected, bos_embed], dim=1)
    outputs = talker.model(inputs_embeds=combined, use_cache=True, return_dict=True)

    logits = talker.codec_head(outputs.last_hidden_state[:, -1:, :])
    first_token = torch.argmax(logits, dim=-1).item()
    past_kv = outputs.past_key_values

    generated_tokens = [first_token]
    current_token = torch.tensor([[first_token]])

    while len(generated_tokens) < MAX_CODEC_TOKENS:
        token_embed = talker.model.codec_embedding(current_token)
        outputs = talker.model(
            inputs_embeds=token_embed,
            past_key_values=past_kv,
            use_cache=True,
            return_dict=True,
        )
        logits = talker.codec_head(outputs.last_hidden_state)
        next_token = torch.argmax(logits, dim=-1).item()
        generated_tokens.append(next_token)
        current_token = torch.tensor([[next_token]])
        past_kv = outputs.past_key_values

        if next_token == EOS_TOKEN:
            break

num_tokens = len(generated_tokens)
print(f"\nGenerated {num_tokens} LM tokens")
print(f"Tokens: {generated_tokens[:10]}...")

# === PyTorch Code Predictor ===
print("\n=== PyTorch Code Predictor ===")
codebook0 = torch.tensor([generated_tokens], dtype=torch.long)
print(f"Input shape: {codebook0.shape}")

with torch.no_grad():
    pytorch_codes = code_predictor.infer(codebook0)
print(f"Output shape: {pytorch_codes.shape}")
print(f"Codebook 0: {pytorch_codes[0, 0, :5].tolist()}")
print(f"Codebook 1: {pytorch_codes[0, 1, :5].tolist()}")
print(f"Codebook 2: {pytorch_codes[0, 2, :5].tolist()}")
print(f"Codebook 15: {pytorch_codes[0, 15, :5].tolist()}")

# === CoreML Code Predictor ===
print("\n=== CoreML Code Predictor ===")
codebook0_np = np.zeros((1, MAX_CODEC_TOKENS), dtype=np.int32)
codebook0_np[0, :num_tokens] = generated_tokens
print(f"Input shape: {codebook0_np.shape}")

coreml_result = coreml_code_predictor.predict({"codebook0": codebook0_np})
coreml_codebooks = coreml_result["all_codebooks"]
print(f"Output shape: {coreml_codebooks.shape}")

# CoreML output is codebooks 1-15 (shape: 1, 15, 125)
print(f"Codebook 1: {coreml_codebooks[0, 0, :5]}")
print(f"Codebook 2: {coreml_codebooks[0, 1, :5]}")
print(f"Codebook 15: {coreml_codebooks[0, 14, :5]}")

# === Compare ===
print("\n=== Comparison ===")
# PyTorch includes codebook 0, CoreML outputs codebooks 1-15
pytorch_codebooks_1_15 = pytorch_codes[0, 1:, :MAX_CODEC_TOKENS].numpy()
coreml_codebooks_trimmed = coreml_codebooks[0, :, :num_tokens]

print(f"PyTorch codebooks 1-15 shape: {pytorch_codebooks_1_15.shape}")
print(f"CoreML codebooks 1-15 shape: {coreml_codebooks_trimmed.shape}")

# Compare codebook 1
for cb in range(15):
    pytorch_cb = pytorch_codebooks_1_15[cb, :num_tokens]
    coreml_cb = coreml_codebooks_trimmed[cb, :]
    match = np.array_equal(pytorch_cb, coreml_cb)
    if not match:
        diff_count = np.sum(pytorch_cb != coreml_cb)
        print(f"Codebook {cb+1}: MISMATCH ({diff_count}/{num_tokens} different)")
        print(f"  PyTorch: {pytorch_cb[:10]}")
        print(f"  CoreML:  {coreml_cb[:10]}")
    else:
        print(f"Codebook {cb+1}: MATCH")
