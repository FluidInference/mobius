# Analyze code_predictor model structure
import torch
from qwen_tts import Qwen3TTSModel
import warnings
warnings.filterwarnings('ignore')

print("Loading model...")
tts_model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
talker = tts_model.model.talker
code_predictor = talker.code_predictor

print("\n=== Code Predictor Config ===")
config = code_predictor.config
print(f"Hidden size: {config.hidden_size}")
print(f"Num layers: {config.num_hidden_layers}")
print(f"Num attention heads: {config.num_attention_heads}")
print(f"Num KV heads: {config.num_key_value_heads}")
print(f"Vocab size: {config.vocab_size}")
print(f"Num code groups: {config.num_code_groups}")

print("\n=== Talker Config (main model) ===")
talker_config = talker.config
print(f"Hidden size: {talker_config.hidden_size}")
print(f"Num layers: {talker_config.num_hidden_layers}")
print(f"Num attention heads: {talker_config.num_attention_heads}")
print(f"Num KV heads: {talker_config.num_key_value_heads}")

# Count parameters
def count_params(model):
    return sum(p.numel() for p in model.parameters())

print("\n=== Parameter Counts ===")
print(f"Code predictor: {count_params(code_predictor):,} params")
print(f"Talker (main): {count_params(talker.model):,} params")
print(f"Total talker: {count_params(talker):,} params")

# Measure forward time
import time

# Prepare sample inputs
past_hidden = torch.randn(1, 1, config.hidden_size)
last_id_hidden = torch.randn(1, 1, config.hidden_size)
inputs_embeds = torch.cat([past_hidden, last_id_hidden], dim=1)

print("\n=== Forward Time (1 full generate) ===")
with torch.no_grad():
    start = time.time()
    for _ in range(10):
        result = code_predictor.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=3,
            do_sample=False,
            return_dict_in_generate=True,
        )
    elapsed = time.time() - start
    print(f"Code predictor generate (avg): {elapsed/10*1000:.2f} ms")

# Compare with main model forward
main_hidden = torch.randn(1, 1, talker_config.hidden_size)
with torch.no_grad():
    start = time.time()
    for _ in range(10):
        out = talker.model(inputs_embeds=main_hidden, use_cache=False)
    elapsed = time.time() - start
    print(f"Main model forward (avg): {elapsed/10*1000:.2f} ms")
