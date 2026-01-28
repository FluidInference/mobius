# Test if code_predictor.generate is deterministic
import torch
import numpy as np
import random
from qwen_tts import Qwen3TTSModel
import warnings
warnings.filterwarnings('ignore')

print("Loading model...")
tts_model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
talker = tts_model.model.talker
config = talker.config

# Create fixed inputs
past_hidden = torch.randn(1, 1, 1024)
last_id_hidden = torch.randn(1, 1, 1024)
inputs_embeds = torch.cat([past_hidden, last_id_hidden], dim=1)

print(f"inputs_embeds: mean={inputs_embeds.mean().item():.6f}, std={inputs_embeds.std().item():.6f}")

# Run code_predictor.generate multiple times
print("\n=== Testing determinism ===")

results = []
for i in range(3):
    torch.manual_seed(42)
    with torch.no_grad():
        result = talker.code_predictor.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=config.num_code_groups - 1,
            do_sample=False,
            return_dict_in_generate=True,
        )
        tokens = result.sequences[0].tolist()
        results.append(tokens)
        print(f"Run {i}: {tokens}")

# Check if all results are the same
all_same = all(r == results[0] for r in results)
print(f"\nAll results identical: {all_same}")

# Now test with slightly different inputs
print("\n=== Testing sensitivity to input ===")

inputs_small_diff = inputs_embeds + 0.00004 * torch.randn_like(inputs_embeds)
print(f"Max diff from original: {(inputs_small_diff - inputs_embeds).abs().max().item():.6f}")

torch.manual_seed(42)
with torch.no_grad():
    result_diff = talker.code_predictor.generate(
        inputs_embeds=inputs_small_diff,
        max_new_tokens=config.num_code_groups - 1,
        do_sample=False,
        return_dict_in_generate=True,
    )
    tokens_diff = result_diff.sequences[0].tolist()
    print(f"Original:       {results[0]}")
    print(f"Small diff:     {tokens_diff}")
    print(f"Same tokens: {tokens_diff == results[0]}")

# Test with exact same input but after some other operations
print("\n=== Testing after other operations ===")

# Do some stuff
dummy = torch.randn(100, 100) @ torch.randn(100, 100)
_ = talker.model(inputs_embeds=torch.randn(1, 10, 1024), use_cache=False)

# Now run code_predictor again with same input
torch.manual_seed(42)
with torch.no_grad():
    result_after = talker.code_predictor.generate(
        inputs_embeds=inputs_embeds.clone(),  # clone to ensure no aliasing
        max_new_tokens=config.num_code_groups - 1,
        do_sample=False,
        return_dict_in_generate=True,
    )
    tokens_after = result_after.sequences[0].tolist()
    print(f"After operations: {tokens_after}")
    print(f"Same as original: {tokens_after == results[0]}")

# Check if code_predictor has any internal state
print("\n=== Checking code_predictor state ===")
print(f"code_predictor type: {type(talker.code_predictor)}")
print(f"Has reset_state: {hasattr(talker.code_predictor, 'reset_state')}")
print(f"Has _reorder_cache: {hasattr(talker.code_predictor, '_reorder_cache')}")

# Try clearing caches
if hasattr(talker.code_predictor, 'model'):
    cp_model = talker.code_predictor.model
    if hasattr(cp_model, 'clear_cache'):
        cp_model.clear_cache()
        print("Cleared code_predictor.model cache")
