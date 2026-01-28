# Explore Code Predictor Architecture
import torch
from qwen_tts import Qwen3TTSModel

print("Loading model...")
model = Qwen3TTSModel.from_pretrained(
    "./model_0.6b",
    device_map="cpu",
    torch_dtype=torch.float32,
)
print("Model loaded!")

talker = model.model.talker
code_predictor = talker.code_predictor

print("\n=== Code Predictor Architecture ===")
print(f"Type: {type(code_predictor)}")

# Print structure
for name, child in code_predictor.named_children():
    params = sum(p.numel() for p in child.parameters())
    print(f"  {name}: {type(child).__name__} ({params/1e6:.1f}M params)")

print("\n=== Talker Config (relevant parts) ===")
talker_config = talker.config
for attr in dir(talker_config):
    if not attr.startswith('_') and not callable(getattr(talker_config, attr)):
        val = getattr(talker_config, attr)
        if isinstance(val, (int, float, str, bool, list)) and len(str(val)) < 100:
            print(f"  {attr}: {val}")

print("\n=== Code Predictor Config ===")
cp_config = code_predictor.config
for attr in dir(cp_config):
    if not attr.startswith('_') and not callable(getattr(cp_config, attr)):
        val = getattr(cp_config, attr)
        if isinstance(val, (int, float, str, bool, list)) and len(str(val)) < 100:
            print(f"  {attr}: {val}")

# Look at the forward signature
import inspect
print("\n=== Code Predictor Forward Signature ===")
sig = inspect.signature(code_predictor.forward)
print(f"Parameters: {list(sig.parameters.keys())}")

# Check the source
print("\n=== Code Predictor Source (first 3000 chars) ===")
try:
    source = inspect.getsource(code_predictor.forward)
    print(source[:3000])
except:
    print("Could not get source")

# Look at internal model structure
print("\n=== Internal Model Structure ===")
inner_model = code_predictor.model
for name, child in inner_model.named_children():
    params = sum(p.numel() for p in child.parameters())
    print(f"  {name}: {type(child).__name__} ({params/1e6:.1f}M params)")

# Look at lm_head
print("\n=== LM Head Structure ===")
lm_head = code_predictor.lm_head
print(f"LM head type: {type(lm_head)}")
print(f"Number of heads: {len(lm_head)}")
if len(lm_head) > 0:
    print(f"First head: {lm_head[0]}")
    print(f"Last head: {lm_head[-1]}")

# Test forward pass
print("\n=== Test Forward Pass ===")
batch_size = 1
seq_len = 10  # 10 codec tokens
hidden_dim = cp_config.hidden_size

print(f"Code predictor hidden size: {hidden_dim}")
print(f"Vocab size: {cp_config.vocab_size}")

# The code predictor takes hidden states and generates remaining codebook layers
# Let's trace through to understand the exact input format

# Create dummy inputs matching what the talker would pass
hidden_states = torch.randn(batch_size, seq_len, hidden_dim)

print(f"Hidden states shape: {hidden_states.shape}")

# Try to understand what inputs it expects
with torch.no_grad():
    try:
        # Try calling forward with just inputs_embeds
        output = code_predictor(inputs_embeds=hidden_states)
        print(f"Output type: {type(output)}")
        if hasattr(output, 'logits'):
            print(f"Logits shape: {output.logits.shape}")
        if hasattr(output, 'hidden_states'):
            print(f"Hidden states: {output.hidden_states}")
    except Exception as e:
        print(f"Forward with inputs_embeds failed: {e}")

# Look at detailed structure
print("\n=== Detailed Module Structure ===")
for name, module in code_predictor.named_modules():
    if name and '.' not in name:  # Only top-level children
        params = sum(p.numel() for p in module.parameters(recurse=True))
        print(f"  {name}: {type(module).__name__} ({params/1e6:.2f}M params)")
