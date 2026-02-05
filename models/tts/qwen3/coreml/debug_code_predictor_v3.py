# Debug Code Predictor V3: PyTorch vs CoreML detailed comparison
import torch
import numpy as np
import coremltools as ct

MAX_CODEC_TOKENS = 125

print("Loading models...")
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained("./model_0.6b", device_map="cpu", torch_dtype=torch.float32)
talker = model.model.talker
config = talker.config

# Get the code predictor
code_predictor = talker.code_predictor

# Load CoreML V3 model
print("Loading CoreML V3 code predictor...")
coreml_cp = ct.models.MLModel("qwen3_tts_code_predictor_v3.mlpackage", compute_units=ct.ComputeUnit.CPU_ONLY)

# Use real codebook0 from a previous run (first 10 tokens from V9)
codebook0_tokens = [1995, 1821, 1821, 1821, 508, 508, 637, 637, 637, 637]
num_tokens = len(codebook0_tokens)

print(f"\n=== Input ===")
print(f"Codebook0 tokens: {codebook0_tokens}")

# === PyTorch Code Predictor (infer method) ===
print("\n=== PyTorch code_predictor.infer() ===")
codebook0 = torch.tensor([codebook0_tokens], dtype=torch.long)
with torch.no_grad():
    pytorch_codes = code_predictor.infer(codebook0)
print(f"Output shape: {pytorch_codes.shape}")
for cb in range(min(5, pytorch_codes.shape[1])):
    print(f"  Codebook {cb}: {pytorch_codes[0, cb, :10].tolist()}")

# === PyTorch Code Predictor (step by step) ===
print("\n=== PyTorch step-by-step generation_steps ===")
pytorch_step_results = {}
with torch.no_grad():
    for gen_steps in range(1, 5):  # Check first 4 steps
        output = code_predictor(input_ids=codebook0, generation_steps=gen_steps)
        tokens = torch.argmax(output.logits, dim=-1)
        pytorch_step_results[gen_steps] = tokens[0].tolist()
        print(f"  gen_steps={gen_steps}: {tokens[0, :10].tolist()}")

# === CoreML Code Predictor V3 ===
print("\n=== CoreML V3 code_predictor ===")
codebook0_np = np.zeros((1, MAX_CODEC_TOKENS), dtype=np.int32)
codebook0_np[0, :num_tokens] = codebook0_tokens

coreml_result = coreml_cp.predict({"codebook0": codebook0_np})
coreml_codebooks = coreml_result["all_codebooks"]
print(f"Output shape: {coreml_codebooks.shape}")
for cb in range(min(5, coreml_codebooks.shape[1])):
    print(f"  Codebook {cb+1}: {coreml_codebooks[0, cb, :10].tolist()}")

# === Detailed Comparison ===
print("\n=== Comparison ===")
# PyTorch infer() includes codebook0, so codebook 1 is at index 1
# CoreML outputs codebooks 1-14, so codebook 1 is at index 0

for cb in range(14):
    pytorch_cb = pytorch_codes[0, cb + 1, :num_tokens].numpy()  # cb+1 because infer includes codebook0
    coreml_cb = coreml_codebooks[0, cb, :num_tokens]

    match_count = np.sum(pytorch_cb == coreml_cb)
    print(f"Codebook {cb+1}: {match_count}/{num_tokens} match")
    if match_count < num_tokens:
        print(f"  PyTorch: {pytorch_cb[:10]}")
        print(f"  CoreML:  {coreml_cb[:10]}")

# === Also compare with generation_steps ===
print("\n=== Compare gen_steps vs infer ===")
for gen_steps in [1, 2, 3, 4]:
    infer_cb = pytorch_codes[0, gen_steps, :num_tokens].tolist()
    step_cb = pytorch_step_results[gen_steps][:num_tokens]
    match = infer_cb == step_cb
    print(f"gen_steps={gen_steps}: {'MATCH' if match else 'MISMATCH'}")
    if not match:
        print(f"  infer:      {infer_cb[:10]}")
        print(f"  gen_steps:  {step_cb[:10]}")
