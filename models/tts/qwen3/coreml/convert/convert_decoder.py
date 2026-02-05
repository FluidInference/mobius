# %% [markdown]
# # Qwen3-TTS 12Hz Tokenizer Decoder → CoreML Conversion
#
# Converts the decoder (codes → audio) component to CoreML.
# Input: [B, 16, T] codec codes
# Output: [B, 1, T*1920] audio at 24kHz

# %%
import torch
import torch.nn as nn
import coremltools as ct
from pathlib import Path
import numpy as np

# Configuration
MAX_CODE_LENGTH = 125  # ~10 seconds at 12.5Hz (125 * 80ms = 10s)
SAMPLE_RATE = 24000
UPSAMPLE_RATE = 1920  # codes to audio samples

print(f"Max code length: {MAX_CODE_LENGTH}")
print(f"Max audio length: {MAX_CODE_LENGTH * UPSAMPLE_RATE / SAMPLE_RATE:.1f}s")

# %%
# Load the tokenizer model
from qwen_tts import Qwen3TTSTokenizer

print("Loading tokenizer...")
tokenizer = Qwen3TTSTokenizer.from_pretrained(
    "./tokenizer_12hz",
    device_map="cpu",
)
decoder = tokenizer.model.decoder
decoder.eval()

print(f"Decoder loaded: {sum(p.numel() for p in decoder.parameters()):,} parameters")

# %%
# Test original decoder
print("\n=== Testing Original Decoder ===")
batch_size = 1
num_quantizers = 16
seq_len = 10

test_codes = torch.randint(0, 2048, (batch_size, num_quantizers, seq_len))
print(f"Input shape: {test_codes.shape}")

with torch.no_grad():
    output = decoder(test_codes)
    print(f"Output shape: {output.shape}")

# %%
# Create CoreML-compatible wrapper
class Qwen3TTSDecoderCoreML(nn.Module):
    """
    CoreML-compatible wrapper for Qwen3-TTS 12Hz decoder.

    Handles:
    - Fixed input shape with masking
    - Deterministic operations
    """

    def __init__(self, decoder, max_length=MAX_CODE_LENGTH):
        super().__init__()
        self.decoder = decoder
        self.max_length = max_length
        self.num_quantizers = 16

    def forward(self, codes: torch.Tensor, code_length: torch.Tensor) -> torch.Tensor:
        """
        Args:
            codes: [B, 16, max_length] - padded codec codes
            code_length: [B] - actual length of codes (for output trimming reference)

        Returns:
            audio: [B, 1, max_length * 1920] - padded audio output
        """
        # Forward through decoder
        # The decoder handles variable length internally via causal convolutions
        audio = self.decoder(codes)

        return audio


# %%
# Wrap the decoder
print("\n=== Creating CoreML Wrapper ===")
coreml_decoder = Qwen3TTSDecoderCoreML(decoder)
coreml_decoder.eval()

# Test wrapper
test_codes_padded = torch.randint(0, 2048, (1, 16, MAX_CODE_LENGTH))
test_length = torch.tensor([10])

with torch.no_grad():
    output = coreml_decoder(test_codes_padded, test_length)
    print(f"Wrapper output shape: {output.shape}")

# %%
# Trace with torch.jit
print("\n=== Tracing with torch.jit ===")

# Create example inputs
example_codes = torch.randint(0, 2048, (1, 16, MAX_CODE_LENGTH))
example_length = torch.tensor([MAX_CODE_LENGTH])

try:
    with torch.no_grad():
        traced_model = torch.jit.trace(
            coreml_decoder,
            (example_codes, example_length),
            strict=False,
        )
    print("Tracing successful!")

    # Verify traced model
    with torch.no_grad():
        traced_output = traced_model(example_codes, example_length)
        original_output = coreml_decoder(example_codes, example_length)

        diff = (traced_output - original_output).abs().max().item()
        print(f"Max diff between traced and original: {diff}")

except Exception as e:
    print(f"Tracing failed: {e}")
    import traceback
    traceback.print_exc()

# %%
# Convert to CoreML
print("\n=== Converting to CoreML ===")

# Define input shapes
# codes: [1, 16, MAX_CODE_LENGTH]
# code_length: [1]

inputs = [
    ct.TensorType(
        name="codes",
        shape=(1, 16, MAX_CODE_LENGTH),
        dtype=np.int32,
    ),
    ct.TensorType(
        name="code_length",
        shape=(1,),
        dtype=np.int32,
    ),
]

outputs = [
    ct.TensorType(name="audio"),
]

try:
    mlmodel = ct.convert(
        traced_model,
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
    )
    print("CoreML conversion successful!")

    # Save the model
    output_path = Path("qwen3_tts_decoder_10s.mlpackage")
    mlmodel.save(str(output_path))
    print(f"Saved to: {output_path}")
    print(f"Model size: {sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / 1e6:.1f} MB")

except Exception as e:
    print(f"CoreML conversion failed: {e}")
    import traceback
    traceback.print_exc()

# %%
# Test CoreML model
print("\n=== Testing CoreML Model ===")

try:
    # Load and run
    import coremltools as ct

    loaded_model = ct.models.MLModel("qwen3_tts_decoder_10s.mlpackage")

    # Create test input
    test_codes_np = np.random.randint(0, 2048, (1, 16, MAX_CODE_LENGTH)).astype(np.int32)
    test_length_np = np.array([50], dtype=np.int32)

    # Run inference
    result = loaded_model.predict({
        "codes": test_codes_np,
        "code_length": test_length_np,
    })

    audio = result["audio"]
    print(f"CoreML output shape: {audio.shape}")
    print(f"Audio duration: {audio.shape[-1] / SAMPLE_RATE:.2f}s")

    # Compare with PyTorch
    with torch.no_grad():
        pt_codes = torch.from_numpy(test_codes_np)
        pt_length = torch.from_numpy(test_length_np)
        pt_output = coreml_decoder(pt_codes, pt_length).numpy()

    diff = np.abs(audio - pt_output).max()
    print(f"Max diff PyTorch vs CoreML: {diff}")

except Exception as e:
    print(f"CoreML test failed: {e}")
    import traceback
    traceback.print_exc()

# %%
# Save test audio
print("\n=== Saving Test Audio ===")

try:
    import soundfile as sf

    # Use the CoreML output
    audio_data = audio.squeeze()  # [T]

    # Normalize
    audio_data = audio_data / np.abs(audio_data).max() * 0.9

    sf.write("test_output.wav", audio_data, SAMPLE_RATE)
    print(f"Saved test audio: test_output.wav ({len(audio_data) / SAMPLE_RATE:.2f}s)")

except Exception as e:
    print(f"Audio save failed: {e}")
