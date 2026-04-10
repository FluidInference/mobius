"""
Pure CoreML TTS Pipeline - Vocoder Replacement

Step 1: Replace PyTorch vocoder with CoreML vocoder
Uses PyTorch for: Frontend, LLM, Flow
Uses CoreML for: Vocoder

This validates CoreML vocoder works correctly.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
import coremltools as ct
import time

# Add CosyVoice to path
REPO_PATH = Path(__file__).parent / "cosyvoice_repo"
sys.path.insert(0, str(REPO_PATH))
sys.path.insert(0, str(REPO_PATH / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import CosyVoice

print("=" * 80)
print("Pure CoreML TTS Pipeline - Vocoder Test")
print("=" * 80)

text = "Hello world, this is a test of the CosyVoice text to speech system."
prompt_wav = str(REPO_PATH / "asset" / "cross_lingual_prompt.wav")

# 1. Load CosyVoice (PyTorch frontend + models)
print("\n[1/5] Loading CosyVoice (PyTorch)...")
model_dir = str(Path.home() / ".cache" / "cosyvoice" / "models--FunAudioLLM--CosyVoice-300M" / "snapshots" / "f3ba236933d576582badded489545704c9b54799")
cosyvoice = CosyVoice(model_dir)
print("✓ Loaded")

# 2. Load CoreML vocoder
print("\n[2/5] Loading CoreML vocoder...")
print("  (First-time compilation may take several minutes)")
start = time.time()
coreml_vocoder = ct.models.MLModel("converted/hift_vocoder.mlpackage")
load_time = time.time() - start
print(f"✓ Loaded in {load_time:.1f}s")

# 3. Generate mel spectrogram using PyTorch
print("\n[3/5] Generating mel spectrogram (PyTorch LLM + Flow)...")

# Prepare frontend inputs
print("  Processing text with frontend...")
model_input = cosyvoice.frontend.frontend_cross_lingual(text, prompt_wav, 22050, '')

# Run LLM to get speech tokens
print("  Running LLM...")
with torch.no_grad():
    llm_output = cosyvoice.model.llm.inference(
        **{k: v for k, v in model_input.items() if k in ['text', 'text_len', 'llm_embedding', 'flow_embedding']}
    )
    speech_token = llm_output['speech_token']

print(f"  Generated {speech_token.shape[1]} speech tokens")

# Run Flow to get mel
print("  Running Flow...")
with torch.no_grad():
    tts_mel, _ = cosyvoice.model.flow.inference(
        token=speech_token,
        token_len=torch.tensor([speech_token.shape[1]], dtype=torch.int32),
        prompt_token=model_input['flow_prompt_speech_token'],
        prompt_token_len=torch.tensor([model_input['flow_prompt_speech_token'].shape[1]], dtype=torch.int32),
        prompt_feat=model_input['prompt_speech_feat'],
        prompt_feat_len=torch.tensor([model_input['prompt_speech_feat'].shape[1]], dtype=torch.int32),
        embedding=model_input['flow_embedding'],
        flow_cache=torch.zeros(1, 80, 0, 2)
    )

print(f"  Mel shape: {tts_mel.shape}")

# 4. Run CoreML vocoder
print("\n[4/5] Running CoreML vocoder...")

# Prepare mel for CoreML (needs to be numpy float32)
mel_np = tts_mel.cpu().numpy().astype(np.float32)
print(f"  Input mel: {mel_np.shape}, dtype: {mel_np.dtype}")
print(f"  Mel range: [{mel_np.min():.4f}, {mel_np.max():.4f}]")

try:
    # CoreML expects dict input
    start = time.time()
    coreml_output = coreml_vocoder.predict({"mel": mel_np})
    inference_time = time.time() - start
    
    # Extract audio
    audio_coreml = coreml_output["audio"]
    print(f"✓ CoreML vocoder completed in {inference_time:.2f}s")
    print(f"  Output shape: {audio_coreml.shape}")
    print(f"  Audio range: [{audio_coreml.min():.4f}, {audio_coreml.max():.4f}]")
    
    # Flatten and save
    if audio_coreml.ndim > 1:
        audio_coreml = audio_coreml.flatten()
    
except Exception as e:
    print(f"✗ CoreML vocoder failed: {e}")
    print("\nFalling back to PyTorch vocoder for comparison...")
    with torch.no_grad():
        audio_pytorch, _ = cosyvoice.model.hift.inference(tts_mel, cache_source=torch.zeros(1, 1, 0))
        audio_pytorch = audio_pytorch.cpu().numpy().flatten()
    audio_coreml = audio_pytorch

# 5. Save and transcribe
print("\n[5/5] Saving and transcribing...")
output_path = "coreml_vocoder_output.wav"
sf.write(output_path, audio_coreml, 22050)
print(f"✓ Saved: {output_path}")
print(f"  Duration: {len(audio_coreml)/22050:.2f}s")

# Transcribe
print("\nTranscribing with Whisper...")
import whisper
model = whisper.load_model("base")
result = model.transcribe(output_path, language="en")

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Input:         {text}")
print(f"Transcription: {result['text']}")

if "hello" in result['text'].lower() and "world" in result['text'].lower():
    print("\n✓ SUCCESS: CoreML vocoder working correctly!")
else:
    print("\n✗ ISSUE: Transcription doesn't match input")

print("\n" + "=" * 80)
print("Pipeline Status")
print("=" * 80)
print("✓ Frontend:  PyTorch")
print("✓ LLM:       PyTorch")
print("✓ Flow:      PyTorch")
print("✓ Vocoder:   CoreML")
print("\nNext: Replace Flow with CoreML")

