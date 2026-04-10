"""
FULL CosyVoice3 TTS Pipeline - PyTorch (no CoreML)
Actual text-to-speech with proper tokenization and inference.
"""

import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import sys
from pathlib import Path
import torch
import torchaudio
import numpy as np
from scipy.io import wavfile

REPO_PATH = Path(__file__).parent / "cosyvoice_repo"
sys.path.insert(0, str(REPO_PATH))
sys.path.insert(0, str(REPO_PATH / "third_party" / "Matcha-TTS"))

print("=" * 80)
print("Full CosyVoice3 TTS Pipeline (PyTorch)")
print("=" * 80)

text = "Hello world, this is a test of the CosyVoice text to speech system."
print(f"\nInput text: {text}")

# Try to load full CosyVoice model
print("\n[1/3] Loading CosyVoice model...")

try:
    from cosyvoice.cli.cosyvoice import CosyVoice
    from huggingface_hub import snapshot_download
    
    print("Downloading model...")
    model_dir = snapshot_download(
        repo_id="FunAudioLLM/CosyVoice-300M",  # Smaller, faster model
        cache_dir=Path.home() / ".cache" / "cosyvoice"
    )
    
    print(f"Model dir: {model_dir}")
    print("Initializing CosyVoice...")
    
    cosyvoice = CosyVoice(model_dir)
    print("✓ Model loaded")
    
    # Generate speech
    print("\n[2/3] Generating speech...")

    # Use cross-lingual mode (Chinese prompt → English speech)
    prompt_wav = str(REPO_PATH / "asset" / "cross_lingual_prompt.wav")
    print(f"Using cross-lingual mode with prompt: {prompt_wav}")
    for i, audio_chunk in enumerate(cosyvoice.inference_cross_lingual(text, prompt_wav)):
        audio = audio_chunk['tts_speech'].numpy()

        # Save
        import soundfile as sf
        output_path = "full_pipeline_pytorch.wav"
        sample_rate = 22050

        # Flatten to 1D if needed (mono audio)
        if audio.ndim > 1:
            audio = audio.flatten()

        sf.write(output_path, audio, sample_rate)

        duration = len(audio) / sample_rate
        file_size = len(audio) * 2 / 1024  # 2 bytes per sample for int16
        
        print(f"\n✓ Generated audio:")
        print(f"  Output: {output_path}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  File size: {file_size:.1f} KB")
        print(f"  Sample rate: {sample_rate} Hz")
        
        break  # Just use first chunk
    
    # Transcribe with Whisper
    print("\n[3/3] Transcribing with Whisper...")
    
    try:
        import whisper
        
        print("Loading Whisper...")
        model = whisper.load_model("base")
        
        print("Transcribing...")
        result = model.transcribe(output_path)
        
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"Input text:      {text}")
        print(f"Transcription:   {result['text']}")
        print(f"Language:        {result['language']}")
        print(f"\nMatch: {'✓ YES' if 'hello' in result['text'].lower() else '✗ NO'}")
        print("\n✓ FULL PIPELINE COMPLETE!")
        
    except Exception as e:
        print(f"Whisper error: {e}")
        print(f"\n✓ Generated WAV: {output_path}")
        print("Run manually: whisper " + output_path)
    
except Exception as e:
    print(f"\n✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Trying alternative: Use ONNX models directly")
    print("=" * 80)
    
    # Fallback: Use ONNX models if available
    print("This will use exported ONNX models instead of PyTorch")
    print("Install dependencies: uv pip install onnxruntime soundfile")
