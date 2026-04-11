"""
Generate training data for MB-MelGAN fine-tuning from CosyVoice3.

This script:
1. Loads CosyVoice3 model
2. Generates audio samples from text
3. Extracts mel spectrograms from the audio
4. Saves (mel, audio) pairs for training
"""

import sys
import torch
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm
import soundfile as sf

# Add CosyVoice paths
sys.path.insert(0, "cosyvoice_repo")
sys.path.insert(0, "cosyvoice_repo/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import AutoModel


def compute_mel_spectrogram(audio, sample_rate=24000, n_fft=2048, hop_length=300, n_mels=80):
    """Compute mel spectrogram matching CosyVoice3's vocoder input"""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=80,
        f_max=7600,
    )

    # Compute mel spectrogram
    mel = mel_transform(audio)

    # Convert to log scale
    mel = torch.log(torch.clamp(mel, min=1e-5))

    return mel


def generate_training_data(
    output_dir="mbmelgan_training_data", num_samples=1000, use_300m=True
):
    """Generate training data from CosyVoice"""

    print("=" * 80)
    print("Generating MB-MelGAN Training Data from CosyVoice")
    print("=" * 80)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    (output_dir / "mels").mkdir(exist_ok=True)
    (output_dir / "audio").mkdir(exist_ok=True)

    print(f"\n1. Loading CosyVoice model...")

    try:
        if use_300m:
            # Use CosyVoice-300M (simpler, more reliable)
            from cosyvoice.cli.cosyvoice import CosyVoice
            from huggingface_hub import snapshot_download

            print("   Downloading CosyVoice-300M...")
            model_dir = snapshot_download(
                repo_id="FunAudioLLM/CosyVoice-300M",
                cache_dir=Path.home() / ".cache" / "cosyvoice"
            )
            print(f"   Model dir: {model_dir}")
            cosyvoice = CosyVoice(model_dir)
        else:
            # Use local Fun-CosyVoice3-0.5B model
            model_dir = "pretrained_models/Fun-CosyVoice3-0.5B-2512"
            print(f"   Model: {model_dir}")
            cosyvoice = AutoModel(model_dir=model_dir)

        print(f"   ✓ CosyVoice loaded")
        print(f"   Sample rate: {cosyvoice.sample_rate} Hz")
    except Exception as e:
        print(f"   ❌ Failed to load CosyVoice: {e}")
        print(f"\n   Error details:")
        import traceback
        traceback.print_exc()
        return False

    # Sample texts for generation (mix of English and Chinese)
    sample_texts = [
        # English
        "Hello, this is a test of the text to speech system.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models are becoming increasingly powerful.",
        "Natural language processing enables computers to understand human language.",
        "Speech synthesis has made significant progress in recent years.",
        # Chinese
        "你好，我是通义生成式语音大模型。",
        "收到好友从远方寄来的生日礼物。",
        "那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。",
        "在面对挑战时，他展现了非凡的勇气与智慧。",
        "八百标兵奔北坡，北坡炮兵并排跑。",
    ]

    # Prompt for cross-lingual generation
    prompt_audio = Path("cosyvoice_repo/asset/cross_lingual_prompt.wav")

    if not prompt_audio.exists():
        print(f"   ❌ Prompt audio not found: {prompt_audio}")
        return False

    print(f"\n2. Generating {num_samples} samples...")
    print(f"   Prompt audio: {prompt_audio}")
    print(f"   Mode: cross-lingual")

    samples_generated = 0
    samples_per_text = num_samples // len(sample_texts)

    with tqdm(total=num_samples, desc="Generating") as pbar:
        for text_idx, text in enumerate(sample_texts):
            for sample_idx in range(samples_per_text):
                try:
                    # Generate audio using CosyVoice (cross-lingual mode)
                    results = list(cosyvoice.inference_cross_lingual(text, str(prompt_audio), stream=False))

                    if not results:
                        continue

                    # Get generated audio
                    audio = results[0]["tts_speech"]  # [1, samples]

                    # Compute mel spectrogram
                    mel = compute_mel_spectrogram(
                        audio, sample_rate=cosyvoice.sample_rate, n_fft=2048, hop_length=300, n_mels=80
                    )  # [1, 80, frames]

                    # Save mel and audio
                    sample_id = f"{text_idx:03d}_{sample_idx:04d}"
                    mel_path = output_dir / "mels" / f"{sample_id}.pt"
                    audio_path = output_dir / "audio" / f"{sample_id}.wav"

                    torch.save(mel, mel_path)

                    # Convert to numpy and save with soundfile
                    audio_np = audio.squeeze().cpu().numpy()
                    sf.write(str(audio_path), audio_np, cosyvoice.sample_rate)

                    samples_generated += 1
                    pbar.update(1)

                    # Save metadata
                    if samples_generated == 1:
                        metadata = {
                            "sample_rate": cosyvoice.sample_rate,
                            "n_fft": 2048,
                            "hop_length": 300,
                            "n_mels": 80,
                            "f_min": 80,
                            "f_max": 7600,
                        }
                        torch.save(metadata, output_dir / "metadata.pt")

                except Exception as e:
                    print(f"\n   ⚠️  Failed to generate sample: {e}")
                    continue

                if samples_generated >= num_samples:
                    break

            if samples_generated >= num_samples:
                break

    print(f"\n" + "=" * 80)
    print(f"✅ Generated {samples_generated} training samples")
    print("=" * 80)

    print(f"\nOutput:")
    print(f"  - Mels: {output_dir}/mels/*.pt")
    print(f"  - Audio: {output_dir}/audio/*.wav")
    print(f"  - Metadata: {output_dir}/metadata.pt")

    # Verify one sample
    if samples_generated > 0:
        print(f"\nVerifying sample {list((output_dir / 'mels').glob('*.pt'))[0].stem}...")
        mel_path = list((output_dir / "mels").glob("*.pt"))[0]
        audio_path = output_dir / "audio" / f"{mel_path.stem}.wav"

        mel = torch.load(mel_path)
        audio, sr = torchaudio.load(audio_path)

        print(f"  - Mel shape: {mel.shape}")
        print(f"  - Audio shape: {audio.shape}")
        print(f"  - Sample rate: {sr} Hz")
        print(f"  - Duration: {audio.shape[1] / sr:.2f}s")

    print(f"\n✅ Ready for fine-tuning!")
    print(f"\nNext step: python train_mbmelgan.py")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="mbmelgan_training_data")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--use-300m", action="store_true", default=True, help="Use CosyVoice-300M (default, more reliable)")
    args = parser.parse_args()

    success = generate_training_data(args.output_dir, args.num_samples, args.use_300m)
    sys.exit(0 if success else 1)
