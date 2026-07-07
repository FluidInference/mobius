"""LuxTTS PyTorch reference inference — parity oracle for the CoreML conversion.

Runs the upstream torch path (not ONNX), saves the generated wav plus
intermediate tensors (prompt features, text tokens, fm_decoder output)
for per-component parity checks.

Usage:
    uv run --no-sync python scripts/reference_infer.py \
        --prompt-audio <ref.wav> --text "..." --output-dir build/oracle
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from huggingface_hub import snapshot_download
from torch.nn.utils import parametrize
from transformers import pipeline

from linacodec.vocoder.vocos import Vocos
from zipvoice.modeling_utils import generate, process_audio
from zipvoice.models.zipvoice_distill import ZipVoiceDistill
from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.feature import VocosFbank


def load_models_cpu_torch():
    """Upstream load_models_gpu, minus the torch.device(device, 0) breakage on CPU."""
    model_path = snapshot_download("YatharthS/LuxTTS")

    tokenizer = EmiliaTokenizer(token_file=f"{model_path}/tokens.txt")
    with open(f"{model_path}/config.json") as f:
        model_config = json.load(f)

    model = ZipVoiceDistill(
        **model_config["model"],
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
    )
    load_checkpoint(filename=f"{model_path}/model.pt", model=model, strict=True)
    model = model.eval()

    vocos = Vocos.from_hparams(f"{model_path}/vocoder/config.yaml")
    parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[0], "weight")
    parametrize.remove_parametrizations(vocos.upsampler.upsample_layers[1], "weight")
    vocos.load_state_dict(torch.load(f"{model_path}/vocoder/vocos.bin", map_location="cpu"))
    vocos = vocos.eval()

    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device="cpu")
    return model, VocosFbank(), vocos, tokenizer, transcriber


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-audio", required=True)
    parser.add_argument("--text", default="The quick brown fox jumps over the lazy dog, and honestly, it felt great.")
    parser.add_argument("--output-dir", default="build/oracle")
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    model, feature_extractor, vocos, tokenizer, transcriber = load_models_cpu_torch()
    vocos.freq_range = 12000
    vocos.return_48k = True

    prompt_tokens, prompt_features_lens, prompt_features, prompt_rms = process_audio(
        args.prompt_audio, transcriber, tokenizer, feature_extractor, "cpu", target_rms=0.01, duration=5
    )

    torch.manual_seed(args.seed)  # reseed so sampling noise is reproducible
    wav = generate(
        prompt_tokens,
        prompt_features_lens,
        prompt_features,
        prompt_rms,
        args.text,
        model,
        vocos,
        tokenizer,
        num_step=args.num_steps,
        guidance_scale=args.guidance_scale,
    )

    wav_np = wav.numpy().squeeze()
    sf.write(out / "reference_48k.wav", wav_np, 48000)

    # Persist oracle tensors for parity checks
    np.save(out / "prompt_features.npy", prompt_features.numpy())
    np.save(out / "prompt_tokens.npy", np.asarray(prompt_tokens[0], dtype=np.int64))
    text_tokens = tokenizer.texts_to_token_ids([args.text])
    np.save(out / "text_tokens.npy", np.asarray(text_tokens[0], dtype=np.int64))
    meta = {
        "text": args.text,
        "num_steps": args.num_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "prompt_rms": float(prompt_rms),
        "prompt_features_lens": int(prompt_features_lens[0]),
        "wav_samples": int(wav_np.shape[-1]),
        "wav_seconds_48k": float(wav_np.shape[-1] / 48000.0),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"wrote {out / 'reference_48k.wav'}")


if __name__ == "__main__":
    main()
