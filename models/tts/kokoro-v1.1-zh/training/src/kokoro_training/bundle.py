"""Export/load an auditable PyTorch checkpoint with its matching voice and frontend."""

import argparse
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import soundfile as sf
import torch

from .artifacts import ROOT, sha256, verify_assets, write_json
from .frontend import Frontend, VERSION
from .model import CompatibleGenerator, select_style


def export(checkpoint: Path, assets: Path, output: Path):
    if output.exists():
        raise ValueError("Refusing to overwrite a model bundle")
    verify_assets(assets)
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    config = json.loads((assets / "config.json").read_text())
    model = CompatibleGenerator(config)
    state = {k.removeprefix("generator."): v for k, v in saved["model"].items() if k.startswith("generator.")}
    model.load_state_dict(state, strict=True)
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Nonfinite checkpoint")
    # Nested module dictionaries also load through the upstream KModel API.
    nested = {name: child.state_dict() for name, child in model.named_children()}
    voices = torch.load(assets / "voices/zf_001.pt", weights_only=True, map_location="cpu")
    voices = voices + saved["model"]["style_offset"][None]
    select_style(voices, 1)
    output.mkdir(parents=True)
    torch.save(nested, output / "model.pth")
    torch.save(voices, output / "voice.pt")
    shutil.copy2(assets / "config.json", output / "config.json")
    shutil.copy2(ROOT / "uv.lock", output / "uv.lock")
    shutil.copy2(ROOT / "LICENSE.kokoro", output / "LICENSE.kokoro")
    write_json(output / "vocab.json", config["vocab"])
    # Carry the exact source required by the source-checkout command documented below.
    source = output / "src/kokoro_training"
    source.mkdir(parents=True)
    for file in (ROOT / "src/kokoro_training").glob("*.py"):
        shutil.copy2(file, source / file.name)
    shutil.copy2(ROOT / "pyproject.toml", output / "pyproject.toml")
    shutil.copy2(ROOT / "baseline.lock.json", output / "baseline.lock.json")
    for name in ("training-assets.lock.json", "evaluation-prompts.json", "evaluation-policy.json", "NOTICE.md"):
        shutil.copy2(ROOT / name, output / name)
    (output / "README.md").write_text(f"""# Kokoro English/Mandarin PyTorch adaptation

Actual trained weights from optimizer update {saved['step']}, adapted from
Kokoro-82M-v1.1-zh using real EMIME MF5 recordings. Output: mono 24 kHz.
Use `model.pth` together with this bundle's `voice.pt`, configuration, vocabulary,
and frontend. The generator retains the 81.8M-parameter Kokoro architecture.

## Inference

Requires Linux, Python 3.12, `uv`, and the system `espeak-ng` library.
From this directory:

```bash
uv sync --frozen
uv run --frozen python -m kokoro_training.bundle infer \\
  --bundle . --text '请打开 API，然后把结果发给我。' \\
  --output example.wav --device cuda
```

Use `--device cpu` for CPU execution. Inference loads local weights only.
Text above 510 content tokens must be split by the caller. Speed must be
within 0.25..4 and synthesis is capped at 4,000 duration frames (100 seconds).
The output WAV is accompanied by tokens, durations, seed, and hashes in JSON.

`model.pth` contains nested module state dictionaries accepted by upstream
`kokoro.KModel(config='config.json', model='model.pth',
repo_id='hexgrad/Kokoro-82M-v1.1-zh')`. The bundled inference path additionally
checks hashes, token bounds, and strict weight mapping. `voice.pt` has shape
510 × 1 × 256; choose row `number_of_content_tokens - 1`, excluding BOS/EOS.

## Scope and provenance

This is a local experimental trained model, not a production-qualified voice.
The corpus provides separate English and Mandarin recordings, no real same-voice
code-switch recordings, and only one session per language. Production voice
consent and bilingual listening/tone/naturalness acceptance are not certified.
Do not treat ASR diagnostics as those certifications. No deployment is included.

`bundle.json` records hashes for all packaged components and the training
checkpoint/configuration. `NOTICE.md` records upstream and corpus attribution.
The base model is Apache-2.0; the corpus README specifies ODbL/DbCL. These
attributions do not independently grant production voice/personality rights.

Full run reports and source: https://github.com/FluidInference/mobius/pull/95
""")
    manifest = {"format": "kokoro-pytorch-bundle-v1", "sample_rate": 24000,
                "model": "Kokoro-82M-v1.1-zh MF5 supervised adaptation", "step": saved["step"],
                "training_checkpoint_sha256": sha256(checkpoint), "training_config": saved["config"],
                "frontend_version": VERSION, "speaker": "EMIME-MF5", "base_style": "zf_001",
                "production_qualified": False,
                "limitations": ["Small real-recording adaptation, not production-qualified",
                                "No real same-speaker code-switch recordings used",
                                "Session-disjoint validation unavailable in this corpus",
                                "Production voice consent/rights not certified; local research artifact"],
                "files": {str(p.relative_to(output)): {"sha256": sha256(p), "bytes": p.stat().st_size}
                          for p in sorted(output.rglob("*")) if p.is_file()}}
    write_json(output / "bundle.json", manifest)
    return manifest


def load_bundle(directory: Path, device="cpu"):
    manifest = json.loads((directory / "bundle.json").read_text())
    if manifest["format"] != "kokoro-pytorch-bundle-v1" or manifest["frontend_version"] != VERSION:
        raise ValueError("Unsupported model/frontend version")
    required = {"model.pth", "voice.pt", "config.json", "vocab.json", "src/kokoro_training/frontend.py"}
    if not required <= manifest["files"].keys():
        raise ValueError("Incomplete bundle manifest")
    for name, expected in manifest["files"].items():
        path = directory / name
        if not path.resolve().is_relative_to(directory.resolve()) or sha256(path) != expected["sha256"]:
            raise ValueError(f"Bundle checksum/path mismatch: {name}")
    if sha256(ROOT / "src/kokoro_training/frontend.py") != manifest["files"]["src/kokoro_training/frontend.py"]["sha256"]:
        raise ValueError("Installed frontend differs from the bundled frontend")
    config = json.loads((directory / "config.json").read_text())
    if json.loads((directory / "vocab.json").read_text()) != config["vocab"]:
        raise ValueError("Bundled vocabulary differs from model configuration")
    generator = CompatibleGenerator(config)
    generator.load_checkpoint(torch.load(directory / "model.pth", weights_only=True, map_location="cpu"))
    voices = torch.load(directory / "voice.pt", weights_only=True, map_location="cpu")
    select_style(voices, 1)
    return generator.to(device).eval(), voices.to(device), manifest


def synthesize(model, voices, front, text, device, seed=1729, speed=1.0):
    inputs = front(text)
    ids = torch.tensor([inputs["input_ids"]], device=device)
    style, row = select_style(voices, ids.shape[1] - 2)
    with torch.random.fork_rng(devices=[torch.device(device).index or 0] if str(device).startswith("cuda") else []):
        torch.manual_seed(seed)
        audio, durations = model(ids, style, speed=speed)
    if not torch.isfinite(audio).all() or audio.numel() != int(durations.sum()) * 600:
        raise ValueError("Invalid synthesized waveform")
    inputs.update(seed=seed, speed=speed, style_row=row, sample_rate=24000,
                  samples=audio.numel(), durations=durations.cpu().tolist(),
                  peak=float(audio.abs().max()), rms=float(audio.square().mean().sqrt()),
                  clipping_fraction=float((audio.abs() >= 1).float().mean()), origin="model_generated_evaluation_audio")
    return audio.cpu().numpy(), inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("export")
    pack.add_argument("--checkpoint", type=Path, required=True)
    pack.add_argument("--assets", type=Path, default=Path(".artifacts/baseline"))
    pack.add_argument("--output", type=Path, required=True)
    infer = commands.add_parser("infer")
    infer.add_argument("--bundle", type=Path, required=True)
    infer.add_argument("--text", required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--device", default="cpu")
    infer.add_argument("--seed", type=int, default=1729)
    infer.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == "export":
        result = export(args.checkpoint, args.assets, args.output)
        print(json.dumps({"bundle": str(args.output), "step": result["step"], "model": result["files"]["model.pth"]}))
    else:
        if args.output.exists() or args.output.with_suffix(".json").exists():
            raise ValueError("Refusing to overwrite synthesis artifacts")
        model, voices, manifest = load_bundle(args.bundle, args.device)
        audio, metadata = synthesize(model, voices, Frontend(model.config["vocab"]), args.text, args.device, args.seed, args.speed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(args.output, audio, 24000, subtype="FLOAT")
        metadata.update(model_sha256=manifest["files"]["model.pth"]["sha256"], wav_sha256=sha256(args.output))
        write_json(args.output.with_suffix(".json"), metadata)
        print(json.dumps({"output": str(args.output), "seconds": len(audio) / 24000, "peak": metadata["peak"]}))


if __name__ == "__main__":
    main()
