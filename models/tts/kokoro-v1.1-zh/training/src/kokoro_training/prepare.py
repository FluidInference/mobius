"""Prepare actual MF5 recordings and explicitly version approximate targets."""

from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import re

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch

from .artifacts import sha256, verify_assets, write_json
from .features import MelFeatures, dtw_durations
from .frontend import Frontend
from .model import CompatibleGenerator, select_style


def prompt_map(path: Path):
    result = {}
    for line in path.read_text().splitlines():
        match = re.match(r"^\s*(\d+)\s+(.+)", line)
        if match:
            result[int(match[1])] = match[2].strip()
    return result


def load_pitch(targets: Path, device):
    lock = json.loads((targets / "lock.json").read_text())
    for source in ["Utils/JDC/model.py", "Utils/JDC/bst.t7"]:
        if sha256(targets / Path(source).name) != lock["files"][source]["sha256"]:
            raise ValueError("Pitch extractor provenance mismatch")
    spec = importlib.util.spec_from_file_location("pinned_jdc", targets / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.JDCNet(num_class=1, seq_len=192)
    weights = torch.load(targets / "bst.t7", map_location="cpu", weights_only=True)
    model.load_state_dict(weights["net"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def prepare(source: Path, assets: Path, targets: Path, output: Path, device="cuda", limit=None):
    if output.exists():
        raise ValueError("Refusing to overwrite prepared evidence")
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.manual_seed(1729)
    lock = verify_assets(assets)
    config = json.loads((assets / "config.json").read_text())
    frontend = Frontend(config["vocab"])
    model = CompatibleGenerator(config)
    model.load_checkpoint(torch.load(assets / "kokoro-v1_1-zh.pth", weights_only=True, map_location="cpu"))
    model.to(device).eval().requires_grad_(False)
    voices = torch.load(assets / "voices/zf_001.pt", weights_only=True)
    pitch = load_pitch(targets, device)
    mel = MelFeatures().to(device)
    texts = {lang: prompt_map(next(source.rglob(name))) for lang, name in
             [("ENG", "english_prompts.txt"), ("MAN", "mandarin_prompts.txt")]}
    official = {lang: {int(m) for m in re.findall(r"^\s*(\d+)", next(source.rglob(name)).read_text(), re.M)}
                for lang, name in [("ENG", "EnglishTestingData"), ("MAN", "MandarinTestingData")]}
    parallel_test = {i for values in official.values() for i in values if i <= 25}
    records, exclusions = [], []
    files = sorted(source.rglob("MF5_*_0.wav"))
    if limit:
        files = files[:limit]
    for i, path in enumerate(files):
        print(f"[{i + 1}/{len(files)}] {path.stem}", flush=True)
        _, language, number, _ = path.stem.split("_")
        number = int(number)
        text = texts[language][number]
        try:
            front = frontend(text)
            audio, rate = sf.read(path, dtype="float32")
            if audio.ndim != 1 or rate != 22050 or not np.isfinite(audio).all():
                raise ValueError("Unexpected or nonfinite source audio")
            clip_fraction = float((np.abs(audio) >= 0.9999).mean())
            if clip_fraction > 0.001 or not 1.0 <= len(audio) / rate <= 14.0:
                raise ValueError("Pilot signal/length exclusion")
            # Original retained. Rational resampler is pinned in scipy; pad <25 ms only.
            real = resample_poly(audio, up=160, down=147).astype(np.float32)
            pad = (-len(real)) % 600
            real = np.pad(real, (0, pad))
            waveform = torch.tensor(real, device=device)[None]
            ids = torch.tensor([front["input_ids"]], device=device)
            style, row = select_style(voices, ids.shape[1] - 2)
            with torch.no_grad():
                generated, teacher_durations = model(ids, style.to(device))
                durations, alignment = dtw_durations(generated.cpu().numpy(), real,
                                                    teacher_durations.cpu().numpy())
                normalized_mel = mel.normalized(waveform)[..., :len(real) // 300]
                f0, _, _ = pitch(normalized_mel[:, None])
                if f0.shape[-1] != normalized_mel.shape[-1]:
                    raise ValueError("Pitch target frame-grid mismatch")
                energy = torch.log(torch.exp(normalized_mel * 4 - 4).norm(dim=1).clamp(min=1e-8))
            if not torch.isfinite(f0).all() or not torch.isfinite(energy).all():
                raise ValueError("Nonfinite acoustic target")
            # Official test utterances and both sides of parallel translations remain held out.
            passage = f"parallel-{number}" if number <= 25 else f"{language}-{number}"
            is_test = number in official[language] or number in parallel_test
            import hashlib
            split = "test" if is_test else "dev" if int(hashlib.sha256(passage.encode()).hexdigest()[:8], 16) % 10 == 0 else "train"
            item = {"id": path.stem, "language": "en" if language == "ENG" else "zh", "split": split,
                    "speaker_id": "EMIME-MF5", "session_id": f"MF5-{language}", "passage_id": passage,
                    "source_sha256": sha256(path), "source_rate": rate, "source_samples": len(audio),
                    "prepared_rate": 24000, "prepared_samples": len(real), "pad_right_samples": pad,
                    "clipping_fraction": clip_fraction, **front, "alignment": alignment,
                    "style_row": row, "targets": f"targets/{path.stem}.pt",
                    "origin": "real_recording", "rights_scope": "local_research_adaptation_only"}
            target_path = output / item["targets"]
            target_path.parent.mkdir(exist_ok=True)
            torch.save({"audio": waveform.cpu(), "ids": ids.cpu(),
                        "durations": torch.tensor(durations)[None], "f0": f0.cpu(), "energy": energy.cpu()}, target_path)
            item["targets_sha256"] = sha256(target_path)
            records.append(item)
        except (ValueError, KeyError, RuntimeError) as exc:
            exclusions.append({"id": path.stem, "reason": str(exc)})
    (output / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    counts = defaultdict(lambda: {"utterances": 0, "seconds": 0})
    for record in records:
        group = counts[f"{record['split']}/{record['language']}"]
        group["utterances"] += 1
        group["seconds"] += record["prepared_samples"] / 24000
    write_json(output / "data-contract.json", {
        "schema_version": 1, "speaker": "EMIME-MF5", "microphone": 0, "baseline": lock,
        "manifest_sha256": sha256(output / "manifest.jsonl"), "counts": dict(counts),
        "exclusions": exclusions, "pitch_extractor": json.loads((targets / "lock.json").read_text()),
        "split_policy": "official test IDs + translated passage grouping; deterministic 10% dev from remaining passages",
        "session_disjoint": False,
        "limitations": ["only one source session per language; session-disjoint bilingual split unavailable",
                        "no real code-switching recordings in EMIME", "approximate teacher-DTW alignment requires calibration",
                        "production voice rights/consent not certified; weights remain local research artifacts"],
        "preprocessing": {"resampler": "scipy.resample_poly 160/147", "sample_rate": 24000,
                          "alignment_hop": 600, "pitch_energy_hop": 300, "right_padding": "to next multiple of 600"},
    })
    return dict(counts)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(".artifacts/emime/mf5-source"))
    parser.add_argument("--assets", type=Path, default=Path(".artifacts/baseline"))
    parser.add_argument("--targets", type=Path, default=Path(".artifacts/targets"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.assets, args.targets, args.output, args.device)))


if __name__ == "__main__":
    main()
