"""Verify exported trained tensors, upstream interoperability, and real inference."""

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from kokoro import KModel

from kokoro_training.artifacts import ROOT, sha256, write_json
from kokoro_training.bundle import load_bundle
from kokoro_training.frontend import Frontend
from kokoro_training.model import CompatibleGenerator, select_style


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=Path(".artifacts/baseline"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new evidence path")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, voices, manifest = load_bundle(args.bundle, "cuda")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor.cpu(), saved["model"]["generator." + key]), key
    original_voices = torch.load(args.assets / "voices/zf_001.pt", weights_only=True)
    assert torch.equal(voices.cpu(), original_voices + saved["model"]["style_offset"][None])
    oracle = KModel(repo_id="hexgrad/Kokoro-82M-v1.1-zh", config=str(args.bundle / "config.json"),
                    model=str(args.bundle / "model.pth")).cuda().eval()
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, oracle.state_dict()[key]), key
    base = CompatibleGenerator(model.config)
    base.load_checkpoint(torch.load(args.assets / "kokoro-v1_1-zh.pth", weights_only=True))
    changes = {}
    for name, tensor in model.state_dict().items():
        delta = tensor.cpu() - base.state_dict()[name]
        group = changes.setdefault(name.split(".")[0], {"changed_tensors": 0, "total_tensors": 0, "max_absolute_change": 0.0})
        group["total_tensors"] += 1
        group["changed_tensors"] += not torch.equal(tensor.cpu(), base.state_dict()[name])
        group["max_absolute_change"] = max(group["max_absolute_change"], float(delta.abs().max()))
    assert changes["bert"]["changed_tensors"] == 0
    assert changes["decoder"]["changed_tensors"] == 0
    assert all(changes[k]["changed_tensors"] > 0 for k in ("bert_encoder", "predictor", "text_encoder"))
    frontend = Frontend(model.config["vocab"])
    cases = []
    for case in json.loads((ROOT / "evaluation-prompts.json").read_text()):
        ids = torch.tensor([frontend(case["text"])["input_ids"]], device="cuda")
        style, _ = select_style(voices, ids.shape[1] - 2)
        torch.manual_seed(1729)
        expected, durations = oracle.forward_with_tokens(ids, style)
        torch.manual_seed(1729)
        torch.cuda.synchronize()
        start = time.perf_counter()
        actual, actual_durations = model(ids, style)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        assert torch.equal(expected.reshape(-1), actual.reshape(-1)), case["id"]
        assert torch.equal(durations, actual_durations)
        assert torch.isfinite(actual).all() and float(actual.square().mean().sqrt()) > 0.001
        assert float((actual.abs() >= 1).float().mean()) < 0.001
        cases.append({"id": case["id"], "sample_count": actual.numel(), "upstream_waveform_max_abs": 0.0,
                      "seconds": actual.numel() / 24000, "inference_seconds": elapsed,
                      "rtf": elapsed / (actual.numel() / 24000), "peak": float(actual.abs().max())})
    write_json(args.output, {"passed": True, "checkpoint_tensors_match": True, "voice_table_matches": True,
                            "all_upstream_state_tensors_match": True, "model_sha256": sha256(args.bundle / "model.pth"),
                            "step": manifest["step"], "changes_from_baseline": changes, "cases": cases,
                            "timing_scope": "warm GPU inference only, excludes frontend and loading; shared GPU run"})
    print(json.dumps({"passed": True, "cases": len(cases), "changes": changes}))


if __name__ == "__main__":
    main()
