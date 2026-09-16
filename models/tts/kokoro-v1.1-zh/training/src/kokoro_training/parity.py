"""Bounded, same-backend parity against an independently instantiated oracle."""

import hashlib
import json
from pathlib import Path
import time

import torch
from kokoro import KModel

from .artifacts import ROOT, environment, sha256, verify_assets, write_json
from .model import CheckpointError, CompatibleGenerator, select_style, validate_tokens

TRACE_MODULES = (
    "bert",
    "bert_encoder",
    "predictor.text_encoder",
    "predictor.lstm",
    "predictor.duration_proj",
    "predictor.F0_proj",
    "predictor.N_proj",
    "text_encoder",
    "decoder.generator.m_source",
    "decoder.generator.conv_post",
    "decoder",
)


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def compare_tensors(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    result = {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
    }
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        return {**result, "passed": False, "reason": "shape/dtype mismatch"}
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        return {**result, "passed": False, "reason": "nonfinite values"}
    diff = (reference.double() - candidate.double()).abs()
    return {
        **result,
        "passed": torch.equal(reference, candidate),
        "max_absolute_error": diff.max().item() if diff.numel() else 0,
        "mean_absolute_error": diff.mean().item() if diff.numel() else 0,
    }


def compare_traces(reference: dict, candidate: dict) -> dict:
    if reference.keys() != candidate.keys():
        return {"passed": False, "reason": "different trace keys"}
    checks = {key: compare_tensors(value, candidate[key]) for key, value in reference.items()}
    return {"passed": all(c["passed"] for c in checks.values()), "checks": checks}


def run_trace(model, ids, style, seed: int, max_frames: int):
    trace, handles = {}, []

    def capture(name, value):
        if isinstance(value, torch.Tensor):
            trace[name] = value.detach().cpu().clone()
        elif isinstance(value, (tuple, list)):
            for i, child in enumerate(value):
                capture(f"{name}.{i}", child)

    def hook(name):
        def save(module, args, output):
            if name == "predictor.duration_proj":
                frames = output.sigmoid().sum(-1).round().clamp(min=1).sum().item()
                if frames > max_frames:
                    raise ValueError(f"Predicted alignment exceeds declared {max_frames}-frame cap")
            capture(name, output)

        return save

    for name in TRACE_MODULES:
        handles.append(model.get_submodule(name).register_forward_hook(hook(name)))
    handles.append(
        model.decoder.register_forward_pre_hook(
            lambda module, args: capture("decoder.inputs", args)
        )
    )
    devices = [ids.device.index or 0] if ids.is_cuda else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(seed)
            if ids.is_cuda:
                torch.cuda.manual_seed_all(seed)
            if isinstance(model, KModel):
                audio, durations = model.forward_with_tokens(ids, style, speed=1)
            else:
                audio, durations = model(ids, style, speed=1)
            capture("audio", audio)
            capture("integer_durations", durations)
            capture("rng.cpu.after", torch.get_rng_state())
            if ids.is_cuda:
                capture("rng.cuda.after", torch.cuda.get_rng_state(ids.device))
    finally:
        for handle in handles:
            handle.remove()
    return trace


def cases_from_manifest(manifest: Path, config: dict) -> list[dict]:
    data = json.loads(manifest.read_text())
    if len(data["clips"]) != 12:
        raise ValueError("Expected the fixed 12-clip development manifest")
    cases = []
    for clip in data["clips"]:
        ids = clip["input_ids"]
        validate_tokens(ids, config["n_token"], config["plbert"]["max_position_embeddings"])
        if (
            clip["unknown_phonemes"]
            or [0, *[config["vocab"][p] for p in clip["phonemes"]], 0] != ids
        ):
            raise ValueError(f"Saved phoneme/token mismatch in {clip['id']}")
        cases.append(
            {
                "id": clip["id"],
                "language": clip["language"],
                "input_ids": ids,
                "origin": "historical development tokens, including known frontend defects",
            }
        )
    # Token-interface probes, not training examples or naturalness evidence.
    cases.extend(
        [
            {
                "id": "boundary-one-phone",
                "language": "probe",
                "input_ids": [0, config["vocab"]["a"], 0],
            },
            {
                "id": "boundary-punctuation",
                "language": "probe",
                "input_ids": [0, config["vocab"]["."], 0],
            },
            {
                "id": "boundary-510-phones",
                "language": "probe",
                "input_ids": [0] + [config["vocab"]["a"]] * 510 + [0],
            },
        ]
    )
    return cases


def parity(assets: Path, output: Path, device: str, seed: int, max_frames: int) -> dict:
    if output.exists():
        raise ValueError("Use a new run directory; existing evidence must remain immutable")
    output.mkdir(parents=True)
    started = time.monotonic()
    lock = verify_assets(assets)
    write_json(output / "environment.json", environment())
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = json.loads((assets / "config.json").read_text())
    candidate = CompatibleGenerator(config)
    checkpoint = torch.load(assets / "kokoro-v1_1-zh.pth", map_location="cpu", weights_only=True)
    try:
        mapping = candidate.load_checkpoint(checkpoint)
    except CheckpointError as exc:
        write_json(output / "checkpoint-map.json", exc.report)
        raise
    write_json(output / "checkpoint-map.json", mapping)
    oracle = KModel(
        repo_id=lock["repo_id"], config=config, model=str(assets / "kokoro-v1_1-zh.pth")
    )
    # Upstream's permissive fallback is not evidence: independently check every loaded tensor.
    state_check = compare_traces(oracle.state_dict(), candidate.state_dict())
    write_json(output / "loaded-state-parity.json", state_check)
    if not state_check["passed"]:
        raise ValueError("Independently loaded upstream and candidate states differ")
    del checkpoint
    candidate.to(device).eval().requires_grad_(False)
    oracle.to(device).eval().requires_grad_(False)
    voices = torch.load(assets / "voices/zf_001.pt", map_location="cpu", weights_only=True)
    manifest = ROOT.parent / "coreml/bilingual-demo/manifest.json"
    cases = cases_from_manifest(manifest, config)
    write_json(
        output / "inputs.json",
        {"assets": lock, "manifest_sha256": sha256(manifest), "cases": cases},
    )
    write_json(
        output / "resolved-config.json",
        {
            "device": device,
            "dtype": "float32",
            "seed": seed,
            "max_alignment_frames": max_frames,
            "speed": 1,
            "style": "zf_001: row = content token count - 1, BOS/EOS excluded",
            "tolerance": {"atol": 0, "rtol": 0},
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
            "tf32": False,
            "optimizer_steps": 0,
            "stochastic_control": "restore full RNG via fork_rng; compare source outputs and final RNG states",
        },
    )
    results = []
    for i, case in enumerate(cases):
        print(f"[{i + 1}/{len(cases)}] {case['id']}", flush=True)
        try:
            style, row = select_style(voices, len(case["input_ids"]) - 2)
            ids = torch.tensor([case["input_ids"]], dtype=torch.long, device=device)
            style = style.to(device)
            reference = run_trace(oracle, ids, style, seed, max_frames)
            repeated = run_trace(oracle, ids, style, seed, max_frames)
            repeat_check = compare_traces(reference, repeated)
            del repeated
            actual = run_trace(candidate, ids, style, seed, max_frames)
            check = compare_traces(reference, actual)
            result = {
                "id": case["id"],
                "passed": repeat_check["passed"] and check["passed"],
                "style_row": row,
                "style_sha256": tensor_hash(style),
                "oracle_repeat": repeat_check,
                "candidate": check,
                "waveform_sha256": tensor_hash(actual["audio"]),
                "audio_samples": actual["audio"].numel(),
                "alignment_frames": int(actual["integer_durations"].sum()),
                "sample_rate": 24000,
                "origin": "real upstream model inference; not training audio",
            }
            del reference, actual
        except (ValueError, RuntimeError) as exc:
            result = {"id": case["id"], "passed": False, "error": str(exc)}
        write_json(output / "cases" / f"{case['id']}.json", result)
        results.append({k: v for k, v in result.items() if k not in {"oracle_repeat", "candidate"}})
    summary = {
        "schema_version": 1,
        "passed": all(r["passed"] for r in results),
        "scope": "untouched FP32 same-backend inference parity; no training or quality claim",
        "cases": results,
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
        if device.startswith("cuda")
        else None,
        "remaining_gates": [
            "frontend parity",
            "real-data/target audit",
            "gradient coverage",
            "micro-overfit/resume",
            "pilot",
            "quality evaluation",
            "export/device acceptance",
        ],
    }
    write_json(output / "summary.json", summary)
    return summary
