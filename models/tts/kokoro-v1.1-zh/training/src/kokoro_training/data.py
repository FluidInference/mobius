"""Acquisition-manifest audit. Not an aligner, preprocessor, or rights certifier."""

from collections import defaultdict
import json
import math
from pathlib import Path
import unicodedata

from jsonschema import Draft202012Validator

from .artifacts import ROOT, sha256


def text_key(text: str) -> str:
    # Conservative exact-text leakage key: retain digits/letters/Hanzi.
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def audit_records(records: list, audio_root: Path | None) -> dict:
    schema = json.loads((ROOT / "schemas/recording.schema.json").read_text())
    validator = Draft202012Validator(schema)
    errors, measurements = [], []
    counts = defaultdict(lambda: {"utterances": 0, "seconds": 0.0})
    groups = {
        k: defaultdict(set)
        for k in ["session", "passage", "duplicate_group", "audio_sha256", "text"]
    }
    ids, speakers, splits, seen_audio = set(), set(), set(), set()
    reviewed_rights = []
    demo = json.loads((ROOT.parent / "coreml/bilingual-demo/manifest.json").read_text())
    development_text = {text_key(c["text"]) for c in demo["clips"]}
    if not records:
        errors.append({"record": None, "reason": "empty manifest"})
    if audio_root is None:
        errors.append({"record": None, "reason": "audio bytes not checked"})
    for index, record in enumerate(records):

        def error(reason):
            errors.append({"record": index + 1, "reason": reason})

        failures = list(validator.iter_errors(record))
        if failures:
            for failure in failures:
                # Do not copy private transcript/path/rights content into the report.
                error(
                    f"schema violation at {'.'.join(map(str, failure.path)) or '<record>'}: {failure.validator}"
                )
            continue
        reviewed_rights.append(
            record["rights"]["status"] == "approved"
            and record["rights"]["scope"] == "production_voice"
        )
        if record["id"] in ids:
            error("duplicate utterance ID")
        ids.add(record["id"])
        speakers.add((record["corpus"], record["speaker_id"]))
        if record["speaker_gender"] != "female":
            error("target female speaker not established")
        if record["origin"] != "real_recording":
            error("not a real target-speaker recording")
        if record["rights"]["status"] != "approved":
            error("rights review not approved")
        if record["qc"]["status"] != "reviewed":
            error("recording/transcript QC not reviewed")
        if record["split"] == "unassigned":
            error("split not assigned")
        duration = record["duration_seconds"]
        if not math.isfinite(duration):
            error("nonfinite duration")
            continue
        if not record["raw_text"].strip() or not record["normalized_text"].strip():
            error("blank transcript")
        split = record["split"]
        splits.add(split)
        if record["audio_sha256"] in seen_audio:
            error("duplicate audio bytes; do not count repeated recordings twice")
        seen_audio.add(record["audio_sha256"])
        key = text_key(record["normalized_text"])
        if key in development_text and split != "dev":
            error("historical demo prompt must remain development-only")
        counts[record["language"]]["utterances"] += 1
        counts[record["language"]]["seconds"] += duration
        for group, value in {
            "session": (record["corpus"], record["speaker_id"], record["session_id"]),
            "passage": (record["corpus"], record["passage_id"]),
            "duplicate_group": (record["corpus"], record["duplicate_group"]),
            "audio_sha256": record["audio_sha256"],
            "text": key,
        }.items():
            groups[group][value].add(split)
        if audio_root is not None:
            relative = Path(record["audio_path"])
            root = audio_root.resolve()
            path = (root / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(root):
                error("audio path escapes supplied root")
                continue
            try:
                if sha256(path) != record["audio_sha256"]:
                    error("audio SHA-256 mismatch")
                    continue
                import numpy as np
                import soundfile as sf

                info = sf.info(path)
                if (
                    info.samplerate != record["sample_rate"]
                    or info.channels != record["channels"]
                    or abs(info.duration - duration) > 1 / info.samplerate
                ):
                    error("audio header/duration differs from manifest")
                peak, nonfinite, clipped, samples = 0.0, 0, 0, 0
                for block in sf.blocks(path, blocksize=65536, dtype="float32", always_2d=True):
                    nonfinite += int((~np.isfinite(block)).sum())
                    peak = max(peak, float(np.nan_to_num(np.abs(block)).max()))
                    clipped += int((np.abs(block) >= 1).sum())
                    samples += block.size
                if nonfinite or samples == 0 or peak == 0:
                    error("nonfinite, empty, or all-zero recording")
                measurements.append(
                    {
                        "record": index + 1,
                        "peak": peak,
                        "clipping_fraction": clipped / samples if samples else 0,
                        "nonfinite_samples": nonfinite,
                    }
                )
            except (OSError, RuntimeError, ValueError):
                error("audio unavailable or undecodable")
    if len(speakers) != 1:
        errors.append({"record": None, "reason": "exactly one verified target speaker required"})
    for group, values in groups.items():
        leaking = sum(len(splits) > 1 for splits in values.values())
        if leaking:
            errors.append({"record": None, "reason": f"{group} crosses splits", "groups": leaking})
    for language in ("en", "zh"):
        if language not in counts:
            errors.append({"record": None, "reason": f"missing {language} recordings"})
    if not {"train", "dev", "test"}.issubset(splits):
        errors.append({"record": None, "reason": "train/dev/test splits not all represented"})
    return {
        "schema_version": 1,
        "passed": not errors,
        "records": len(records),
        "scope": "acquisition metadata and audio integrity only",
        "training_ready": False,
        "counts_by_language": dict(counts),
        "speaker_count": len(speakers),
        "errors": errors,
        "audio_measurements": measurements,
        "production_rights_recorded": bool(records)
        and len(reviewed_rights) == len(records)
        and all(reviewed_rights),
        "limitations": [
            "rights/speaker/QC evidence references require external review",
            "near-duplicate/paraphrase and semantic audio-text checks not implemented",
            "mixed-language labels are declarations, not acoustic verification",
            "frontend, prepared-audio, alignment and targets not yet audited",
        ],
    }


def audit_data(manifest: Path, audio_root: Path) -> dict:
    if not manifest.is_file():
        return {
            "passed": False,
            "training_ready": False,
            "records": 0,
            "errors": [{"reason": "recording manifest missing; no training data acquired"}],
        }
    try:
        records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        report = audit_records(records, audio_root)
    except (ValueError, TypeError) as exc:
        return {
            "passed": False,
            "training_ready": False,
            "errors": [{"reason": f"invalid JSONL manifest ({type(exc).__name__})"}],
        }
    report["manifest_sha256"] = sha256(manifest)
    return report
