"""Verify saved demo artifacts only: no model loading, network, or inference."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_manifest(manifest: dict) -> list[dict]:
    clips = manifest["clips"]
    require(manifest["count"] == len(clips) == 12, "Expected exactly 12 clips")
    require(Counter(c["language"] for c in clips) == {"en": 4, "zh": 4, "mixed": 4},
            "Expected four clips per language slice")
    require(manifest["language_counts"] == {"en": 4, "zh": 4, "mixed": 4}, "Incorrect language counts")
    require(len({c["id"] for c in clips}) == 12, "Duplicate clip ID")
    require([c["order"] for c in clips] == list(range(1, 13)), "Invalid clip order")
    for clip in clips:
        key = clip["id"]
        require(re.fullmatch(r"[a-z0-9_-]+", key) is not None, "Unsafe clip ID")
        require(clip["audio"] == f"audio/{key}.wav", f"Unexpected audio path: {key}")
        require(clip["voice"] == "zf_001" and clip["speed"] == 1, f"Voice/speed mismatch: {key}")
        require(clip["sample_rate"] == 24000, f"Sample-rate mismatch: {key}")
        require(re.fullmatch(r"[0-9a-f]{64}", clip["audio_sha256"]) is not None,
                f"Invalid audio SHA: {key}")
        require(bool(clip["text"].strip()) and bool(clip["phonemes"]), f"Empty prompt/phonemes: {key}")
        require(math.isfinite(clip["duration_seconds"]) and clip["duration_seconds"] > 0,
                f"Invalid duration: {key}")
        for stage in ("render", "asr", "naturalness", "signal"):
            directory = "renders" if stage == "render" else stage
            require(clip["evidence"][stage] == f"evidence/{directory}/{key}.json",
                    f"Unexpected evidence path: {key}/{stage}")
    duration = sum(c["duration_seconds"] for c in clips)
    require(math.isclose(duration, manifest["total_duration_seconds"], abs_tol=1e-6),
            "Total duration mismatch")
    return clips


def verify(root: Path, audio_dir: Path | None = None) -> dict:
    clips = validate_manifest(read_json(root / "manifest.json"))
    config_hashes = {stage: sha256(root / "provenance" / f"{stage}-config.json")
                     for stage in ("asr", "naturalness", "signal")}
    for clip in clips:
        key = clip["id"]
        render = read_json(root / clip["evidence"]["render"])
        require(render["status"] == "ok", f"Failed render: {key}")
        for field in ("id", "text", "phonemes", "input_ids", "unknown_phonemes", "voice",
                      "speed", "sample_rate", "duration_seconds", "audio_sha256"):
            require(render[field] == clip[field], f"Render mismatch: {key}/{field}")
        records = {}
        for stage in ("asr", "naturalness", "signal"):
            record = read_json(root / clip["evidence"][stage])
            require(record["status"] == "ok" and record["id"] == key, f"Invalid {stage}: {key}")
            require(record["audio_sha256"] == clip["audio_sha256"], f"Stale {stage} audio: {key}")
            require(record["config_sha256"] == config_hashes[stage], f"Stale {stage} config: {key}")
            records[stage] = record["result"]
        asr = records["asr"]
        for field in ("metric", "rate", "errors", "reference_tokens", "hypothesis"):
            require(clip["saved_asr"][field] == asr[field], f"ASR summary mismatch: {key}/{field}")
        require(clip["saved_asr"]["auto_language_diagnostic"] == asr.get("auto_language_diagnostic"),
                f"Alternate ASR summary mismatch: {key}")
        require(clip["saved_predicted_mos"] == records["naturalness"]["predicted_mos"],
                f"Naturalness summary mismatch: {key}")
        for field in ("peak", "clipping_fraction"):
            require(clip["saved_signal"][field] == records["signal"][field],
                    f"Signal summary mismatch: {key}/{field}")
        if audio_dir is not None:
            audio = audio_dir / f"{key}.wav"
            require(audio.is_file(), f"Missing WAV: {key}")
            require(sha256(audio) == clip["audio_sha256"], f"WAV hash mismatch: {key}")
    playlist = [line for line in (root / "playlist.m3u8").read_text(encoding="utf-8").splitlines()
                if line and not line.startswith("#")]
    require(playlist == [c["audio"] for c in clips], "Playlist does not match manifest order")
    return {"metadata_clips_verified": len(clips), "audio_clips_verified": len(clips) if audio_dir else 0,
            "audio_check": "verified" if audio_dir else "not requested; WAVs are not committed",
            "scope": "artifact integrity only; not a model-quality test"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--audio-dir", type=Path, help="Existing WAV directory; no files are generated")
    args = parser.parse_args()
    try:
        result = verify(args.demo_dir, args.audio_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Demo integrity check failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
