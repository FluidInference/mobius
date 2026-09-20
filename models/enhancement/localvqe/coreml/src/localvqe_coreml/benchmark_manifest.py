"""Fail-closed input coverage checks for the LocalVQE blind benchmark."""

from pathlib import Path


def require_coverage(actual: list[str], expected: list[str]) -> None:
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("Expected manifest is empty or contains duplicate stems")
    if len(actual) != len(set(actual)):
        raise ValueError("Input dataset contains duplicate stems")
    missing, extra = set(expected) - set(actual), set(actual) - set(expected)
    if missing or extra:
        raise ValueError(f"Manifest mismatch: {len(missing)} missing, {len(extra)} unexpected; "
                         f"missing={sorted(missing)[:5]}, unexpected={sorted(extra)[:5]}")


def collect_jobs(blind_dir: Path, enh_dir: str, manifest: Path, scenarios: dict, limit: int | None = None):
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be a positive number of clips per scenario")
    expected = [line.strip() for line in manifest.read_text().splitlines() if line.strip()]
    microphones = sorted(blind_dir.glob("*/*_mic.flac"))
    stems = [path.name.removesuffix("_mic.flac") for path in microphones]
    require_coverage(stems, expected)
    jobs = []
    selected_counts = {}
    for mic, stem in zip(microphones, stems):
        scenario = stem.split("_", 1)[-1]
        if scenario not in scenarios:
            raise ValueError(f"Unknown scenario in {stem}")
        lpb = mic.with_name(stem + "_lpb.flac")
        if not mic.is_file() or not lpb.is_file():
            raise ValueError(f"Missing microphone or loopback for {stem}")
        # Validate source coverage before selecting a deterministic diagnostic subset.
        if limit is not None and selected_counts.get(scenario, 0) >= limit:
            continue
        enhanced = None if enh_dir == "unprocessed" else Path(enh_dir) / mic.parent.name / f"{stem}_enh.wav"
        if enhanced is not None and not enhanced.is_file():
            raise ValueError(f"Missing render: {enhanced}")
        jobs.append((scenario, stem, mic, lpb, enhanced))
        selected_counts[scenario] = selected_counts.get(scenario, 0) + 1
    return jobs, expected
