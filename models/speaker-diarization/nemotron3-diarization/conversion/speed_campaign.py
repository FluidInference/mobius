#!/usr/bin/env python3
"""Generate, execute, and rank Nemotron 3 diarization speed sweeps.

This runner deliberately separates model conversion, Core ML profiling, and corpus
evaluation. Timing jobs are sequential and resumable so multiple candidates never
contend for the ANE and a stopped campaign can continue from its receipts.

Examples:
    uv run python speed_campaign.py generate --family large-chunk --output build/large-chunk.json
    uv run python speed_campaign.py convert --manifest build/large-chunk.json \
        --models-dir ~/Documents/nemotron3-models
    uv run python speed_campaign.py run --manifest build/large-chunk.json \
        --cli /path/to/fluidaudiocli --models-dir ~/Documents/nemotron3-models \
        --results-dir build/campaign-results --files ES2004a,TS3003b
    uv run python speed_campaign.py rank build/campaign-results/*.receipt.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


SUBSAMPLING = 8
NUM_SPEAKERS = 8


@dataclass(frozen=True, slots=True)
class Candidate:
    """One fixed-shape streaming model candidate."""

    name: str
    spkcache_len: int
    fifo_len: int
    chunk_len: int
    right_context: int
    left_context: int
    update_period: int

    @property
    def audio_per_call(self) -> float:
        return self.chunk_len * 0.08

    @property
    def latency(self) -> float:
        return (self.chunk_len + self.right_context) * 0.08

    @property
    def packed_frames(self) -> int:
        return self.spkcache_len + self.fifo_len + self.left_context + self.chunk_len + self.right_context

    @property
    def mel_frames(self) -> int:
        return (self.left_context + self.chunk_len + self.right_context) * SUBSAMPLING

    def manifest_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["audio_per_call"] = self.audio_per_call
        value["latency"] = self.latency
        value["packed_frames"] = self.packed_frames
        value["mel_frames"] = self.mel_frames
        return value

    def conversion_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResultSummary:
    """Aggregate corpus metrics used for gating and Pareto ranking."""

    name: str
    files: int
    der: float
    miss: float
    false_alarm: float
    confusion: float
    rtfx: float
    speaker_exact: int
    speaker_absolute_error: int
    latency: float | None = None
    packed_frames: int | None = None

    @property
    def speaker_exact_rate(self) -> float:
        return self.speaker_exact / self.files if self.files else 0.0


def candidate_name(chunk: int, rc: int, fifo: int, cache: int, update: int) -> str:
    return f"g-c{chunk}-r{rc}-f{fifo}-sc{cache}-u{update}"


def make_candidate(chunk: int, rc: int, fifo: int, cache: int, update: int) -> Candidate:
    return Candidate(
        name=candidate_name(chunk, rc, fifo, cache, update),
        spkcache_len=cache,
        fifo_len=fifo,
        chunk_len=chunk,
        right_context=rc,
        left_context=0,
        update_period=update,
    )


def update_periods(chunk: int, fifo: int) -> tuple[int, ...]:
    """Useful cache-pop schedules, deduplicated and bounded by available frames."""
    available = max(1, fifo + chunk)
    values = {
        max(chunk, min(40, available)),
        max(chunk, min(222, available)),
        max(chunk, min(max(1, fifo), available)),
        chunk,
    }
    return tuple(sorted(values))


def generate_candidates(family: str) -> list[Candidate]:
    """Generate a bounded stage or the complete selected-axis Cartesian grid."""
    candidates: list[Candidate] = []

    if family in {"large-chunk", "all"}:
        for chunk in (40, 48, 64, 80, 96, 128):
            candidates.append(make_candidate(chunk, 4, 40, 264, max(40, chunk)))

    if family in {"extended-chunk", "all"}:
        for chunk in (160, 192, 224, 256, 288, 320, 340, 384, 448):
            candidates.append(make_candidate(chunk, 4, 40, 264, chunk))

    if family in {"context-frontier", "all"}:
        for chunk in (128, 288):
            for rc in (0, 1, 2, 4, 8, 16, 32, 40):
                candidates.append(make_candidate(chunk, rc, 40, 264, chunk))

    if family in {"context", "all"}:
        for chunk in (16, 24, 32, 40, 48, 64, 80, 96, 128):
            for rc in (0, 1, 2, 4, 8):
                candidates.append(make_candidate(chunk, rc, 40, 264, max(40, chunk)))

    if family in {"state", "all"}:
        for chunk in (24, 32, 48, 64):
            for fifo in (0, 8, 16, 24, 32, 40, 64, 128, 264):
                for cache in (64, 96, 128, 160, 192, 224, 264):
                    for update in update_periods(chunk, fifo):
                        candidates.append(make_candidate(chunk, 4, fifo, cache, update))

    if family == "exhaustive":
        for chunk in (9, 12, 16, 24, 32, 40, 48, 64, 80, 96, 128):
            for rc in (0, 1, 2, 4, 8):
                for fifo in (0, 8, 16, 24, 32, 40, 64, 128, 264):
                    for cache in (64, 96, 128, 160, 192, 224, 264):
                        for update in update_periods(chunk, fifo):
                            candidates.append(make_candidate(chunk, rc, fifo, cache, update))

    unique = {candidate.conversion_dict().__repr__(): candidate for candidate in candidates}
    return sorted(unique.values(), key=lambda value: (value.latency, value.packed_frames, value.name))


def load_manifest(path: Path) -> list[Candidate]:
    payload = json.loads(path.read_text())
    items = payload["variants"] if isinstance(payload, dict) else payload
    candidates: list[Candidate] = []
    for item in items:
        fields = {
            key: item[key]
            for key in (
                "name",
                "spkcache_len",
                "fifo_len",
                "chunk_len",
                "right_context",
                "left_context",
                "update_period",
            )
        }
        candidates.append(Candidate(**fields))
    return candidates


def write_manifest(path: Path, family: str, candidates: Sequence[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "family": family,
        "variants": [candidate.conversion_dict() for candidate in candidates],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def command_output(command: Sequence[str], cwd: Path | None = None) -> str:
    result = subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True)
    if result.returncode:
        detail = (result.stdout + result.stderr).strip()
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}\n{detail}")
    return result.stdout.strip()


def git_commit(path: Path) -> str | None:
    try:
        return command_output(("git", "rev-parse", "HEAD"), cwd=path)
    except RuntimeError:
        return None


def directory_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        with item.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def summarize_rows(name: str, rows: Sequence[dict[str, Any]], candidate: Candidate | None = None) -> ResultSummary:
    if not rows:
        raise ValueError(f"No benchmark rows for {name}")

    def average(key: str) -> float:
        return statistics.fmean(float(row[key]) for row in rows)

    exact = sum(int(row["detectedSpeakers"]) == int(row["groundTruthSpeakers"]) for row in rows)
    absolute_error = sum(abs(int(row["detectedSpeakers"]) - int(row["groundTruthSpeakers"])) for row in rows)
    return ResultSummary(
        name=name,
        files=len(rows),
        der=average("der"),
        miss=average("missRate"),
        false_alarm=average("falseAlarmRate"),
        confusion=average("speakerErrorRate"),
        rtfx=average("rtfx"),
        speaker_exact=exact,
        speaker_absolute_error=absolute_error,
        latency=candidate.latency if candidate else None,
        packed_frames=candidate.packed_frames if candidate else None,
    )


def dominates(left: ResultSummary, right: ResultSummary) -> bool:
    """True when left is no worse on every available frontier metric."""
    no_worse = [
        left.der <= right.der,
        left.rtfx >= right.rtfx,
        left.speaker_absolute_error <= right.speaker_absolute_error,
    ]
    strict = [
        left.der < right.der,
        left.rtfx > right.rtfx,
        left.speaker_absolute_error < right.speaker_absolute_error,
    ]
    if left.latency is not None and right.latency is not None:
        no_worse.append(left.latency <= right.latency)
        strict.append(left.latency < right.latency)
    return all(no_worse) and any(strict)


def pareto_frontier(summaries: Sequence[ResultSummary]) -> list[ResultSummary]:
    return [
        candidate
        for candidate in summaries
        if not any(other != candidate and dominates(other, candidate) for other in summaries)
    ]


def infer_name(path: Path) -> str:
    if path.name.endswith(".receipt.json"):
        return path.name.removesuffix(".receipt.json")
    stem = path.stem
    match = re.search(r"(?:n3_(?:ami|voxconverse)_)?(.+)$", stem)
    return match.group(1) if match else stem


def read_result(path: Path, candidates: dict[str, Candidate]) -> ResultSummary:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "summary" in payload:
        value = payload["summary"]
        return ResultSummary(**value)
    if not isinstance(payload, list):
        raise ValueError(f"Unsupported result format: {path}")
    name = infer_name(path)
    return summarize_rows(name, payload, candidates.get(name))


def candidate_args(candidate: Candidate) -> list[str]:
    return [
        "--variant",
        candidate.name,
        "--chunk-len",
        str(candidate.chunk_len),
        "--rc",
        str(candidate.right_context),
        "--fifo",
        str(candidate.fifo_len),
        "--spkcache",
        str(candidate.spkcache_len),
        "--update-period",
        str(candidate.update_period),
    ]


def machine_metadata(repo: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "git_commit": git_commit(repo),
    }
    try:
        metadata["macos"] = command_output(("sw_vers", "-productVersion"))
        metadata["hardware"] = command_output(("sysctl", "-n", "machdep.cpu.brand_string"))
    except RuntimeError:
        pass
    return metadata


def run_conversion(args: argparse.Namespace) -> None:
    conversion_dir = Path(__file__).resolve().parent
    manifest = Path(args.manifest).resolve()
    candidates = load_manifest(manifest)
    if args.names:
        selected = set(args.names.split(","))
        candidates = [candidate for candidate in candidates if candidate.name in selected]
        missing = selected - {candidate.name for candidate in candidates}
        if missing:
            raise ValueError(f"Names missing from manifest: {', '.join(sorted(missing))}")

    models_dir = Path(args.models_dir).expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    silence_embedding = conversion_dir / "build" / "learnable_sil_emb.bin"
    if silence_embedding.exists():
        shutil.copy2(silence_embedding, models_dir / silence_embedding.name)
    build_dir = Path(args.build_dir).expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        candidate
        for candidate in candidates
        if not (args.resume and (models_dir / f"Nemotron3Diarizer_{candidate.name}.mlmodelc").exists())
    ]
    if not pending:
        print("Every requested model is already compiled.")
        return

    conversion_pending = [
        candidate
        for candidate in pending
        if not (args.resume and (build_dir / f"Nemotron3Diarizer_{candidate.name}.mlpackage").exists())
    ]
    if args.dry_run:
        if conversion_pending:
            print("Convert:", ", ".join(candidate.name for candidate in conversion_pending))
        print("Compile:", ", ".join(candidate.name for candidate in pending))
        return
    if conversion_pending:
        converter = (
            Path(args.converter_python).expanduser().resolve() if args.converter_python else Path(sys.executable)
        )
        command = [
            str(converter),
            "convert.py",
            "--manifest",
            str(manifest),
            "--variants",
            *[candidate.name for candidate in conversion_pending],
            "--output-dir",
            str(build_dir),
        ]
        print("Conversion:", " ".join(command))
        subprocess.run(command, cwd=conversion_dir, check=True)

    for candidate in pending:
        package = build_dir / f"Nemotron3Diarizer_{candidate.name}.mlpackage"
        print(f"Compile: {package}")
        target = models_dir / f"Nemotron3Diarizer_{candidate.name}.mlmodelc"
        compiler = subprocess.run(
            ("xcrun", "--find", "coremlcompiler"), check=False, capture_output=True, text=True
        )
        if compiler.returncode == 0:
            subprocess.run((compiler.stdout.strip(), "compile", str(package), str(models_dir)), check=True)
            continue

        # Command Line Tools installations do not include coremlcompiler. coremltools
        # delegates compilation to the system Core ML compiler.
        import coremltools as ct

        compiled = Path(ct.utils.compile_model(str(package)))
        shutil.copytree(compiled, target)


def run_benchmarks(args: argparse.Namespace) -> None:
    manifest = Path(args.manifest).resolve()
    candidates = load_manifest(manifest)
    if args.names:
        selected = set(args.names.split(","))
        candidates = [candidate for candidate in candidates if candidate.name in selected]
    cli = Path(args.cli).expanduser().resolve()
    models_dir = Path(args.models_dir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(args.repo).expanduser().resolve() if args.repo else cli.parents[3]
    metadata = machine_metadata(repo)

    for candidate in candidates:
        model = models_dir / f"Nemotron3Diarizer_{candidate.name}.mlmodelc"
        if not model.exists():
            print(f"SKIP {candidate.name}: missing {model}")
            continue
        raw_path = results_dir / f"{candidate.name}.{args.dataset}.json"
        receipt_path = results_dir / f"{candidate.name}.{args.dataset}.receipt.json"
        log_path = results_dir / f"{candidate.name}.{args.dataset}.log"
        if args.resume and receipt_path.exists():
            print(f"SKIP {candidate.name}: receipt exists")
            continue

        command = [
            str(cli),
            "nemotron3-benchmark",
            "--models",
            str(models_dir),
            *candidate_args(candidate),
            "--dataset",
            args.dataset,
            "--compute-units",
            args.compute_units,
            "--output",
            str(raw_path),
        ]
        if args.files:
            command.extend(("--files", args.files))
        if args.max_files:
            command.extend(("--max-files", str(args.max_files)))
        print("Benchmark:", " ".join(command))
        if args.dry_run:
            continue

        started = datetime.now(UTC)
        with log_path.open("w") as log:
            process = subprocess.run(
                command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, text=True, check=False
            )
        if process.returncode:
            raise RuntimeError(f"Benchmark failed for {candidate.name}; see {log_path}")
        rows = json.loads(raw_path.read_text())
        summary = summarize_rows(candidate.name, rows, candidate)
        receipt = {
            "schema_version": 1,
            "candidate": candidate.manifest_dict(),
            "dataset": args.dataset,
            "files": args.files,
            "compute_units": args.compute_units,
            "started_at": started.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "model_sha256": directory_digest(model),
            "machine": metadata,
            "summary": asdict(summary),
            "raw_results": os.path.relpath(raw_path, receipt_path.parent),
            "log": os.path.relpath(log_path, receipt_path.parent),
        }
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(
            f"{candidate.name}: DER {summary.der:.2f}% / {summary.rtfx:.1f}x / "
            f"speakers {summary.speaker_exact}/{summary.files}"
        )


def run_profile(args: argparse.Namespace) -> None:
    candidates = load_manifest(Path(args.manifest).resolve())
    if args.names:
        selected = set(args.names.split(","))
        candidates = [candidate for candidate in candidates if candidate.name in selected]
    models_dir = Path(args.models_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    tool_dir = Path(args.coreml_cli_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        model = models_dir / f"Nemotron3Diarizer_{candidate.name}.mlmodelc"
        output = output_dir / f"{candidate.name}.coreml-cli.json"
        if args.resume and output.exists() and output.stat().st_size > 0:
            print(f"SKIP {candidate.name}: profile exists")
            continue
        command = [
            "uv",
            "run",
            "coreml-cli",
            str(model),
            "--units",
            args.units,
            "--iterations",
            str(args.iterations),
            "--ops",
            "--json",
        ]
        print("Profile:", " ".join(command))
        if args.dry_run:
            continue
        temporary_output = output.with_suffix(output.suffix + ".incomplete")
        error_output = output.with_suffix(".error.log")
        with temporary_output.open("w") as destination, error_output.open("w") as errors:
            process = subprocess.run(
                command, cwd=tool_dir, stdout=destination, stderr=errors, check=False, text=True
            )
        if process.returncode == 0:
            temporary_output.replace(output)
            continue
        print(f"PROFILE FAILED {candidate.name}: see {error_output}")


def rank_results(args: argparse.Namespace) -> None:
    candidates = {}
    if args.manifest:
        candidates = {candidate.name: candidate for candidate in load_manifest(Path(args.manifest))}
    summaries = [read_result(Path(path), candidates) for path in args.results]
    frontier = {summary.name for summary in pareto_frontier(summaries)}
    print("| Variant | Files | DER | Miss/FA/Conf | RTFx | Speakers exact | Latency | Frontier |")
    print("|---|---:|---:|---:|---:|---:|---:|:---:|")
    for summary in sorted(summaries, key=lambda value: (value.der, -value.rtfx)):
        latency = "—" if summary.latency is None else f"{summary.latency:.2f}s"
        print(
            f"| {summary.name} | {summary.files} | {summary.der:.2f}% | "
            f"{summary.miss:.2f}/{summary.false_alarm:.2f}/{summary.confusion:.2f} | "
            f"{summary.rtfx:.1f}x | {summary.speaker_exact}/{summary.files} | {latency} | "
            f"{'yes' if summary.name in frontier else ''} |"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="write a variant manifest")
    generate.add_argument(
        "--family",
        choices=(
            "large-chunk",
            "extended-chunk",
            "context-frontier",
            "context",
            "state",
            "all",
            "exhaustive",
        ),
        default="large-chunk",
    )
    generate.add_argument("--max-packed", type=int, help="drop candidates above this packed sequence length")
    generate.add_argument("--output", required=True)

    convert = subparsers.add_parser("convert", help="convert and compile manifest variants")
    convert.add_argument("--manifest", required=True)
    convert.add_argument("--models-dir", required=True)
    convert.add_argument("--build-dir", default="build/campaign-models")
    convert.add_argument("--converter-python")
    convert.add_argument("--names", help="comma-separated variant names")
    convert.add_argument("--resume", action="store_true")
    convert.add_argument("--dry-run", action="store_true")

    run = subparsers.add_parser("run", help="run real-corpus FluidAudio benchmarks sequentially")
    run.add_argument("--manifest", required=True)
    run.add_argument("--cli", required=True)
    run.add_argument("--repo", help="FluidAudio checkout used for commit metadata")
    run.add_argument("--models-dir", required=True)
    run.add_argument("--results-dir", required=True)
    run.add_argument("--dataset", choices=("ami", "voxconverse"), default="ami")
    run.add_argument("--files", help="comma-separated real-corpus recording names")
    run.add_argument("--max-files", type=int)
    run.add_argument("--compute-units", choices=("all", "ane", "gpu", "cpu"), default="all")
    run.add_argument("--names", help="comma-separated variant names")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")

    profile = subparsers.add_parser("profile", help="run coreml-cli sequentially")
    profile.add_argument("--manifest", required=True)
    profile.add_argument("--models-dir", required=True)
    profile.add_argument("--coreml-cli-dir", required=True)
    profile.add_argument("--output-dir", required=True)
    profile.add_argument("--units", default="cpu_and_neural_engine")
    profile.add_argument("--iterations", type=int, default=50)
    profile.add_argument("--names", help="comma-separated variant names")
    profile.add_argument("--resume", action="store_true")
    profile.add_argument("--dry-run", action="store_true")

    rank = subparsers.add_parser("rank", help="rank raw results or campaign receipts")
    rank.add_argument("results", nargs="+")
    rank.add_argument("--manifest")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "generate":
        candidates = generate_candidates(args.family)
        if args.max_packed:
            candidates = [candidate for candidate in candidates if candidate.packed_frames <= args.max_packed]
        write_manifest(Path(args.output), args.family, candidates)
        print(f"Wrote {len(candidates)} variants to {args.output}")
    elif args.command == "convert":
        run_conversion(args)
    elif args.command == "run":
        run_benchmarks(args)
    elif args.command == "profile":
        run_profile(args)
    elif args.command == "rank":
        rank_results(args)
    else:
        parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
