"""Command-line entry point shared by the Kai and Lex Core ML toolkits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from typed_coreml import (
    KINDS,
    compare_native,
    export_coreml,
    load_native,
    verify_coreml,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolkit", required=True, type=Path)
    parser.add_argument("--native-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--kind", choices=KINDS)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--candidates", type=int)
    parser.add_argument("--fixture-index", type=int, default=0)
    parser.add_argument("--flexible", action="store_true")
    parser.add_argument("--marker-map", action="store_true", help="use an FP16 one-hot marker map instead of indices")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--verify-limit", type=int, help="maximum pinned real rows to verify per type")
    parser.add_argument("--verify-package", type=Path)
    parser.add_argument("--verify-packages-dir", type=Path)
    args = parser.parse_args()
    if args.marker_map and args.flexible:
        parser.error("--marker-map currently requires a fixed token and candidate shape")
    if args.verify_limit is not None and args.verify_limit < 1:
        parser.error("--verify-limit must be positive")
    if args.verify_package is not None and args.kind is None:
        parser.error("--verify-package requires --kind")
    if args.verify_package is not None and args.verify_packages_dir is not None:
        parser.error("Choose one package verification mode")
    toolkit = args.toolkit.resolve()
    lock = json.loads((toolkit / "assets.lock.json").read_text())
    native_dir = args.native_dir
    if native_dir is None:
        snapshot = snapshot_download(
            repo_id=lock["source_repo"],
            revision=lock["source_revision"],
            allow_patterns=["native/**"],
        )
        native_dir = Path(snapshot) / "native"
    model, collator, _ = load_native(native_dir)
    records = []
    for fixture in ("upstream-decisions.jsonl", "upstream-system-one-rows.jsonl"):
        records.extend(json.loads(line) for line in (toolkit / fixture).read_text().splitlines() if line)
    kinds = [args.kind] if args.kind else list(KINDS)
    failed_packages = []
    for kind in kinds:
        selected = [record for record in records if record["question"]["type"].lower() == kind]
        if not selected:
            raise ValueError(f"Missing upstream {kind} fixture")
        if args.verify_only:
            print(json.dumps(compare_native(
                model, collator, selected, kind, tokens=args.tokens, candidates=args.candidates,
                marker_map=args.marker_map,
            )))
            continue
        package = args.verify_package
        if args.verify_packages_dir is not None:
            package = args.verify_packages_dir / (kind + ".mlpackage")
        if package is not None:
            report = verify_coreml(
                model, collator, selected[:args.verify_limit], kind, package
            )
            output = package.parent / (kind + "-validation.json")
            output.write_text(json.dumps(report, indent=2) + "\n")
            metadata_path = package.with_suffix(".json")
            metadata = json.loads(metadata_path.read_text())
            passed = report["choice_agreement"] and report["max_abs_probability_error"] <= 0.02
            metadata["validated"] = passed
            metadata["validation_report"] = output.name
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            print(json.dumps(report))
            if not passed:
                failed_packages.append(kind)
            continue
        if args.output_dir is None:
            raise ValueError("--output-dir is required for conversion")
        if args.fixture_index < 0 or args.fixture_index >= len(selected):
            raise ValueError("--fixture-index is outside the upstream fixture set")
        suffix = "-flex" if args.flexible else "-marker-map" if args.marker_map else ""
        report = export_coreml(
            model,
            collator,
            [selected[args.fixture_index]],
            kind,
            args.output_dir / (kind + suffix + ".mlpackage"),
            source_repo=lock["source_repo"],
            source_revision=lock["source_revision"],
            tokens=args.tokens,
            candidates=args.candidates,
            flexible=args.flexible,
            marker_map=args.marker_map,
        )
        print(json.dumps(report))
    if failed_packages:
        raise SystemExit(f"Core ML native-parity gate failed for: {', '.join(failed_packages)}")


if __name__ == "__main__":
    main()
