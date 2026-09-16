"""Each command performs one explicit stage; none starts an optimizer."""

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    fetch = sub.add_parser(
        "fetch-baseline", help="Explicitly download and verify pinned public assets"
    )
    fetch.add_argument("--assets", type=Path, required=True)
    env = sub.add_parser("environment")
    env.add_argument("--output", type=Path, required=True)
    par = sub.add_parser("parity")
    par.add_argument("--assets", type=Path, required=True)
    par.add_argument("--output", type=Path, required=True)
    par.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    par.add_argument("--seed", type=int, default=1729)
    par.add_argument("--max-frames", type=int, default=4000)
    audit = sub.add_parser("audit-data")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--audio-root", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    emime = sub.add_parser(
        "inventory-emime", help="Inspect a supplied archive without extracting it"
    )
    emime.add_argument("--archive", type=Path, required=True)
    emime.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Must be set before CUDA initialization, including environment inspection.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    from .artifacts import environment, fetch_assets, write_json

    try:
        if args.stage == "fetch-baseline":
            report = fetch_assets(args.assets)
        elif args.stage == "environment":
            report = environment()
            write_json(args.output, report)
        elif args.stage == "parity":
            if args.max_frames < 1:
                raise ValueError("max-frames must be positive")
            from .parity import parity

            report = parity(args.assets, args.output, args.device, args.seed, args.max_frames)
        elif args.stage == "inventory-emime":
            from .emime import inventory_emime

            report = inventory_emime(args.archive)
            write_json(args.output, report)
        else:
            from .data import audit_data

            report = audit_data(args.manifest, args.audio_root)
            write_json(args.output, report)
        print(
            json.dumps(
                report
                if args.stage != "parity"
                else {k: v for k, v in report.items() if k != "cases"},
                indent=2,
            )
        )
        return 0 if report.get("passed", True) else 1
    except (ValueError, OSError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
