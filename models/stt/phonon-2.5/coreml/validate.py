"""Run the unmodified FluidAudio probe and compare it with an fp32 reference.

Uses only public fixtures listed in a caller-provided JSON manifest. Reported WER
is a small smoke sample, not a reproduction of the upstream leaderboard.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import unicodedata


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("models", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--units", choices=["cpu", "ane", "gpu"], default="ane")
    args = parser.parse_args()
    rows = json.loads(args.manifest.read_text())
    subprocess.run([
        str(args.executable), str(args.models), args.units, str(args.output),
        *(row["path"] for row in rows),
    ], check=True)
    report = json.loads(args.output.read_text())
    reference = {row["id"]: row for row in json.loads(args.reference.read_text())}
    assert len(report["rows"]) == len(rows)
    import jiwer
    grouped = {}
    for source, output in zip(rows, report["rows"]):
        assert output["file"] == source["path"] and output["text"].strip()
        ref = reference[source["id"]]
        output.update(id=source["id"], language=source["language"], ref=source["ref"], fp32_text=ref["text"])
        grouped.setdefault(source["language"], []).append(output)
    report["scores"] = {}
    for lang, values in grouped.items():
        truth = [normalize(row["ref"]) for row in values]
        fp32 = [normalize(row["fp32_text"]) for row in values]
        converted = [normalize(row["text"]) for row in values]
        report["scores"][lang] = {
            "clips": len(values), "fp32_wer": jiwer.wer(truth, fp32),
            "coreml_wer": jiwer.wer(truth, converted),
            "coreml_vs_fp32_wer": jiwer.wer(fp32, converted),
            "identical_normalized_transcripts": sum(a == b for a, b in zip(fp32, converted)),
        }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report["scores"], indent=2))


if __name__ == "__main__":
    main()
