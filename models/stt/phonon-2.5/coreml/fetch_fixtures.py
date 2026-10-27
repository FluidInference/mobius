"""Fetch a bounded public test sample; audio stays in a user-chosen cache."""
import argparse
import csv
import io
import json
from pathlib import Path
import tarfile
import urllib.request

FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
parser.add_argument("--per-language", type=int, default=20)
args = parser.parse_args()
if not 1 <= args.per_language <= 100:
    parser.error("--per-language must be between 1 and 100")
args.output.mkdir(parents=True, exist_ok=True)
rows = []
base = "https://huggingface.co/FermionResearch/Phonon-2-CoreML/resolve/main/ci"
expected = json.load(urllib.request.urlopen(base + "/expected_texts.json", timeout=60))
for clip, ref in expected.items():
    path = args.output / (clip + ".flac")
    path.write_bytes(urllib.request.urlopen(base + "/clips/" + clip + ".flac", timeout=60).read())
    rows.append({"id": clip, "path": str(path.resolve()), "language": "en", "ref": ref})
for lang in ["en_us", "fr_fr", "de_de", "es_419"]:
    base = f"https://huggingface.co/datasets/google/fleurs/resolve/{FLEURS_REVISION}/data/{lang}"
    tsv = urllib.request.urlopen(base + "/test.tsv", timeout=60).read().decode()
    refs = {row[1]: row[2] for row in csv.reader(io.StringIO(tsv), delimiter="\t")}
    count = 0
    with urllib.request.urlopen(base + "/audio/test.tar.gz", timeout=60) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".wav"):
                    continue
                name = Path(member.name).name
                if name not in refs:
                    raise ValueError(f"No reference for {lang}/{name}")
                path = args.output / f"{lang}-{name}"
                with archive.extractfile(member) as file:
                    path.write_bytes(file.read())
                rows.append({"id": f"{lang}-{name}", "path": str(path.resolve()), "language": lang, "ref": refs[name]})
                count += 1
                if count == args.per_language:
                    break
    (args.output / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
print(f"Saved {len(rows)} public clips and rows.json")
