"""Explicit, checksum-pinned acquisition of real MF5 data, JDC, and evaluation ASR."""

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import urllib.request

from .artifacts import ROOT, sha256, write_json


def acquire(part, output):
    lock = json.loads((ROOT / "training-assets.lock.json").read_text())
    for entry in lock["files"]:
        if entry["part"] != part:
            continue
        path = output / entry["path"]
        if path.exists():
            if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
                raise ValueError(f"Existing asset differs: {entry['path']}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        try:
            with urllib.request.urlopen(entry["url"], timeout=60) as stream, temporary.open("wb") as destination:
                shutil.copyfileobj(stream, destination)
            if temporary.stat().st_size != entry["bytes"] or sha256(temporary) != entry["sha256"]:
                raise ValueError(f"Acquisition checksum mismatch: {entry['path']}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if part == "targets":
        write_json(output / "targets/lock.json", lock["target_model"])


def extract_mf5(archive, output):
    lock = json.loads((ROOT / "training-assets.lock.json").read_text())
    expected = next(e for e in lock["files"] if e["part"] == "data")
    if sha256(archive) != expected["sha256"]:
        raise ValueError("Unexpected corpus archive")
    if output.exists():
        raise ValueError("Refusing to overwrite extracted recordings")
    output.mkdir(parents=True)
    documents = {"README_mandarin_1.0.txt", "README_mandarin_1.1.txt", "odbl-10.txt",
                 "english_prompts.txt", "mandarin_prompts.txt", "EnglishTestingData",
                 "MandarinTestingData", "EMIME_MANDARIN_DATABASE_ACCENTS.pdf"}
    count = 0
    with tarfile.open(archive, "r|bz2") as tar:
        for member in tar:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or member.issym() or member.islnk():
                raise ValueError("Unsafe corpus member")
            if not member.isfile():
                continue
            is_recording = re.fullmatch(r"MF5_(ENG|MAN)_\d+_0.wav", name.name)
            if not is_recording and name.name not in documents:
                continue
            if "Mandarin_test_set_segmentations" in name.parts or member.size > 100_000_000:
                raise ValueError("Unexpected selected corpus member")
            path = output / str(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as destination, tar.extractfile(member) as source:
                shutil.copyfileobj(source, destination)
            count += bool(is_recording)
    if count != 314:
        raise ValueError(f"Expected 314 single-microphone MF5 recordings, got {count}")
    print(f"Extracted {count} real MF5 recordings and supporting documents")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", choices=("data", "targets", "asr", "extract-mf5"))
    parser.add_argument("--output", type=Path, default=Path(".artifacts"))
    args = parser.parse_args()
    if args.part == "extract-mf5":
        extract_mf5(args.output / "emime/UEDIN_mandarin_bilingual_data_v1.1.tar.bz2", args.output / "emime/mf5-source")
    else:
        acquire(args.part, args.output)


if __name__ == "__main__":
    main()
