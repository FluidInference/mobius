"""Assemble the HuggingFace upload directory for the Nemotron 3 diarization CoreML models.

DOES NOT UPLOAD. Stages everything locally so publishing after NVIDIA's public release
is a single command. A DO_NOT_UPLOAD_YET marker is written into the directory and must
be removed manually as the final pre-flight check.

Usage:
    uv run python stage_hf.py                       # stage to build/hf-staging
    # after NVIDIA public release + license check + repo confirmation:
    #   rm build/hf-staging/DO_NOT_UPLOAD_YET
    #   hf upload FluidInference/nemotron-3-diarization-coreml build/hf-staging .
"""

import shutil
from pathlib import Path

BUILD = Path("build")
COMPILED = BUILD / "compiled"
STAGING = BUILD / "hf-staging"

# Curated preset set: the four model-card profiles, the speed-campaign presets, the
# int8 size option, and the split-graph family (with its projection weights).
MODELS = [
    # (file name in compiled/, subdir in repo) — curated set; dominated variants
    # stay local in build/ (regenerable via convert.py / convert_split.py).
    ("Nemotron3Diarizer_low.mlmodelc", "monolithic"),
    ("Nemotron3Diarizer_offline.mlmodelc", "monolithic"),
    ("Nemotron3Diarizer_fast.mlmodelc", "monolithic"),
    ("Nemotron3Diarizer_fast32.mlmodelc", "monolithic"),
    ("Nemotron3Diarizer_fast128.mlmodelc", "monolithic"),
    ("Nemotron3Diarizer_s32_split_w8a8.mlmodelc", "split"),
    ("Nemotron3Diarizer_c128_split_w8a8.mlmodelc", "split"),
]
SUPPORT = ["learnable_sil_emb.bin", "pre_encode_proj_t.bin"]


def main():
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)

    for name, subdir in MODELS:
        src = COMPILED / name
        if not src.exists():
            print(f"  [MISSING] {name}")
            continue
        dst = STAGING / subdir / name
        dst.parent.mkdir(exist_ok=True)
        shutil.copytree(src, dst)
        print(f"  [ok] {subdir}/{name}")

    for name in SUPPORT:
        shutil.copy(BUILD / name, STAGING / name)
        print(f"  [ok] {name}")

    readme = Path("hf_readme.md")
    shutil.copy(readme, STAGING / "README.md")
    print("  [ok] README.md")

    (STAGING / "DO_NOT_UPLOAD_YET").write_text(
        "GUARD FILE — the source checkpoint is under the NVIDIA eval license "
        "(no redistribution of the model or derivatives) until the public release.\n"
        "Before uploading:\n"
        "  1. Confirm nvidia's public release happened and check the release license.\n"
        "  2. Update the `license`/base-model fields in README.md accordingly.\n"
        "  3. Confirm the target repo name with Alex.\n"
        "  4. Delete this file.\n"
    )
    total = sum(f.stat().st_size for f in STAGING.rglob("*") if f.is_file()) / 1e6
    print(f"\nStaged {total:.0f} MB at {STAGING} — DO_NOT_UPLOAD_YET guard in place.")


if __name__ == "__main__":
    main()
