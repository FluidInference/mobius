#!/usr/bin/env python3
"""Build (and optionally upload) the v2.1 optimized PocketTTS packs.

v2.1 is NOT a new model — same weights as v2, re-converted for speed:
fused flow decoder (ANE), one-shot cond prefill, fp16 flowlm. Each
`v2.1/<lang>/` is SELF-CONTAINED (no runtime fallback to v2): it ships the
optimized models plus the unchanged mimi/int8/constants copied from v2, so a
pack loads entirely from its own folder.

Per pack `v2.1/<lang>/`:
    flow_decoder_fused.{mlpackage,mlmodelc}   NEW  (ANE, 8-step fused)
    cond_prefill.{mlpackage,mlmodelc}         NEW  (one-shot prefill)
    flowlm_step.{mlpackage,mlmodelc}          NEW  (fp16 default)
    flowlm_stepv2.{mlpackage,mlmodelc}        copied from v2 (int8, fast option)
    mimi_decoder.{mlpackage,mlmodelc}         copied from v2 (unchanged)
    constants/ , constants_bin/               copied from v2 (tokenizer/bos/voices)
    manifest.json                             { version 2.1, base v2, ... }
  (NO cond_step / flow_decoder 1-step — v2.1 is the optimized path, no fallback.)

Usage:
    python build_v2_1.py --languages english               # build english only
    python build_v2_1.py --languages all                   # build every pack
    HF_TOKEN=hf_xxx python build_v2_1.py --languages all --upload   # build + PR

Run inside the conversion venv (uv sync first). Upload is OFF unless --upload.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent                       # .../pocket_tts/coreml
_CONVERT = _HERE / "convert_models" / "convert"
_BUILD = _HERE / "build"                                       # convert scripts write here
_V21 = _HERE / "v2.1"                                          # assembled output tree
_REPO_ID = "FluidInference/pocket-tts-coreml"

ALL_LANGS = [
    "english", "spanish", "french_24l", "german", "german_24l",
    "italian", "italian_24l", "portuguese", "portuguese_24l", "spanish_24l",
]

# Files copied UNCHANGED from v2/<lang>/ into v2.1/<lang>/ (self-contained).
COPY_FROM_V2 = [
    "flowlm_stepv2.mlpackage", "flowlm_stepv2.mlmodelc",
    "mimi_decoder.mlpackage", "mimi_decoder.mlmodelc",
    "constants", "constants_bin",
]
# Newly-converted files (produced into build/<lang>/ by the convert scripts).
NEW_FILES = ["flow_decoder_fused", "cond_prefill", "flowlm_step"]


def _run(cmd: list[str]) -> None:
    print("  $", " ".join(cmd))
    env = dict(os.environ, OS_ACTIVITY_DT_MODE="disable")
    subprocess.run(cmd, check=True, env=env)


def _compile_mlmodelc(pkg: Path) -> Path:
    """Compile a .mlpackage → sibling .mlmodelc via coremltools."""
    import coremltools as ct
    m = ct.models.MLModel(str(pkg))
    cp = m.get_compiled_model_path()
    dst = pkg.with_suffix(".mlmodelc")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(cp, dst)
    return dst


def convert_new(lang: str) -> None:
    """Run the 3 optimized conversions for one language into build/<lang>/."""
    print(f"[{lang}] converting optimized models...")
    py = sys.executable
    _run([py, str(_CONVERT / "convert_flow_decoder_fused.py"), "--language", lang, "--num-steps", "8"])
    _run([py, str(_CONVERT / "convert_cond_prefill.py"), "--language", lang, "--t-max", "256"])
    _run([py, str(_CONVERT / "convert_flowlm_step.py"), "--language", lang])  # fp16 default
    # the fused converter writes flow_decoder_fused.mlpackage; ensure .mlmodelc exists
    bdir = _BUILD / lang
    for base in NEW_FILES:
        pkg = bdir / f"{base}.mlpackage"
        if not pkg.exists():
            raise FileNotFoundError(f"expected {pkg} from conversion")
        _compile_mlmodelc(pkg)


def fetch_v2_unchanged(lang: str, dst: Path) -> None:
    """Download the unchanged v2/<lang>/ files and copy them into v2.1/<lang>/."""
    from huggingface_hub import snapshot_download
    patterns = [f"v2/{lang}/{name}/*" if not name.endswith((".bin",)) else f"v2/{lang}/{name}"
                for name in COPY_FROM_V2]
    cache = snapshot_download(_REPO_ID, allow_patterns=patterns, local_dir=str(_HERE / ".v2_cache"))
    src_lang = Path(cache) / "v2" / lang
    for name in COPY_FROM_V2:
        s = src_lang / name
        if not s.exists():
            print(f"  WARN: v2/{lang}/{name} not found upstream — skipping")
            continue
        d = dst / name
        if d.exists():
            shutil.rmtree(d) if d.is_dir() else d.unlink()
        shutil.copytree(s, d) if s.is_dir() else shutil.copy2(s, d)


def assemble(lang: str) -> Path:
    """Assemble the self-contained v2.1/<lang>/ pack."""
    out = _V21 / lang
    out.mkdir(parents=True, exist_ok=True)
    bdir = _BUILD / lang
    # copy newly-converted artifacts
    for base in NEW_FILES:
        for ext in (".mlpackage", ".mlmodelc"):
            s = bdir / f"{base}{ext}"
            d = out / f"{base}{ext}"
            if d.exists():
                shutil.rmtree(d)
            shutil.copytree(s, d)
    # copy unchanged from v2
    fetch_v2_unchanged(lang, out)
    # manifest
    manifest = {
        "version": "2.1",
        "base": "v2",
        "kind": "optimization-reconvert",
        "self_contained": True,
        "note": "Same weights as v2. Re-converted for speed: fused flow decoder "
                "(100% ANE), one-shot cond prefill, fp16 flowlm. NOT a finetune.",
        "new": NEW_FILES,
        "copied_from_v2": [c for c in COPY_FROM_V2],
        "compute_units": {
            "flow_decoder_fused": "all (ANE)",
            "cond_prefill": "all (GPU)",
            "flowlm_step": "all (GPU, fp16)",
            "flowlm_stepv2": "cpuAndGpu (GPU, int8 — fastest flowlm)",
            "mimi_decoder": "cpuOnly",
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[{lang}] assembled {out}")
    return out


def upload(create_pr: bool = True) -> None:
    from huggingface_hub import upload_folder
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        raise SystemExit("HF_TOKEN not set — refusing to upload. `export HF_TOKEN=...` (rotate the leaked one first).")
    print(f"Uploading {_V21} → {_REPO_ID}:v2.1  (create_pr={create_pr})")
    upload_folder(repo_id=_REPO_ID, folder_path=str(_V21), path_in_repo="v2.1",
                  token=tok, create_pr=create_pr,
                  commit_message="Add v2.1 optimized packs (fused decoder + cond prefill + fp16 flowlm)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--languages", default="english",
                    help="comma list, or 'all'. Default: english (canary).")
    ap.add_argument("--skip-convert", action="store_true",
                    help="assemble from existing build/<lang>/ without re-running conversions")
    ap.add_argument("--upload", action="store_true", help="upload assembled v2.1 to HF as a PR (needs $HF_TOKEN)")
    ap.add_argument("--no-pr", action="store_true", help="commit straight to main instead of a PR (discouraged)")
    args = ap.parse_args()

    langs = ALL_LANGS if args.languages == "all" else [s.strip() for s in args.languages.split(",")]
    for lang in langs:
        if not args.skip_convert:
            convert_new(lang)
        assemble(lang)

    if args.upload:
        upload(create_pr=not args.no_pr)
    else:
        print(f"\nBuilt {len(langs)} pack(s) under {_V21}. Upload skipped (pass --upload).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
