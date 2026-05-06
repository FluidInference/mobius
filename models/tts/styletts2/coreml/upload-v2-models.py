"""Upload the rebuilt StyleTTS2-ANE postbert + vocoder bundles to HuggingFace.

The 7-graph ANE-pinned StyleTTS2 backend has two bundles whose weights
diverged from what is currently published on HF (probed via
`x-linked-size` against the repo's main branch on 2026-05-05):

    postbert.weight.bin   HF live: 28,324,480 B   build: 14,200,320 B
    vocoder.weight.bin    HF live: 54,836,160 B   build: 217,140,224 B

The other five graphs (plbert, alignment, diffusion_step, prosody, noise)
match what is on HF byte-for-byte and are NOT touched by this script.

To avoid disrupting clients that still pin to the existing names, the new
bundles are uploaded under a `_v2` suffix:

    build/hf-upload-v2/ANE/styletts2_ane_postbert.mlmodelc
        -> ANE/styletts2_ane_postbert_v2.mlmodelc
    build/hf-upload-v2/ANE/styletts2_ane_postbert.mlpackage
        -> ANE/styletts2_ane_postbert_v2.mlpackage
    build/hf-upload-v2/ANE/styletts2_ane_vocoder.mlmodelc
        -> ANE/styletts2_ane_vocoder_v2.mlmodelc
    build/hf-upload-v2/ANE/styletts2_ane_vocoder.mlpackage
        -> ANE/styletts2_ane_vocoder_v2.mlpackage

Both bundle types are directories, so we use HfApi.upload_folder() with
`path_in_repo` set to the v2-suffixed bundle path on the Hub. The local
files are uploaded as-is — the rename happens only at the destination.

Usage:
    uv run python upload-v2-models.py --token hf_xxxxxxxxxxxx
    uv run python upload-v2-models.py --token hf_xxx --dry-run

Optional flags:
    --repo-id      override target repo
                   (default: FluidInference/StyleTTS-2-coreml)
    --build-dir    override local staging root
                   (default: build/hf-upload-v2/ANE)
    --dry-run      validate inputs and print plan, do not upload
    --commit-msg   override commit message
"""
import argparse
import pathlib
import sys
import time

from huggingface_hub import HfApi, whoami
from huggingface_hub.utils import HfHubHTTPError


REPO_ID = "FluidInference/StyleTTS-2-coreml"
ANE_PREFIX = "ANE"

# (local_basename, hub_basename) — local stays as-is, hub gets `_v2` suffix
# inserted before the extension.
BUNDLES: tuple[tuple[str, str], ...] = (
    ("styletts2_ane_postbert.mlmodelc", "styletts2_ane_postbert_v2.mlmodelc"),
    ("styletts2_ane_postbert.mlpackage", "styletts2_ane_postbert_v2.mlpackage"),
    ("styletts2_ane_vocoder.mlmodelc", "styletts2_ane_vocoder_v2.mlmodelc"),
    ("styletts2_ane_vocoder.mlpackage", "styletts2_ane_vocoder_v2.mlpackage"),
)

DEFAULT_COMMIT_MSG = (
    "Add StyleTTS2-ANE postbert_v2 + vocoder_v2 bundles\n\n"
    "Publishes the rebuilt postbert (14.2 MB weights, was 28.3 MB) and\n"
    "vocoder (217 MB weights, was 54.8 MB) bundles from the 7-graph\n"
    "ANE-pinned re-cut (FluidInference/mobius#56) under a `_v2` suffix\n"
    "to avoid breaking clients pinned to the existing filenames.\n\n"
    "End-to-end validation against PyTorch reference on \"Hello world,\n"
    "this is a test of the text to speech system.\" using ref_s_Vinay:\n"
    "  log-mel cos: 0.9767    F0 cos: 0.9941    N cos: 0.9860\n"
    "Parakeet-TDT ASR transcribes both CoreML-ANE and PyTorch outputs\n"
    "verbatim (conf 0.959 / 0.974).\n\n"
    "Five other graphs (plbert, alignment, diffusion_step, prosody,\n"
    "noise) match HF main byte-for-byte and are not touched."
)


def fail(msg: str, code: int = 1) -> None:
    print(f"[upload-v2] ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def info(msg: str) -> None:
    print(f"[upload-v2] {msg}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Upload StyleTTS2-ANE postbert_v2 + vocoder_v2 CoreML bundles "
            "to HuggingFace."
        )
    )
    p.add_argument(
        "--token",
        required=True,
        help="HuggingFace access token with write access to the repo.",
    )
    p.add_argument(
        "--repo-id",
        default=REPO_ID,
        help=f"Target HF repo (default: {REPO_ID}).",
    )
    p.add_argument(
        "--build-dir",
        type=pathlib.Path,
        default=pathlib.Path("build/hf-upload-v2/ANE"),
        help=(
            "Local directory containing the rebuilt bundles "
            "(default: build/hf-upload-v2/ANE)."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print plan; do not upload.",
    )
    p.add_argument(
        "--commit-msg",
        default=DEFAULT_COMMIT_MSG,
        help="Override the commit message for the HF revision.",
    )
    return p.parse_args()


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n = n / 1024
    return f"{n:.1f} GB"


def main() -> None:
    args = parse_args()

    if not args.build_dir.is_dir():
        fail(f"build dir does not exist: {args.build_dir}")

    # Validate every bundle exists and is non-empty before touching HF.
    plans: list[tuple[pathlib.Path, str]] = []
    for local_name, hub_name in BUNDLES:
        local = args.build_dir / local_name
        if not local.is_dir():
            fail(
                f"missing bundle: {local}  "
                f"(re-run the ANE coreml export pipeline first)"
            )
        files = sorted(p for p in local.rglob("*") if p.is_file())
        if not files:
            fail(f"bundle is empty: {local}")
        path_in_repo = f"{ANE_PREFIX}/{hub_name}"
        plans.append((local, path_in_repo))
        total = sum(p.stat().st_size for p in files)
        info(
            f"  found {local_name:38s}  {len(files):3d} files  "
            f"{fmt_bytes(total):>10s}  ->  {args.repo_id}/{path_in_repo}"
        )

    # Validate token and repo access.
    info("Authenticating with HF...")
    try:
        identity = whoami(token=args.token)
    except HfHubHTTPError as e:
        fail(f"token is not valid: {e}")
    info(f"  authenticated as: {identity.get('name', identity)}")

    api = HfApi(token=args.token)
    try:
        repo_info = api.repo_info(args.repo_id, repo_type="model")
    except HfHubHTTPError as e:
        fail(
            f"cannot access repo {args.repo_id}: {e}\n"
            f"  (the token must have write access to this repo)"
        )
    info(f"  repo accessible: {args.repo_id} (last sha: {repo_info.sha[:12]})")

    if args.dry_run:
        info("--dry-run: stopping before any upload.")
        info("Plan:")
        for local, path_in_repo in plans:
            info(f"  {local}  ->  {args.repo_id}/{path_in_repo}")
        return

    for local, path_in_repo in plans:
        info(f"Uploading {local}  ->  {args.repo_id}/{path_in_repo} ...")
        t0 = time.perf_counter()
        try:
            # delete_patterns="*" wipes any orphan files at the destination
            # before re-uploading. For brand-new `_v2` paths this is a
            # no-op, but it keeps the script idempotent on re-runs.
            commit = api.upload_folder(
                folder_path=str(local),
                path_in_repo=path_in_repo,
                repo_id=args.repo_id,
                repo_type="model",
                commit_message=args.commit_msg,
                delete_patterns="*",
            )
        except HfHubHTTPError as e:
            fail(f"upload failed: {e}")
        dt = time.perf_counter() - t0
        commit_url = getattr(commit, "commit_url", None) or commit
        info(f"  done in {dt:.1f}s  ->  {commit_url}")

    info("All uploads complete.")
    info(
        f"View at: https://huggingface.co/{args.repo_id}/tree/main/{ANE_PREFIX}"
    )
    info(
        "FluidAudio will only see these once ModelNames.swift is updated "
        "to reference the `_v2` filenames (or once a separate StyleTTS2Ane "
        "downloader is added that resolves to the v2 paths)."
    )


if __name__ == "__main__":
    main()
