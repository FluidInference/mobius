#!/usr/bin/env bash
# Upload converted PocketTTS language packs to a HuggingFace model repo.
#
# This script is run BY THE USER (not by any agent) — requires `hf auth
# login` (or HF_TOKEN env var) with write access to the target repo.
# Default repo: FluidInference/pocket-tts-coreml. Override with HF_REPO.
#
# Pushes the 10 converted language packs into:
#   <repo>/languages/<lang>/cond_step.{mlpackage,mlmodelc}
#   <repo>/languages/<lang>/flowlm_step.{mlpackage,mlmodelc}
#   <repo>/languages/<lang>/flow_decoder.{mlpackage,mlmodelc}
#   <repo>/languages/<lang>/mimi_decoder.{mlpackage,mlmodelc}
#   <repo>/languages/<lang>/constants_bin/
#
# Both .mlpackage (source/debuggable) and .mlmodelc (Swift-loadable) are
# uploaded — Swift's `MLModel(contentsOf:)` consumes the .mlmodelc form,
# while .mlpackage is preserved for reproducibility / re-tracing.
#
# Run `./mlmodelc_all.sh` first to produce the .mlmodelc artifacts.
#
# Existing root-level English files (cond_step.mlpackage at repo root,
# etc.) are LEFT UNTOUCHED to preserve backward compatibility for current
# FluidAudio Swift clients. The new uniform layout duplicates English
# under `languages/english/` so the Swift multi-language code path can
# treat every language identically.
#
# Trial-14 caveat: the root-level mimi_decoder.mlpackage in
# FluidInference/pocket-tts-coreml predates the cap-256 modulo-wrap fix
# (TRIALS.md Trial 12) and produces robotic audio after ~2 s of long-form
# English. To roll the fix out to existing clients without changing the
# layout, additionally refresh the root mimi by passing
# UPDATE_ROOT_MIMI=1 (otherwise root mimi is left in place).
#
# Usage:
#   ./upload_languages.sh                         # interactive, all 10 langs
#   LANGUAGES="spanish italian" ./upload_languages.sh
#   HF_REPO=myorg/myrepo ./upload_languages.sh
#   UPDATE_ROOT_MIMI=1 ./upload_languages.sh      # also refresh root mimi
#   SKIP_MLMODELC=1 ./upload_languages.sh         # mlpackage-only (debug)
#   SKIP_MLPACKAGE=1 ./upload_languages.sh        # mlmodelc-only (slim push)
#   DRY_RUN=1 ./upload_languages.sh               # print plan without uploading
#   YES=1 ./upload_languages.sh                   # skip confirmation
#
# Requires: `hf` CLI (huggingface_hub>=0.24); resolved through uv so the
# host doesn't need a system-wide install.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REPO="${HF_REPO:-FluidInference/pocket-tts-coreml}"

# All 10 language packs (matches convert_all_languages.sh + mlmodelc_all.sh).
DEFAULT_LANGUAGES=(
    english
    spanish
    spanish_24l
    french_24l
    german
    german_24l
    italian
    italian_24l
    portuguese
    portuguese_24l
)

# Per-language model artifacts. For each base name we upload the
# .mlpackage (source) and .mlmodelc (compiled, Swift-loadable) variants
# unless SKIP_MLPACKAGE / SKIP_MLMODELC is set.
MODEL_BASENAMES=(
    cond_step
    flowlm_step
    flow_decoder
    mimi_decoder
)

# Non-model artifacts (always uploaded as-is, no compile variant).
EXTRA_ARTIFACTS=(
    constants_bin
)

if [[ -n "${LANGUAGES:-}" ]]; then
    # shellcheck disable=SC2206
    TARGETS=(${LANGUAGES})
else
    TARGETS=("${DEFAULT_LANGUAGES[@]}")
fi

# Resolve the `hf` CLI lazily through uv (same pattern as the other scripts
# in this directory, so the host doesn't need a global huggingface_hub).
HF_CLI=("uv" "run" "--no-project" "--python" "3.10"
        "--with" "huggingface_hub>=0.24"
        "hf")

upload_path() {
    # upload_path <local_path> <remote_path> <commit_message>
    local local_path="$1"
    local remote_path="$2"
    local message="$3"

    if [[ ! -e "$local_path" ]]; then
        echo "  [skip] $local_path not found"
        return 0
    fi

    if [[ -n "${DRY_RUN:-}" ]]; then
        echo "  [dry-run] hf upload $REPO $local_path $remote_path"
        return 0
    fi

    echo "  [upload] $local_path → $REPO:$remote_path"
    "${HF_CLI[@]}" upload \
        "$REPO" \
        "$local_path" \
        "$remote_path" \
        --repo-type model \
        --commit-message "$message"
}

# Build the artifact list for a single language, expanding model basenames
# into .mlpackage / .mlmodelc according to SKIP_* env flags.
artifacts_for_lang() {
    local items=()
    for base in "${MODEL_BASENAMES[@]}"; do
        if [[ -z "${SKIP_MLPACKAGE:-}" ]]; then
            items+=("$base.mlpackage")
        fi
        if [[ -z "${SKIP_MLMODELC:-}" ]]; then
            items+=("$base.mlmodelc")
        fi
    done
    items+=("${EXTRA_ARTIFACTS[@]}")
    printf '%s\n' "${items[@]}"
}

# Plan + size preview
echo "Plan:"
echo "  repo  : $REPO"
echo "  langs : ${TARGETS[*]}"
echo "  forms : $([[ -z "${SKIP_MLPACKAGE:-}" ]] && echo -n "mlpackage ")$([[ -z "${SKIP_MLMODELC:-}" ]] && echo -n "mlmodelc ")constants_bin"

missing_mlmodelc=()
for lang in "${TARGETS[@]}"; do
    src="build/$lang"
    if [[ ! -d "$src" ]]; then
        echo "    [warn] build/$lang missing — run convert_all_languages.sh first"
        continue
    fi
    sz="$(du -sh "$src" 2>/dev/null | awk '{print $1}')"
    echo "    $lang ($sz)"

    # Pre-flight: warn if mlmodelc inclusion is enabled but artifacts are missing.
    if [[ -z "${SKIP_MLMODELC:-}" ]]; then
        for base in "${MODEL_BASENAMES[@]}"; do
            if [[ ! -d "$src/$base.mlmodelc" ]]; then
                missing_mlmodelc+=("$src/$base.mlmodelc")
            fi
        done
    fi
done

if [[ "${#missing_mlmodelc[@]}" -gt 0 ]]; then
    echo ""
    echo "  [warn] ${#missing_mlmodelc[@]} expected .mlmodelc artifact(s) missing:"
    for m in "${missing_mlmodelc[@]}"; do echo "         $m"; done
    echo "  Run ./mlmodelc_all.sh to produce them, or pass SKIP_MLMODELC=1."
fi

if [[ -n "${UPDATE_ROOT_MIMI:-}" ]]; then
    echo "  + root: refresh mimi_decoder.mlpackage from build/english/ (Trial-14 fix)"
fi
echo ""

if [[ -z "${HF_TOKEN:-}" && -z "${DRY_RUN:-}" ]]; then
    echo "  [info] HF_TOKEN not set; relying on cached login from \`hf auth login\`."
    echo ""
fi

if [[ -z "${DRY_RUN:-}" && -z "${YES:-}" ]]; then
    read -r -p "Proceed with upload to $REPO? [y/N] " ans
    case "$ans" in
        y|Y|yes|YES) ;;
        *) echo "Aborted."; exit 1 ;;
    esac
fi

# Per-language uploads
for lang in "${TARGETS[@]}"; do
    src="build/$lang"
    if [[ ! -d "$src" ]]; then
        echo ""
        echo "[skip] $lang — no $src"
        continue
    fi

    echo ""
    echo "=============================================================="
    echo "Language: $lang"
    echo "=============================================================="

    while IFS= read -r artifact; do
        upload_path \
            "$src/$artifact" \
            "languages/$lang/$artifact" \
            "Add $lang/$artifact"
    done < <(artifacts_for_lang)
done

# Optional root-level mimi refresh (Trial-14 fix for legacy clients)
if [[ -n "${UPDATE_ROOT_MIMI:-}" ]]; then
    echo ""
    echo "=============================================================="
    echo "Refreshing root-level mimi_decoder (Trial-14)"
    echo "=============================================================="
    upload_path \
        "build/english/mimi_decoder.mlpackage" \
        "mimi_decoder.mlpackage" \
        "Refresh mimi_decoder with Trial-12 cap-256 modulo-wrap fix (Trial 14 in TRIALS.md)"
    if [[ -z "${SKIP_MLMODELC:-}" && -d "build/english/mimi_decoder.mlmodelc" ]]; then
        upload_path \
            "build/english/mimi_decoder.mlmodelc" \
            "mimi_decoder.mlmodelc" \
            "Refresh mimi_decoder.mlmodelc (Trial-14, recompiled)"
    fi
fi

echo ""
echo "=============================================================="
echo "Upload complete."
echo "  Browse: https://huggingface.co/$REPO/tree/main/languages"
echo "=============================================================="
