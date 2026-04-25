#!/usr/bin/env bash
# Compile every PocketTTS *.mlpackage into a sibling *.mlmodelc.
#
# FluidAudio's Swift loader (Sources/FluidAudio/ModelNames.swift, see
# `condStepFile = condStep + ".mlmodelc"` etc.) consumes the compiled
# .mlmodelc form, not the source .mlpackage. We ship BOTH on HF so:
#   - .mlpackage  → reproducible, debuggable, what conversion produces
#   - .mlmodelc   → what FluidAudio actually loads at runtime
#
# Compilation is done with `xcrun coremlc compile` which emits
#   <output_dir>/<basename>.mlmodelc
# for an input <basename>.mlpackage.
#
# Idempotent: skips when the .mlmodelc exists and is newer than the
# source .mlpackage. Pass FORCE=1 to recompile everything.
#
# Usage:
#   ./mlmodelc_all.sh                              # every supported language
#   LANGUAGES="english spanish_24l" ./mlmodelc_all.sh
#   FORCE=1 ./mlmodelc_all.sh                      # recompile, ignore caches
#
# Requires: Xcode command-line tools (`xcrun coremlc` ships with them).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Must stay in sync with convert_all_languages.sh DEFAULT_LANGUAGES.
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

MLPACKAGES=(
    cond_step.mlpackage
    flowlm_step.mlpackage
    flow_decoder.mlpackage
    mimi_decoder.mlpackage
)

if [[ -n "${LANGUAGES:-}" ]]; then
    # shellcheck disable=SC2206
    TARGETS=(${LANGUAGES})
else
    TARGETS=("${DEFAULT_LANGUAGES[@]}")
fi

if ! xcrun --find coremlc >/dev/null 2>&1; then
    echo "ERROR: xcrun coremlc not found. Install Xcode command-line tools:" >&2
    echo "  xcode-select --install" >&2
    exit 1
fi

# is_stale <mlpackage_path> <mlmodelc_path>
# Returns 0 (true) when mlmodelc must be (re)compiled.
is_stale() {
    local pkg="$1"
    local mlc="$2"

    [[ -n "${FORCE:-}" ]] && return 0
    [[ ! -d "$mlc" ]] && return 0

    # Compare mtimes via stat (BSD/macOS).
    local pkg_mtime mlc_mtime
    pkg_mtime="$(stat -f %m "$pkg" 2>/dev/null || echo 0)"
    mlc_mtime="$(stat -f %m "$mlc" 2>/dev/null || echo 0)"
    [[ "$pkg_mtime" -gt "$mlc_mtime" ]]
}

compile_one() {
    local pkg="$1"
    local out_dir="$2"
    local basename
    basename="$(basename "$pkg" .mlpackage)"
    local mlc="$out_dir/${basename}.mlmodelc"

    if [[ ! -d "$pkg" ]]; then
        echo "  [skip] $pkg not found"
        return 0
    fi

    if ! is_stale "$pkg" "$mlc"; then
        echo "  [skip] ${basename}.mlmodelc up-to-date"
        return 0
    fi

    echo "  [compile] $(basename "$pkg") → ${basename}.mlmodelc"
    # Remove any stale partial output first.
    rm -rf "$mlc"
    xcrun coremlc compile "$pkg" "$out_dir" >/dev/null
}

total=0
compiled=0
skipped=0
failed=0
START_TS=$(date +%s)

for lang in "${TARGETS[@]}"; do
    BUILD_DIR="build/$lang"
    if [[ ! -d "$BUILD_DIR" ]]; then
        echo "[warn] $BUILD_DIR missing — run convert_all_languages.sh first"
        continue
    fi

    echo ""
    echo "=============================================================="
    echo "Language: $lang"
    echo "=============================================================="

    for pkg_name in "${MLPACKAGES[@]}"; do
        total=$((total + 1))
        pkg_path="$BUILD_DIR/$pkg_name"
        out_path="$BUILD_DIR/$(basename "$pkg_name" .mlpackage).mlmodelc"

        before=$compiled
        if compile_one "$pkg_path" "$BUILD_DIR"; then
            if [[ -d "$out_path" ]] && is_stale "$pkg_path" "$out_path"; then
                # Shouldn't happen — compile succeeded but output not present
                failed=$((failed + 1))
            elif [[ -d "$out_path" ]]; then
                # Determine if this run actually compiled (skip leaves $compiled unchanged).
                # Since `is_stale` was false on entry for skips, simplest accounting is
                # by-path mtime check post-call. Cheap heuristic: count compiled if
                # mtime updated since START_TS.
                mtime="$(stat -f %m "$out_path" 2>/dev/null || echo 0)"
                if [[ "$mtime" -ge "$START_TS" ]]; then
                    compiled=$((compiled + 1))
                else
                    skipped=$((skipped + 1))
                fi
            fi
        else
            failed=$((failed + 1))
        fi
    done
done

echo ""
echo "=============================================================="
echo "Compilation summary"
echo "  total     : $total"
echo "  compiled  : $compiled"
echo "  skipped   : $skipped"
echo "  failed    : $failed"
echo "=============================================================="

if [[ "$failed" -gt 0 ]]; then
    exit 1
fi
