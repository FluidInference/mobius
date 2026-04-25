#!/usr/bin/env bash
# Convert every upstream PocketTTS language pack to CoreML.
#
# For each language under kyutai/pocket-tts/languages/<id>/, runs:
#   1. convert_cond_step.py        → build/<lang>/cond_step.mlpackage
#   2. convert_flowlm_step.py      → build/<lang>/flowlm_step.mlpackage
#   3. convert_flow_decoder_v2.py  → build/<lang>/flow_decoder.mlpackage
#   4. convert_mimi_decoder.py     → build/<lang>/mimi_decoder.mlpackage
#   5. export_constants.py         → build/<lang>/constants/*.npy
#   6. pack_constants_bin.py       → build/<lang>/constants_bin/*.bin
#
# Idempotent: skips steps whose outputs already exist. Safe to re-run.
# Pass LANGUAGES="english spanish" to limit the set.
#
# Usage:
#   ./convert_all_languages.sh                      # every supported language
#   LANGUAGES="italian spanish" ./convert_all_languages.sh
#   FORCE=1 ./convert_all_languages.sh              # ignore existing outputs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Supported language IDs (must stay in sync with _language_arg.py)
DEFAULT_LANGUAGES=(
    english
    german
    italian
    portuguese
    spanish
    french_24l
    german_24l
    italian_24l
    portuguese_24l
    spanish_24l
)

# IMPORTANT: mimi_decoder weights are PER-LANGUAGE, not shared.
# Trial 14 (see TRIALS.md) disproved the earlier "shared mimi" assumption.
# Per-language `parity_mimi.py` showed `decoder_transformer.self_attn.in_proj.weight`
# differs by abs_max≈1.92 between English and Italian, and using English's
# mimi.mlpackage everywhere degraded all 5 non-English langs (1/6 verify pass).
# Tracing mimi per-language (via the loop below) yields 6/6 verify pass.

if [[ -n "${LANGUAGES:-}" ]]; then
    # shellcheck disable=SC2206
    TARGETS=(${LANGUAGES})
else
    TARGETS=("${DEFAULT_LANGUAGES[@]}")
fi

run_step() {
    local lang="$1"
    local script="$2"
    local expected_output="$3"

    if [[ -z "${FORCE:-}" && -e "$expected_output" ]]; then
        echo "  [skip] $script (found $expected_output)"
        return 0
    fi

    # --no-project avoids the broken editable install of this repo (no local
    # pocket_tts/ source; upstream was intentionally removed in b47105f).
    # Dependencies for the conversion scripts are declared inline via --with.
    echo "  [run]  uv run --no-project python $script --language $lang"
    uv run --no-project \
        --python 3.10 \
        --with "pocket-tts>=1.0.3" \
        --with "coremltools>=8.0" \
        --with "safetensors>=0.4.0" \
        --with "sentencepiece>=0.2.1" \
        --with "scipy>=1.5.0" \
        --with "numpy>=2" \
        --with "torch>=2.5.0" \
        --with "huggingface_hub>=0.10" \
        --with "einops>=0.4.0" \
        python "$script" --language "$lang"
}

for lang in "${TARGETS[@]}"; do
    echo ""
    echo "=============================================================="
    echo "Language: $lang"
    echo "=============================================================="

    BUILD_DIR="build/$lang"
    mkdir -p "$BUILD_DIR"

    run_step "$lang" "convert_models/convert/convert_cond_step.py"       "$BUILD_DIR/cond_step.mlpackage"
    run_step "$lang" "convert_models/convert/convert_flowlm_step.py"     "$BUILD_DIR/flowlm_step.mlpackage"
    run_step "$lang" "convert_models/convert/convert_flow_decoder_v2.py" "$BUILD_DIR/flow_decoder.mlpackage"
    run_step "$lang" "convert_models/convert/convert_mimi_decoder.py"    "$BUILD_DIR/mimi_decoder.mlpackage"
    run_step "$lang" "convert_assets/export_constants.py"                "$BUILD_DIR/constants/bos_emb.npy"
    run_step "$lang" "convert_assets/pack_constants_bin.py"              "$BUILD_DIR/constants_bin/bos_emb.bin"
done

echo ""
echo "=============================================================="
echo "All requested language packs converted."
echo "  build/  contains one subdirectory per language, ready to upload."
echo "  See upload_languages.sh for the HF upload helper."
echo "=============================================================="
