#!/usr/bin/env bash
# Build a `build/_listen/` folder with descriptive filenames so you can
# A/B the 24L packs at two durations:
#   - <lang>_10s.wav  (~10-second 40-word phrase, copied from existing verify.wav)
#   - <lang>_1s.wav   (~1-second short greeting, freshly synthesized)
#
# Targets the 5 verified 24L packs by default. Override with LANGUAGES.
#
# Usage:
#   ./make_listening_samples.sh
#   LANGUAGES="spanish_24l italian_24l" ./make_listening_samples.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VOICE="${VOICE:-alba}"
LISTEN_DIR="build/_listen"
mkdir -p "$LISTEN_DIR"

DEFAULT_LANGUAGES=(
    spanish_24l
    french_24l
    german_24l
    italian_24l
    portuguese_24l
)

# Short greetings that should produce ~1s of audio (3-4 words each).
short_text() {
    case "$1" in
        spanish|spanish_24l)        echo "Hola, ¿cómo estás?" ;;
        french_24l)                 echo "Bonjour, comment ça va?" ;;
        german|german_24l)          echo "Guten Tag, alles gut?" ;;
        italian|italian_24l)        echo "Ciao, come stai?" ;;
        portuguese|portuguese_24l)  echo "Olá, tudo bem?" ;;
        english)                    echo "Hello, how are you?" ;;
        *)                          echo "Hello." ;;
    esac
}

if [[ -n "${LANGUAGES:-}" ]]; then
    # shellcheck disable=SC2206
    TARGETS=(${LANGUAGES})
else
    TARGETS=("${DEFAULT_LANGUAGES[@]}")
fi

for lang in "${TARGETS[@]}"; do
    BUILD_DIR="build/$lang"
    if [[ ! -d "$BUILD_DIR" ]]; then
        echo "[skip] $BUILD_DIR missing"
        continue
    fi

    echo ""
    echo "=============================================================="
    echo "Listen samples: $lang"
    echo "=============================================================="

    # 10-second clip: copy the existing verify.wav (already produced by
    # verify_all_languages.sh, ~10s 40-word phrase).
    SRC_10S="$BUILD_DIR/verify.wav"
    DST_10S="$LISTEN_DIR/${lang}_10s.wav"
    if [[ -f "$SRC_10S" ]]; then
        cp "$SRC_10S" "$DST_10S"
        sz=$(du -h "$DST_10S" | awk '{print $1}')
        echo "  [copy] $SRC_10S -> $DST_10S  ($sz)"
    else
        echo "  [warn] $SRC_10S not found — run verify_all_languages.sh first"
    fi

    # 1-second clip: freshly synthesize a short greeting.
    DST_1S="$LISTEN_DIR/${lang}_1s.wav"
    SHORT="$(short_text "$lang")"
    echo "  [gen]  $DST_1S  (text: \"$SHORT\")"

    uv run --no-project \
        --python 3.10 \
        --with "pocket-tts>=1.0.3" \
        --with "coremltools>=8.0" \
        --with "safetensors>=0.4.0" \
        --with "sentencepiece>=0.2.1" \
        --with "scipy>=1.5.0" \
        --with "numpy>=2" \
        python generate_coreml_v4.py \
            --language "$lang" \
            --voice "$VOICE" \
            --text "$SHORT" \
            --output "$DST_1S"

    if [[ -f "$DST_1S" ]]; then
        sz=$(du -h "$DST_1S" | awk '{print $1}')
        echo "  [done] $DST_1S  ($sz)"
    fi
done

echo ""
echo "=============================================================="
echo "Listen folder contents:"
echo "=============================================================="
ls -1 "$LISTEN_DIR" | sort
echo ""
echo "Open in Finder:"
echo "  open $LISTEN_DIR"
