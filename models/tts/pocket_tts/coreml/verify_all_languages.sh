#!/usr/bin/env bash
# Verify every converted PocketTTS language pack with Whisper.
#
# For each language in the target set, runs:
#   1. generate_coreml_v4.py --language <lang> --text "<lang sample>" --output build/<lang>/verify.wav
#   2. verify_with_whisper.py --audio build/<lang>/verify.wav --language <iso>
#
# Prints both the reference prompt and the Whisper transcription so a human can
# compare. A failed generation aborts the script (set -e); a failed whisper call
# continues so we still see output for the other languages.
#
# Usage:
#   ./verify_all_languages.sh                          # 5 non-English packs
#   LANGUAGES="spanish italian" ./verify_all_languages.sh
#   VOICE=anna ./verify_all_languages.sh
#   WHISPER_MODEL=mlx-community/whisper-small-mlx ./verify_all_languages.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VOICE="${VOICE:-alba}"
WHISPER_MODEL="${WHISPER_MODEL:-}"

# Default to the 6 verified language packs (Phase 2 targets + English).
DEFAULT_LANGUAGES=(
    english
    spanish
    french_24l
    german
    italian
    portuguese
)

# Per-language (sample_text, whisper_iso_code) pairs.
# Sample texts target roughly 10 seconds of synthesized audio
# (PocketTTS speaks at ~4 words/sec, so ~40-word phrases land near 10 s).
sample_text() {
    case "$1" in
        english)
            echo "Hello, this is a text to speech system. It can read sentences in many different languages with a natural sounding voice. The neural network was trained on a large collection of recordings from professional speakers."
            ;;
        spanish|spanish_24l)
            echo "Hola, este es un sistema de síntesis de voz. Puede leer frases en muchos idiomas diferentes con una voz natural y agradable. La red neuronal fue entrenada con una gran colección de grabaciones de locutores profesionales."
            ;;
        french_24l)
            echo "Bonjour, ceci est un système de synthèse vocale. Il peut lire des phrases dans de nombreuses langues différentes avec une voix naturelle. Le réseau de neurones a été entraîné sur une grande collection d'enregistrements de locuteurs professionnels."
            ;;
        german|german_24l)
            echo "Hallo, das ist ein Sprachsynthesesystem. Es kann Sätze in vielen verschiedenen Sprachen mit einer natürlich klingenden Stimme vorlesen. Das neuronale Netz wurde mit einer großen Sammlung von Aufnahmen professioneller Sprecher trainiert."
            ;;
        italian|italian_24l)
            echo "Ciao, questo è un sistema di sintesi vocale. Può leggere frasi in molte lingue diverse con una voce dal suono naturale. La rete neurale è stata addestrata su una grande raccolta di registrazioni di parlatori professionisti."
            ;;
        portuguese|portuguese_24l)
            echo "Olá, este é um sistema de síntese de voz. Ele pode ler frases em muitos idiomas diferentes com uma voz de som natural. A rede neural foi treinada com uma grande coleção de gravações de locutores profissionais."
            ;;
        *) echo "Hello world." ;;
    esac
}

whisper_code() {
    case "$1" in
        english)                    echo "en" ;;
        spanish|spanish_24l)        echo "es" ;;
        french_24l)                 echo "fr" ;;
        german|german_24l)          echo "de" ;;
        italian|italian_24l)        echo "it" ;;
        portuguese|portuguese_24l)  echo "pt" ;;
        *) echo "en" ;;
    esac
}

if [[ -n "${LANGUAGES:-}" ]]; then
    # shellcheck disable=SC2206
    TARGETS=(${LANGUAGES})
else
    TARGETS=("${DEFAULT_LANGUAGES[@]}")
fi

RESULTS_DIR="build/_verify_results"
mkdir -p "$RESULTS_DIR"
SUMMARY="$RESULTS_DIR/summary.txt"
: > "$SUMMARY"

for lang in "${TARGETS[@]}"; do
    echo ""
    echo "=============================================================="
    echo "Verify: $lang"
    echo "=============================================================="

    BUILD_DIR="build/$lang"
    if [[ ! -d "$BUILD_DIR" ]]; then
        echo "  [skip] $BUILD_DIR missing; run convert_all_languages.sh first"
        continue
    fi

    TEXT="$(sample_text "$lang")"
    ISO="$(whisper_code "$lang")"
    WAV="$BUILD_DIR/verify.wav"
    LOG="$RESULTS_DIR/${lang}.log"

    echo "  text   : $TEXT"
    echo "  voice  : $VOICE"
    echo "  output : $WAV"

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
            --text "$TEXT" \
            --output "$WAV" 2>&1 | tee "$LOG"

    WHISPER_ARGS=(--audio "$WAV" --language "$ISO" --reference "$TEXT")
    if [[ -n "$WHISPER_MODEL" ]]; then
        WHISPER_ARGS+=(--model "$WHISPER_MODEL")
    fi

    set +e
    uv run --no-project \
        --python 3.11 \
        --with "mlx-whisper>=0.4.0" \
        python verify_with_whisper.py "${WHISPER_ARGS[@]}" \
        2>&1 | tee -a "$LOG"
    rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
        echo "  [warn] whisper exited $rc for $lang"
    fi

    TRANSCRIPT="$(grep -E '^\[verify\] whisper' "$LOG" | tail -1 || true)"
    printf '%-20s | ref=%s\n%-20s | got=%s\n' \
        "$lang" "$TEXT" "" "${TRANSCRIPT:-<no whisper output>}" >> "$SUMMARY"
done

echo ""
echo "=============================================================="
echo "Summary ($SUMMARY):"
echo "=============================================================="
cat "$SUMMARY"
