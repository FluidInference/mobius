#!/usr/bin/env bash
# Render every mic/lpb pair of an AEC-Challenge blind-set layout
# (<scenario>/<stem>_mic.flac + _lpb.flac) to <out>/<scenario>/<stem>_enh.wav.
#
#   ./render-blind.sh coreml <blind_dir> <out_dir> <fluidaudiocli> [variant] [jobs]
#   ./render-blind.sh ggml   <blind_dir> <out_dir> <localvqe-bin> <model.gguf> [jobs]
set -uo pipefail
ENGINE="${1:?coreml|ggml}"
BLIND="${2:?blind_dir}"
OUT="${3:?out_dir}"
BIN="${4:?binary}"
if [ "$ENGINE" = coreml ]; then
    VARIANT="${5:-v1.3}"; JOBS="${6:-4}"
else
    GGUF="${5:?model.gguf}"; JOBS="${6:-4}"
fi
export ENGINE BIN OUT VARIANT GGUF

mkdir -p "$OUT"
for d in "$BLIND"/*/; do mkdir -p "$OUT/$(basename "$d")"; done

find -L "$BLIND" -name '*_mic.flac' | sort | xargs -P "$JOBS" -I{} bash -c '
    mic="$1"; stem="${mic%_mic.flac}"
    scen="$(basename "$(dirname "$mic")")"
    out="$OUT/$scen/$(basename "$stem")_enh.wav"
    [ -f "$out" ] && exit 0
    if [ "$ENGINE" = coreml ]; then
        "$BIN" enhance "$mic" --reference "${stem}_lpb.flac" --output "$out" --variant "$VARIANT" >/dev/null 2>&1 \
            || echo "FAIL: $mic" >&2
    else
        "$BIN" "$GGUF" --in-wav "$mic" "${stem}_lpb.flac" --out-wav "$out" >/dev/null 2>&1 \
            || echo "FAIL: $mic" >&2
    fi
' _ {}
echo "done: $(find "$OUT" -name '*_enh.wav' | wc -l) files in $OUT"
