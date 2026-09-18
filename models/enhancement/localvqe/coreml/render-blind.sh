#!/usr/bin/env bash
# Render every mic/lpb pair of an AEC-Challenge blind-set layout
# (<dir>/<stem>_mic.flac + _lpb.flac) to <out>/<dir>/<stem>_enh.wav.
#
#   ./render-blind.sh coreml <blind_dir> <out_dir> <fluidaudiocli> [variant] [jobs]
#   ./render-blind.sh ggml   <blind_dir> <out_dir> <localvqe-bin> <model.gguf> [jobs]
#
# Outputs are written to a temp name and moved into place only on success, so
# an interrupted or failed render never becomes a cached "done" file. Exits
# non-zero if any clip failed.
set -uo pipefail
ENGINE="${1:?coreml|ggml}"
BLIND="${2:?blind_dir}"
OUT="${3:?out_dir}"
BIN="${4:?binary}"
if [ "$ENGINE" = coreml ]; then
    VARIANT="${5:-v1.3}"; JOBS="${6:-4}"; GGUF=""
else
    GGUF="${5:?model.gguf}"; JOBS="${6:-4}"; VARIANT=""
fi
export ENGINE BIN OUT VARIANT GGUF

mkdir -p "$OUT"
for d in "$BLIND"/*/; do mkdir -p "$OUT/$(basename "$d")"; done

find -L "$BLIND" -name '*_mic.flac' | sort | xargs -P "$JOBS" -I{} bash -c '
    mic="$1"; stem="${mic%_mic.flac}"
    scen="$(basename "$(dirname "$mic")")"
    out="$OUT/$scen/$(basename "$stem")_enh.wav"
    tmp="$out.part.$$.wav"
    [ -f "$out" ] && exit 0
    if [ "$ENGINE" = coreml ]; then
        "$BIN" enhance "$mic" --reference "${stem}_lpb.flac" --output "$tmp" --variant "$VARIANT" >/dev/null 2>&1
    else
        "$BIN" "$GGUF" --in-wav "$mic" "${stem}_lpb.flac" --out-wav "$tmp" >/dev/null 2>&1
    fi
    if [ $? -eq 0 ] && [ -s "$tmp" ]; then
        mv -f "$tmp" "$out"
    else
        rm -f "$tmp"; echo "FAIL: $mic" >&2; exit 1
    fi
' _ {}
status=$?
n_expected=$(find -L "$BLIND" -name '*_mic.flac' | wc -l | tr -d " ")
n_done=$(find "$OUT" -name '*_enh.wav' | wc -l | tr -d " ")
echo "rendered $n_done / $n_expected clips in $OUT"
if [ "$status" -ne 0 ] || [ "$n_done" -ne "$n_expected" ]; then
    echo "render-blind: FAILED (xargs status $status)" >&2
    exit 1
fi
