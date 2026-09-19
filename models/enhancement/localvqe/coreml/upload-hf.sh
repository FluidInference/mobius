#!/usr/bin/env bash
# Upload the compiled bundles + model card to the FluidInference HF repo.
#   ./upload-hf.sh [repo]   (default FluidInference/localvqe-coreml)
set -euo pipefail
REPO="${1:-FluidInference/localvqe-coreml}"
HERE="$(cd "$(dirname "$0")" && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp "$HERE/hf-model-card.md" "$STAGE/README.md"
cp "$HERE/LICENSE-upstream" "$STAGE/LICENSE"
for m in "$HERE"/build/*.mlmodelc; do cp -R "$m" "$STAGE/"; done
ls -la "$STAGE"
hf repo create "$REPO" --type model 2>/dev/null || true
hf upload "$REPO" "$STAGE" . --commit-message "LocalVQE v1.3/v1.2 fp32 streaming Core ML exports (16 ms + 256 ms)"
