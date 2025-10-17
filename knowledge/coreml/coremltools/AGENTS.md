# coremltools 9.0b1 Snapshot

## Vendored upstream release
- Source: [apple/coremltools](https://github.com/apple/coremltools)
- Release tag: [9.0b1](https://github.com/apple/coremltools/releases/tag/9.0b1)
- Commit: [`511e360f51b8c84aa8e0bc8fb059adf456858be5`](https://github.com/apple/coremltools/commit/511e360f51b8c84aa8e0bc8fb059adf456858be5) (2025-07-28)
- Primary maintainers: Apple Core ML Tools team
- License: See `knowledge/coreml/coremltools/LICENSE.txt`

This directory captures the documentation slice of the upstream 9.0b1 release so we can reference guides, API docs, and release notes without shipping the full toolchain. CI configs, lint settings, and branding assets were removed. Keep the license and NOTICE files intact if you redistribute any portion of this package.

## Layout overview
```text
coremltools/
├── docs/          # Sphinx documentation sources (reStructuredText)
├── docs-guides/   # Markdown tutorials and workflow walk-throughs
├── README.md      # Upstream project overview and installation guidance
├── LICENSE.txt    # Upstream BSD-style license
├── NOTICE.txt     # Copyright / attribution notice
└── AGENTS.md      # Local snapshot notes (this file)
```

## Document summaries
- `README.md`: Overview of coremltools, supported features, and installation instructions.
- `docs/`: Sphinx documentation sources (`docs/source/index.rst`) that feed the published API and MIL reference site.
- `docs-guides/`: Markdown tutorials covering installation, conversion workflows, optimization (quantization, pruning, palettization), and troubleshooting.
- `LICENSE.txt`, `NOTICE.txt`: Upstream licensing terms that must accompany any redistribution.
- *(Upstream source trees, build scripts, and tests were intentionally omitted to keep this knowledge snapshot documentation-focused. Fetch the official release if you need the full toolchain.)*

## Docs & guides highlights
- Quickstarts: `docs-guides/source/introductory-quickstart.md`, `installing-coremltools.md`, and `overview-coremltools.md` walk through environment setup and the unified conversion API.
- Conversion playbooks: PyTorch (`convert-pytorch-workflow.md`, `model-scripting.md`, `convert-a-torchvision-model-from-pytorch.md`), TensorFlow 1/2 (`tensorflow-1-workflow.md`, `convert-tensorflow-2-bert-transformer-models.md`), ONNX interoperability (`target-conversion-formats.md`).
- Optimization Tool (OPT) guides: Compression workflows (`opt-overview.md`, `opt-workflow.md`), quantization/pruning/palettization APIs (`opt-quantization-api.md`, `opt-pruning-algos.md`, `opt-palettization-overview.md`) with performance benchmark summaries.
- Advanced topics: Typed execution (`typed-execution.md`), stateful and multifunction models (`stateful-models.md`, `multifunction-models.md`), custom operators and composite graphs (`custom-operators.md`, `composite-operators.md`), model debugging utilities (`mlmodel-debugging-perf-utilities.md`).
- Deployment aides: Xcode previews (`xcode-model-preview-types.md`), flexible inputs (`flexible-inputs.md`), and metadata utilities (`mlmodel-utilities.md`).

## Re-vendoring checklist
1. Download or clone the desired release tag.
2. Copy over only the documentation assets (`docs/`, `docs-guides/`, `README.md`, license files) into `knowledge/coreml/coremltools/` and keep this file alongside them.
3. Remove upstream VCS directories and tooling-specific folders to maintain a docs-only snapshot.
4. Update the commit hash, date, and release tag above.
5. Review release notes for breaking changes and capture key highlights below.

## 9.0b1 release highlights (per upstream notes)
- Support for int8 model inputs/outputs plus model state read/write APIs.
- Adds iOS 26 / macOS 26 / watchOS 26 / tvOS 26 deployment targets and GPU low-precision accumulation hints.
- Extends coverage for PyTorch 2.7 and ExecuTorch 0.5, enriches converted-model metadata, and optimizes `im2col` alongside other fixes.

Refer to the official release page for the full changelog and compatibility table.
