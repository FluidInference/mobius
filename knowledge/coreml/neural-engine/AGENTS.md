# Neural Engine Reference Notes

## Vendored upstream snapshot
- Source: [hollance/neural-engine](https://github.com/hollance/neural-engine)
- Commit: [`10d30481b21ef12e88ca5cea2c886bb72b297de4`](https://github.com/hollance/neural-engine/commit/10d30481b21ef12e88ca5cea2c886bb72b297de4) (2025-03-07)
- Primary author: Matthijs Hollemans and the Neural Engine contributors
- License: See `knowledge/coreml/neural-engine/LICENSE`

The contents of `knowledge/coreml/neural-engine/` are a direct snapshot of the commit above to keep the docs close to our Core ML tooling. Please credit the original authors when citing, and review the upstream repository for updates before re-vendoring.

## Layout overview
```text
neural-engine/
├── README.md              # Upstream overview of the ANE research notes
├── LICENSE                # Original MIT license from hollance/neural-engine
├── .gitignore             # Ignore rules retained from the upstream repo
└── docs/                  # Topic-specific ANE articles and how-tos
    ├── 16-bit.md
    ├── ane-vs-gpu.md
    ├── internals.md
    ├── is-model-using-ane.md
    ├── model-surgery.md
    ├── os-log.md
    ├── other.md
    ├── prevent-running-on-ane.md
    ├── programming-ane.md
    ├── reverse-engineering.md
    ├── running-on-ane.md
    ├── supported-devices.md
    ├── unsupported-layers.md
    └── why-care.md
```

## Document summaries
- `README.md`: High-level primer on the Apple Neural Engine, why Core ML models may not hit it, and links to the topical guides in `docs/`.
- `LICENSE`: MIT terms from hollance/neural-engine; retain this when re-vendoring or distributing derivatives.
- `.gitignore`: Mirrors the upstream ignore rules for generated build artifacts and temporary files.
- `docs/16-bit.md`: Explains ANE precision characteristics (float16 throughout) and the resulting risks for activations or quantized layers.
- `docs/ane-vs-gpu.md`: Compares the ANE and GPU architectures, frameworks, and memory trade-offs to clarify when Core ML hops between them.
- `docs/internals.md`: Collects references on how NPUs operate internally with pointers to TPU/Tensor Core resources for deeper study.
- `docs/is-model-using-ane.md`: Playbook for detecting ANE execution using configuration toggles, debugger breakpoints, Instruments, and powermetrics.
- `docs/model-surgery.md`: Provides a Core ML Tools script that swaps unsupported broadcastable add layers for ANE-friendly alternatives.
- `docs/os-log.md`: Shows how to capture detailed Core ML diagnostics via `os_log`, log archives, and Console filters.
- `docs/other.md`: Notes anecdotal quirks such as float16 weight slowdowns and scheduler decisions that may divert work from the ANE.
- `docs/prevent-running-on-ane.md`: Swift snippet for forcing `.cpuAndGPU` or `.cpuOnly` execution when the ANE must be sidelined.
- `docs/programming-ane.md`: Confirms there is no public API for direct ANE programming beyond Core ML today and references private frameworks.
- `docs/reverse-engineering.md`: Links to community reverse-engineering efforts (tinygrad, livestreams) for those exploring private ANE interfaces.
- `docs/running-on-ane.md`: Covers how to request ANE usage with `MLModelConfiguration` and caveats around unsupported layers and fallbacks.
- `docs/supported-devices.md`: Enumerates Apple SoCs with ANEs, their core counts/TOPS, and which devices include (or lack) the accelerator.
- `docs/unsupported-layers.md`: Lists Core ML layers known to miss ANE support and strategies to refactor models to stay on-ANE.
- `docs/why-care.md`: Motivates targeting the ANE for latency, energy, and thermal advantages while acknowledging compatibility hurdles.

## Updating the snapshot
1. Clone the upstream repository.
2. Check out the desired commit or tag.
3. Replace the contents of `knowledge/coreml/neural-engine/` with the new snapshot (preserve the license and this file).
4. Record the new commit hash in this document.

## Usage notes
These docs cover Apple Neural Engine (ANE) behavior, layer support, and debugging guidance. They are helpful when building iOS 17+ Core ML agents that target `.CpuOnly` or `.ane` variants. Start with `docs/running-on-ane.md` for deployment tips and `docs/programming-ane.md` for lower-level diagnostics.
