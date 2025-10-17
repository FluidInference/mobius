# Core ML Agents Snapshot Index

This folder aggregates third-party knowledge bases and tooling snapshots that support our Core ML agent workflows. Each snapshot captures an upstream repository or release alongside local notes for attribution, updates, and usage.

## Current snapshots
- `core-ml-on-device-llama.md` — Apple ML Research highlight detailing the Llama-3.1-8B-Instruct Core ML export, GPU tuning, KV-cache support, and Int4 quantization strategy for ~33 tok/s on M1 Max-class devices.
- `neural-engine/` — Vendored documentation from [hollance/neural-engine](https://github.com/hollance/neural-engine) at commit [`10d30481b21ef12e88ca5cea2c886bb72b297de4`](https://github.com/hollance/neural-engine/commit/10d30481b21ef12e88ca5cea2c886bb72b297de4).
  - Reference: `neural-engine/AGENTS.md` for the full file index, summaries, and re-vendoring workflow.
  - New: `neural-engine/docs/neural-engine-transformers.md` captures Apple’s 2022 ANE transformer deployment guide, covering distilbert export, ANE-optimized kernels, and performance validation tips.
- `coremltools/` — Source snapshot of [apple/coremltools 9.0b1](https://github.com/apple/coremltools/releases/tag/9.0b1) (`511e360f51b8c84aa8e0bc8fb059adf456858be5`), trimmed to documentation assets.
  - Reference: `coremltools/AGENTS.md` for directory summaries, re-vendoring steps, and release highlights; the `docs/` and `docs-guides/` subfolders host the full API reference and conversion/optimization guides.

## Planned additions
- _(none yet)_ Use this section to track upcoming toolkits to vendor.

## Vendoring checklist
1. Clone or download the upstream source/release artifact.
2. Copy the relevant files into a subdirectory here (e.g., `coremltools/`).
3. Strip upstream `.git` metadata, retain LICENSE files, and document provenance in a colocated `AGENTS.md`.
4. Update this index with the new snapshot’s location, version, and any maintenance reminders.

Keep large binaries on external storage (e.g., Hugging Face) and double-check upstream licensing before distributing derivatives.

## Knowledge base workflow
1. Pull context via the shared `knowledge_base` before you start new Core ML work so you inherit existing notes, pitfalls, and asset locations.
2. While you vendor docs or refine conversion pipelines, capture any new findings (unsupported layers, tooling quirks, benchmarking deltas) and save them back into the `knowledge_base` once confirmed.
3. Treat knowledge base updates as part of the deliverable: add high-signal summaries when you publish a fresh snapshot, finish a debugging session, or discover runtime behaviors that others should know about.
4. Reference the stored notes in your `AGENTS.md` updates so future contributors can trace where guidance originated.

## Adding new knowledge snapshots
1. Create a directory named after the asset (`<tool-or-reference>/[version]/` if multiple releases will coexist).
2. Vendor only the materials we need (docs, configs, lightweight scripts) and remove upstream build/test artifacts unless they are essential for Möbius workflows.
3. Keep upstream `LICENSE`/`NOTICE` files alongside the snapshot and cite the exact commit or release tag.
4. Author an `AGENTS.md` inside the new directory summarizing contents, layout, re-vendoring steps, and any high-impact references.
5. Append a bullet under **Current snapshots** (or **Planned additions** while work is in flight) linking to the new folder and its `AGENTS.md`.
