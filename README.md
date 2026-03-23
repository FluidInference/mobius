![banner.png](banner.png)

# möbius

[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-7289da.svg)](https://discord.gg/WNsvaCtmDe)
[![All Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/FluidInference)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/FluidInference/mobius)
[![Blog](https://img.shields.io/badge/Blog-Read%20More-blue)](https://inference.plus/)

A möbius strip is a surface with only one side and one boundary. What looks like two sides is really one continuous path — the same way models stay unified as they move across formats. Its quite easy to run AI on NVIDIA GPUs, we want to bring the same experience to the edge, across a fragmented set of devices and chips.

Running AI on the edge is messy and slow. It’s easy to spin up models on NVIDIA GPUs, but the edge is a different story — fragmented devices, weird accelerators, and lots of friction. `möbius` is our attempt to smooth that out, with coding agents and SFT models doing the heavy lifting. Developers just can integrate it with just a couple lines of code, easier than making an API call.

Right now we’re focused on small models for AI accelerators (ANE, NPUs) and LLMs on edge GPUs.

We also share bits of the journey on our [blog](https://inference.plus/).

## Models Folder Structure

The models in this repository are organized using a standardized path structure. `uv` solves the problem with different models and package dependencies, each desintation for each model will have their own `pyproject.toml` script, like a self-encompased repository.

```text
models/
│   ├── {class}/
│   │   ├── {name}/
│   │   │   └── {destination}
```

- **class**: The model class or variant (e.g., `vad`, `llm`, `vllm`, `tts`, `stt`, etc..)
- **name**: Specific model name or version identifier
- **destination**: Target runtime or format (e.g., `coreml`, `onnx`, `openvino`)

## Usage

For usage, look at our other repos like [FluidAudio](https://github.com/FluidInference/FluidAudio) and [fluid-server](https://github.com/FluidInference/fluid-server)

Things to keep in mind:

- Trace with .CpuOnly
- Target iOS17+ (most users are on iOS17/macOS 14 right now)
- Use `uv` to manage dependencies

## Tools

### [`coreml-cli`](tools/coreml-cli/)

Profile CoreML models from the command line — see per-operation compute device assignments (CPU/GPU/ANE), compilation time, and prediction latency across all `MLComputeUnits` configurations. Useful after converting a model to verify ANE compatibility and measure performance.

```bash
uvx --from "coreml-cli @ git+https://github.com/FluidInference/mobius.git#subdirectory=tools/coreml-cli" coreml-cli path/to/model.mlmodelc
```

See [`tools/coreml-cli/README.md`](tools/coreml-cli/README.md) for full usage.

## Citations

```code
@misc{Mobius,
  author = {Fluid Inference},
  title = {möbius},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/FluidInference/mobius}},
  commit = {insert_some_commit_here},
  email = {hello@fluidinference.com}
}
```
