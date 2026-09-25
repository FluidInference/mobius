# Kev-0.8B for Core ML

Core ML export of [jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b) (Jared Palmer, Apache-2.0): a LoRA
adapter and pointer head on [Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) at revision
`dc7cdfe2`. The LoRA is folded in fp32 with Kev's own loader (`merge_reference.py`), the path every published Kev number
uses. The goal of this port is fidelity: the same answers as Kev's reference model. Speed is not tuned yet.

## What the package computes

Kev asks each question as its own causal row: the state, then `<q>` instruction, one `<opt> … </opt>` span per option,
and `<decide>` (Qwen3.5's Gated DeltaNet layers cannot honour Kev's packed block mask, so Kev itself runs rows on these
backbones). `KevRow` is one fixed-length row: the Qwen3.5 decoder from `qwen35_export.py` (copied from the Cua-S1-4B
conversion), the final norm at `<decide>` and at each `</opt>`, selected with one-hot maps, and Kev's pointer head with
the checkpoint's calibration temperature (2.351). Token embeddings are gathered on the host (`embeddings.f16`).

| Package | Tokens per row | Options | Used for |
| --- | ---: | ---: | --- |
| `L512_K16/KevRow_fp16.mlpackage` | 512 | 16 | every transfer-v4 row, 92% of decision-v7 rows |
| `L1024_K80/KevRow_fp16.mlpackage` | 1,024 | 80 | longer rows and the 78-way banking intent task |

## Fidelity

Kev's own benchmark (`kev.benchmark`, unchanged) scored the Core ML packages through a stand-in model
(`kev_benchmark_coreml.py`) and Kev's fp32 PyTorch model (`KEV_DTYPE=fp32`, Apple GPU) on the same development splits:

| Suite | Questions | Top answer differs | Max \|Δp\| | Kev fp32 | Published | Core ML fp16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| transfer-v4 dev (new sources) | 764 | 1 | 0.013 | 0.6479 | 0.648 | 0.6463 |
| decision-v7 dev (trained sources) | 1,468 | 2 | 0.029 | 0.8267 | 0.827 | 0.8252 |

Accuracy is Kev's "clean" metric (656 and 1,264 questions). Kev's shipped MLX path differs from the same fp32 reference
on 4 of 1,264 questions (max \|Δp\| 0.054, `runs/r4-mlx-parity-0.8b` in the Kev repo). The export wrapper in fp32
PyTorch matches Kev's model to 3.0e-6 (`reports/parity-wrapper-torch.json`). The `result.json` shipped in the Hub
snapshot reports 0.8252 for decision-v7; Kev's reference on the same snapshot gives 0.8267, which matches the model card.

## Speed (not a goal of this port)

Kev's MLX server on an M5 Pro answers five questions about a 370-token state in 69.6 ms (36.4 ms when the state is
cached; `reports/mlx-serving-baseline-m5pro.json`, `mac_bench.py`). This export has no state cache and runs one full
row per question, about 114 ms per row from Python at 512 tokens on the GPU, so it is several times slower than Kev's
MLX path today. A state-cache export would be needed to compete.

## Reproduce

```bash
# in a clone of github.com/jaredpalmer/kev at 3d9973b, with `uv sync --extra serve`
uv run python <this dir>/merge_reference.py --run jaredpalmer/kev-0.8b --out <this dir>/build/merged
# conversion needs torch 2.7 (coremltools 9.0 does not convert torch 2.8 traces of this graph)
cd <this dir> && uv run --no-project --python 3.12 --with torch==2.7.0 --with coremltools==9.0 --with safetensors \
    --with "numpy<2.3" python convert-coreml.py --length 512 --max-options 16
# ... and --length 1024 --max-options 80
cd <kev repo> && PYTHONPATH=<this dir> uv run --with coremltools==9.0 python <this dir>/kev_benchmark_coreml.py \
    --build <this dir>/build -- --run jaredpalmer/kev-0.8b --suite evals/v4/transfer-v4 --out runs/coreml-transfer-v4
```
