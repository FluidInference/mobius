# Kev-0.8B for Core ML

Core ML export of [jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b) (Jared Palmer, Apache-2.0): a LoRA
adapter and pointer head on [Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) at revision
`dc7cdfe2`. The LoRA is folded in fp32 with Kev's own loader (`merge_reference.py`), the path every published Kev number
uses. The port first matched Kev's reference answers (row packages below), then got fast with a fused export that reads
the state once and answers every question in the same Core ML call.

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

## Fused path: state + packed questions in one call

The row export re-reads the state for every question. `kev_stages.py` `FusedPass` runs the state and all of a request's
questions in one call:

- Questions are packed end to end after the state. A segment mask `[P, P]` is at once the attention mask, the
  segment-local decay operator of the delta rule, and its WY triangle, so every question restarts from the state's
  recurrent state and never sees another question. A question's first conv lags read the state's conv tail
  (`lag_keep` / `lag_tail`). Questions never cross a 128-token lane.
- Projections, norms, MLPs and attention run once over all `S + P` positions; only the delta rule splits (chunked over
  the state, chunk 64, then one segment-masked chunk for the questions).
- The WY inverse is `_masked_block_inverse`: the same 2×2 blocking as the row export's `unit_lower_inverse`, kept as one
  matrix (`D_2s = D_s − D_s (t ⊙ M_s) D_s`), 4 ops per level instead of block extraction and concatenation. The block
  recursion was 15 of 29 ms of a 64 + 64 token call. A doubling (Neumann) product is not an option: powers of the
  strictly lower matrix reach 3e35 on real records, even in fp32.

`convert-stages.py --stage fused` builds `fused_S<state>_P<packed>_B16_K16` functions (iOS 18 / macOS 15, fp16 I/O) and
`combine-stages.py` merges them into one multifunction package that shares weights. Every function in a package adds to
each function's load time (Core ML processes the whole MIL program per load), so ship the buckets you need: the published
package has `S ∈ {32, 64, 128, 192, 256, 384} × P192` (Kev's evaluation state cap is 384 tokens; 12 yes/no questions fit
192 packed tokens).

Fidelity: in fp32 PyTorch the fused pass matches `KevRow` on development records of all three suites with every
question packed three times (segment isolation): 0 top-answer flips, max |Δp| 3e-6 (`parity_packed.py`). The fp16
Core ML functions, driven from Swift (FluidUse `KevFastManager`), agree with the fp16 row packages on 191 questions:
0 flips, max |Δp| 0.004.

Speed, M5 Pro, GPU (`CPU_AND_GPU`), warmed; Kev's `scripts/serving_bench.py` request shapes, new state each request:

| Request | Core ML fused (Swift) | Kev MLX server |
| --- | ---: | ---: |
| 2 questions, short state | 18.1 ms | 26.6 ms |
| 6 questions, short state | 37.8 ms | 51.5 ms |
| 5 questions, 370-token state | 64.1 ms | 69.6 ms |

(Core ML measured with the `P ∈ {32, 64, 128, 256}` buckets; MLX from `reports/mlx-serving-baseline-m5pro.json`.)
One 128 + 192 token call: GPU 30.8 ms, CPU only 186 ms, CPU + Neural Engine 740 ms, so this backbone does not belong on
the ANE. Weight-only int8 / int4 (`linear_quantize_weights`) leave GPU time unchanged (31.2 / 30.6 ms) and only shrink
the weights (955 → 480 / 271 MB). A function left idle for a while pays a 0.3–0.8 s re-setup on its next call,
independent of the inputs; warm before latency-sensitive work.

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
# fused functions (fp32 parity first), then one multifunction package
cd <kev repo> && PYTHONPATH=<this dir> uv run python <this dir>/parity_packed.py --merged <this dir>/build/merged \
    --packed-len 192 --repeat 3
for s in 32 64 128 192 256 384; do (cd <this dir> && uv run ... python convert-stages.py --stage fused --length $s \
    --question-len 192 --batch 16 --chunk-size 64 --lane 128 --build build/fused/functions); done
python combine-stages.py --functions build/fused/functions --output build/KevFused.mlpackage
```
