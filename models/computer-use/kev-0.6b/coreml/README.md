# Kev 0.6B on Core ML

This toolkit converts the trained [Kev 0.6B](https://huggingface.co/jaredpalmer/kev-0.6b) LoRA adapter and pointer head, merged with [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base), to a fixed-shape Core ML decision model. The exporter includes Qwen3 query/key normalization and the trained head. It does not substitute the stock base model.

The current packages accept one typed question per call, up to 128 tokens and 32 option slots. Conversion parity uses upstream `kev.data.materialize` and `model.encode`. The standalone runtime renders unlabelled requests with upstream `kev.api.to_record` and the weight-free `kev.model.encode` function. The 128-token bucket may truncate the right edge of a long state; questions that cannot fit raise an error. Longer buckets and packed multiquestion requests are not represented by these packages.

```bash
uv sync --frozen
HF_HUB_DISABLE_XET=1 uv run python convert-coreml.py --length 128 --max-options 32
HF_HUB_DISABLE_XET=1 uv run python verify.py build/kev_0_6b_fp16_L128_options32.mlpackage --units cpu-ne
uv run python quantize.py --precision w8 --length 128 --max-options 32
HF_HUB_DISABLE_XET=1 uv run python verify.py build/kev_0_6b_w8_L128_options32.mlpackage --units cpu-ne
uv run pytest -q
```

`assets.lock.json` pins the adapter, pointer head, Qwen base, and upstream Kev code. `assets.py` verifies the two trained weight hashes before loading. The first conversion downloads about 1.2 GB of base weights plus the adapter and head. Generated packages stay in ignored `build/` and are distributed through a separate Hugging Face repo.

The Hub release includes `source/runtime.py`, `tokenizer/`, `config/training_config.json`, both Core ML packages, and `LICENSE`. Download it and run `uv run --project source python source/runtime.py --model-dir . --request-json request.json` from its root. The request needs one Kev `choice`, `noul`, or `score` question and no training label. `source/verify.py` instead loads the original model for conversion parity.

`verify.py` writes a `reports/verification-<package>-cpu-ne.json` file after a passing CPU+ANE run. `publish-hf.py` requires that report for each uploaded variant, checks at least four native decisions and the 0.02 probability-error gate, and compares the report's package SHA-256 inventory with the exact files being uploaded. Run the verification command again after any conversion or quantization change.

## Validation

On Apple Silicon, the real merged PyTorch model and the explicit export wrapper agreed within `9.54e-7` logits on the export fixture. Both Core ML packages agreed on the chosen option for four local request fixtures (two Choice, one Noul, one Score). FP16 had maximum probability error `0.002031` and a 1,194 MB package. W8 had maximum probability error `0.008554` and a 599 MB package. `CPU_AND_NE` model-call medians varied across tiny runs: FP16 `13.10–15.35 ms`, W8 `12.27–23.15 ms`; these runs do not establish a speed ordering or full ANE residency. Inspect `coreml-cli --ops --fallback` before making a device-residency claim. The four requests are a parity smoke test, not the Decision Index or a 2048 benchmark.

The upstream adapter/head and Qwen base are Apache-2.0. See the original [Kev model card](https://huggingface.co/jaredpalmer/kev-0.6b) and [Kev source](https://github.com/jaredpalmer/kev).
