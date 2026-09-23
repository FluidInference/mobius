# NanoJev Core ML conversion

This toolkit targets the **current unified-games** [NanoJev](https://huggingface.co/C-Tianyu/NanoJev) checkpoint at the immutable revision in `assets.lock.json`. It uses the trained `best.safetensors`, the bundled tokenizer and backbone configuration, and the original `DecisionModel` source. The current release differs from older NanoJev snapshots, so this package must not be presented as a reproduction of the tracker’s 26.19 result until the evaluated historical checkpoint is identified.

NanoJev encodes one path per candidate with Qwen3, then runs a trained scalar head. Choice requests additionally run an attention head over the **whole candidate set**. `export_model.py` preserves that set operation and the separate Boolean/Score behavior. The conversion writes one shared candidate-encoder package and one small head package, rather than replicating the 0.6B backbone for every question type.

```bash
uv sync --frozen
HF_HUB_DISABLE_XET=1 uv run python convert-coreml.py --length 128 --candidates 4 --parity-only
HF_HUB_DISABLE_XET=1 uv run python convert-coreml.py --length 128 --candidates 4
HF_HUB_DISABLE_XET=1 uv run python verify.py --length 128 --candidates 4
```

The L128/K4 bucket accepts one question per call. Native rendering rejects rather than truncates requests longer than 128 tokens. Larger lengths and candidate counts need separate exports and validation. `assets.py` allowlists files, disables remote model code, checks the full checkpoint hash, and loads all trained parameters strictly. All generated packages stay in ignored `build/`.

The upstream code includes an MIT license. The model card says this covers source code, but does not state a license for the trained checkpoint. **Do not redistribute Core ML weights** until the model owner grants or clarifies weight redistribution. The Qwen base retains its upstream license. Local conversion and parity checks are allowed; public model weights are held at this gate.
