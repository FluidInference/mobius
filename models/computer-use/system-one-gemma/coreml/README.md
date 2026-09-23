# System One Gemma scorer on Core ML — conversion blocked at base access

This toolkit pins the real trained [`akash-kamat/system-one-gemma`](https://github.com/akash-kamat/system-one-gemma) scorer at commit `cc75aa8042dec965003210fdba033fee8759735e` and its `google/gemma-3-270m` base at revision `9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1`. The 15 MB adapter contains a trained BF16 `score.weight` with shape `[1, 640]`; `modules_to_save` includes `score`. Stock Gemma or a replacement scalar head would not be the tracked model. `assets.lock.json` pins SHA-256 hashes for the adapter, metadata, tokenizer and audited runtime.

**Current status: no Core ML model has been produced or validated.** Google gates the base. The authenticated Hugging Face account on this Mac receives HTTP 403 even for its tiny `config.json`. The exact next step is for the account owner to [review and accept the Gemma terms on the base model page](https://huggingface.co/google/gemma-3-270m), then run `uv run python assets.py` again. Do not use another account's token or an ungated mirror. The source-only [FluidInference/system-one-gemma-coreml](https://huggingface.co/FluidInference/system-one-gemma-coreml) repository contains no Gemma or trained adapter weights.

The upstream web app is the serving reference: it scores each state/question/option string with a Gemma sequence-classification head, right-pads the candidate batch, uses a **256-token** cap, and applies **temperature 2.35** before softmax. It keeps the question/option tail and truncates the end of the state. Its CLI (`infer.py`) defaults to 384 tokens and temperature 1.0 instead; that is a different configuration. This conversion targets the released web-app behavior. It exposes a complete choice probability distribution and chosen option. The upstream scorer has no separate trained Boolean, ordered-score, or abstention head, so this toolkit does not claim those native decision types or the historical Decision Index score of 17.09. The historical tracker checkpoint/adapter pair is unverified.

After legitimate access is granted, on Apple Silicon:

```bash
uv sync --frozen
uv run python assets.py
uv run python convert-coreml.py
uv run python verify-parity.py
uv run pytest -q
```

`convert-coreml.py` is a prepared, **untested** FP16 L256/K16 export. It checks that PEFT's merge retained the exact trained scalar head and writes 16 raw choice logits; host code in `scorer.py` batches options in groups of 16 and applies the upstream temperature/softmax across the full option set. `verify-parity.py` is prepared to compare the native model and Core ML on the eight real upstream demo questions. A produced package must not be published as validated until this check passes and a new report records its size, latency, and compute placement. The prepared export has not been profiled on the Neural Engine.

The upstream repository licenses its code under Apache-2.0 but says its pretrained scorer weights inherit a **noncommercial restriction** from training data. The trained-weight redistribution permission is not sufficiently clear for us to rehost them. Google's [Gemma Terms](https://ai.google.dev/gemma/terms) also govern a converted derivative; section 3.1 specifies downstream restrictions, terms copies, modification notices, and a Notice file. Access to the base and permission to publish the complete converted artifact are separate conditions. This toolkit does not include or redistribute either weight set.
