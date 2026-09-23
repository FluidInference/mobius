# Decision 1.0 Kai 0.6B → Core ML

Source: [Decision-1.0-Kai-0.6B](https://huggingface.co/llm-semantic-router/Decision-1.0-Kai-0.6B), immutable revision in `assets.lock.json`.

The complete native release has 571,909,635 unique parameters across its encoder, Choice path, Score path, and trained heads. Its tokenizer retains the upstream Gemma-origin tokenizer terms; see the source repository's `LICENSING_STATUS.md` and `NOTICE`. The 307.8M count on the older Decision Index snapshot describes an estimated encoder, not this complete release. Its evaluated revision has not been established.

Run from this directory with its sibling `decision-vela/coreml` shared exporter present:

```bash
uv sync
uv run python convert-coreml.py --verify-only
uv run python convert-coreml.py --kind choice --output-dir build/kai
uv run python convert-coreml.py --verify-packages-dir build/kai
DECISION_TOOLKIT_DIR="$PWD" DECISION_NATIVE_DIR="/path/to/materialized/native" \
    uv run pytest ../../decision-vela/coreml/tests/test_native_parity.py
```

The verifier uses six real rows from the source repository: `examples/decisions.jsonl` and its `examples/system-one.json` rendered by the pinned upstream System One code. The upstream integrity checker requires a materialized directory without Hugging Face cache symlinks; on macOS, `cp -cRL source/native destination/native` makes a copy-on-write copy. Verification checks the full native model against each typed wrapper. Conversion traces one selected row and shape; `--fixture-index 1` selects the System One row, including a two-option Choice request. `--verify-packages-dir` compares every saved path to the full native model on both source requests per kind. Generated reports record Core ML logit difference and candidate agreement. See `STATUS.md` for measured results and remaining shape limits.

Optional embedding-only W8 packages for Choice, Noul and Score each reduce the fixed L128 package from 643.8 to 448.0 MB. All six pinned native decisions remain unchanged, with worst probability difference 0.00887. Use `conversion/run_coreml.py --embedding-w8-kinds choice noul score` with the published repository to select them; default inference stays FP16. The profiled Choice W8 graph places 926/938 executable operations on ANE under CPU+ANE, versus 926/934 for FP16. Small AC timing runs did not establish a repeatable speed gain, and W8 cold load was slower. [Detailed report](reports/embedding-w8.json).
