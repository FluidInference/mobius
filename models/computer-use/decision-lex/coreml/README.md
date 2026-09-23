# Decision 1.0 Lex 0.6B → Core ML

Source: [Decision-1.0-Lex-0.6B](https://huggingface.co/llm-semantic-router/Decision-1.0-Lex-0.6B), immutable revision in `assets.lock.json`.

The complete native release has 571,909,635 unique parameters across its encoder, Choice path, Score path, and trained heads. Its tokenizer retains the upstream Gemma-origin tokenizer terms; see the source repository's `LICENSING_STATUS.md` and `NOTICE`. The 307.8M count on the older Decision Index snapshot describes an estimated encoder, not this complete release. Its evaluated revision has not been established.

Run from this directory with its sibling `decision-vela/coreml` shared exporter present:

```bash
uv sync
uv run python convert-coreml.py --verify-only
uv run python convert-coreml.py --kind choice --output-dir build/lex
uv run python convert-coreml.py --verify-packages-dir build/lex
DECISION_TOOLKIT_DIR="$PWD" DECISION_NATIVE_DIR="/path/to/materialized/native" \
    uv run pytest ../../decision-vela/coreml/tests/test_native_parity.py
```

The verifier uses six real rows from the source repository: `examples/decisions.jsonl` and its `examples/system-one.json` rendered by the pinned upstream System One code. The upstream integrity checker requires a materialized directory without Hugging Face cache symlinks; on macOS, `cp -cRL source/native destination/native` makes a copy-on-write copy. Verification checks the full native model against each typed wrapper. Conversion traces one selected row and shape; `--fixture-index 1` selects the System One row, including a two-option Choice request. Generated reports record Core ML logit difference and candidate agreement. The public [Core ML preview](https://huggingface.co/FluidInference/decision-1.0-lex-coreml) is limited to 128 tokens and three Choice/Score candidate slots (two for Noul). Wider shapes and Core ML performance profiling remain open; the preview must not be treated as a reproduction of the historical Decision Index result.

Optional embedding-only W8 packages for Noul and Score each reduce the fixed L128 package from 643.8 to 448.0 MB. Both paths retain their two pinned native decisions; worst probability differences were 0.00323 and 0.00102. Use `conversion/run_coreml.py --embedding-w8-kinds noul score` with the published repository. **Choice stays FP16** because both symmetric and asymmetric W8 changed the pinned `route` decision. The profiled Noul W8 graph places 926/938 executable operations on ANE under CPU+ANE, with no repeatable speed gain in the small AC timing check. [Detailed report](reports/embedding-w8.json).
