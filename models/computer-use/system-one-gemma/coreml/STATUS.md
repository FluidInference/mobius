# System One Gemma status — 2026-09-22

- `blocked_gated_base`: authenticated `hf_hub_download("google/gemma-3-270m", "config.json", revision="9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1")` returned `GatedRepoError` HTTP 403.
- The exact upstream trained adapter, score-head tensor, SHA-256 hashes, serving renderer, 256-token limit, and 2.35 temperature are pinned and source-audited.
- The Core ML exporter and native-vs-Core ML parity runner are prepared but **have not executed**. There is no validated Core ML package, speed number, ANE placement, or Decision Index score.
- Trained-weight redistribution needs an explicit rights determination. This public toolkit excludes the adapter and base weights.
- Next: account owner accepts the [Gemma access terms](https://huggingface.co/google/gemma-3-270m). Then fetch exact base, run conversion and parity, review Gemma and upstream weight terms before publishing any converted weights.
