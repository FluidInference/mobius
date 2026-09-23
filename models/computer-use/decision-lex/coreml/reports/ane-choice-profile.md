# Lex Choice L128 ANE profile

The published fixed-shape FP16 Choice package (L128, K3) was profiled on an
Apple M5 Pro with 24 GB RAM and macOS 27.0. The machine was running on battery,
so the times are exploratory, not a release latency benchmark.

| Measure | Result |
| --- | ---: |
| Package bytes | 643,804,060 |
| ANE executable ops | 926 / 934 (99.1%) |
| CPU fallback ops | 8 |
| Full-request p50, CPU + ANE | 4.162 ms |
| Full-request p95, CPU + ANE | 4.591 ms |

The CPU fallback is the same Int32 mask, embedding, and marker-selection island
seen in Kai Choice. There were two warmups and 10 timed full-host requests,
alternating the two pinned real Choice rows in the conversion fixtures. The
profile includes tokenizer, padded request assembly, Core ML prediction, and
result decoding. The other Lex typed paths were not timed. Kai's marker-map
variant was slower and Kai's per-channel W8 variant changed a real decision,
so neither optimization has been applied to Lex. The published Lex package is
unchanged.

To reproduce placement, run the shared `profile_compute_plan.py` helper with
`--package` pointing to the published Choice `.mlpackage`. For request timing,
use `profile_coreml_requests.py` with its staged repository, Choice package,
and `--units cpu-ne`.
