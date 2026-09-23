# Kai/Lex L128 ANE experiments

These are local exploratory measurements on an Apple M5 Pro with 24 GB RAM,
macOS 27.0, and battery power. They are not a leaderboard or release benchmark.
The pinned source and package provenance remain in each model's `assets.lock.json`
and `reports/` directory. Only the Kai Choice path has been profiled so far;
Lex and the other typed paths have not been measured on device in this study.

| Kai Choice L128, K3 | Published FP16 | FP16 one-hot marker map |
| --- | ---: | ---: |
| Package bytes | 643,804,060 | 643,803,269 |
| ANE executable ops | 926 / 934 (99.1%) | 926 / 931 (99.5%) |
| CPU fallback ops | 8 | 5 |
| CPU fallback cause | Int32 mask, embedding, marker gather | Int32 mask and embedding |
| Full-request p50, CPU + ANE | 4.214 ms | 4.785 ms |
| Full-request p95, CPU + ANE | 4.450 ms | 5.069 ms |

The marker-map experiment moves candidate selection from a position gather to
an FP16 matrix product. Its traced wrapper exactly matched the full native
model on the export request; saved Core ML logits differed by at most 0.005079
on that request, with the same chosen candidate. The first two complete
published-host requests produced the same probability vectors as the FP16
baseline. The paired full-request check alternated two pinned real Choice
requests for 10 timed calls per package after two warmups. The marker map was
13.6% slower at the median despite three fewer CPU operations, so it is
rejected. The first timing attempt included a repeated Core ML specification
lookup in the experimental host; only the corrected paired values above are
used. Ten battery-power calls are too few for a general latency claim.

The remaining fallback is an Int32 island around the embedding lookup and
attention-mask preparation. Weight-only W8 per-channel compression cut the Kai
Choice package from 643,804,060 to 323,256,927 bytes, but failed the first
pinned real request: the full native model and Core ML disagreed on the chosen
candidate (maximum logit error 1.1753 and probability error 0.4280). W8 is
rejected and was neither timed nor published. A future compression attempt
would need selective exclusions and native decision checks before placement or
speed work. The published Kai and Lex FP16 packages are unchanged.
