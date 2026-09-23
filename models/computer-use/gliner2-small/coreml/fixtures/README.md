# Selected optimization requests

`optimization-selected20.jsonl` contains the first five requests from each of
`jev.ag_news`, `jev.emotion`, `app.phishing`, and `app.model_routing_domain` in the existing
[`laya` application suite](../../../laya/coreml/benchmark/suites.jsonl).
Selection uses source order only; gold labels were removed before scoring compression variants.
The 20 rows are a small, reproducible parity and latency check, not a Decision
Index or 2048 score. Use the same requests, bucket (L128/K8), and compute units
for the FP16 reference and compressed candidate.
`verify-compression.py` requires all selected decisions to agree and defaults to
a maximum 0.02 absolute probability difference; a pass applies only to these
requests and should be followed by the larger existing 100-request native check.
