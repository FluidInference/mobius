# Pinned upstream reference

`cua_s1/__init__.py`, `cua_s1/model.py`, and `cua_s1/checkpoint.py` are copied
without modification from `trycua/cua` commit
`83f142c4290a0f7d9ed545ae8532858c6e4f8145`, under
`libs/cua-s1/python/src/cua_s1/`. Their download URLs and SHA-256 checksums are in
`../assets.lock.json`. Keeping this small reference local avoids installing the
GUI, document extraction, training, or server components.

The checkpoint loader validates the upstream safetensors metadata and the
signature covering the actual tensors and configuration, and loads all tensor
keys strictly. No original `.pt` pickle file is used.

`CUA-LICENSE` is the package's MIT license. `THIRD_PARTY_NOTICES.md` is copied
from `libs/cua-s1/THIRD_PARTY_NOTICES.md` at the same revision and includes the
Minimal Labs attribution and license for the adapted attention-model code.

`cua_metrics.py` is copied without modification from `libs/cua-s1/evals/metrics.py`
at that same revision. `score-report.py` verifies its SHA-256 before loading it
and adapts the saved option indices into the evaluator's action/target records.
