# Decision model Core ML profiles

`profile_requests.py` measures one warmed, complete typed request at a time: upstream request rendering/tokenization, Core ML prediction, and decoding. Package load/compilation is reported separately. Each invocation uses four Jeff or Kev 0.6B fixtures, or three Kev 0.5B fixtures, with one untimed warmup and at most two timed rounds. This is a small local latency and correctness probe, not a Decision Index or application-suite benchmark. `p95` is the nearest-rank observation from 6–8 calls, so it is descriptive rather than a population latency guarantee.

Run the script with the corresponding toolkit's `uv` environment and pass the published repo's tokenizer/config snapshot as `--assets`. The package can be in a different local directory. For example:

```bash
cd models/computer-use/kev-0.6b/coreml
uv run python ../../../../tools/decision-coreml-profile/profile_requests.py \
  --model kev06 --assets /path/to/kev-0.6b-coreml-snapshot \
  --package /path/to/kev_0_6b_w8_L128_options32.mlpackage \
  --units cpu-ne --output build/full-request-cpu-ne.json
```

Run each model/variant/compute-unit combination in a separate process so runtime and tokenizer caches do not cross-contaminate results. For Kev 0.5B, the source in this branch fixes the previously published `runtime.py` serving path, which required a training label and source field. `profile_requests.py` uses the new unlabelled `KevCoreML` session; comparisons to earlier model-call-only figures must be labelled accordingly. See each model's source revision and native/Core ML parity report before accepting an optimized package.
