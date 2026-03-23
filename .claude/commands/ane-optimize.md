---
description: Profile ANE utilization and validate numerical parity after an optimization pass on a CoreML model conversion.
---

You are running the ANE optimization validation loop. The user has provided a model conversion directory path: $ARGUMENTS

This skill validates that an ANE optimization change (a) improved ANE utilization and (b) did not regress numerical parity with the original PyTorch model.

## Step 1: Locate the conversion directory and its artifacts

The argument should be a conversion directory path (e.g., `models/stt/parakeet-tdt-v3-0.6b/coreml/`). From this directory, identify:

1. **Build output directory** — look for a `build/` subdirectory or read the conversion script's `--output-dir` default. Common patterns: `build/`, `<model>_coreml/`, `coreml_models/`.
2. **Compare script** — look for `compare-components.py`, `compare-models.py`, or similar in the conversion directory. Read its `--help` or source to understand its CLI interface.
3. **Compiled models** — find `.mlmodelc` or `.mlpackage` files in the build output directory.

If any of these cannot be found, stop and ask the user to clarify.

## Step 2: Save or retrieve the baseline

Check if a baseline fallback analysis already exists at `<build-dir>/ane-baseline.json`.

**If no baseline exists**, this is the first optimization pass. Capture the baseline:

```bash
cd tools/coreml-cli && uv run coreml-cli <path-to-build-dir> --fallback --json > <build-dir>/ane-baseline.json
```

Also capture the benchmark baseline:

```bash
cd tools/coreml-cli && uv run coreml-cli <path-to-build-dir> --units cpu_and_neural_engine --json > <build-dir>/ane-benchmark-baseline.json
```

Tell the user: "Baseline captured. Make your optimization changes, reconvert, then invoke /ane-optimize again."

**Stop here on a baseline capture run** — do not proceed to steps 3-5.

**If baseline exists**, proceed to Step 3.

## Step 3: Profile the optimized model

Run these commands from `tools/coreml-cli/`:

**a) Fallback analysis (structured JSON for comparison):**
```bash
cd tools/coreml-cli && uv run coreml-cli <path-to-build-dir> --fallback --json
```

Save the output — you will compare it against the baseline in Step 5.

**b) Full benchmark for the ANE config:**
```bash
cd tools/coreml-cli && uv run coreml-cli <path-to-build-dir> --units cpu_and_neural_engine
```

**c) Human-readable fallback table (for the user):**
```bash
cd tools/coreml-cli && uv run coreml-cli <path-to-build-dir> --fallback
```

## Step 4: Validate numerical parity

This is CRITICAL. ANE optimization changes must not break model accuracy.

1. Read the compare script in the conversion directory to understand its interface.
2. Run the compare script from the conversion directory using `uv run`. Adapt to the actual script's interface — example patterns:
   ```bash
   # Typer-based (parakeet):
   cd <conversion-dir> && uv run python compare-components.py compare --output-dir <build-dir> --runs 5

   # argparse-based (silero-vad, pyannote):
   cd <conversion-dir> && uv run python compare-models.py --coreml-dir <build-dir> --num-tests 5
   ```
3. Examine the output for:
   - **Max absolute error** per component/output
   - **Max relative error** per component/output
   - **Pass/fail** against rtol/atol thresholds

**FLAG A REGRESSION if any of these occur:**
- Max absolute error increased by more than 10x compared to known good values
- Any component that previously passed parity now fails
- End-to-end output changed meaningfully (wrong transcription, garbled audio, incorrect labels)

If no compare script exists in the conversion directory, tell the user: "No comparison script found. You must manually verify that outputs match the PyTorch baseline before accepting this optimization."

## Step 5: Compare against baseline and report

Read the saved baseline JSON files and compare against the current profiling results. Produce this report:

```
## ANE Optimization Report

### Model: <model-name>

### Device Assignment (cpu_and_neural_engine config)
| Component        | Baseline ANE% | Current ANE% | Delta  |
|------------------|---------------|--------------|--------|
| <component-name> | XX.X%         | XX.X%        | +X.X%  |

### Fallback Ops
- Baseline: X ops on CPU (out of Y total)
- Current:  X ops on CPU (out of Y total)
- Eliminated reasons: <reasons no longer present>
- Remaining reasons: <reasons with op counts>

### Latency (cpu_and_neural_engine)
| Component        | Baseline (ms) | Current (ms) | Speedup |
|------------------|---------------|--------------|---------|
| <component-name> | XX.XX         | XX.XX        | X.Xx    |

### Numerical Parity
| Component        | Max Abs Error | Max Rel Error | Status |
|------------------|---------------|---------------|--------|
| <component-name> | X.XXe-XX      | X.XXe-XX      | PASS/FAIL |

### Verdict
PASS — ANE% improved with no parity regression
WARN — ANE% improved but parity degraded, review errors above
FAIL — Parity regression detected, revert this change
NEUTRAL — No meaningful change in ANE utilization
```

## Rules

- **NEVER skip the parity check.** An optimization that breaks accuracy is worse than no optimization.
- If the compare script fails to run, do NOT report the optimization as successful. Fix the issue first.
- Quote fallback reasons exactly from coreml-cli output — these are the actionable hints for the next optimization pass.
- For multi-component models, profile and validate EACH component separately.
- Save current results as the new baseline only if the verdict is PASS: overwrite `ane-baseline.json` and `ane-benchmark-baseline.json`.
