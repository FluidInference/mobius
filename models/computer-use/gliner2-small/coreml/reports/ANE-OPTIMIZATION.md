# GLiNER2.5 small L128/K8 compression checks (2026-09-22)

The initial results are from one Apple M5 Pro on battery. The fixed, gold-free
[`optimization-selected20.jsonl`](../fixtures/optimization-selected20.jsonl)
contains five source-order requests from each of four application suites. Six
requests exceed this package's L128 bucket, leaving 14 checked. Timed calls
alternate which model runs first; cold load and preprocessing are recorded
separately. These checks do not establish a Decision Index or game score.

| Variant | Package bytes | FP16 choices | Worst absolute probability difference | CPU+ANE total request p50 | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| FP16 reference | 151,542,752 | — | — | 3.701 ms | Validated classification export |
| Whole-model per-channel W8 | 76,664,593 | 14/14 | 0.06544 on ALL | not run | Rejected: exceeds 0.02 gate and ALL p50 slowed 4.414→5.099 ms |
| Token-embedding-only W8 | 102,771,476 | 14/14 on ALL and CPU+ANE | 0.01741 ALL, 0.01758 CPU+ANE | 3.719 ms | Experimental size variant; no speed claim |

The embedding-only W8 variant also matched the native classifier on **100/100**
eligible requests from the existing source-order application check; 300 rows
had too many labels and seven exceeded L128 before the 100 were collected.
Worst chosen-label confidence difference was **0.01685**. Its
CPU+ANE compute plan assigned 496/516 executable operations to ANE. The 20
CPU operations include 17 int32 input/mask operations, two unresolved
comparisons, and a zero-runtime weight dequantization constant. The FP16 plan
assigned 496/512 operations to ANE with 16 CPU operations. Their measured
request medians were effectively tied; the main CPU boundary remains.

After the Mac switched to AC power, one additional paired CPU+ANE check on
the **same 14 valid requests** again kept all choices and the 0.01758 maximum
probability difference. Full-request p50 was 3.702 ms FP16 and 3.668 ms
embedding int8; model-call p50 was 3.285 and 3.289 ms respectively. These
differences are smaller than this small sample can resolve, so there is still
no demonstrated speed win. See `optimization-embedding-w8-selected20-cpu-ane-ac.json`.

The exact package shapes, requests, per-request timings, skips, power state,
and conversion metadata are in adjacent JSON reports. These variants were
tested as classification paths only; the original checkpoint's extraction
heads are outside this package. The FP16 package remains the recommended
release artifact until a controlled AC timing pass. The embedding-only W8
package may be distributed as an explicitly experimental size option.
