# Verdict Core ML benchmark

`heman10x/rlcd-modernbert-151m` was converted from its pinned safetensors checkpoint at revision
`8af2496eb63c7fa66d7d234e1f62629380030eb4`. The export uses the same fixed L128, 25-option shape as
the GLiClass Edge and laya short-input measurements. Every latency result is five warmups followed by
100 timed predictions on an Apple M5 Pro. Package size includes the complete `.mlpackage`.

| Weights | Package | Fastest median | Fastest units | FP16 choice agreement | Application micro accuracy |
| --- | ---: | ---: | --- | ---: | ---: |
| FP16 | 303.2 MB | 3.565 ms | CPU+ANE | 100% | 44.52% |
| LUT8 per-tensor | 151.9 MB | 3.769 ms | All / CPU+ANE | 98.26% | 44.70% |
| LUT6 per-tensor | 114.0 MB | **3.451 ms** | CPU+ANE | 87.00% | 42.09% |
| LUT4 per-tensor | **76.1 MB** | 3.565 ms | All | 49.22% | 35.86% |
| LUT4 group-32, iOS 18 | 247.1 MB | 6.244 ms | All | not completed | not completed |

LUT8 is a useful download-size compression but makes inference 5.7% slower. LUT6 is 3.2% faster than
FP16, but changes 13.0% of decisions and loses 2.44 application-suite accuracy points. Per-tensor
LUT4 fails outright. Grouped-channel LUT4 was stopped after its controlled speed profile: it shrinks
only 18.5%, runs 75% slower in automatic mode and takes 66 ms when forced to CPU+ANE.

Verdict is not a better FluidUse candidate. Its FP16 latency is effectively laya-class (3.565 ms
against laya's 3.6 ms L128 model profile), while GLiClass Edge Apps v2 runs in 0.843 ms FP16 and
0.903 ms LUT8. Verdict's 44.52% application-suite micro accuracy is also far below GLiClass Edge's
72.89%, consistent with Verdict's lower public Decision Index.

The compressed parity run uses the same 3,899 application requests and natural-language label
mapping as the existing GLiClass benchmark. These application percentages are not the public
Decision Index. The complete consolidated measurements are in
[`reports/verdict-coreml-compression.json`](reports/verdict-coreml-compression.json).
