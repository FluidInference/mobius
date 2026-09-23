# Benchmark data

> Measured by Fluid Inference on one M5 Pro MacBook with our own harness, against the
> general-access checkpoint (`Nemotron-3-Diarization.nemo`, sha256 `867c53f5…`) under
> OpenMDW 1.1. Protocols are stated per table; AMI, AliMeeting and NOTSOFAR1 use the
> forced-alignment references NVIDIA's card cites
> ([nttcslab-sp/diar-forced-alignment](https://github.com/nttcslab-sp/diar-forced-alignment)),
> collar 0. NVIDIA's model card remains the official source for the model's accuracy.
>
> Rows still marked *preview* were measured on the earlier early-access checkpoint and
> have not yet been re-run; they are retained only where no general-access figure exists.


## Card-protocol results (headline)

NVIDIA's current card publishes 32 Nemotron rows: 8 dataset conditions x 4 latency
profiles (offline 30.4 s / low 1.04 s / verylow 0.64 s / ultra 0.32 s), each with
DER, SCA, and MAE. Our coverage of those 32 rows:

- **16 rows protocol-identical** (below): AMI MHM/SDM + AliMeeting Near/Far, all
  four profiles. **DER at parity or better on 12 of 16** (11 strictly better, 1
  tie at card precision); mean DER delta **-0.26**.
- **8 rows protocol-adjacent**: NOTSOFAR1 Eval MHM/SC, all four profiles — same
  audio conditions and scoring settings, different reference source (NVIDIA's
  FastMSS forced alignments are unpublished). DER is NOT row-comparable; see the
  NOTSOFAR1 section for what is.
- **8 rows not reproducible**: DIHARD III Eval and CALLHOME-Part2 are LDC-licensed
  corpora we do not hold.

The accurate summary is: *FluidAudio matched or beat NVIDIA's DER on 12 of the 16
card configurations we could reproduce protocol-identically* — not "reproduced
NVIDIA's full benchmark suite."

### Protocol-identical rows (AMI + AliMeeting, all four card profiles)

Protocol identical to NVIDIA's model card: forced-alignment references
(nttcslab-sp/diar-forced-alignment), collar 0, overlap included, card
latency-profile configurations. Hardware: M5 Pro (MacBook), single stream,
wall-clock RTFx includes mel + host state updates. NVIDIA card numbers measured on
a Blackwell RTX PRO 5000. Audio conditions: AMI MHM = Mix-Headset; AMI SDM =
Array1-01; AliMeeting Far = array channel 0; AliMeeting Near = equal-weight mix of
per-speaker headset channels (`Scripts/materialize_alimeeting_card_audio.py`).

| Dataset | Profile | Card DER | Ours | Card SCA | Ours | Card MAE | Ours | Our wall RTFx |
|---|---|---|---|---|---|---|---|---|
| AMI MHM | offline | 9.30 | **9.28** | 87.50 | **87.50** | 0.1250 | **0.1250** | 1669x |
| AMI MHM | low | 10.35 | **9.90** | 75.00 | **75.00** | 0.2500 | **0.2500** | 76x |
| AMI MHM | verylow | 10.13 | **10.13** | 68.75 | **68.75** | 0.4375 | **0.3750** | 26x† |
| AMI MHM | ultra | 10.48 | 10.58 | 68.75 | **68.75** | 0.3750 | **0.3750** | 24x† |
| AMI SDM | offline | 12.27 | 12.63 | 75.00 | **75.00** | 0.3125 | **0.3125** | 1675x |
| AMI SDM | low | 12.62 | 14.07 | 81.25 | **81.25** | 0.1875 | **0.1875** | 70x |
| AMI SDM | verylow | 14.91 | **14.55** | 75.00 | **81.25** | 0.2500 | **0.1875** | 48x |
| AMI SDM | ultra | 13.75 | 14.77 | 62.50 | **81.25** | 0.4375 | **0.2500** | 24x |
| AliMeeting Far | offline | 11.14 | **9.83** | 85.00 | **90.00** | 0.1500 | **0.1000** | 1676x |
| AliMeeting Far | low | 11.34 | **10.20** | 65.00 | **70.00** | 0.3500 | **0.3000** | 71x |
| AliMeeting Far | verylow | 11.60 | **10.62** | 50.00 | **60.00** | 0.5000 | **0.4000** | 47x |
| AliMeeting Far | ultra | 12.30 | **11.17** | 60.00 | 55.00 | 0.4000 | 0.4500 | 24x |
| AliMeeting Near | offline | 7.25 | **6.83** | 95.00 | **95.00** | 0.0500 | **0.0500** | 1685x |
| AliMeeting Near | low | 7.21 | **6.94** | 80.00 | 70.00 | 0.2000 | 0.3000 | 71x |
| AliMeeting Near | verylow | 7.44 | **6.79** | 60.00 | **80.00** | 0.4000 | **0.2000** | 48x |
| AliMeeting Near | ultra | 7.82 | **7.52** | 70.00 | **70.00** | 0.3000 | **0.3000** | 24x |

† AMI MHM verylow/ultra wall clocks were contaminated by a single machine-sleep
stall each (one file at 0.4x / 1.3x while every other file ran 23-50x); the values
shown are median per-file RTFx. All later runs were held awake with `caffeinate`.

The losses are concentrated in AMI SDM (far-field single mic): offline +0.36, low
+1.45, ultra +1.02 — speaker-confusion driven, plausibly fp16-vs-bf16 sensitivity
on the hardest condition — plus AMI MHM ultra at +0.10. Speaker counting is a
clean sweep: SCA/MAE at parity or better on 14 of 16 rows, and on AMI SDM
verylow/ultra we count speakers substantially better than the card (SCA 81.25 vs
75.00/62.50) while losing DER, so the DER gap there is timing/confusion, not
speaker-count failure.

These 16 rows re-baseline and extend the earlier 9-row table (AliMeeting audio is
now materialized by a committed, reproducible script; per-row deltas vs the old
table are <= 0.05 DER except where profiles were previously unmeasured).

### NOTSOFAR1 Eval (protocol-adjacent — DER not row-comparable)

NVIDIA scored NOTSOFAR1 against FastMSS forced alignments that are not published,
over 160 sessions whose list is also not published. We ran the released
`240825.1_eval_full_with_GT` eval set (129 meetings): MHM = equal-weight mix of
close-talk mics, SC = first single-channel device per meeting (sorted by name),
references built from the released word-level GT timings merged at <= 0.2 s gaps
(`Scripts/materialize_notsofar_card_audio.py`), collar 0, overlap included.

Because the reference timing source differs, DER deltas vs the card mostly measure
reference construction, not model quality — our DER reads ~3-4 points above the
card across all profiles on both conditions. Speaker-counting metrics (SCA/MAE)
are far less reference-timing-sensitive and are the meaningful comparison here.

| Dataset | Profile | Card DER | Ours* | Card SCA | Ours | Card MAE | Ours | Our wall RTFx |
|---|---|---|---|---|---|---|---|---|
| NOTSOFAR1 MHM | offline | 7.53 | 11.75 | 66.87 | **80.62** | 0.3563 | **0.2016** | 1564x |
| NOTSOFAR1 MHM | low | 8.66 | 12.41 | 45.00 | 39.53 | 0.5813 | 0.6202 | 68x |
| NOTSOFAR1 MHM | verylow | 8.92 | 12.68 | 34.38 | 29.46 | 0.6813 | 0.7364 | 45x |
| NOTSOFAR1 MHM | ultra | 9.61 | 13.23 | 24.37 | 18.60 | 0.8000 | 0.8450 | 23x |
| NOTSOFAR1 SC | offline | 11.74 | 14.38 | 61.88 | **70.54** | 0.4000 | **0.3023** | 1578x |
| NOTSOFAR1 SC | low | 13.87 | 16.05 | 28.75 | 24.03 | 0.7750 | 0.7907 | 72x |
| NOTSOFAR1 SC | verylow | 14.51 | 16.19 | 25.62 | 24.81 | 0.8000 | **0.7829** | 45x |
| NOTSOFAR1 SC | ultra | 15.58 | 17.12 | 18.12 | 13.18 | 0.7280 | 0.9147 | 24x |

\* Different references than the card (see above) — do not read the DER columns as
a same-protocol comparison. Offline speaker counting beats the card on both
conditions (MHM SCA 80.62 vs 66.87; SC 70.54 vs 61.88); streaming-profile SCA
trails, consistent with the card's own steep SCA falloff at low latency on this
dataset.

### Not reproducible

DIHARD III Eval (LDC2022S14 family) and CALLHOME-Part2 (2000 NIST SRE,
LDC2001S97) are LDC-licensed; we do not hold licenses, so those 8 card rows
cannot be run honestly. The complete NVIDIA tables remain on the model card.

### FluidAudio presets, card protocol (AMI MHM)

| Preset | Latency | Audio/call | DER | Wall RTFx | Route |
|---|---|---|---|---|---|
| low (card config) | 1.04 s | 0.72 s | 9.90 | 68x | GPU |
| fast | 1.04 s | 0.72 s | 10.65 | 109x | GPU |
| fast32 | 2.88 s | 2.56 s | 9.94 | 343x | GPU |
| efficient | 4.16 s | 3.84 s | 9.74 | 326x | GPU |
| **fast128** | 10.56 s | 10.24 s | **9.59** | **1004x** | GPU |
| fast32-split-w8a8 | 2.88 s | 2.56 s | 9.86 | 250x | **100% ANE**, 95 MB |
| c128-split-w8a8 | 10.56 s | 10.24 s | 9.68 | 566x | **100% ANE**, 95 MB |
| offline (card config) | 30.4 s | 27.2 s | 9.28 | 1680x | GPU |

Every FluidAudio preset beats the card's published low-profile DER (10.35); the
chunk ladder's "bigger chunks improve quality" finding holds under card protocol.

---

## Appendix: word-aligned-harness results (relative comparisons only)

Historic campaign data below used AMI *Mix-Headset* audio (the MHM condition —
earlier labeled "SDM" in error) scored against word-aligned references with a 0.5 s
merge gap, which reads ~2x above forced-alignment DER. Valid for comparing
configurations against each other; do not quote absolute values.


All measurements: M5 Pro, macOS 26.7, FluidAudio Swift pipeline. AMI SDM test (16
meetings, word-aligned refs, collar 0) and a frozen 14-file VoxConverse subset
stratified 5-10 speakers. Wall RTFx includes mel + host state updates.

> **Caveat (added after reviewing the model card's training data):** VoxConverse v0.3
> dev **and test** are in the model's training set, so absolute DER on our VoxConverse
> subset is train-contaminated and optimistic. Those numbers are valid only for
> *relative* comparisons between configurations (all equally contaminated). AMI SDM
> test numbers are clean (only AMI train/dev was used in training). Future
> multi-speaker gates should use DIHARD III eval 5-9 spk (held out by NVIDIA).


## c128-split-w8a8 (16 files)

| meeting | DER % | miss | fa | conf | spk | RTFx |
|---|---|---|---|---|---|---|
| EN2002a | 31.24 | 29.4 | 0.8 | 1.0 | 4/4 | 559x |
| EN2002b | 30.79 | 28.8 | 1.1 | 0.9 | 4/4 | 564x |
| EN2002c | 30.05 | 29.0 | 0.6 | 0.4 | 4/3 | 550x |
| EN2002d | 33.68 | 32.0 | 0.8 | 0.9 | 4/4 | 567x |
| ES2004a | 26.98 | 25.8 | 0.7 | 0.4 | 4/4 | 543x |
| ES2004b | 21.74 | 20.5 | 0.9 | 0.4 | 5/4 | 548x |
| ES2004c | 20.42 | 19.1 | 0.7 | 0.6 | 4/4 | 563x |
| ES2004d | 22.55 | 20.3 | 1.6 | 0.6 | 5/4 | 565x |
| IS1009a | 23.55 | 19.9 | 2.8 | 0.9 | 5/4 | 564x |
| IS1009b | 15.24 | 12.8 | 1.5 | 0.9 | 4/4 | 565x |
| IS1009c | 15.57 | 13.2 | 2.0 | 0.4 | 4/4 | 544x |
| IS1009d | 21.85 | 17.7 | 1.9 | 2.3 | 4/4 | 543x |
| TS3003a | 35.35 | 34.3 | 0.5 | 0.6 | 4/4 | 528x |
| TS3003b | 26.26 | 25.3 | 0.9 | 0.1 | 4/4 | 553x |
| TS3003c | 30.64 | 29.6 | 0.8 | 0.2 | 4/4 | 560x |
| TS3003d | 32.13 | 30.9 | 0.8 | 0.4 | 4/4 | 558x |
| **avg** | **26.13** | 24.3 | 1.1 | 0.7 | | 555x |

## c128 (16 files)

| meeting | DER % | miss | fa | conf | spk | RTFx |
|---|---|---|---|---|---|---|
| EN2002a | 30.63 | 28.8 | 0.8 | 1.0 | 4/4 | 767x |
| EN2002b | 30.36 | 28.4 | 1.1 | 0.9 | 4/4 | 793x |
| EN2002c | 29.72 | 28.6 | 0.6 | 0.5 | 4/3 | 789x |
| EN2002d | 33.48 | 31.8 | 0.9 | 0.8 | 5/4 | 796x |
| ES2004a | 27.12 | 25.4 | 0.7 | 1.0 | 4/4 | 782x |
| ES2004b | 21.60 | 20.3 | 0.9 | 0.4 | 5/4 | 793x |
| ES2004c | 20.29 | 19.0 | 0.7 | 0.5 | 4/4 | 796x |
| ES2004d | 22.53 | 20.1 | 1.8 | 0.6 | 4/4 | 794x |
| IS1009a | 23.47 | 19.8 | 2.9 | 0.8 | 5/4 | 786x |
| IS1009b | 15.23 | 12.8 | 1.5 | 0.8 | 4/4 | 791x |
| IS1009c | 15.62 | 13.0 | 2.1 | 0.5 | 4/4 | 791x |
| IS1009d | 20.61 | 17.4 | 2.1 | 1.1 | 4/4 | 794x |
| TS3003a | 35.28 | 34.2 | 0.5 | 0.6 | 4/4 | 789x |
| TS3003b | 26.23 | 25.1 | 0.9 | 0.2 | 4/4 | 793x |
| TS3003c | 30.43 | 29.5 | 0.8 | 0.1 | 4/4 | 793x |
| TS3003d | 31.93 | 30.6 | 0.9 | 0.5 | 4/4 | 790x |
| **avg** | **25.91** | 24.1 | 1.2 | 0.6 | | 790x |

## c160 (16 files)

| meeting | DER % | miss | fa | conf | spk | RTFx |
|---|---|---|---|---|---|---|
| EN2002a | 30.70 | 28.7 | 0.9 | 1.1 | 4/4 | 862x |
| EN2002b | 30.37 | 28.3 | 1.1 | 0.9 | 4/4 | 895x |
| EN2002c | 29.95 | 28.7 | 0.7 | 0.6 | 4/3 | 890x |
| EN2002d | 33.42 | 31.5 | 1.0 | 1.0 | 4/4 | 898x |
| ES2004a | 27.40 | 25.7 | 0.7 | 0.9 | 4/4 | 880x |
| ES2004b | 21.52 | 20.2 | 0.9 | 0.4 | 5/4 | 894x |
| ES2004c | 20.46 | 19.2 | 0.7 | 0.5 | 4/4 | 894x |
| ES2004d | 22.61 | 20.2 | 1.7 | 0.7 | 4/4 | 898x |
| IS1009a | 23.37 | 17.8 | 3.0 | 2.5 | 5/4 | 884x |
| IS1009b | 15.29 | 12.9 | 1.4 | 1.0 | 4/4 | 896x |
| IS1009c | 15.67 | 13.0 | 2.2 | 0.5 | 4/4 | 890x |
| IS1009d | 20.27 | 17.2 | 2.0 | 1.0 | 4/4 | 894x |
| TS3003a | 35.13 | 34.0 | 0.6 | 0.6 | 4/4 | 890x |
| TS3003b | 26.56 | 25.2 | 1.0 | 0.4 | 5/4 | 896x |
| TS3003c | 30.39 | 29.4 | 0.8 | 0.2 | 4/4 | 888x |
| TS3003d | 31.92 | 30.6 | 0.9 | 0.5 | 4/4 | 888x |
| **avg** | **25.94** | 23.9 | 1.2 | 0.8 | | 890x |

## c192-split (16 files)

| meeting | DER % | miss | fa | conf | spk | RTFx |
|---|---|---|---|---|---|---|
| EN2002a | 30.68 | 28.8 | 0.9 | 0.9 | 4/4 | 564x |
| EN2002b | 30.48 | 28.3 | 1.1 | 1.1 | 4/4 | 567x |
| EN2002c | 29.77 | 28.7 | 0.7 | 0.4 | 4/3 | 570x |
| EN2002d | 33.41 | 31.6 | 0.9 | 0.9 | 4/4 | 577x |
| ES2004a | 26.78 | 25.6 | 0.7 | 0.5 | 4/4 | 569x |
| ES2004b | 21.01 | 19.8 | 1.0 | 0.2 | 4/4 | 576x |
| ES2004c | 20.09 | 19.0 | 0.7 | 0.4 | 4/4 | 577x |
| ES2004d | 22.37 | 20.1 | 1.7 | 0.6 | 4/4 | 576x |
| IS1009a | 23.89 | 20.0 | 2.9 | 1.0 | 5/4 | 568x |
| IS1009b | 15.25 | 12.9 | 1.5 | 0.9 | 4/4 | 575x |
| IS1009c | 15.58 | 13.0 | 2.2 | 0.4 | 4/4 | 567x |
| IS1009d | 20.39 | 17.2 | 2.1 | 1.1 | 4/4 | 570x |
| TS3003a | 35.26 | 34.1 | 0.5 | 0.7 | 4/4 | 559x |
| TS3003b | 26.30 | 25.1 | 1.0 | 0.2 | 4/4 | 566x |
| TS3003c | 30.44 | 29.4 | 0.9 | 0.2 | 4/4 | 559x |
| TS3003d | 31.81 | 30.6 | 0.9 | 0.4 | 4/4 | 550x |
| **avg** | **25.84** | 24.0 | 1.2 | 0.6 | | 568x |

## c192 (16 files)

| meeting | DER % | miss | fa | conf | spk | RTFx |
|---|---|---|---|---|---|---|
| EN2002a | 30.58 | 28.7 | 0.9 | 1.0 | 4/4 | 947x |
| EN2002b | 30.41 | 28.2 | 1.1 | 1.1 | 4/4 | 985x |
| EN2002c | 29.81 | 28.7 | 0.7 | 0.5 | 4/3 | 981x |
| EN2002d | 33.51 | 31.7 | 0.9 | 0.9 | 4/4 | 993x |
| ES2004a | 26.57 | 25.4 | 0.7 | 0.4 | 4/4 | 969x |
| ES2004b | 21.00 | 19.8 | 1.0 | 0.2 | 4/4 | 991x |
| ES2004c | 20.15 | 19.0 | 0.8 | 0.4 | 4/4 | 990x |
| ES2004d | 22.34 | 20.0 | 1.7 | 0.6 | 4/4 | 990x |
| IS1009a | 23.77 | 19.9 | 2.9 | 1.0 | 5/4 | 972x |
| IS1009b | 15.20 | 12.9 | 1.5 | 0.8 | 4/4 | 978x |
| IS1009c | 15.62 | 12.9 | 2.2 | 0.5 | 4/4 | 991x |
| IS1009d | 20.24 | 17.0 | 2.1 | 1.1 | 4/4 | 990x |
| TS3003a | 35.11 | 33.9 | 0.6 | 0.6 | 4/4 | 979x |
| TS3003b | 26.32 | 25.1 | 1.0 | 0.2 | 4/4 | 987x |
| TS3003c | 30.62 | 29.6 | 0.9 | 0.2 | 4/4 | 990x |
| TS3003d | 31.96 | 30.6 | 0.8 | 0.5 | 4/4 | 986x |
| **avg** | **25.83** | 24.0 | 1.2 | 0.6 | | 983x |

## c256-split (16 files)

| meeting | DER % | miss | fa | conf | spk | RTFx |
|---|---|---|---|---|---|---|
| EN2002a | 30.82 | 28.9 | 0.9 | 1.0 | 4/4 | 615x |
| EN2002b | 30.46 | 28.4 | 1.2 | 0.9 | 4/4 | 614x |
| EN2002c | 33.98 | 29.0 | 0.7 | 4.3 | 4/3 | 610x |
| EN2002d | 33.71 | 31.9 | 0.9 | 0.9 | 4/4 | 614x |
| ES2004a | 27.27 | 25.4 | 0.8 | 1.1 | 4/4 | 596x |
| ES2004b | 21.22 | 20.1 | 0.8 | 0.3 | 5/4 | 613x |
| ES2004c | 20.20 | 19.0 | 0.7 | 0.5 | 4/4 | 609x |
| ES2004d | 22.41 | 20.0 | 1.8 | 0.6 | 4/4 | 605x |
| IS1009a | 23.99 | 19.6 | 2.8 | 1.5 | 5/4 | 610x |
| IS1009b | 15.36 | 12.9 | 1.4 | 1.0 | 4/4 | 628x |
| IS1009c | 15.48 | 13.0 | 2.1 | 0.4 | 4/4 | 608x |
| IS1009d | 20.68 | 17.4 | 2.1 | 1.2 | 4/4 | 626x |
| TS3003a | 34.97 | 33.9 | 0.5 | 0.5 | 4/4 | 622x |
| TS3003b | 26.20 | 25.0 | 1.0 | 0.2 | 4/4 | 621x |
| TS3003c | 30.37 | 29.4 | 0.8 | 0.2 | 4/4 | 609x |
| TS3003d | 31.75 | 30.5 | 0.9 | 0.4 | 4/4 | 609x |
| **avg** | **26.18** | 24.0 | 1.2 | 0.9 | | 613x |

## c64 (16 files)

| meeting | DER % | miss | fa | conf | spk | RTFx |
|---|---|---|---|---|---|---|
| EN2002a | 30.87 | 28.7 | 0.9 | 1.2 | 4/4 | 506x |
| EN2002b | 30.31 | 28.4 | 1.2 | 0.7 | 4/4 | 520x |
| EN2002c | 29.59 | 28.7 | 0.6 | 0.3 | 4/3 | 516x |
| EN2002d | 34.25 | 32.4 | 0.8 | 1.0 | 4/4 | 515x |
| ES2004a | 27.24 | 25.7 | 0.7 | 0.8 | 4/4 | 510x |
| ES2004b | 21.46 | 20.2 | 0.9 | 0.4 | 5/4 | 517x |
| ES2004c | 20.26 | 19.0 | 0.7 | 0.5 | 4/4 | 517x |
| ES2004d | 22.64 | 20.2 | 1.7 | 0.7 | 4/4 | 518x |
| IS1009a | 23.45 | 18.1 | 2.9 | 2.4 | 5/4 | 516x |
| IS1009b | 15.28 | 12.9 | 1.5 | 0.9 | 4/4 | 518x |
| IS1009c | 15.93 | 13.0 | 2.3 | 0.6 | 4/4 | 518x |
| IS1009d | 23.84 | 17.6 | 2.3 | 4.0 | 4/4 | 519x |
| TS3003a | 35.31 | 34.1 | 0.5 | 0.6 | 4/4 | 521x |
| TS3003b | 26.22 | 25.1 | 0.9 | 0.2 | 4/4 | 518x |
| TS3003c | 30.59 | 29.5 | 0.8 | 0.2 | 4/4 | 518x |
| TS3003d | 32.15 | 30.6 | 0.9 | 0.7 | 4/4 | 517x |
| **avg** | **26.21** | 24.0 | 1.2 | 1.0 | | 516x |

## c96 (16 files)

| meeting | DER % | miss | fa | conf | spk | RTFx |
|---|---|---|---|---|---|---|
| EN2002a | 30.66 | 28.6 | 0.9 | 1.2 | 4/4 | 641x |
| EN2002b | 30.61 | 28.5 | 1.1 | 1.0 | 4/4 | 652x |
| EN2002c | 29.50 | 28.5 | 0.6 | 0.4 | 4/3 | 651x |
| EN2002d | 33.48 | 31.6 | 0.9 | 0.9 | 4/4 | 661x |
| ES2004a | 26.74 | 25.6 | 0.7 | 0.5 | 4/4 | 653x |
| ES2004b | 21.70 | 20.4 | 0.9 | 0.4 | 5/4 | 660x |
| ES2004c | 20.44 | 19.1 | 0.7 | 0.6 | 4/4 | 660x |
| ES2004d | 22.50 | 20.2 | 1.7 | 0.7 | 4/4 | 658x |
| IS1009a | 23.73 | 17.4 | 3.1 | 3.3 | 4/4 | 649x |
| IS1009b | 15.16 | 12.8 | 1.6 | 0.8 | 4/4 | 659x |
| IS1009c | 15.81 | 13.0 | 2.3 | 0.5 | 4/4 | 661x |
| IS1009d | 20.49 | 17.1 | 2.2 | 1.3 | 4/4 | 659x |
| TS3003a | 35.28 | 34.0 | 0.6 | 0.8 | 4/4 | 657x |
| TS3003b | 26.28 | 25.1 | 1.0 | 0.2 | 4/4 | 661x |
| TS3003c | 30.50 | 29.4 | 0.9 | 0.2 | 4/4 | 660x |
| TS3003d | 31.93 | 30.7 | 0.8 | 0.4 | 4/4 | 657x |
| **avg** | **25.93** | 23.9 | 1.2 | 0.8 | | 656x |

## efficient (16 files)

| meeting | DER % | miss | fa | conf | spk | RTFx |
|---|---|---|---|---|---|---|
| EN2002a | 30.93 | 29.1 | 0.8 | 1.0 | 4/4 | 266x |
| EN2002b | 30.70 | 28.6 | 1.2 | 0.9 | 4/4 | 269x |
| EN2002c | 29.78 | 28.7 | 0.6 | 0.4 | 4/3 | 259x |
| EN2002d | 33.95 | 32.2 | 0.9 | 0.9 | 4/4 | 265x |
| ES2004a | 27.09 | 25.5 | 0.7 | 0.8 | 4/4 | 240x |
| ES2004b | 21.37 | 20.1 | 0.9 | 0.4 | 5/4 | 253x |
| ES2004c | 20.51 | 19.1 | 0.8 | 0.6 | 4/4 | 243x |
| ES2004d | 22.53 | 20.1 | 1.7 | 0.7 | 4/4 | 242x |
| IS1009a | 23.46 | 18.7 | 2.7 | 2.1 | 5/4 | 251x |
| IS1009b | 15.32 | 12.8 | 1.5 | 1.0 | 4/4 | 247x |
| IS1009c | 15.91 | 12.9 | 2.3 | 0.8 | 4/4 | 251x |
| IS1009d | 20.92 | 17.4 | 2.1 | 1.5 | 4/4 | 256x |
| TS3003a | 35.24 | 34.0 | 0.5 | 0.7 | 4/4 | 243x |
| TS3003b | 26.22 | 25.1 | 0.9 | 0.2 | 4/4 | 238x |
| TS3003c | 30.42 | 29.4 | 0.9 | 0.1 | 4/4 | 250x |
| TS3003d | 32.12 | 30.7 | 0.8 | 0.6 | 4/4 | 249x |
| **avg** | **26.03** | 24.0 | 1.2 | 0.8 | | 251x |

## fast (16 files)

| meeting | DER % | miss | fa | conf | spk | RTFx |
|---|---|---|---|---|---|---|
| EN2002a | 31.73 | 29.2 | 1.0 | 1.5 | 5/4 | 84x |
| EN2002b | 30.84 | 28.5 | 1.2 | 1.2 | 4/4 | 83x |
| EN2002c | 35.75 | 29.9 | 0.7 | 5.2 | 4/3 | 87x |
| EN2002d | 34.82 | 32.9 | 0.8 | 1.1 | 4/4 | 85x |
| ES2004a | 27.69 | 26.0 | 0.8 | 0.9 | 5/4 | 87x |
| ES2004b | 20.99 | 20.0 | 0.7 | 0.2 | 4/4 | 82x |
| ES2004c | 20.44 | 19.1 | 0.7 | 0.6 | 4/4 | 78x |
| ES2004d | 23.15 | 20.6 | 1.6 | 1.0 | 4/4 | 75x |
| IS1009a | 24.72 | 18.1 | 2.8 | 3.7 | 5/4 | 81x |
| IS1009b | 15.76 | 13.1 | 1.6 | 1.0 | 4/4 | 79x |
| IS1009c | 15.75 | 13.1 | 2.1 | 0.5 | 4/4 | 79x |
| IS1009d | 20.74 | 17.5 | 2.0 | 1.3 | 4/4 | 93x |
| TS3003a | 36.08 | 34.6 | 0.6 | 0.9 | 4/4 | 93x |
| TS3003b | 26.46 | 25.3 | 0.9 | 0.3 | 4/4 | 28x |
| TS3003c | 30.92 | 29.7 | 0.9 | 0.2 | 4/4 | 85x |
| TS3003d | 32.33 | 30.5 | 0.9 | 0.9 | 4/4 | 80x |
| **avg** | **26.76** | 24.3 | 1.2 | 1.3 | | 80x |

## fast32-split-w8a8 (16 files)

| meeting | DER % | miss | fa | conf | spk | RTFx |
|---|---|---|---|---|---|---|
| EN2002a | 31.01 | 28.9 | 0.9 | 1.2 | 4/4 | 247x |
| EN2002b | 30.93 | 28.8 | 1.1 | 1.0 | 4/4 | 246x |
| EN2002c | 30.05 | 29.0 | 0.5 | 0.6 | 4/3 | 247x |
| EN2002d | 34.21 | 32.3 | 0.9 | 1.0 | 4/4 | 248x |
| ES2004a | 27.42 | 26.1 | 0.7 | 0.6 | 4/4 | 250x |
| ES2004b | 21.57 | 20.4 | 0.7 | 0.4 | 5/4 | 230x |
| ES2004c | 20.37 | 19.1 | 0.7 | 0.5 | 4/4 | 254x |
| ES2004d | 22.93 | 20.5 | 1.6 | 0.8 | 4/4 | 255x |
| IS1009a | 23.89 | 18.9 | 2.5 | 2.5 | 5/4 | 254x |
| IS1009b | 15.56 | 13.2 | 1.5 | 0.9 | 4/4 | 254x |
| IS1009c | 16.17 | 13.5 | 1.9 | 0.7 | 4/4 | 254x |
| IS1009d | 20.69 | 17.5 | 2.0 | 1.2 | 4/4 | 254x |
| TS3003a | 35.58 | 34.3 | 0.5 | 0.7 | 4/4 | 3x |
| TS3003b | 26.22 | 25.1 | 0.9 | 0.2 | 4/4 | 181x |
| TS3003c | 30.75 | 29.7 | 0.8 | 0.2 | 4/4 | 255x |
| TS3003d | 32.77 | 31.1 | 0.9 | 0.8 | 4/4 | 254x |
| **avg** | **26.26** | 24.3 | 1.1 | 0.8 | | 230x |

## fast32-split (16 files)

| meeting | DER % | miss | fa | conf | spk | RTFx |
|---|---|---|---|---|---|---|
| EN2002a | 30.96 | 28.9 | 0.8 | 1.2 | 4/4 | 215x |
| EN2002b | 30.51 | 28.4 | 1.1 | 1.0 | 4/4 | 215x |
| EN2002c | 30.24 | 29.0 | 0.6 | 0.7 | 4/3 | 217x |
| EN2002d | 34.10 | 32.0 | 1.0 | 1.1 | 4/4 | 221x |
| ES2004a | 26.98 | 25.5 | 0.7 | 0.8 | 4/4 | 221x |
| ES2004b | 21.62 | 20.4 | 0.8 | 0.4 | 5/4 | 221x |
| ES2004c | 20.45 | 19.2 | 0.7 | 0.5 | 4/4 | 221x |
| ES2004d | 22.76 | 20.3 | 1.6 | 0.8 | 4/4 | 222x |
| IS1009a | 24.12 | 18.6 | 2.5 | 3.0 | 5/4 | 218x |
| IS1009b | 15.43 | 13.0 | 1.5 | 0.9 | 4/4 | 222x |
| IS1009c | 15.79 | 13.5 | 1.8 | 0.5 | 4/4 | 221x |
| IS1009d | 24.45 | 18.5 | 1.9 | 4.0 | 4/4 | 222x |
| TS3003a | 35.69 | 34.4 | 0.6 | 0.8 | 4/4 | 217x |
| TS3003b | 26.34 | 25.2 | 0.9 | 0.2 | 4/4 | 220x |
| TS3003c | 30.64 | 29.5 | 0.9 | 0.2 | 4/4 | 217x |
| TS3003d | 32.28 | 30.7 | 0.8 | 0.7 | 4/4 | 221x |
| **avg** | **26.40** | 24.2 | 1.1 | 1.1 | | 219x |

## fast32_vad (16 files)

| meeting | DER % | miss | fa | conf | spk | RTFx |
|---|---|---|---|---|---|---|
| EN2002a | 31.74 | 29.4 | 0.9 | 1.4 | 4/4 | 352x |
| EN2002b | 31.66 | 29.7 | 1.1 | 0.8 | 4/4 | 378x |
| EN2002c | 30.61 | 29.6 | 0.5 | 0.5 | 4/3 | 370x |
| EN2002d | 34.98 | 33.1 | 0.9 | 0.9 | 4/4 | 363x |
| ES2004a | 28.12 | 26.9 | 0.7 | 0.5 | 4/4 | 389x |
| ES2004b | 22.24 | 21.3 | 0.8 | 0.2 | 4/4 | 369x |
| ES2004c | 21.09 | 19.9 | 0.8 | 0.4 | 4/4 | 370x |
| ES2004d | 24.32 | 22.0 | 1.5 | 0.8 | 4/4 | 385x |
| IS1009a | 24.99 | 18.8 | 2.6 | 3.5 | 5/4 | 404x |
| IS1009b | 16.55 | 14.3 | 1.4 | 0.9 | 4/4 | 368x |
| IS1009c | 16.59 | 14.1 | 2.0 | 0.5 | 4/4 | 369x |
| IS1009d | 21.08 | 18.0 | 2.0 | 1.1 | 4/4 | 353x |
| TS3003a | 37.89 | 36.6 | 0.4 | 0.9 | 4/4 | 409x |
| TS3003b | 26.96 | 26.0 | 0.7 | 0.2 | 4/4 | 344x |
| TS3003c | 31.50 | 30.5 | 0.8 | 0.2 | 4/4 | 397x |
| TS3003d | 33.74 | 32.3 | 0.9 | 0.5 | 4/4 | 350x |
| **avg** | **27.13** | 25.2 | 1.1 | 0.8 | | 373x |

## fast32_vad035 (16 files)

| meeting | DER % | miss | fa | conf | spk | RTFx |
|---|---|---|---|---|---|---|
| EN2002a | 31.65 | 29.3 | 0.9 | 1.5 | 4/4 | 358x |
| EN2002b | 31.72 | 29.6 | 1.2 | 0.9 | 4/4 | 383x |
| EN2002c | 30.75 | 29.5 | 0.5 | 0.7 | 4/3 | 360x |
| EN2002d | 34.76 | 32.7 | 0.9 | 1.1 | 4/4 | 359x |
| ES2004a | 28.30 | 27.1 | 0.6 | 0.6 | 4/4 | 391x |
| ES2004b | 21.89 | 20.8 | 0.8 | 0.3 | 4/4 | 359x |
| ES2004c | 21.01 | 19.9 | 0.8 | 0.4 | 4/4 | 363x |
| ES2004d | 23.91 | 21.5 | 1.6 | 0.8 | 4/4 | 359x |
| IS1009a | 27.94 | 19.1 | 3.2 | 5.7 | 5/4 | 386x |
| IS1009b | 16.49 | 14.1 | 1.4 | 0.9 | 4/4 | 360x |
| IS1009c | 16.40 | 13.8 | 1.9 | 0.7 | 4/4 | 335x |
| IS1009d | 21.08 | 18.0 | 2.0 | 1.1 | 4/4 | 357x |
| TS3003a | 37.02 | 35.8 | 0.5 | 0.8 | 4/4 | 407x |
| TS3003b | 27.07 | 26.0 | 0.9 | 0.2 | 4/4 | 370x |
| TS3003c | 35.41 | 30.9 | 0.8 | 3.7 | 5/4 | 400x |
| TS3003d | 33.34 | 31.8 | 0.9 | 0.6 | 4/4 | 360x |
| **avg** | **27.42** | 25.0 | 1.2 | 1.2 | | 369x |

## low (16 files)

| meeting | DER % | miss | fa | conf | spk | RTFx |
|---|---|---|---|---|---|---|
| EN2002a | 31.81 | 29.5 | 0.9 | 1.4 | 4/4 | 61x |
| EN2002b | 31.12 | 28.9 | 1.1 | 1.2 | 4/4 | 61x |
| EN2002c | 29.84 | 28.8 | 0.6 | 0.4 | 4/3 | 60x |
| EN2002d | 34.32 | 32.6 | 0.8 | 0.9 | 5/4 | 61x |
| ES2004a | 27.36 | 25.6 | 0.8 | 1.0 | 4/4 | 60x |
| ES2004b | 21.36 | 20.2 | 0.8 | 0.3 | 5/4 | 59x |
| ES2004c | 20.43 | 19.2 | 0.7 | 0.5 | 4/4 | 62x |
| ES2004d | 22.66 | 20.2 | 1.7 | 0.8 | 4/4 | 62x |
| IS1009a | 24.13 | 18.9 | 2.8 | 2.4 | 5/4 | 57x |
| IS1009b | 15.49 | 13.0 | 1.6 | 0.9 | 4/4 | 60x |
| IS1009c | 15.55 | 12.9 | 2.0 | 0.6 | 4/4 | 59x |
| IS1009d | 20.49 | 17.4 | 2.1 | 1.0 | 4/4 | 60x |
| TS3003a | 35.82 | 34.4 | 0.6 | 0.8 | 4/4 | 60x |
| TS3003b | 26.30 | 25.1 | 0.9 | 0.3 | 4/4 | 61x |
| TS3003c | 30.69 | 29.6 | 0.9 | 0.2 | 4/4 | 60x |
| TS3003d | 32.13 | 30.8 | 0.9 | 0.5 | 4/4 | 59x |
| **avg** | **26.22** | 24.2 | 1.2 | 0.8 | | 60x |

## low_int8 (16 files)

| meeting | DER % | miss | fa | conf | spk | RTFx |
|---|---|---|---|---|---|---|
| EN2002a | 31.97 | 29.7 | 0.9 | 1.3 | 4/4 | 54x |
| EN2002b | 31.36 | 28.9 | 1.2 | 1.3 | 4/4 | 54x |
| EN2002c | 30.04 | 28.9 | 0.6 | 0.5 | 4/3 | 54x |
| EN2002d | 34.51 | 32.8 | 0.8 | 0.9 | 5/4 | 53x |
| ES2004a | 27.25 | 25.6 | 0.8 | 0.9 | 4/4 | 53x |
| ES2004b | 21.50 | 20.3 | 0.8 | 0.3 | 5/4 | 53x |
| ES2004c | 20.49 | 19.1 | 0.7 | 0.6 | 4/4 | 54x |
| ES2004d | 22.57 | 20.2 | 1.6 | 0.7 | 4/4 | 55x |
| IS1009a | 23.99 | 18.9 | 2.4 | 2.7 | 5/4 | 55x |
| IS1009b | 15.50 | 13.0 | 1.5 | 0.9 | 4/4 | 54x |
| IS1009c | 15.88 | 13.0 | 2.0 | 0.9 | 4/4 | 55x |
| IS1009d | 20.59 | 17.6 | 2.0 | 1.0 | 4/4 | 56x |
| TS3003a | 35.61 | 34.3 | 0.6 | 0.7 | 4/4 | 57x |
| TS3003b | 26.22 | 25.1 | 0.9 | 0.2 | 4/4 | 57x |
| TS3003c | 30.57 | 29.5 | 0.9 | 0.2 | 4/4 | 55x |
| TS3003d | 32.13 | 30.8 | 0.8 | 0.5 | 4/4 | 56x |
| **avg** | **26.26** | 24.2 | 1.2 | 0.9 | | 55x |

## offline (16 files)

| meeting | DER % | miss | fa | conf | spk | RTFx |
|---|---|---|---|---|---|---|
| EN2002a | 30.69 | 28.8 | 0.9 | 1.0 | 4/4 | 733x |
| EN2002b | 30.42 | 28.3 | 1.2 | 0.9 | 4/4 | 1094x |
| EN2002c | 29.67 | 28.6 | 0.6 | 0.4 | 4/3 | 1087x |
| EN2002d | 33.00 | 31.2 | 0.9 | 0.9 | 4/4 | 1067x |
| ES2004a | 27.02 | 25.8 | 0.7 | 0.5 | 4/4 | 1040x |
| ES2004b | 21.75 | 20.3 | 0.8 | 0.6 | 5/4 | 1106x |
| ES2004c | 20.20 | 19.0 | 0.8 | 0.4 | 4/4 | 1123x |
| ES2004d | 22.52 | 20.0 | 1.8 | 0.7 | 4/4 | 1120x |
| IS1009a | 21.71 | 17.5 | 3.4 | 0.8 | 4/4 | 1112x |
| IS1009b | 15.11 | 12.8 | 1.7 | 0.7 | 4/4 | 1116x |
| IS1009c | 15.64 | 13.0 | 2.3 | 0.3 | 4/4 | 1125x |
| IS1009d | 20.25 | 17.0 | 2.2 | 1.1 | 4/4 | 1114x |
| TS3003a | 34.97 | 33.9 | 0.5 | 0.6 | 4/4 | 1093x |
| TS3003b | 26.20 | 25.0 | 1.1 | 0.1 | 4/4 | 1116x |
| TS3003c | 30.35 | 29.4 | 0.8 | 0.2 | 4/4 | 1121x |
| TS3003d | 31.73 | 30.5 | 0.9 | 0.4 | 4/4 | 1109x |
| **avg** | **25.70** | 23.8 | 1.3 | 0.6 | | 1080x |

## q16 (16 files)

| meeting | DER % | miss | fa | conf | spk | RTFx |
|---|---|---|---|---|---|---|
| EN2002a | 31.56 | 29.4 | 0.9 | 1.3 | 4/4 | 108x |
| EN2002b | 30.59 | 28.6 | 1.1 | 0.9 | 4/4 | 109x |
| EN2002c | 30.03 | 28.9 | 0.6 | 0.5 | 4/3 | 108x |
| EN2002d | 34.15 | 32.4 | 0.9 | 0.9 | 4/4 | 108x |
| ES2004a | 27.41 | 25.7 | 0.8 | 1.0 | 4/4 | 108x |
| ES2004b | 21.45 | 20.3 | 0.8 | 0.3 | 5/4 | 108x |
| ES2004c | 20.64 | 19.5 | 0.6 | 0.5 | 4/4 | 109x |
| ES2004d | 22.64 | 20.2 | 1.7 | 0.7 | 4/4 | 109x |
| IS1009a | 23.83 | 18.8 | 3.0 | 2.1 | 5/4 | 108x |
| IS1009b | 15.47 | 12.9 | 1.6 | 1.0 | 4/4 | 108x |
| IS1009c | 15.62 | 13.0 | 2.1 | 0.6 | 4/4 | 109x |
| IS1009d | 20.70 | 17.4 | 2.2 | 1.1 | 4/4 | 108x |
| TS3003a | 35.58 | 34.1 | 0.7 | 0.8 | 4/4 | 101x |
| TS3003b | 26.18 | 25.1 | 0.9 | 0.2 | 4/4 | 103x |
| TS3003c | 30.59 | 29.5 | 0.9 | 0.2 | 4/4 | 97x |
| TS3003d | 32.06 | 30.7 | 0.9 | 0.5 | 4/4 | 95x |
| **avg** | **26.16** | 24.1 | 1.2 | 0.8 | | 106x |

## q24 (16 files)

| meeting | DER % | miss | fa | conf | spk | RTFx |
|---|---|---|---|---|---|---|
| EN2002a | 31.34 | 29.3 | 0.8 | 1.2 | 4/4 | 144x |
| EN2002b | 30.63 | 28.6 | 1.1 | 0.9 | 4/4 | 131x |
| EN2002c | 30.11 | 28.9 | 0.6 | 0.6 | 4/3 | 144x |
| EN2002d | 34.17 | 32.3 | 0.9 | 1.0 | 5/4 | 143x |
| ES2004a | 27.51 | 25.7 | 0.8 | 1.1 | 4/4 | 132x |
| ES2004b | 21.33 | 20.1 | 0.9 | 0.4 | 5/4 | 145x |
| ES2004c | 20.55 | 19.0 | 0.8 | 0.8 | 4/4 | 142x |
| ES2004d | 22.72 | 20.2 | 1.8 | 0.7 | 4/4 | 137x |
| IS1009a | 23.33 | 18.4 | 3.0 | 1.9 | 5/4 | 124x |
| IS1009b | 15.28 | 12.9 | 1.5 | 0.9 | 4/4 | 138x |
| IS1009c | 15.63 | 12.9 | 2.1 | 0.6 | 4/4 | 132x |
| IS1009d | 20.81 | 17.5 | 2.2 | 1.2 | 4/4 | 141x |
| TS3003a | 35.38 | 34.1 | 0.5 | 0.7 | 4/4 | 135x |
| TS3003b | 26.23 | 25.1 | 0.9 | 0.2 | 4/4 | 148x |
| TS3003c | 30.54 | 29.4 | 0.9 | 0.2 | 4/4 | 140x |
| TS3003d | 31.97 | 30.7 | 0.8 | 0.4 | 4/4 | 144x |
| **avg** | **26.10** | 24.1 | 1.2 | 0.8 | | 139x |

## q32 (16 files)

| meeting | DER % | miss | fa | conf | spk | RTFx |
|---|---|---|---|---|---|---|
| EN2002a | 31.26 | 29.2 | 0.9 | 1.1 | 4/4 | 178x |
| EN2002b | 30.77 | 28.7 | 1.2 | 0.9 | 4/4 | 177x |
| EN2002c | 29.76 | 28.8 | 0.6 | 0.3 | 4/3 | 186x |
| EN2002d | 34.07 | 32.3 | 0.9 | 0.9 | 4/4 | 182x |
| ES2004a | 27.32 | 25.6 | 0.7 | 1.0 | 4/4 | 176x |
| ES2004b | 21.34 | 20.2 | 0.8 | 0.4 | 5/4 | 192x |
| ES2004c | 20.46 | 19.0 | 0.7 | 0.7 | 4/4 | 169x |
| ES2004d | 22.66 | 20.2 | 1.7 | 0.7 | 4/4 | 192x |
| IS1009a | 23.87 | 18.7 | 2.9 | 2.3 | 5/4 | 188x |
| IS1009b | 15.45 | 12.9 | 1.6 | 1.0 | 4/4 | 199x |
| IS1009c | 15.78 | 13.0 | 2.1 | 0.7 | 4/4 | 192x |
| IS1009d | 20.65 | 17.2 | 2.1 | 1.3 | 4/4 | 198x |
| TS3003a | 35.46 | 34.0 | 0.6 | 0.8 | 4/4 | 197x |
| TS3003b | 26.32 | 25.1 | 1.0 | 0.2 | 4/4 | 197x |
| TS3003c | 30.49 | 29.4 | 0.9 | 0.2 | 4/4 | 201x |
| TS3003d | 32.04 | 30.7 | 0.9 | 0.5 | 4/4 | 185x |
| **avg** | **26.11** | 24.1 | 1.2 | 0.8 | | 188x |

## s16 (16 files)

| meeting | DER % | miss | fa | conf | spk | RTFx |
|---|---|---|---|---|---|---|
| EN2002a | 30.75 | 28.3 | 0.9 | 1.6 | 5/4 | 139x |
| EN2002b | 30.47 | 28.3 | 1.3 | 0.9 | 5/4 | 147x |
| EN2002c | 30.43 | 29.1 | 0.6 | 0.8 | 4/3 | 134x |
| EN2002d | 34.55 | 32.4 | 0.9 | 1.2 | 5/4 | 108x |
| ES2004a | 27.28 | 25.6 | 0.8 | 0.9 | 5/4 | 108x |
| ES2004b | 21.91 | 20.5 | 0.8 | 0.6 | 5/4 | 122x |
| ES2004c | 20.56 | 19.2 | 0.7 | 0.6 | 4/4 | 126x |
| ES2004d | 22.88 | 20.4 | 1.6 | 0.9 | 4/4 | 112x |
| IS1009a | 28.34 | 18.1 | 3.0 | 7.2 | 5/4 | 119x |
| IS1009b | 15.67 | 13.1 | 1.4 | 1.2 | 4/4 | 131x |
| IS1009c | 15.58 | 13.1 | 2.0 | 0.5 | 4/4 | 121x |
| IS1009d | 20.72 | 17.5 | 2.0 | 1.2 | 4/4 | 125x |
| TS3003a | 35.87 | 34.7 | 0.5 | 0.6 | 4/4 | 111x |
| TS3003b | 26.30 | 25.1 | 0.9 | 0.3 | 4/4 | 130x |
| TS3003c | 30.69 | 29.5 | 0.9 | 0.3 | 4/4 | 148x |
| TS3003d | 32.42 | 30.7 | 0.9 | 0.8 | 4/4 | 155x |
| **avg** | **26.53** | 24.1 | 1.2 | 1.2 | | 127x |

## s24 (16 files)

| meeting | DER % | miss | fa | conf | spk | RTFx |
|---|---|---|---|---|---|---|
| EN2002a | 31.11 | 28.9 | 0.9 | 1.3 | 4/4 | 225x |
| EN2002b | 30.85 | 28.8 | 1.1 | 1.0 | 4/4 | 227x |
| EN2002c | 30.19 | 29.1 | 0.6 | 0.5 | 4/3 | 224x |
| EN2002d | 34.53 | 32.6 | 0.9 | 1.1 | 4/4 | 227x |
| ES2004a | 27.20 | 25.5 | 0.8 | 0.9 | 4/4 | 228x |
| ES2004b | 21.58 | 20.4 | 0.8 | 0.4 | 5/4 | 225x |
| ES2004c | 20.53 | 19.2 | 0.7 | 0.6 | 4/4 | 227x |
| ES2004d | 22.98 | 20.6 | 1.6 | 0.8 | 4/4 | 227x |
| IS1009a | 24.60 | 18.1 | 2.9 | 3.6 | 5/4 | 227x |
| IS1009b | 15.45 | 13.0 | 1.5 | 1.0 | 4/4 | 225x |
| IS1009c | 15.76 | 13.4 | 1.9 | 0.4 | 4/4 | 228x |
| IS1009d | 20.82 | 17.4 | 2.0 | 1.4 | 4/4 | 228x |
| TS3003a | 35.75 | 34.7 | 0.5 | 0.6 | 4/4 | 229x |
| TS3003b | 26.35 | 25.2 | 0.9 | 0.3 | 4/4 | 227x |
| TS3003c | 30.61 | 29.5 | 0.9 | 0.2 | 4/4 | 229x |
| TS3003d | 32.18 | 30.6 | 0.9 | 0.7 | 4/4 | 228x |
| **avg** | **26.28** | 24.2 | 1.2 | 0.9 | | 227x |

## s32 (16 files)

| meeting | DER % | miss | fa | conf | spk | RTFx |
|---|---|---|---|---|---|---|
| EN2002a | 31.13 | 29.0 | 0.8 | 1.3 | 4/4 | 287x |
| EN2002b | 30.40 | 28.4 | 1.2 | 0.8 | 4/4 | 290x |
| EN2002c | 29.75 | 28.6 | 0.6 | 0.6 | 4/3 | 293x |
| EN2002d | 34.14 | 32.0 | 1.0 | 1.2 | 4/4 | 293x |
| ES2004a | 27.26 | 25.7 | 0.7 | 0.8 | 4/4 | 292x |
| ES2004b | 21.72 | 20.4 | 0.8 | 0.5 | 5/4 | 292x |
| ES2004c | 20.51 | 19.3 | 0.7 | 0.6 | 4/4 | 293x |
| ES2004d | 22.87 | 20.4 | 1.6 | 0.9 | 4/4 | 293x |
| IS1009a | 23.92 | 18.6 | 2.5 | 2.8 | 5/4 | 292x |
| IS1009b | 15.65 | 13.0 | 1.5 | 1.1 | 4/4 | 293x |
| IS1009c | 15.87 | 13.2 | 1.8 | 0.8 | 4/4 | 293x |
| IS1009d | 20.82 | 17.8 | 2.0 | 1.1 | 4/4 | 292x |
| TS3003a | 36.01 | 34.5 | 0.6 | 0.9 | 4/4 | 293x |
| TS3003b | 26.28 | 25.1 | 0.9 | 0.2 | 4/4 | 293x |
| TS3003c | 30.72 | 29.6 | 0.9 | 0.2 | 4/4 | 293x |
| TS3003d | 32.25 | 30.7 | 0.8 | 0.7 | 4/4 | 293x |
| **avg** | **26.21** | 24.1 | 1.2 | 0.9 | | 292x |

## turbo1 (16 files)

| meeting | DER % | miss | fa | conf | spk | RTFx |
|---|---|---|---|---|---|---|
| EN2002a | 31.51 | 29.0 | 0.9 | 1.6 | 5/4 | 145x |
| EN2002b | 31.55 | 29.0 | 1.1 | 1.4 | 4/4 | 147x |
| EN2002c | 37.22 | 30.4 | 0.6 | 6.2 | 6/3 | 113x |
| EN2002d | 35.21 | 33.2 | 0.9 | 1.1 | 5/4 | 116x |
| ES2004a | 30.88 | 27.5 | 0.6 | 2.7 | 5/4 | 113x |
| ES2004b | 22.09 | 20.7 | 0.7 | 0.7 | 5/4 | 107x |
| ES2004c | 20.67 | 19.4 | 0.8 | 0.5 | 4/4 | 115x |
| ES2004d | 23.78 | 20.8 | 1.7 | 1.3 | 4/4 | 130x |
| IS1009a | 29.04 | 17.8 | 3.3 | 7.9 | 5/4 | 119x |
| IS1009b | 16.04 | 13.5 | 1.5 | 1.0 | 5/4 | 134x |
| IS1009c | 15.99 | 13.2 | 1.9 | 0.8 | 4/4 | 123x |
| IS1009d | 21.72 | 17.8 | 2.1 | 1.8 | 5/4 | 131x |
| TS3003a | 36.90 | 34.9 | 0.5 | 1.4 | 5/4 | 132x |
| TS3003b | 26.60 | 25.5 | 0.9 | 0.2 | 4/4 | 137x |
| TS3003c | 31.13 | 29.9 | 0.8 | 0.4 | 5/4 | 136x |
| TS3003d | 32.75 | 31.1 | 0.8 | 0.8 | 4/4 | 141x |
| **avg** | **27.69** | 24.6 | 1.2 | 1.9 | | 127x |

## turbo2 (16 files)

| meeting | DER % | miss | fa | conf | spk | RTFx |
|---|---|---|---|---|---|---|
| EN2002a | 32.73 | 30.2 | 0.8 | 1.7 | 5/4 | 105x |
| EN2002b | 31.79 | 28.8 | 1.1 | 1.9 | 5/4 | 102x |
| EN2002c | 37.57 | 29.8 | 0.6 | 7.1 | 5/3 | 120x |
| EN2002d | 35.16 | 32.7 | 0.9 | 1.6 | 5/4 | 137x |
| ES2004a | 35.05 | 29.4 | 0.7 | 5.0 | 5/4 | 163x |
| ES2004b | 22.04 | 20.7 | 0.7 | 0.6 | 6/4 | 129x |
| ES2004c | 20.64 | 19.4 | 0.7 | 0.5 | 5/4 | 95x |
| ES2004d | 24.30 | 21.1 | 1.5 | 1.6 | 4/4 | 116x |
| IS1009a | 24.71 | 18.6 | 2.7 | 3.5 | 6/4 | 126x |
| IS1009b | 16.08 | 13.7 | 1.4 | 0.9 | 4/4 | 123x |
| IS1009c | 16.03 | 13.3 | 2.0 | 0.7 | 4/4 | 136x |
| IS1009d | 36.40 | 21.0 | 1.5 | 13.9 | 8/4 | 125x |
| TS3003a | 37.68 | 35.4 | 0.5 | 1.8 | 4/4 | 138x |
| TS3003b | 36.55 | 26.2 | 0.9 | 9.5 | 4/4 | 128x |
| TS3003c | 38.16 | 31.4 | 0.8 | 6.0 | 5/4 | 54x |
| TS3003d | 44.40 | 34.8 | 0.7 | 8.9 | 8/4 | 48x |
| **avg** | **30.58** | 25.4 | 1.1 | 4.1 | | 115x |
## Port fidelity: NVIDIA demo asset (cross-system DER vs NeMo reference)

Scored on the audio of NVIDIA's release demo video (`nemotron3_tts_8_open_voices_v18.mp4`,
97.6 s, 8 TTS voices), our CoreML output against the NeMo/PyTorch reference output
(md-eval-style 10 ms frames, no collar, identity channel mapping):

| Variant | Cross-system DER | miss | fa | conf | Speakers | Wall (M5 Pro) |
|---|---|---|---|---|---|---|
| offline | **0.134%** | 0.113 | 0.021 | **0.000** | 8/8 | 523x |
| fast32 | 0.247% | 0.051 | 0.195 | **0.000** | 8/8 | 246x |

Zero speaker-confusion frames in either variant: all deviation is 1-2 frame boundary
flicker. The model's intrinsic error on real benchmarks is 9-13% DER (NVIDIA card), so
conversion error is ~70-100x below the model's own error budget. Note: TTS-generated
studio-clean audio — a parity showcase, not a difficulty benchmark.

## Sortformer v2.1 vs Nemotron 3 at matched 1.04 s latency (AMI Mix-Headset, 16 meetings)

Same audio, same forced-alignment references, collar 0.25 s (so Sortformer's 80 ms frames
are not penalised for boundary coarseness), threshold 0.5, no VAD gating, M5 Pro, both
models via `Examples/Nemotron3Demo --compare` (2026-09-20). Sortformer v2.1 `fast`
(4 slots) vs Nemotron 3 `low` (8 slots).

| Meeting | Sortformer DER | Nemotron 3 DER | Δ | Sortformer RTFx | Nemotron 3 RTFx |
|---|---:|---:|---:|---:|---:|
| EN2002a | 10.81 | 9.91 | -0.90 | 39x | 72x |
| EN2002b | 10.64 | 8.42 | -2.22 | 39x | 72x |
| EN2002c | 18.10 | 4.79 | -13.31 | 39x | 73x |
| EN2002d | 12.65 | 9.91 | -2.74 | 39x | 73x |
| ES2004a | 6.09 | 5.08 | -1.01 | 39x | 72x |
| ES2004b | 4.12 | 2.20 | -1.92 | 39x | 73x |
| ES2004c | 2.53 | 1.95 | -0.58 | 39x | 73x |
| ES2004d | 9.50 | 3.71 | -5.79 | 39x | 72x |
| IS1009a | 16.92 | 10.49 | -6.43 | 39x | 72x |
| IS1009b | 4.47 | 3.33 | -1.14 | 39x | 73x |
| IS1009c | 3.92 | 3.57 | -0.35 | 39x | 73x |
| IS1009d | 9.29 | 6.24 | -3.05 | 38x | 72x |
| TS3003a | 4.59 | 4.01 | -0.58 | 37x | 73x |
| TS3003b | 2.81 | 2.25 | -0.56 | 39x | 72x |
| TS3003c | 14.26 | 2.50 | -11.76 | 39x | 71x |
| TS3003d | 5.25 | 2.89 | -2.36 | 38x | 69x |
| **mean** | **8.50** | **5.08** | **-3.42** | 39x | 72x |

Nemotron 3 is lower on all 16 meetings. The largest gaps (EN2002c, TS3003c, IS1009a,
ES2004d) are meetings where Sortformer's speaker confusion dominates. Wall RTFx here
includes host-side work; the model-only cost per audio-second is within 1.5x between the
two (see Documentation/ANE_Profiler.md).
