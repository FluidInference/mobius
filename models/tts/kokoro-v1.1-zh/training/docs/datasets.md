# Dataset acquisition investigation — 2026-09-15

The source-workspace handoff confirms that no previous selected dataset,
recording location, or audited manifest was recovered. The 300 evaluation
prompts and generated samples are development material, not recordings of the
target speaker. This implementation does not fabricate a training manifest.

## EMIME Mandarin/English v1.1

Candidate for small same-speaker bilingual experiments. The official
[README](https://www.emime.org/participate/emime-bilingual-database/the-emime-mandarin-english-bilingual-database-1.html)
lists seven female and seven male Mandarin speakers with separate English and
Mandarin recordings. It declares ODbL 1.0 for the database and DbCL 1.0 for
individual contents. Those declarations are recorded here, not interpreted as
verified production voice consent. Speaker choice, accent/quality review,
rights for the intended derived voice, and split design remain open.

The [official download page](https://www.emime.org/participate/emime-bilingual-database.html)
links a v1.1 archive. HTTPS HEAD on 2026-09-15 returned HTTP 200 and
1,384,527,144 bytes for
`https://data.cstr.ed.ac.uk/emime/UEDIN_mandarin_bilingual_data_v1.1.tar.bz2`.
The separate 96-kHz archive is 10,959,510,325 bytes; it was not downloaded.
The v1.1 archive was acquired locally for header/inventory auditing only, under
ignored `.artifacts/emime/`; no audio or prompt text is redistributed in Git.
The completed [inventory](../evidence/emime-inventory.json) records archive hash
`c6d268abb998172d9ecc7e8bb7080cc51c3e2d3cca758044b99cd39109e14b93`.
It found 9,044 WAV members: 8,792 named primary microphone recordings and
252 additional/segmented test WAVs. No header errors were found among the
primary recordings. This is header validation, not decoded signal/transcript QC.

Measured female-speaker volume for microphone **0 only**, excluding segmented
test copies (all primary files inspected are mono, 22,050 Hz):

| Speaker | English clips / minutes | Mandarin clips / minutes | Total minutes |
| --- | ---: | ---: | ---: |
| MF1 | 145 / 10.26 | 169 / 16.87 | 27.14 |
| MF2 | 145 / 8.09 | 169 / 17.52 | 25.62 |
| MF3 | 145 / 10.25 | 169 / 15.91 | 26.16 |
| MF4 | 145 / 9.78 | 169 / 16.68 | 26.47 |
| MF5 | 145 / 10.02 | 169 / 18.34 | 28.36 |
| MF6 | 145 / 10.23 | 169 / 16.61 | 26.84 |
| MF7 | 145 / 9.47 | 169 / 17.34 | 26.81 |

These are total recorded durations before exclusions or split reservations,
not usable training hours. Roughly 26–28 minutes per speaker supports assessing
whether a small harness experiment is practical; it does not establish a
production-data solution. No speaker was selected, no audio was listened to
in this audit, and no recording was used for optimization. Archive README,
license, prompt files, and paper hashes are included without redistributing
their text. Both 1.0 and 1.1 README versions are present in the v1.1 archive.

Audit pitfalls to preserve:

- Two microphone views represent the same spoken event. Choose a consistent
  channel, group both views together for splits, and do not double usable hours.
- Segmented Mandarin test files overlap full source sentences. Keep their
  parent passages in the same split and preserve the original designated tests.
- Separate English/Mandarin files do not establish within-utterance switching.
- The published prompts include translated Europarl and news/SUS material;
  audit passage overlap and prompt rights before constructing derived corpora.
- Session labels, usable per-speaker volume, and accent fit need inspection.
  A voice with bilingual recordings is not automatically the desired voice.

The [authors' report](https://www.cstr.ed.ac.uk/downloads/publications/2011/wester_mandarin_2011.pdf)
describes accent evaluation. Its speaker comparisons are candidate-selection
context, not our listening assessment or quality certification.

## ASCEND

The [authors' repository](https://github.com/HLTCHKUST/ASCEND) describes
10.62 hours from 23 bilingual speakers in spontaneous conversation. This makes
it a candidate for code-switching diagnostics, not an already-selected single
production voice. The [dataset card](https://huggingface.co/datasets/CAiRE/ASCEND)
declares **CC-BY-SA-4.0**; the repository's MIT code license must not be mistaken
for the dataset license. Speaker, session, and topic fields are available for
auditing. No ASCEND recordings were acquired or used in this work.

## Acquisition decision still required

Choose/approve one real female speaker with suitable English/Mandarin delivery
and a valid intended-use record. Establish genuine mixed-utterance coverage or
document the gap before a pilot. A promising public corpus and a locally
downloaded archive are evidence of availability, not production data approval.
