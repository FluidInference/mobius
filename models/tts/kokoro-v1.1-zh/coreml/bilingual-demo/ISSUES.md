# Twelve-clip issue list

Scope: selected existing samples and saved records only. No new listening
assessment, inference, learned scoring, frontend fixes, or training was
performed. A saved ASR mismatch is not by itself a confirmed pronunciation
error. This is a development issue list, not an official evaluation report.

## Priorities

| ID | Priority | Layer | Evidence strength | Next action |
| --- | --- | --- | --- | --- |
| F1 | First | Mandarin frontend | Saved phonemes plus matching source behavior | Stop treating numeral 二 as an erhua suffix |
| F2 | Next | English frontend | Saved API phonemes are word-like | Define explicit API letter-name pronunciation |
| F3 | Next | English frontend | Saved GitHub phonemes contain θ | Define exact-token GitHub pronunciation |
| E1 | Keep separate | Evaluator | Same WAV, two different language-mode transcripts | Retain both diagnostics; never choose by best reference match |
| U1/U2 | Investigate later | Acoustic model or evaluator; unresolved | Words exist in phonemes but ASR omits/merges them | Inspect narrow segments before attributing a training gap |
| C1 | Coverage limitation | Mandarin frontend/evaluation | Fallback log; zero CER without tone labels | Track fallback and do not certify tone accuracy |

## F1 — 二 is eligible for an incorrect erhua merge

Clip 8: [challenge_zh_018](audio/challenge_zh_018.wav).

- Reference: `这本书的价格是二十三元五角。`
- Saved ASR: `这本书的价格是13元五角` (CER 23.08%).
- Saved input contains `ㄕ十ㄦ4ㄕ十2ㄙㄢ1`, with the `ㄦ` attached to the preceding
  syllable instead of a separately toned numeral 二.
- `MandarinErhua.merge` merges a syllable whenever `cur.base == "er"` and
  the previous base is neither empty nor `er`. It does not check the original
  character, word boundary, or whether the syllable is the numeral 二.
- `MandarinG2P` accumulates pinyin in a shared syllable buffer and invokes this
  merge before tone sandhi. `MandarinBopomofoMap.encode` then places `ㄦ` before
  the surviving syllable's tone digit. This is consistent with the saved input.

**Conclusion:** there is concrete frontend evidence to fix before considering
training. The exact audible realization is not established by this inspection.
The `二十三` versus `13` mismatch is also not merely an equivalent number-format
change; avoid hiding it behind aggressive text normalization.

**Proposed narrow experiment, not run:** preserve source-character/word-role
information through segmentation and restrict erhua eligibility accordingly.
Add text-only regressions for numeral 二, genuine suffix 儿, and independent
儿 words. Do not blindly remove all `er` syllables or change a tone digit as a
workaround. Then regenerate only this affected price clip for a saved-original
A/B comparison, if a local sample run is requested.

Evidence: [render](evidence/renders/challenge_zh_018.json),
[ASR](evidence/asr/challenge_zh_018.json),
[merge source](https://github.com/FluidInference/FluidAudio/blob/main/Sources/FluidAudio/TTS/KokoroAne/G2P/Mandarin/MandarinErhua.swift),
[pipeline source](https://github.com/FluidInference/FluidAudio/blob/main/Sources/FluidAudio/TTS/KokoroAne/G2P/Mandarin/MandarinG2P.swift).

## F2 — API is encoded as a word, not letter names

Clip 12: [challenge_mixed_002](audio/challenge_mixed_002.wav).

- Reference: `这个 API 已经更新了，请 check the documentation。`
- Saved API phonemes: `ˈæpi` — a word-like “appy” sequence.
- Saved ASR: `这个RP已经更新了,请check the documentation。` (MER 8.33%).

**Conclusion:** for the proposed A–P–I letter-name policy, the wrong input is
already present before acoustic synthesis. The ASR spelling `RP` is supporting
evidence, not a precise account of the audible phonemes.

**Proposed narrow experiment, not run:** use an exact-token pronunciation
override for `API`, with a text-only check of the resulting vocabulary IDs.
Preserve normal words and do not spell out every uppercase token. The existing
English resolver checks custom lexicon entries first, so investigate that hook
before introducing a new acoustic model or retraining.

Evidence: [render](evidence/renders/challenge_mixed_002.json),
[ASR](evidence/asr/challenge_mixed_002.json),
[English resolver](https://github.com/FluidInference/FluidAudio/blob/main/Sources/FluidAudio/TTS/KokoroAne/G2P/English/KokoroAneEnglishPhonemizer.swift).

## F3 — GitHub contains a frontend consonant mismatch

Clip 11: [challenge_mixed_001](audio/challenge_mixed_001.wav).

- Reference: `我正在用 GitHub train a small speech model，然后部署到 Mac 上。`
- Saved GitHub phonemes: `ɡˈɪθʌb`, containing `θ`, rather than the proposed
  separate `t` + `h` sequence for “git hub.”
- Saved ASR: `我正在用Gate of Train Small Speed Model,然后部署到Mac上。`
  (MER 23.53%). It also omits `a` and changes `speech` to `Speed`.

**Conclusion:** the GitHub sequence is a specific frontend correction candidate.
Do not attribute the entire sentence's ASR error to that one token, or claim a
voice-model defect from this transcript alone.

**Proposed narrow experiment, not run:** an exact-token `GitHub` override using
the existing vocabulary; compare only the affected token/sentence and leave
the rest of the utterance unchanged. Record which lexicon/fallback path
produced the original sequence; the saved render does not contain that trace.

Evidence: [render](evidence/renders/challenge_mixed_001.json),
[ASR](evidence/asr/challenge_mixed_001.json).

## E1 — ASR language mode can look like missing TTS speech

Clip 9: [parallel_mixed_001](audio/parallel_mixed_001.wav).

- Reference: `离开之前，please close the window。`
- Fixed `zh` decode: `離開之前, please close the window.` (MER 0%).
- Automatic-language decode: `Please close the window.` (MER 50%).
- Both records score the same saved WAV. The Chinese phonemes are also present
  in the synthesis input.

**Conclusion:** this is demonstrated evaluator sensitivity, not evidence that
the TTS dropped the Mandarin phrase. Two modes of the same recognizer are not
two independent judges, and neither transcript is acoustic ground truth.

**Next action:** retain a fixed primary decoding policy plus the alternate
diagnostic. Do not select whichever hypothesis matches the reference best.
Future formal evaluation needs an independently checked multilingual protocol;
do not rerun the local suite to chase a better percentage.

Evidence: [render](evidence/renders/parallel_mixed_001.json),
[both transcripts](evidence/asr/parallel_mixed_001.json),
[ASR configuration](provenance/asr-config.json).

## U1/U2 — Short English words: acoustic model or recognizer?

Clip 2: [parallel_en_012](audio/parallel_en_012.wav).
The reference has `fixes a pronunciation problem`; ASR omits `a` (WER 14.29%),
but the saved phonemes retain `A` between `fixes` and `pronunciation`.

Clip 10: [parallel_mixed_002](audio/parallel_mixed_002.wav).
The reference starts `The meeting`; ASR emits `Dmeeting` (MER 25%). The input
retains `ði mˈiɾɪŋ`. Its saved predicted MOS is about 2.55, but that uncalibrated
proxy does not identify the cause or establish poor human-rated naturalness.

**Conclusion:** possible acoustic/duration/prosody weaknesses remain plausible,
but frontend omission is not supported by these two records. Recognition and
tokenization errors remain alternatives. These are not confirmed training needs.

**Proposed narrow experiment, not run:** inspect the specific word spans in the
existing WAVs first; if needed, use an independent recognizer on those two
clips in the agreed testing environment. Preserve the original text and avoid
using whole-utterance error as a substitute for a local diagnosis.

Evidence: [clip 2 render](evidence/renders/parallel_en_012.json),
[clip 2 ASR](evidence/asr/parallel_en_012.json),
[clip 10 render](evidence/renders/parallel_mixed_002.json),
[clip 10 ASR](evidence/asr/parallel_mixed_002.json).

## C1 — Correct transcript does not establish Mandarin tone quality

Clips 6 and 7 cover tone-sandhi/polyphone contexts and have zero saved CER.
That supports text recovery only. No labeled acoustic tone judge or listening
assessment was used for this assembly.

The source rendering log records `jieba=true, g2pw=off` after the optional
g2pW vocabulary request returned 404. This is a run-time availability fact from
the earlier run, not a fresh network check. It does not prove that these clips
contain polyphone errors, and it does not by itself explain F1.

**Next action:** keep the fallback state attached to artifacts. Before official
training/preprocessing, pin a valid frontend asset set or explicitly approve
the fallback. Do not change tokenization silently between runs.

Evidence: [clip 6 ASR](evidence/asr/challenge_zh_003.json),
[clip 7 ASR](evidence/asr/challenge_zh_008.json),
[source render log](provenance/frontend-log.txt).

## Decision

The first implementation candidate is **F1, the numeral/erhua bug**, with
text-only regression coverage. F2/F3 are bounded pronunciation-policy changes.
These findings do not yet justify starting fine-tuning. The unresolved acoustic
questions and formal quality assessment remain separate work.
