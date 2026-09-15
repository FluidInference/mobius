# Upstream implementation attribution

The compatible generator assembly and inference orchestration follow
[hexgrad/kokoro](https://github.com/hexgrad/kokoro), commit
`dfb907a02bba8152ca444717ca5d78747ccb4bec`, especially `kokoro/model.py`.
Kokoro's source is Apache-2.0 licensed; see [LICENSE.kokoro](LICENSE.kokoro).
The actual layer modules are installed from that pinned upstream source,
including its StyleTTS2/iSTFTNet attribution. They are not vendored here.

The model checkpoint, voice data, training-corpus candidates, and source code
have separate provenance and use requirements. This file does not grant new
rights to recordings or a derived production voice.

Acoustic target extraction follows the published StyleTTS2 training convention
and loads the JDC pitch model from
[yl4579/StyleTTS2](https://github.com/yl4579/StyleTTS2/tree/5cedc71c333f8d8b8551ca59378bdcc7af4c9529),
pinned in `training-assets.lock.json`. Acquisition downloads its MIT license
alongside the target model. The JDC source/checkpoint is not redistributed in Git.

EMIME Mandarin/English recordings are credited to Mirjam Wester, Hui Liang,
and the Centre for Speech Technology Research, University of Edinburgh.
The corpus archive's own README specifies ODbL 1.0 and DbCL 1.0. Preserve its
README/license with acquired data. This experiment uses MF5 microphone 0;
the toolkit does not grant publicity/personality rights or certify consent
for a production synthetic voice.

The evaluation ASR model is OpenAI Whisper large-v3-turbo (MIT). Misaki and
its language resources, eSpeak, OpenCC, spaCy, and other installed dependencies
retain their respective licenses. `uv.lock` records the actual packages;
bundling a trained checkpoint does not relicense those components.
