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
