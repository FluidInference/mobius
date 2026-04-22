#!/usr/bin/env python3
"""Quick start example - transcribe audio in 10 lines of code.

Usage:
    python quickstart.py audio.wav

Note: First load takes ~20s for ANE compilation, then cached for instant reuse.
"""

import sys
import numpy as np
import coremltools as ct
import soundfile as sf
import json
from cohere_mel_spectrogram import CohereMelSpectrogram

# Load models (ML Program format requires .mlpackage)
encoder = ct.models.MLModel("cohere_encoder.mlpackage")
decoder = ct.models.MLModel("cohere_decoder_stateful.mlpackage")
vocab = {int(k): v for k, v in json.load(open("vocab.json")).items()}

# Load audio (16kHz mono)
audio, _ = sf.read(sys.argv[1], dtype="float32")

# Preprocess
mel_processor = CohereMelSpectrogram()
mel = mel_processor(audio)
mel_padded = np.pad(mel, ((0, 0), (0, 0), (0, max(0, 3500 - mel.shape[2]))))[:, :, :3500]

# Encode
encoder_out = encoder.predict({
    "input_features": mel_padded.astype(np.float32),
    "feature_length": np.array([min(mel.shape[2], 3500)], dtype=np.int32)
})
encoder_hidden = encoder_out["hidden_states"]

# Decode
state = decoder.make_state()
PROMPT = [13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]  # English
tokens = []
last_token = None
cross_mask = np.ones((1, 1, 1, encoder_hidden.shape[1]), dtype=np.float16)

for step in range(108):
    current_token = PROMPT[step] if step < len(PROMPT) else last_token

    decoder_out = decoder.predict({
        "input_id": np.array([[current_token]], dtype=np.int32),
        "encoder_hidden_states": encoder_hidden.astype(np.float16),
        "attention_mask": np.zeros((1, 1, 1, step + 1), dtype=np.float16),
        "cross_attention_mask": cross_mask,
        "position_ids": np.array([[step]], dtype=np.int32),
    }, state=state)

    next_token = int(np.argmax(decoder_out["logits"][0]))
    last_token = next_token

    if step >= len(PROMPT) - 1:
        tokens.append(next_token)
        if next_token == 3:  # EOS
            break

# Convert to text
text = "".join([vocab.get(t, "") for t in tokens if t > 4]).replace("▁", " ").strip()
print(text)
