#!/usr/bin/env python3
"""Load parakeet-unified-en-0.6b, dump architecture facts, run reference offline transcription."""
from __future__ import annotations

import torch

import nemo.collections.asr as nemo_asr

NEMO_PATH = "parakeet-unified-en-0.6b.nemo"
AUDIO = "audio/yc_first_minute_16k_15s.wav"


def main() -> None:
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(NEMO_PATH, map_location="cpu")
    model.eval()

    enc = model.encoder
    print("=== encoder ===")
    print("type:", type(enc).__name__)
    print("att_context_style:", enc.att_context_style)
    print("att_context_size:", enc.att_context_size)
    print("att_context_size_all:", enc.att_context_size_all)
    print("att_chunk_context_size:", getattr(enc, "att_chunk_context_size", None))
    print("conv_context_style:", getattr(enc, "conv_context_style", None))
    print("conv_context_size:", enc.conv_context_size)
    print("subsampling_factor:", enc.subsampling_factor)
    print("=== decoder/joint ===")
    print("pred_hidden:", model.decoder.pred_hidden, "layers:", model.decoder.pred_rnn_layers)
    print("blank_idx:", model.decoder.blank_idx)
    print("vocab_size:", model.tokenizer.vocab_size)
    print("joint num_classes:", model.joint.num_classes_with_blank)
    print("joint num_extra_outputs:", model.joint.num_extra_outputs)

    # The released .nemo has no validation_ds section; transcribe() dereferences it.
    from omegaconf import OmegaConf, open_dict

    with open_dict(model.cfg):
        if model.cfg.get("validation_ds") is None:
            model.cfg.validation_ds = OmegaConf.create({})

    with torch.inference_mode():
        out = model.transcribe([AUDIO])
    print("=== offline reference transcript ===")
    print(out[0].text)


if __name__ == "__main__":
    main()
