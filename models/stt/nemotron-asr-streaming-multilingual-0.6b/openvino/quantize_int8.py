#!/usr/bin/env python
"""INT8 PTQ of the Nemotron streaming encoder via NNCF.

- Captures REAL encoder input dicts (incl. evolved cache tensors + per-lang
  prompt_id) by running the FP32 streaming pipeline over a small multilingual
  calibration set streamed from FLEURS (no full-config download).
- Quantizes only the encoder (the ~1.25GB bulk) with model_type=TRANSFORMER
  (keeps attention softmax in higher precision). Decoder/joint/preprocessor
  stay FP16.
- Writes build_ov_int8/ = int8 encoder + fp16 {decoder,joint,preprocessor}
  + vocab + metadata.
"""
import os, sys, shutil, itertools, json
from pathlib import Path
import numpy as np
import openvino as ov
import nncf
from transcribe_ov import NemotronOV

SRC_FP32 = Path("build_ov")          # quantize encoder from FP32 source
SRC_FP16 = Path("build_ov_fp16")     # reuse fp16 for the rest
OUT      = Path("build_ov_int8")
OUT.mkdir(exist_ok=True)

# (fleurs config, nemo lang tag) — diverse scripts/prompt_ids
CALIB = [("en_us","en-US"), ("es_419","es-ES"), ("fr_fr","fr-FR"),
         ("cmn_hans_cn","zh-CN"), ("ja_jp","ja-JP")]
FILES_PER_LANG = 6
SUBSET = 400   # encoder-call samples used by NNCF

def capture_samples():
    import io
    import soundfile as sf
    from datasets import load_dataset, Audio
    model = NemotronOV(str(SRC_FP32), device="CPU")
    samples = []
    orig_enc = model.encoder
    def recorder(d):
        samples.append({k: np.array(v, copy=True) for k, v in d.items()})
        return orig_enc(d)
    model.encoder = recorder
    for cfg, lang in CALIB:
        print(f"[calib] streaming {cfg} ({lang}) ...", flush=True)
        # Audio(decode=False)+soundfile: datasets 5.0 default decode needs torchcodec
        ds = load_dataset("google/fleurs", cfg, split="validation",
                          streaming=True, token=os.environ.get("HF_TOKEN"))
        ds = ds.cast_column("audio", Audio(decode=False))
        n = 0
        for ex in itertools.islice(ds, FILES_PER_LANG):
            audio, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            model.transcribe_streaming(np.asarray(audio, dtype=np.float32), target_lang=lang)
            n += 1
        print(f"[calib]   {cfg}: {n} files, total enc-calls so far={len(samples)}", flush=True)
    print(f"[calib] captured {len(samples)} encoder-input samples", flush=True)
    return samples

def main():
    samples = capture_samples()
    if not samples:
        print("ERROR: no calibration samples captured"); sys.exit(1)
    core = ov.Core()
    enc = core.read_model(str(SRC_FP32 / "nemotron_encoder.xml"))
    calib = nncf.Dataset(samples, lambda x: x)
    print(f"[nncf] quantizing encoder (subset={min(SUBSET,len(samples))}) ...", flush=True)
    q = nncf.quantize(enc, calib, subset_size=min(SUBSET, len(samples)),
                      model_type=nncf.ModelType.TRANSFORMER)
    ov.save_model(q, str(OUT / "nemotron_encoder.xml"), compress_to_fp16=True)
    print("[nncf] encoder int8 saved", flush=True)
    # rest from fp16
    for f in ["nemotron_decoder.xml","nemotron_decoder.bin",
              "nemotron_joint.xml","nemotron_joint.bin",
              "nemotron_preprocessor.xml","nemotron_preprocessor.bin",
              "nemotron_vocab.json","metadata.json"]:
        shutil.copy(SRC_FP16 / f, OUT / f)
    # mark precision
    md = json.load(open(OUT / "metadata.json")); md["precision"] = "INT8 (encoder) + FP16"
    json.dump(md, open(OUT / "metadata.json","w"), indent=2)
    print("DONE -> build_ov_int8", flush=True)

if __name__ == "__main__":
    main()
