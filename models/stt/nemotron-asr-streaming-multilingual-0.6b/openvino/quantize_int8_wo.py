#!/usr/bin/env python
"""Weight-only INT8 quantization of the Nemotron streaming encoder (OpenVINO).

Mirrors the mobius CoreML approach (`linear_quantize_weights`, per-channel
linear-symmetric, encoder only): only the weight consts become int8 with
per-output-channel fp16 scales; activations stay fp16. This is DATA-FREE
(no calibration) and far less aggressive than full PTQ (nncf.quantize),
which also quantizes activations and cost us ~+7 WER on English.

NNCF equivalent: compress_weights(mode=INT8_SYM) — per-channel symmetric
int8 weight compression. Decoder/joint/preprocessor copied from FP16.

Output: build_ov_int8/ = int8-weight encoder + fp16 {decoder,joint,preproc}.
"""
import json, shutil
from pathlib import Path
import openvino as ov
import nncf

SRC_FP32 = Path("build_ov")        # quantize encoder weights from FP32 master
SRC_FP16 = Path("build_ov_fp16")   # reuse fp16 for the rest
OUT      = Path("build_ov_int8")
OUT.mkdir(exist_ok=True)

def main():
    core = ov.Core()
    enc = core.read_model(str(SRC_FP32 / "nemotron_encoder.xml"))
    print("[nncf] compress_weights INT8_SYM (per-channel, weight-only) ...", flush=True)
    q = nncf.compress_weights(enc, mode=nncf.CompressWeightsMode.INT8_SYM)
    ov.save_model(q, str(OUT / "nemotron_encoder.xml"), compress_to_fp16=True)
    print("[nncf] encoder int8-weight saved", flush=True)

    for f in ["nemotron_decoder.xml", "nemotron_decoder.bin",
              "nemotron_joint.xml", "nemotron_joint.bin",
              "nemotron_preprocessor.xml", "nemotron_preprocessor.bin",
              "nemotron_vocab.json", "metadata.json"]:
        shutil.copy(SRC_FP16 / f, OUT / f)

    md = json.load(open(OUT / "metadata.json"))
    md["precision"] = "INT8 weight-only (encoder) + FP16"
    json.dump(md, open(OUT / "metadata.json", "w"), indent=2)
    print("DONE -> build_ov_int8", flush=True)

if __name__ == "__main__":
    main()
