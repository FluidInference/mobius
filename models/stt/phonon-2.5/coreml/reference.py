"""Independent fp32 NeMo greedy TDT decode of the released checkpoint."""
from __future__ import annotations
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from components import load_components


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("container", type=Path)
    parser.add_argument("rows", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = load_components(args.container, args.container.parent / "config.json")
    from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import GreedyTDTInfer
    greedy = GreedyTDTInfer(
        model.decoder, model.joint, blank_index=8192,
        durations=[0, 1, 2, 3, 4], max_symbols_per_step=10,
    )
    labels = list(model.cfg.labels)
    report = []
    for row in json.loads(args.rows.read_text()):
        audio, rate = sf.read(row["path"], dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if rate != 16000:
            from scipy.signal import resample_poly
            from math import gcd
            factor = gcd(rate, 16000)
            audio = resample_poly(audio, 16000 // factor, rate // factor).astype(np.float32)
        started = time.perf_counter()
        with torch.inference_mode():
            mel, length = model.preprocessor(input_signal=torch.from_numpy(audio)[None], length=torch.tensor([len(audio)]))
            encoded, encoded_length = model.encoder(audio_signal=mel, length=length)
            hypothesis = greedy(encoder_output=encoded, encoded_lengths=encoded_length)[0][0]
        ids = hypothesis.y_sequence.tolist()
        text = "".join(labels[i] for i in ids).replace("▁", " ").strip()
        text = re.sub(r"<\|[^>]*\|>|<unk>|<pad>", "", text).strip()
        result = {**row, "text": text, "token_ids": ids, "wall_seconds": time.perf_counter() - started}
        report.append(result)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(row["id"], text, flush=True)


if __name__ == "__main__":
    main()
