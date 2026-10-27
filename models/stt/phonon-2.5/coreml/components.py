"""Build the existing NeMo component tree directly from the released checkpoint.

Uses the Phonon-2 reader/name mapping recovered from mobius PR #111. Loading
components directly avoids downloading a 2.5 GB teacher whose weights we replace.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
from fermion_container import FORMAT, _five_value, _intn
from name_mapping import hf_to_nemo_name


def load_components(container: Path, config_path: Path, parts=("preprocessor", "encoder", "decoder", "joint")):
    from nemo.collections.asr.modules import (
        AudioToMelSpectrogramPreprocessor, ConformerEncoder, RNNTDecoder, RNNTJoint,
    )
    cfg = json.loads(config_path.read_text())
    def params(key):
        return {k: v for k, v in cfg[key].items() if k != "_target_"}
    model = torch.nn.Module()
    if "preprocessor" in parts:
        print("Constructing preprocessor", flush=True)
        model.preprocessor = AudioToMelSpectrogramPreprocessor(**params("preprocessor"))
    if "encoder" in parts:
        print("Constructing encoder", flush=True)
        model.encoder = ConformerEncoder(**params("encoder"))
    if "decoder" in parts:
        print("Constructing decoder", flush=True)
        model.decoder = RNNTDecoder(**params("decoder"))
    joint = params("joint")
    joint["fuse_loss_wer"] = False
    if "joint" in parts:
        print("Constructing joint", flush=True)
        model.joint = RNNTJoint(**joint)
    model.cfg = OmegaConf.create(cfg)
    model.tokenizer = SimpleNamespace(vocab_size=cfg["decoder"]["vocab_size"])
    print("Reading checkpoint", flush=True)
    expected = model.state_dict()
    seen = {k for k in expected if k.startswith("preprocessor.")}
    counts = {}
    with container.open("rb") as file, torch.no_grad():
        header_size = int.from_bytes(file.read(8), "little")
        if not 0 < header_size < 10_000_000:
            raise ValueError("Invalid container header size")
        header = json.loads(file.read(header_size))
        if header["format"] != FORMAT:
            raise ValueError("Unsupported checkpoint format")
        for record in header["index"]:
            kind, shape = record["k"], tuple(record["shape"])
            blob = file.read(record["b"])
            if len(blob) != record["b"]:
                raise ValueError("Truncated checkpoint")
            counts[kind] = counts.get(kind, 0) + 1
            name = record["n"] + ".weight" if kind == "five_value" else record["n"]
            key = hf_to_nemo_name(name)
            component = key.split(".", 1)[0]
            if component in {"encoder", "decoder", "joint", "preprocessor"} and component not in parts:
                continue
            if key not in expected or key in seen:
                raise ValueError(f"Unexpected or duplicate checkpoint tensor: {key}")
            if kind == "five_value":
                array = _five_value(blob, shape)
                if ".pointwise_conv" in name:
                    array = array[:, :, None]
            elif kind.startswith("int"):
                array = _intn(blob, shape, int(kind[3:]))
            elif kind == "fp16":
                array = np.frombuffer(blob, dtype=np.float16).reshape(shape)
            else:
                raise ValueError(f"Unknown tensor encoding: {kind}")
            if key.endswith("num_batches_tracked") and not np.isfinite(array).all():
                array = np.zeros_like(array)
            if array.shape != tuple(expected[key].shape):
                raise ValueError(f"Shape mismatch for {key}: {array.shape} vs {expected[key].shape}")
            expected[key].copy_(torch.from_numpy(np.array(array, copy=True)))
            seen.add(key)
        if file.read(1):
            raise ValueError("Trailing checkpoint bytes")
    missing = set(expected) - seen
    if missing:
        raise ValueError(f"Missing checkpoint tensors: {sorted(missing)}")
    model.eval()
    print(f"Strictly loaded {len(seen)} tensors; encodings {counts}", flush=True)
    return model


if __name__ == "__main__":
    torch.set_num_threads(4)
    root = Path(sys.argv[1])
    m = load_components(root / "model.fermion", root / "config.json")
    print("parameters", sum(p.numel() for p in m.parameters()), flush=True)
