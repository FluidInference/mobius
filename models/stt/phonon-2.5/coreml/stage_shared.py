"""Stage the shared frontend, exact vocabulary and compiled prediction components."""
import argparse
import json
from pathlib import Path
import shutil

import coremltools as ct
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument("config", type=Path, help="config.json unpacked beside model.fermion")
parser.add_argument("components", type=Path)
parser.add_argument("--preprocessor-revision", default="7dd20fe6b1797d35f5e3307e8b1732d9a178edfe")
args = parser.parse_args()
cfg = json.loads(args.config.read_text())
vocab = cfg["joint"]["vocabulary"]
if len(vocab) != 8192 or vocab != cfg["labels"]:
    raise ValueError("Checkpoint does not have the expected v3 vocabulary contract")
args.components.mkdir(parents=True, exist_ok=True)
for name in ["Decoder", "JointDecisionv3"]:
    destination = args.components / (name + ".mlmodelc")
    if destination.exists():
        raise FileExistsError(f"Use a fresh component output directory: {destination}")
    compiled = Path(ct.utils.compile_model(str(args.components / (name + ".mlpackage"))))
    shutil.move(str(compiled), destination)
snapshot_download(
    "FluidInference/parakeet-tdt-0.6b-v3-coreml", revision=args.preprocessor_revision,
    allow_patterns=["Preprocessor.mlmodelc/**"], local_dir=args.components,
)
(args.components / "parakeet_vocab.json").write_text(json.dumps({str(i): token for i, token in enumerate(vocab)}, ensure_ascii=False))
(args.components / "preprocessor-source.json").write_text(json.dumps({"repository": "FluidInference/parakeet-tdt-0.6b-v3-coreml", "revision": args.preprocessor_revision}, indent=2))
