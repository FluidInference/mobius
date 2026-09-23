"""Inspect Jeff's pinned, trained GLiFormer decision boundary using a real request."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from jeff.backends.torch_backend import TorchBackend
from jeff.core.backend import Group

SOURCE_REPO = "knowledgator/gliformer-large-v1"
SOURCE_REVISION = "d0a4e53d09cebe6bc963dd9be319d4279084bb2d"


def main() -> None:
    torch.set_num_threads(2)
    checkpoint = snapshot_download(SOURCE_REPO, revision=SOURCE_REVISION)
    backend = TorchBackend(checkpoint, device="cpu", dtype="float32", attn_kernel="eager", batch_size=1)
    text = "The invoice was charged twice and the customer asks for a refund."
    group = Group(
        key="route",
        labels=("billing: invoice or payment issue", "support: technical product issue"),
        name="Choose the correct support queue",
    )
    native = backend.score([text], [[group]])[0]
    tokens, _, _ = backend.model.prepare_inputs([text])
    batch = backend._collator(
        [
            {
                "tokenized_text": tokens[0],
                "classification": [
                    {
                        "name": group.name,
                        "description": group.description,
                        "all_labels": list(group.labels),
                        "true_labels": [],
                    }
                ],
            }
        ]
    )
    report = {
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "backend": backend.info(),
        "scores": native.scores,
        "input_tokens": native.input_tokens,
        "model_parameters": sum(parameter.numel() for parameter in backend.model.model.parameters()),
        "batch_tensors": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in batch.items()
            if isinstance(value, torch.Tensor)
        },
        "batch_other": {name: str(type(value)) for name, value in batch.items() if not isinstance(value, torch.Tensor)},
    }
    output = Path("build/native-probe.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
