"""Small, real-checkpoint parity test for Jeff's classification path."""

import torch
from huggingface_hub import snapshot_download
from jeff.backends.torch_backend import TorchBackend

from export import REVISION, SOURCE, native_report
from jeff_decision import JeffDecision


def test_trained_classification_path_matches_native_logits():
    torch.set_num_threads(2)
    checkpoint = snapshot_download(SOURCE, revision=REVISION, local_files_only=True)
    backend = TorchBackend(checkpoint, device="cpu", dtype="float32", attn_kernel="eager", batch_size=1)
    records, _ = native_report(backend, JeffDecision(backend.model.model).eval())
    assert len(records) == 4
    assert max(record["max_logit_error"] for record in records) < 1e-3
