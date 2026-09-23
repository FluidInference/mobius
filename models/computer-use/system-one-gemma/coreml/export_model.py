"""Load the exact trained Gemma scorer and expose its scalar logit path."""

from __future__ import annotations

from assets import LOCK, fetch_base, fetch_source


def load_trained_scorer():
    """Return tokenizer and merged scorer, refusing an absent or altered trained head."""
    # Access check happens before any model allocation or download of large base weights.
    base = fetch_base()
    source = fetch_source()

    import torch
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Gemma3TextForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = Gemma3TextForSequenceClassification.from_pretrained(
        base, num_labels=1, dtype=torch.float32, local_files_only=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    peft = PeftModel.from_pretrained(model, source / "pretrained-scorer", local_files_only=True)
    merged = peft.merge_and_unload().eval()
    trained = load_file(source / "pretrained-scorer" / "adapter_model.safetensors")
    trained_head = trained[LOCK["native_serving"]["trained_score_tensor"]].float()
    actual_head = merged.score.weight.detach().float()
    if tuple(trained_head.shape) != tuple(LOCK["native_serving"]["score_shape"]):
        raise ValueError("locked trained head shape changed")
    if not torch.equal(trained_head, actual_head):
        raise ValueError("merged model lost the trained scalar score head")
    return tokenizer, merged


def export_wrapper(model):
    """Make a traceable wrapper that returns only the trained choice logit."""
    import torch

    class ScalarScorer(torch.nn.Module):
        def __init__(self, native):
            super().__init__()
            self.native = native

        def forward(self, input_ids, attention_mask):
            return self.native(input_ids=input_ids.long(), attention_mask=attention_mask.long()).logits.float()

    return ScalarScorer(model).eval()
