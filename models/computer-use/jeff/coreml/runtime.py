"""Standalone Jeff classification inference from the Core ML package.

No original PyTorch checkpoint is loaded. Tokenization and the classification
prompt formatting use the pinned GLiFormer processor and tokenizer files.
"""

from __future__ import annotations

import json
from pathlib import Path

import coremltools as ct
import numpy as np
from gliformer.config import GLiFormerConfig
from gliformer.processing.collator import resolve_gliformer_collator_class
from gliformer.processing.processor import resolve_gliformer_processor_class
from gliner.data_processing.tokenizer import WordsSplitter
from transformers import AutoTokenizer

from jeff_decision import marker_positions


class JeffCoreML:
    """One-group, one-text Jeff classifier for 1–8 labels and at most 128 tokens."""

    def __init__(self, asset_dir: str | Path, package: str | Path, compute_units=ct.ComputeUnit.ALL):
        asset_dir = Path(asset_dir)
        config = GLiFormerConfig(**json.loads((asset_dir / "gliner_config.json").read_text()))
        tokenizer = AutoTokenizer.from_pretrained(asset_dir, local_files_only=True)
        splitter = WordsSplitter(config.words_splitter_type)
        processor_cls = resolve_gliformer_processor_class(config)
        processor = processor_cls(config, tokenizer, splitter)
        collator_cls = resolve_gliformer_collator_class(config)
        self.collator = collator_cls(config, data_processor=processor, return_tokens=True, prepare_labels=False)
        self.splitter = splitter
        self.config = config
        self.model = ct.models.MLModel(str(package), compute_units=compute_units)
        self.output_name = self.model.get_spec().description.output[0].name

    def score(self, text: str, labels: list[str], name: str = "", description: str = "") -> list[float]:
        if not 1 <= len(labels) <= 8:
            raise ValueError("JeffCoreML accepts 1 to 8 labels")
        tokens = [word for word, _, _ in self.splitter(text)]
        batch = self.collator([{
            "tokenized_text": tokens,
            "classification": [{
                "name": name,
                "description": description,
                "all_labels": labels,
                "true_labels": [],
            }],
        }])
        ids = batch["input_ids"]
        mask = batch["attention_mask"]
        if ids.shape[1] > 128:
            raise ValueError(f"JeffCoreML L128 bucket cannot fit {ids.shape[1]} tokens")
        parent, children, count = marker_positions(ids, self.config, 8)
        if count != len(labels):
            raise ValueError("tokenizer label markers disagree with the supplied label count")
        inputs = {
            "input_ids": np.pad(ids.numpy().astype(np.int32), ((0, 0), (0, 128 - ids.shape[1]))),
            "attention_mask": np.pad(mask.numpy().astype(np.int32), ((0, 0), (0, 128 - mask.shape[1]))),
            "parent_position": parent.numpy(),
            "category_positions": children.numpy(),
        }
        logits = self.model.predict(inputs)[self.output_name][0, :count].astype(np.float32)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        return probabilities.tolist()
