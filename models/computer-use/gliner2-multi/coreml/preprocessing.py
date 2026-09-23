"""Native GLiNER2 schema preprocessing for a fixed Core ML bucket."""

import numpy as np
from gliner2 import Schema
from gliner2.models.base import load_extractor_tokenizer
from gliner2.processor import SchemaTransformer
from gliner2.training.trainer import ExtractorCollator


def load_processor(tokenizer_dir: str):
    """Load only the tokenizer and schema formatter needed by the Core ML model."""
    return SchemaTransformer(tokenizer=load_extractor_tokenizer(tokenizer_dir), token_pooling="first")


def native_batch(native, text: str, task: str, labels: list[str], length: int):
    schema = Schema().classification(task, labels)
    collator = ExtractorCollator(native.processor, is_training=False, max_len=length, architecture=native.architecture)
    return collator([(text, schema.build())])


def prepare_classification(native, text: str, task: str, labels: list[str], length: int, max_options: int):
    return prepare_with_processor(native.processor, text, task, labels, length, max_options)


def prepare_with_processor(processor, text: str, task: str, labels: list[str], length: int, max_options: int):
    if not 1 <= len(labels) <= max_options:
        raise ValueError(f"Expected 1..{max_options} labels, got {len(labels)}")
    schema = Schema().classification(task, labels)
    collator = ExtractorCollator(processor, is_training=False, max_len=length, architecture="boundary")
    batch = collator([(text, schema.build())])
    ids = batch.input_ids.numpy()
    attention = batch.attention_mask.numpy()
    indices = batch.cls_marker_indices.numpy()
    mask = batch.cls_marker_mask.numpy()
    if ids.shape[1] > length or indices.shape[1] != len(labels) or int(mask.sum()) != len(labels):
        raise ValueError("Input exceeds bucket or classification markers were truncated")
    ids = np.pad(ids, ((0, 0), (0, length - ids.shape[1])), constant_values=processor.tokenizer.pad_token_id)
    attention = np.pad(attention, ((0, 0), (0, length - attention.shape[1])))
    indices = np.pad(indices, ((0, 0), (0, max_options - indices.shape[1])))
    mask = np.pad(mask, ((0, 0), (0, max_options - mask.shape[1])))
    return {
        "input_ids": ids.astype(np.int32),
        "attention_mask": attention.astype(np.int32),
        "marker_indices": indices.astype(np.int32),
        "marker_mask": mask.astype(np.float32),
    }


def prepare_extraction(
    processor,
    text: str,
    schema,
    length: int,
    max_words: int,
    max_queries: int,
    max_choices: int = 8,
):
    """Prepare an extractive schema without allowing upstream word truncation."""
    if min(length, max_words, max_queries, max_choices) < 1:
        raise ValueError("Extraction bucket dimensions must all be positive")
    built_schema = schema.build() if hasattr(schema, "build") else schema
    collator = ExtractorCollator(processor, is_training=False, max_len=None, architecture="boundary")
    batch = collator([(text, built_schema)])
    if batch.input_ids.shape[1] > length:
        raise ValueError(f"Schema and text require {batch.input_ids.shape[1]} subwords; bucket holds {length}")
    if batch.text_word_indices.shape[1] > max_words:
        raise ValueError(f"Text requires {batch.text_word_indices.shape[1]} words; bucket holds {max_words}")
    if batch.query_marker_indices.shape[1] > max_queries:
        raise ValueError(f"Schema requires {batch.query_marker_indices.shape[1]} queries; bucket holds {max_queries}")
    if batch.cls_marker_indices.shape[1] > max_choices:
        raise ValueError(f"Schema requires {batch.cls_marker_indices.shape[1]} choices; bucket holds {max_choices}")
    if batch.query_marker_indices.shape[1] == 0 and batch.cls_marker_indices.shape[1] == 0:
        raise ValueError("Schema has no extraction or classification queries")

    def padded(values, width, fill=0):
        array = values.numpy()
        return np.pad(array, ((0, 0), (0, width - array.shape[1])), constant_values=fill)

    arrays = {
        "input_ids": padded(batch.input_ids, length, processor.tokenizer.pad_token_id).astype(np.int32),
        "attention_mask": padded(batch.attention_mask, length).astype(np.int32),
        "text_indices": padded(batch.text_word_indices, max_words).astype(np.int32),
        "text_mask": padded(batch.text_word_mask, max_words).astype(np.float32),
        "query_indices": padded(batch.query_marker_indices, max_queries).astype(np.int32),
        "query_mask": padded(batch.query_marker_mask, max_queries).astype(np.float32),
        "cls_indices": padded(batch.cls_marker_indices, max_choices).astype(np.int32),
        "cls_mask": padded(batch.cls_marker_mask, max_choices).astype(np.float32),
    }
    return arrays, batch
