"""Check the conversion boundary against the pinned upstream tokenizer and prompt."""

import runpy

from transformers import AutoTokenizer

from decision import source_path
from preprocessing import Shape, prepare_candidates, prompt

SCHEMA = {
    "type": "object",
    "properties": {"route": {"type": "string", "enum": ["billing", "support"]}},
    "required": ["route"],
    "additionalProperties": False,
}
CONTEXT = "The invoice has a duplicate charge."


def test_pinned_prompt_and_candidate_tokens():
    path = source_path()
    upstream = runpy.run_path(str(path / "rlcd" / "engine.py"))
    tokenizer = AutoTokenizer.from_pretrained(path)
    engine = upstream["Engine"].__new__(upstream["Engine"])
    engine.tokenizer = tokenizer
    expected = engine.prompt(CONTEXT, SCHEMA)
    assert prompt(tokenizer, CONTEXT, SCHEMA) == expected

    candidates = prepare_candidates(tokenizer, CONTEXT, SCHEMA, Shape(length=256))
    prefix = engine.encode(expected)
    suffix = engine.encode('  "route": ')
    assert len(candidates) == 2
    for candidate in candidates:
        value = engine.encode('"' + candidate.value + '"\n')
        assert candidate.ids == prefix + suffix + value
        start = len(prefix) + len(suffix) - 1
        assert candidate.positions == list(range(start, start + len(value)))
        assert candidate.targets == value
