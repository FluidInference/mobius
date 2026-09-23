"""Check benchmark accounting without loading any model package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from profile_requests import percentile_ms, unlabelled


def test_nearest_rank_percentiles_preserve_small_sample_tail():
    assert percentile_ms([4.0, 1.0, 3.0, 2.0], 0.5) == 2.0
    assert percentile_ms([4.0, 1.0, 3.0, 2.0], 0.95) == 4.0


def test_unlabelled_copy_preserves_original_fixture():
    original = {"questions": {"q": {"type": "choice", "label": "a", "src": "fixture"}}}
    serving = unlabelled(original)
    assert serving["questions"]["q"] == {"type": "choice"}
    assert original["questions"]["q"]["label"] == "a"
