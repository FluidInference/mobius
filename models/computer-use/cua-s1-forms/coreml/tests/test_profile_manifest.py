import importlib.util
from pathlib import Path

import pytest

from assets import load_demo

spec = importlib.util.spec_from_file_location("profile_coreml", Path(__file__).parents[1] / "profile-coreml.py")
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def test_default_profile_manifest_covers_all_three_real_forms():
    selected = profile.select_rows(load_demo(), profile.DEFAULT_ROWS)
    assert [row["meta"]["page"] for row in selected] == ["patient-registration", "job-application", "auto-claim"]
    assert [len(row["options"]) for row in selected] == [27, 21, 19]
    assert all(row["options"][row["label"]].startswith("fill ") for row in selected)


@pytest.mark.parametrize("indices", [[], [0, 0], [-1], [196]])
def test_invalid_profile_manifests_are_rejected(indices):
    with pytest.raises(ValueError):
        profile.select_rows(load_demo(), indices)
