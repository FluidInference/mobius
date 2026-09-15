import json

import pytest

from kokoro_training.artifacts import ROOT
from kokoro_training.frontend import Frontend


@pytest.fixture(scope="module")
def frontend():
    config = ROOT / ".artifacts/baseline/config.json"
    if not config.exists():
        pytest.skip("baseline acquisition required for frontend integration tests")
    return Frontend(json.loads(config.read_text())["vocab"])


def test_numeral_is_not_erhua(frontend):
    result = frontend("这本书的价格是二十三元五角。")
    assert "ㄕ十4/ㄦ4ㄕ十2ㄙㄢ1" in result["phonemes"]
    assert not result["unknown_symbols"]


def test_real_erhua_and_independent_er(frontend):
    result = frontend("小孩儿在这儿，儿童有二十人。")
    assert "ㄏㄞR2" in result["phonemes"]
    assert "ㄦ2ㄊ中2" in result["phonemes"]
    assert "ㄦ4ㄕ十2" in result["phonemes"]


def test_exact_api_github_policy(frontend):
    phones = frontend("这个 API 在 GitHub 上。 ")["phonemes"]
    assert "ˌA pˌi ˈI" in phones
    assert "ɡˈɪt hˌʌb" in phones
    assert "θ" not in phones


def test_pure_english_does_not_become_chinese_numbers(frontend):
    assert frontend("We have 23 books.")["language"] == "en"


def test_empty_and_overlength_rejected(frontend):
    with pytest.raises(ValueError):
        frontend(" ")
    with pytest.raises(ValueError):
        frontend("hello world " * 120)
