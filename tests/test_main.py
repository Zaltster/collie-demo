from __future__ import annotations

import pytest

from collie_demo.main import env_class_thresholds


def test_env_class_thresholds_parses_label_value_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "COLLIE_TEST_THRESHOLDS", "apple=0.50, banana=0.30,pear=0.05"
    )

    assert env_class_thresholds("COLLIE_TEST_THRESHOLDS") == {
        "apple": 0.5,
        "banana": 0.3,
        "pear": 0.05,
    }


def test_env_class_thresholds_rejects_malformed_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COLLIE_TEST_THRESHOLDS", "apple:0.5")

    with pytest.raises(ValueError, match="label=value"):
        env_class_thresholds("COLLIE_TEST_THRESHOLDS")
