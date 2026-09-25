"""Phase 6 statistics (D-032): Wilson intervals and the paired bootstrap."""

import pytest

from danalm.eval.stats import paired_bootstrap, wilson_interval


def test_wilson_interval_matches_known_values_and_stays_in_range():
    lo, hi = wilson_interval(58, 64)  # 90.6%
    assert lo == pytest.approx(0.8101, abs=1e-3) and hi == pytest.approx(0.9563, abs=1e-3)
    assert wilson_interval(64, 64)[1] == 1.0 and wilson_interval(0, 64)[0] == 0.0
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_paired_bootstrap_is_zero_for_identical_systems_and_signed_for_better_ones():
    def acc(g: list, p: list) -> float:
        return sum(x == y for x, y in zip(g, p, strict=True)) / len(g)

    gold = list("aabbccddee") * 4
    same = paired_bootstrap(gold, gold, gold, acc, 500, seed=1)
    assert same == {"diff": 0.0, "lo": 0.0, "hi": 0.0}
    worse = ["x" if i % 4 == 0 else g for i, g in enumerate(gold)]  # 75% right
    res = paired_bootstrap(gold, gold, worse, acc, 2000, seed=1)
    assert res["diff"] == pytest.approx(0.25)
    assert 0.1 <= res["lo"] < res["diff"] < res["hi"] <= 0.45  # steps of 1/40
