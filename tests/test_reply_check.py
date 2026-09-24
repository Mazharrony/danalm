"""The D-029 reply check: parsing the judge's verdicts and estimating a set-wide rate."""

import pytest

from danalm.data.reply_checks import parse_verdicts, weighted_rate


def test_verdicts_are_parsed_from_json_lines_and_bad_lines_skipped():
    answer = "\n".join(
        [
            '{"n": 1, "verdict": "ok", "reason": ""}',
            '{"n": 2, "verdict": "BROKEN", "reason": "wrong verb"}',
            '{"n": 3, "verdict": "maybe"}',  # not a verdict
            '{"n": 7, "verdict": "OK"}',  # outside 1..n
            "not json at all",
        ]
    )
    assert parse_verdicts(answer, 3) == {1: ("OK", ""), 2: ("BROKEN", "wrong verb")}


def test_weighted_rate_weights_each_sampled_variety_by_its_share():
    hits = {"a": 1, "b": 10, "c": 0}
    judged = {"a": 100, "b": 100, "c": 0}  # c was not sampled: left out and renormalized
    share = {"a": 0.6, "b": 0.2, "c": 0.2}
    assert weighted_rate(hits, judged, share) == pytest.approx((0.6 * 0.01 + 0.2 * 0.10) / 0.8)
