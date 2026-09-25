"""Uncertainty for small test sets (D-032): Wilson intervals for rates and a paired bootstrap for
the difference of a metric between two systems scored on the same messages."""

import math
import random
from collections.abc import Callable, Sequence
from typing import Any


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% (z=1.96) Wilson score interval for k successes out of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_bootstrap(
    gold: Sequence[Any],
    pred_a: Sequence[Any],
    pred_b: Sequence[Any],
    metric: Callable[[list, list], float],
    n_resamples: int,
    seed: int,
) -> dict[str, float]:
    """metric(a) - metric(b) on the full set, and the 2.5th and 97.5th percentiles of that
    difference over resamples of the same message indices for both systems."""
    rng = random.Random(seed)
    n = len(gold)
    diffs = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        g = [gold[i] for i in idx]
        diffs.append(metric(g, [pred_a[i] for i in idx]) - metric(g, [pred_b[i] for i in idx]))
    diffs.sort()
    return {
        "diff": metric(list(gold), list(pred_a)) - metric(list(gold), list(pred_b)),
        "lo": diffs[int(0.025 * n_resamples)],
        "hi": diffs[int(0.975 * n_resamples) - 1],
    }
