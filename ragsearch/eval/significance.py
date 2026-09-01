"""Paired significance tests for comparing two configs on the same queries.

Both tests are paired: they consume two equal-length arrays of per-query
scores (``a[i]`` and ``b[i]`` are the same query under two configs).

* ``paired_randomization_test`` -- the standard IR test. Under the null the
  two configs are exchangeable, so for each query we flip a coin to swap
  ``a[i]``/``b[i]`` and rebuild the mean difference; the p-value is how often
  the shuffled |difference| reaches the observed one.
* ``paired_bootstrap_test`` -- resample queries with replacement; report the
  mean difference, a percentile confidence interval, and the fraction of
  resamples whose sign disagrees with the observed effect.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SigResult:
    metric: str
    mean_a: float
    mean_b: float
    delta: float  # mean_b - mean_a
    p_value: float
    method: str
    n: int
    ci_low: float | None = None
    ci_high: float | None = None

    def __str__(self) -> str:
        ci = ""
        if self.ci_low is not None:
            ci = f"  95% CI [{self.ci_low:+.4f}, {self.ci_high:+.4f}]"
        return (
            f"{self.metric}: {self.mean_a:.4f} -> {self.mean_b:.4f} "
            f"(Δ{self.delta:+.4f}, p={self.p_value:.4f}, {self.method}, n={self.n})"
            + ci
        )


def _mean(xs) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def paired_randomization_test(
    a, b, *, metric: str = "metric", iterations: int = 10000, seed: int = 12345
) -> SigResult:
    if len(a) != len(b):
        raise ValueError("paired test needs equal-length inputs")
    a = list(map(float, a))
    b = list(map(float, b))
    n = len(a)
    diffs = [bi - ai for ai, bi in zip(a, b)]
    observed = abs(_mean(diffs))
    rng = random.Random(seed)
    hits = 0
    for _ in range(iterations):
        total = 0.0
        for d in diffs:
            total += d if rng.random() < 0.5 else -d
        if abs(total / n) >= observed - 1e-12:
            hits += 1
    p = (hits + 1) / (iterations + 1)
    return SigResult(metric, _mean(a), _mean(b), _mean(diffs), p, "randomization", n)


def paired_bootstrap_test(
    a, b, *, metric: str = "metric", iterations: int = 10000, seed: int = 12345,
    alpha: float = 0.05,
) -> SigResult:
    if len(a) != len(b):
        raise ValueError("paired test needs equal-length inputs")
    a = list(map(float, a))
    b = list(map(float, b))
    n = len(a)
    diffs = [bi - ai for ai, bi in zip(a, b)]
    observed = _mean(diffs)
    rng = random.Random(seed)
    resampled = []
    opposite = 0
    for _ in range(iterations):
        idx = [rng.randrange(n) for _ in range(n)]
        m = _mean([diffs[i] for i in idx])
        resampled.append(m)
        if (m > 0) != (observed > 0):
            opposite += 1
    resampled.sort()
    lo = resampled[int((alpha / 2) * iterations)]
    hi = resampled[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    p = 2 * min(opposite, iterations - opposite) / iterations
    p = min(1.0, max(p, 1 / iterations))
    return SigResult(
        metric, _mean(a), _mean(b), observed, p, "bootstrap", n, lo, hi
    )
