"""Paired significance test behaviour."""

import pytest

from ragsearch.eval.significance import (
    paired_bootstrap_test,
    paired_randomization_test,
)


def test_identical_runs_are_not_significant():
    a = [0.1, 0.5, 0.9, 0.3, 0.7]
    r = paired_randomization_test(a, list(a), iterations=2000)
    assert r.delta == pytest.approx(0.0)
    assert r.p_value == pytest.approx(1.0, abs=1e-6)


def test_large_consistent_gain_is_significant():
    a = [0.20, 0.25, 0.22, 0.19, 0.30, 0.21, 0.24, 0.23, 0.26, 0.20]
    b = [x + 0.15 for x in a]  # candidate wins every single query
    r = paired_randomization_test(a, b, iterations=5000)
    assert r.delta == pytest.approx(0.15)
    assert r.p_value < 0.01


def test_noise_difference_is_not_significant():
    a = [0.5, 0.4, 0.6, 0.55, 0.45, 0.5, 0.48, 0.52]
    b = [0.51, 0.39, 0.61, 0.54, 0.46, 0.49, 0.49, 0.51]  # tiny wobble both ways
    r = paired_randomization_test(a, b, iterations=5000)
    assert r.p_value > 0.1


def test_randomization_is_deterministic_with_seed():
    a = [0.1, 0.4, 0.2, 0.8, 0.5]
    b = [0.2, 0.3, 0.4, 0.7, 0.6]
    r1 = paired_randomization_test(a, b, iterations=3000, seed=7)
    r2 = paired_randomization_test(a, b, iterations=3000, seed=7)
    assert r1.p_value == r2.p_value


def test_bootstrap_reports_ci_bracketing_estimate():
    a = [0.2, 0.25, 0.22, 0.19, 0.30, 0.21, 0.24, 0.23]
    b = [x + 0.1 for x in a]
    r = paired_bootstrap_test(a, b, iterations=5000)
    assert r.method == "bootstrap"
    assert r.ci_low <= r.delta <= r.ci_high
    assert r.p_value < 0.05


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        paired_randomization_test([0.1, 0.2], [0.1])
