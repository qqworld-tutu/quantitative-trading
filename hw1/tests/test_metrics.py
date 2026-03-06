import math

from assignment1_part2.metrics import lag1_autocorr, log_returns, summary_stats


def test_log_returns_matches_log_diff():
    result = log_returns([100.0, 110.0, 121.0])
    assert len(result) == 2
    assert math.isclose(result[0], math.log(1.1), rel_tol=1e-9)
    assert math.isclose(result[1], math.log(1.1), rel_tol=1e-9)


def test_summary_stats_contains_required_fields():
    stats = summary_stats([0.01, -0.02, 0.03, 0.01])
    assert set(stats) == {
        "mean",
        "median",
        "std",
        "skew",
        "kurtosis",
        "min",
        "max",
        "autocorr1",
    }
    assert math.isclose(stats["autocorr1"], lag1_autocorr([0.01, -0.02, 0.03, 0.01]))

