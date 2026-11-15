"""Operational capacity sweep has deterministic and mathematically valid outputs."""

import pytest

from baf_fraud.evaluation import DEFAULT_BUDGETS, budget_sweep, capacity_metrics


def test_budget_sweep_review_counts_are_monotonic():
    y = [1, 0, 1, 0] * 25
    scores = [i / 100 for i in range(100)]
    result = budget_sweep(y, scores)
    assert len(result) == len(DEFAULT_BUDGETS) == 5
    cases = [value["reviewed"] for value in result.values()]
    assert cases == sorted(cases)
    assert result["1.00%"]["reviewed"] == 1
    assert result["5.00%"]["reviewed"] == 5


def test_budget_sweep_matches_capacity_function():
    y = [0, 1, 0, 1, 0]
    scores = [0.1, 0.9, 0.2, 0.8, 0.3]
    result = budget_sweep(y, scores, (0.4,))
    expected = capacity_metrics(y, scores, 0.4)
    assert result["40.00%"]["frauds_detected"] == expected["frauds_detected"]
    assert result["40.00%"]["precision_at_capacity"] == pytest.approx(1.0)
