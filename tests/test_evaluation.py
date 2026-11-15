import pytest

from baf_fraud.evaluation import capacity_metrics, classification_metrics


def test_capacity_math_and_lift():
    y = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    scores = [.1, .9, .01, .8, .2, .3, .4, .5, .6, .95]
    r = capacity_metrics(y, scores, capacity=.2)
    assert r["reviewed"] == 2
    assert r["frauds_detected"] == 1
    assert r["precision_at_capacity"] == pytest.approx(.5)
    assert r["recall_at_capacity"] == pytest.approx(.5)
    assert r["lift_over_prevalence"] == pytest.approx(2.5)


def test_non_round_capacity_ceil():
    r = capacity_metrics([0] * 9 + [1], [.01] * 9 + [.99], capacity=.11)
    assert r["reviewed"] == 2


def test_ties_deterministic():
    r = capacity_metrics([1, 0, 1, 0], [.7] * 4, capacity=.5)
    assert r["frauds_detected"] == 1


def test_zero_fraud_graceful():
    r = capacity_metrics([0, 0, 0], [.1, .2, .3])
    assert r["recall_at_capacity"] == r["lift_over_prevalence"] == 0


def test_invalid_capacity():
    with pytest.raises(ValueError):
        capacity_metrics([0, 1], [.1, .9], capacity=0)


def test_confusion_matrix_orientation():
    r = classification_metrics([0, 0, 1, 1], [.1, .9, .8, .2])
    assert r["confusion_matrix"] == [[1, 1], [1, 1]]
    assert r["precision_at_threshold"] == pytest.approx(.5)
