"""Fraud metrics for binary classification under investigation-capacity limits."""

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

DEFAULT_BUDGETS = (0.0025, 0.005, 0.01, 0.02, 0.05)


def capacity_metrics(y_true, scores, capacity=0.01):
    """Review top ceil(n*capacity) scores; break ties by original row order."""
    if not 0 < capacity <= 1:
        raise ValueError("Capacity must be within (0, 1]")
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(scores, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("Nonempty, equally sized 1D inputs required")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError("Labels must be binary and scores must be finite")
    k = math.ceil(len(y) * capacity)
    positions = np.arange(len(y))
    selected = np.lexsort((positions, -p))[:k]
    detected = int(y[selected].sum())
    positives = int(y.sum())
    p_at_k = detected / k
    prevalence = positives / len(y)
    return {
        "reviewed": int(k),
        "frauds_detected": detected,
        "precision_at_capacity": p_at_k,
        "recall_at_capacity": detected / positives if positives else 0.0,
        "lift_over_prevalence": p_at_k / prevalence if prevalence else 0.0,
    }


def classification_metrics(y_true, scores, threshold=0.5):
    """Threshold metrics plus the threshold-free Average Precision (AP)."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(scores, dtype=float)
    predicted = (p >= threshold).astype(int)
    cm = confusion_matrix(y, predicted, labels=[0, 1])
    return {
        "precision_at_threshold": float(precision_score(y, predicted, zero_division=0)),
        "recall_at_threshold": float(recall_score(y, predicted, zero_division=0)),
        "f1_at_threshold": float(f1_score(y, predicted, zero_division=0)),
        "average_precision": float(average_precision_score(y, p)) if y.sum() else 0.0,
        "confusion_matrix": cm.tolist(),
    }


def evaluate_model(y_true, scores, capacity=0.01):
    """Evaluate AP, 0.5-threshold metrics and top-budget investigation metrics."""
    return {
        **classification_metrics(y_true, scores),
        **capacity_metrics(y_true, scores, capacity),
    }


def budget_sweep(y_true, scores, capacities=DEFAULT_BUDGETS):
    """Describe the same ranking at several review budgets, no model selection."""
    return {
        f"{capacity:.2%}": {
            "capacity_fraction": capacity,
            **capacity_metrics(y_true, scores, capacity),
        }
        for capacity in capacities
    }
