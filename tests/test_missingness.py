"""Sentinel handling preserves valid negative values and works within pipelines."""

import numpy as np
import pandas as pd

from baf_fraud.modeling import SentinelMissingness, build_candidates, build_sentinel_model


def test_sentinel_is_selective_and_preserves_numeric_negatives():
    x = pd.DataFrame(
        {
            "prev_address_months_count": [-1, 12, 0],
            "bank_months_count": [3, -1, 20],
            "credit_risk_score": [-12, -1, 37],
            "intended_balcon_amount": [-1.7, -1.0, 8.5],
            "source": ["APP", "APP", "INTERNET"],
        }
    )
    out = SentinelMissingness().transform(x)
    assert np.isnan(out.loc[0, "prev_address_months_count"])
    assert np.isnan(out.loc[1, "bank_months_count"])
    assert out["prev_address_months_count__missing"].tolist() == [1, 0, 0]
    assert out["bank_months_count__missing"].tolist() == [0, 1, 0]
    assert out["credit_risk_score"].tolist() == [-12, -1, 37]
    assert out["intended_balcon_amount"].tolist() == [-1.7, -1.0, 8.5]
    assert x.loc[0, "prev_address_months_count"] == -1  # no mutation


def test_sentinel_model_handles_unseen_missing_test_cases():
    train = pd.DataFrame(
        {
            "bank_months_count": [1, 5, -1, 3, 10, -1, 9, 4],
            "source": ["APP", "WEB"] * 4,
        }
    )
    y = [0, 1] * 4
    model = build_sentinel_model(train, c=0.3).fit(train, y)
    values = model.predict_proba(pd.DataFrame({"bank_months_count": [-1, 12], "source": ["WEB", "X"]}))
    assert values.shape == (2, 2)
    assert np.isfinite(values).all()


def test_candidate_grid_contains_baselines_and_two_experiments():
    train = pd.DataFrame({"amount": [1.0, -1.0, 2.0], "source": ["WEB"] * 3})
    candidates = build_candidates(train)
    assert set(candidates) == {
        "Logistic Regression",
        "Random Forest",
        "Sentinel LR (C=0.3)",
        "Sentinel LR (C=1.0)",
    }
