import pandas as pd
import pytest
from baf_fraud.data import chronological_split


def test_chronological_split_no_target_or_time_leakage():
    df = pd.DataFrame({"month": [0, 1, 2, 3, 4, 5, 6, 7],
                       "fraud_bool": [0, 1, 0, 0, 0, 1, 1, 0],
                       "amount": range(8), "channel": ["web", "mobile"] * 4})
    (xtr, ytr), (xv, yv), (xt, yt) = chronological_split(df)
    assert (len(xtr), len(xv), len(xt)) == (6, 1, 1)
    assert {"fraud_bool", "month"}.isdisjoint(xtr.columns)
    assert xtr["amount"].tolist() == list(range(6))
    assert xv["amount"].tolist() == [6]
    assert xt["amount"].tolist() == [7]
    assert ytr.sum() == 2 and yv.sum() == 1 and yt.sum() == 0


def test_bad_month_fails():
    df = pd.DataFrame({"month": [0, 5, 6, 8], "fraud_bool": [0, 1, 0, 1]})
    with pytest.raises(ValueError):
        chronological_split(df)
