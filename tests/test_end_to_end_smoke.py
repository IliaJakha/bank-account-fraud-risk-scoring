"""Small synthetic dry run, checking outputs without requiring the BAF dataset."""

import numpy as np
import pandas as pd

from baf_fraud.run import run_experiment


def test_end_to_end_creates_evidence_files(tmp_path):
    n = 80
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "month": np.repeat(np.arange(8), 10),
            "fraud_bool": (np.arange(n) % 5 == 0).astype(int),
            "bank_months_count": np.where(np.arange(n) % 7 == 0, -1, np.arange(n) % 22),
            "credit_risk_score": rng.standard_normal(n) * 40,
            "source": np.where(np.arange(n) % 2 == 0, "APP", "INTERNET"),
        }
    )
    result = run_experiment(df, tmp_path)
    assert result["provenance"]["kind"] == "computed_from_local_dataset"
    assert len(result["validation"]) == 4
    assert result["selected_model"] in result["validation"]
    assert result["test"]["reviewed"] == 1
    assert (tmp_path / "metrics.json").is_file()
    for name in ["validation_budget_sweep", "test_budget_sweep", "test_precision_recall_curve"]:
        assert (tmp_path / "figures" / f"{name}.png").is_file()
