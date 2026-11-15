"""Time-forward BAF evaluation; compare on month 6, test once on month 7."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from .data import chronological_split, load_base
from .evaluation import DEFAULT_BUDGETS, budget_sweep, evaluate_model
from .modeling import SENTINEL_COLUMNS, build_candidates
from .plotting import (
    plot_budget_sweep,
    plot_distributions,
    plot_test,
    plot_validation,
)


def resolve_data_path(path: str | None, download: bool) -> Path:
    """Use local Base.csv or explicitly download through kagglehub."""
    if path:
        return Path(path)
    if download:
        try:
            import kagglehub
        except ImportError as exc:
            raise SystemExit("Install downloader: pip install -e '.[notebook]'") from exc
        root = Path(kagglehub.dataset_download("sgpjesus/bank-account-fraud-dataset-neurips-2022"))
        csv = root / "Base.csv"
        if not csv.is_file():
            matches = list(root.rglob("Base.csv"))
            if len(matches) != 1:
                raise FileNotFoundError("Expected exactly one Base.csv in downloaded dataset")
            csv = matches[0]
        return csv
    return Path("data/Base.csv")


def run_experiment(df, output, capacity=0.01):
    """Train on months 0–5; tune on 6; evaluate one selected model on 7."""
    if not 0 < capacity <= 1:
        raise ValueError("Review capacity must be in (0,1]")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = chronological_split(df)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    models = build_candidates(x_train)

    validation = {}
    validation_budgets = {}
    for name, model in models.items():
        print(f"Fitting {name}...", flush=True)
        model.fit(x_train, y_train)
        val_scores = model.predict_proba(x_val)[:, 1]
        validation[name] = evaluate_model(y_val, val_scores, capacity)
        validation_budgets[name] = budget_sweep(y_val, val_scores)
        print(
            f"  validation AP={validation[name]['average_precision']:.4f}; "
            f"Precision@{capacity:.1%}={validation[name]['precision_at_capacity']:.4f}"
        )

    # All selection metrics originate from validation, not test.
    selected = max(
        sorted(validation),
        key=lambda name: (
            validation[name]["average_precision"],
            validation[name]["precision_at_capacity"],
        ),
    )
    test_scores = models[selected].predict_proba(x_test)[:, 1]
    test_metrics = evaluate_model(y_test, test_scores, capacity)
    test_budgets = budget_sweep(y_test, test_scores)
    report = {
        "provenance": {
            "kind": "computed_from_local_dataset",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "data_file": "Base.csv (local path not recorded)",
            "pipeline": "baf_fraud.run.run_experiment",
            "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
        },
        "dataset": "BAF Base.csv (synthetic benchmark)",
        "data_rows": len(df),
        "selection_capacity_fraction": capacity,
        "preprocessing_experiment": {
            "sentinel_columns": list(SENTINEL_COLUMNS),
            "rule": "replace exact -1 with NaN, add indicators, median impute within train pipeline",
            "other_negative_numbers": "preserved unchanged",
        },
        "splits": {
            "train": {"months": "0–5", "rows": len(y_train), "fraud_count": int(y_train.sum())},
            "validation": {"months": "6", "rows": len(y_val), "fraud_count": int(y_val.sum())},
            "test": {"months": "7", "rows": len(y_test), "fraud_count": int(y_test.sum())},
        },
        "selection_rule": "highest validation AP; validation Precision@capacity tie-break",
        "validation": validation,
        "validation_budget_sweep": validation_budgets,
        "selected_model": selected,
        "test": test_metrics,
        "test_budget_sweep": test_budgets,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    figures = output / "figures"
    plot_distributions(df, figures)
    plot_validation(validation, figures)
    plot_budget_sweep(validation_budgets[selected], figures, split="validation")
    plot_budget_sweep(test_budgets, figures, split="test")
    plot_test(y_test.to_numpy(), test_scores, figures)
    print(
        f"Selected from validation: {selected}; test AP={test_metrics['average_precision']:.4f}; "
        f"fraud found in {test_metrics['reviewed']} reviews: {test_metrics['frauds_detected']}"
    )
    print(f"Saved metrics and figures to {output.resolve()}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Path to Base.csv; default data/Base.csv")
    parser.add_argument("--download", action="store_true", help="Fetch dataset using kagglehub")
    parser.add_argument("--output", default="results/generated", help="Run output directory")
    parser.add_argument("--capacity", type=float, default=0.01, help="Selection review fraction")
    args = parser.parse_args(argv)
    df = load_base(resolve_data_path(args.data, args.download))
    return run_experiment(df, args.output, args.capacity)


if __name__ == "__main__":
    main()
