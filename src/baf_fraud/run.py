"""Train, choose, and evaluate the BAF fraud-risk baseline on a time-forward split."""
import argparse
import json
from pathlib import Path
from .data import load_base, chronological_split
from .modeling import build_models
from .evaluation import evaluate_model
from .plotting import plot_distributions, plot_validation, plot_test


def resolve_data_path(path: str | None, download: bool) -> Path:
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Path to Base.csv; default data/Base.csv")
    parser.add_argument("--download", action="store_true", help="Fetch dataset using kagglehub")
    parser.add_argument("--output", default="results/generated", help="Local output directory")
    parser.add_argument("--capacity", type=float, default=.01, help="Fraction for human review")
    args = parser.parse_args(argv)
    df = load_base(resolve_data_path(args.data, args.download))
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = chronological_split(df)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    print(f"Train: {len(X_train):,}  |  validation: {len(X_val):,}  |  test: {len(X_test):,}")
    models = build_models(X_train)
    val_metrics = {}
    for name, model in models.items():
        print(f"Training {name} ...", flush=True)
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        val_metrics[name] = evaluate_model(y_val, val_prob, args.capacity)
        print(f"  validation AP={val_metrics[name]['average_precision']:.4f}  "
              f"P@capacity={val_metrics[name]['precision_at_capacity']:.4f}")
    selected = max(sorted(val_metrics), key=lambda name: (
        val_metrics[name]["average_precision"],
        val_metrics[name]["precision_at_capacity"],
    ))
    test_prob = models[selected].predict_proba(X_test)[:, 1]
    test_metrics = evaluate_model(y_test, test_prob, args.capacity)
    report = {
        "source": "BAF synthetic Base.csv",
        "data_rows": len(df), "review_capacity_fraction": args.capacity,
        "splits": {
            "train": {"months": "0–5", "rows": len(y_train), "fraud_count": int(y_train.sum())},
            "validation": {"months": "6", "rows": len(y_val), "fraud_count": int(y_val.sum())},
            "test": {"months": "7", "rows": len(y_test), "fraud_count": int(y_test.sum())},
        },
        "model_selection_rule": "highest validation average precision; P@capacity tie-break",
        "validation": val_metrics, "selected_model": selected, "test": test_metrics,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_distributions(df, output / "figures")
    plot_validation(val_metrics, output / "figures")
    plot_test(y_test.to_numpy(), test_prob, output / "figures")
    print(f"Selected on validation: {selected}")
    print(f"TEST AP: {test_metrics['average_precision']:.4f} | "
          f"frauds found at {args.capacity:.1%} capacity: {test_metrics['frauds_detected']} "
          f"of {test_metrics['reviewed']} reviews")
    print(f"Saved reproducible outputs to {output.resolve()}")
    return report


if __name__ == "__main__":
    main()
