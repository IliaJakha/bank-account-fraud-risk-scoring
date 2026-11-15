"""Conventional Matplotlib evidence plots; no decorative or synthetic curves."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _axes():
    """Simple white charts with default scientific styling."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.5, color="0.85", zorder=0)
    ax.set_axisbelow(True)
    return fig, ax


def plot_distributions(df, output_dir):
    """Descriptive class balance and month-index prevalence, from actual rows."""
    output_dir = Path(output_dir)
    counts = df["fraud_bool"].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = _axes()
    percentages = counts / counts.sum() * 100
    bars = ax.bar(["Legitimate", "Fraud"], percentages, color=["0.55", "0.25"], width=0.55)
    for bar, n, pct in zip(bars, counts, percentages):
        ax.annotate(
            f"{n:,} ({pct:.2f}%)",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=10,
            xytext=(0, 5),
            textcoords="offset points",
        )
    ax.set_ylim(0, 110)
    ax.set_ylabel("Applications (%)")
    ax.set_title("Class distribution · BAF Base", loc="left")
    _save(fig, output_dir / "class_imbalance.png")

    monthly = df.groupby("month")["fraud_bool"].agg(fraud_rate="mean", applications="size")
    fig, ax = _axes()
    ax.plot(monthly.index, monthly["fraud_rate"] * 100, "o-", color="0.2", linewidth=1.6)
    ax.set_xticks(monthly.index)
    ax.set_xlabel("Dataset month index (0–7; not calendar months)")
    ax.set_ylabel("Fraud prevalence (%)")
    ax.set_title("Observed monthly fraud prevalence", loc="left")
    ax.set_ylim(bottom=0)
    _save(fig, output_dir / "monthly_fraud_rate.png")


def plot_validation(results, output_dir):
    """Model comparison computed on the validation split, not on test data."""
    metrics = [
        ("average_precision", "Average precision"),
        ("precision_at_capacity", "Precision at capacity"),
        ("recall_at_capacity", "Recall at capacity"),
    ]
    names = list(results)
    fig, ax = _axes()
    x = np.arange(len(metrics))
    width = 0.75 / len(names)
    shades = np.linspace(0.25, 0.75, num=len(names))
    for i, (name, shade) in enumerate(zip(names, shades)):
        vals = [results[name][key] * 100 for key, _ in metrics]
        ax.bar(x + (i - (len(names) - 1) / 2) * width, vals, width * 0.92,
               color=str(shade), label=name)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylabel("Metric (%)")
    ax.set_ylim(bottom=0)
    ax.set_title("Validation comparison · Month 6", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    _save(fig, Path(output_dir) / "validation_model_comparison.png")


def plot_budget_sweep(sweep, output_dir, split="validation"):
    """Plot only budgets evaluated from actual model predictions."""
    budgets = [100 * item["capacity_fraction"] for item in sweep.values()]
    precision = [100 * item["precision_at_capacity"] for item in sweep.values()]
    recall = [100 * item["recall_at_capacity"] for item in sweep.values()]
    fig, ax = _axes()
    ax.plot(budgets, precision, "o-", color="0.2", label="Precision at budget")
    ax.plot(budgets, recall, "s--", color="0.6", label="Recall at budget")
    ax.set_xlabel("Applications reviewed (%)")
    ax.set_ylabel("Metric (%)")
    ax.set_title(f"Investigation budget sensitivity · {split.title()}", loc="left")
    ax.legend(frameon=False)
    _save(fig, Path(output_dir) / f"{split}_budget_sweep.png")


def plot_test(y_true, scores, output_dir):
    """Plot actual held-out score-ranking curve and 0.5-threshold confusion matrix."""
    out = Path(output_dir)
    fig, ax = _axes()
    PrecisionRecallDisplay.from_predictions(
        y_true, scores, ax=ax, name="Selected validation candidate", color="0.2"
    )
    ax.axhline(np.mean(y_true), linestyle="--", color="0.6", label="Test prevalence")
    ax.set_title("Held-out test · Precision–recall curve", loc="left")
    ax.legend(frameon=False)
    _save(fig, out / "test_precision_recall_curve.png")

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        scores >= 0.5,
        display_labels=["Legitimate", "Fraud"],
        cmap="Greys",
        ax=ax,
        values_format="d",
        colorbar=False,
    )
    ax.set_title("Held-out test · Threshold 0.5", loc="left")
    _save(fig, out / "test_confusion_matrix.png")
