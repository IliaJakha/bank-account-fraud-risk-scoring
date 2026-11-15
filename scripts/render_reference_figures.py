"""Render restrained plots from historical recorded notebook summary values.

These visuals DO NOT represent a rerun of the model on the full Base.csv file.
They are completely reproducible from results/reference_notebook_results.json.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results" / "reference_notebook_results.json"
OUT = ROOT / "assets"


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def axes(figsize=(7.4, 4.3)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(axis="y", color="0.87", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    return fig, ax


def main():
    summary = json.loads(REFERENCE.read_text(encoding="utf-8"))
    counts = [summary["dataset"]["legitimate"], summary["dataset"]["fraud"]]
    fig, ax = axes()
    bars = ax.bar(["Legitimate", "Fraud"], [v / sum(counts) * 100 for v in counts],
                  color=["0.55", "0.25"], width=0.55)
    for bar, n in zip(bars, counts):
        ax.annotate(f"{n:,} ({n / sum(counts):.2%})",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", xytext=(0, 6), textcoords="offset points")
    ax.set_ylabel("Share of applications (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Class balance · Historical notebook results", loc="left")
    save(fig, "class_imbalance.png")

    months = sorted(summary["months"], key=int)
    fig, ax = axes()
    ax.plot([int(m) for m in months], [summary["months"][m] for m in months],
            "o-", color="0.25", lw=1.8, ms=5)
    ax.set_xticks(range(8))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("BAF month index (not calendar months)")
    ax.set_ylabel("Fraud prevalence (%)")
    ax.set_title("Fraud prevalence over time · Recorded observations", loc="left")
    save(fig, "monthly_fraud_rate.png")

    fig, ax = axes((8.3, 4.6))
    metrics = [("average_precision", "Average precision"),
               ("precision_at_1pct", "Precision@1%"),
               ("recall_at_1pct", "Recall@1%")]
    x = np.arange(len(metrics))
    for i, name in enumerate(["Logistic Regression", "Random Forest"]):
        vals = [summary["validation"][name][k] * 100 for k, _ in metrics]
        ax.bar(x + (i - 0.5) * 0.34, vals, width=0.32,
               label=name, color=["0.25", "0.65"][i])
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylabel("Metric (%)")
    ax.set_ylim(0, 31)
    ax.legend(frameon=False)
    ax.set_title("Validation month 6 · Recorded baseline results", loc="left")
    save(fig, "validation_comparison.png")

    matrix = np.asarray(summary["test"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.3, 4.5))
    ax.imshow(matrix, cmap="Greys", vmin=0, vmax=100000)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center",
                    color="white" if matrix[i, j] > 55000 else "black", fontsize=12)
    ax.set_xticks(range(2), ["Legitimate", "Fraud"])
    ax.set_yticks(range(2), ["Legitimate", "Fraud"])
    ax.set_xlabel("Predicted label (threshold = 0.5)")
    ax.set_ylabel("True label")
    ax.set_title("Test confusion matrix · Recorded baseline", loc="left")
    save(fig, "test_confusion_matrix.png")


if __name__ == "__main__":
    main()
