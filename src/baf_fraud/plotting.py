"""Report figures from actual data or predictions; never reconstructed model curves."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import PrecisionRecallDisplay, ConfusionMatrixDisplay

NAVY, BLUE, TEAL, ORANGE = "#15284B", "#3F70D2", "#109C93", "#EA8A4C"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#35445A", labelsize=10)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E7EDF5")


def plot_distributions(df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = df["fraud_bool"].value_counts().reindex([0, 1], fill_value=0)
    total = counts.sum()
    fig, ax = plt.subplots(figsize=(9, 3.7), layout="constrained")
    fig.patch.set_facecolor("white")
    ax.barh([0], [counts.loc[0] / total * 100], color=BLUE, height=.5)
    ax.barh([0], [counts.loc[1] / total * 100], left=[counts.loc[0] / total * 100],
            color=ORANGE, height=.5)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Share of applications (%)")
    ax.set_title("Class imbalance  |  BAF Base", color=NAVY, fontsize=16, weight="bold", pad=20)
    ax.text(.5, .88, f"Legitimate: {counts.loc[0]:,} ({counts.loc[0]/total:.2%})",
            transform=ax.transAxes, color=BLUE, weight="bold", ha="center")
    ax.text(.5, -.40, f"Fraud: {counts.loc[1]:,} ({counts.loc[1]/total:.2%})",
            transform=ax.transAxes, color=ORANGE, weight="bold", ha="center")
    _style(ax)
    ax.set_ylim(-.6, 1.1)
    fig.savefig(output_dir / "class_imbalance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    series = df.groupby("month")["fraud_bool"].mean().mul(100).sort_index()
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    fig.patch.set_facecolor("white")
    ax.plot(series.index, series.values, marker="o", linewidth=3, markersize=8, color=BLUE)
    ax.fill_between(series.index, 0, series.values, color=BLUE, alpha=.07)
    ax.set_ylim(0, max(1.75, series.max() * 1.12))
    ax.set_xticks(series.index)
    ax.set_xlabel("Dataset month index (not calendar months)")
    ax.set_ylabel("Fraud rate (%)")
    ax.set_title("Fraud prevalence changes over time", color=NAVY, fontsize=16, weight="bold", pad=16)
    for x, y in zip(series.index, series.values):
        ax.annotate(f"{y:.2f}%", (x, y), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=9)
    _style(ax)
    fig.savefig(output_dir / "monthly_fraud_rate.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_validation(results, output_dir):
    labels = ["Avg. precision", "Precision @ 1%", "Recall @ 1%"]
    keys = ["average_precision", "precision_at_capacity", "recall_at_capacity"]
    models = list(results)
    if len(models) != 2:
        return
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    fig.patch.set_facecolor("white")
    for i, model in enumerate(models):
        values = [results[model][k] * 100 for k in keys]
        bars = ax.bar(x + (i - .5) * .33, values, .31,
                      label=model, color=(BLUE if i == 0 else TEAL))
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 32)
    ax.set_title("Validation  |  Month 6", color=NAVY, fontsize=16, weight="bold", pad=18)
    ax.legend(frameon=False, loc="upper right")
    _style(ax)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "validation_model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_test(y_true, scores, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    PrecisionRecallDisplay.from_predictions(y_true, scores, ax=ax, name="Selected baseline", color=BLUE)
    ax.axhline(np.mean(y_true), color=ORANGE, ls="--", label="Test fraud prevalence")
    ax.set_title("Held-out test precision–recall", color=NAVY, fontsize=15, weight="bold", pad=15)
    ax.legend(frameon=False)
    fig.savefig(out / "test_precision_recall_curve.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    ConfusionMatrixDisplay.from_predictions(y_true, scores >= .5,
                    display_labels=["Legitimate", "Fraud"], cmap="Blues", ax=ax,
                    values_format="d", colorbar=False)
    ax.set_title("Held-out test confusion matrix", color=NAVY, fontsize=15, weight="bold", pad=15)
    fig.savefig(out / "test_confusion_matrix.png", dpi=180)
    plt.close(fig)
