"""Report figure generation reading directly from computed CSV outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve


def plot_lift_comparison(results_raw: pd.DataFrame, out_dir: Path) -> Path:
    """Plot 3-seed AP lift across all four transfer directions (all vs size-restricted)."""
    order = [
        ("ibm", "ibm", "IBM → IBM"),
        ("elliptic2", "elliptic2", "Elliptic2 → Elliptic2"),
        ("ibm", "elliptic2", "IBM → Elliptic2"),
        ("elliptic2", "ibm", "Elliptic2 → IBM"),
    ]
    mnames = ["SizeOnly", "LogReg", "RandomForest", "GradBoosting"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for ax, sm in zip(axes, (False, True), strict=False):
        sub = results_raw[results_raw["size_restricted"] == sm]
        agg = sub.groupby(["train_on", "test_on", "model"])["lift"].agg(["mean", "std"])
        w = 0.8 / len(mnames)
        for j, m in enumerate(mnames):
            vals = [
                agg.loc[(s, t, m), "mean"] if (s, t, m) in agg.index else np.nan
                for s, t, _ in order
            ]
            errs = [
                agg.loc[(s, t, m), "std"] if (s, t, m) in agg.index else 0.0 for s, t, _ in order
            ]
            ax.bar(np.arange(4) + j * w - 0.4 + w / 2, vals, w, yerr=errs, capsize=3, label=m)
        ax.axhline(1, color="grey", ls="--", lw=1)
        ax.set_xticks(range(4), [o[2] for o in order], fontsize=9)
        ax.set_title("Size-restricted (≤10 nodes)" if sm else "All subgraphs (no size restriction)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("PR-AUC ÷ prevalence  (1 = random)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out_path = out_dir / "results_lift.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_archival_from_raw_csv(results_raw: pd.DataFrame, out_dir: Path) -> Path:
    """Compute unrestricted 3-seed means directly from results_raw.csv (no hard-coded literals)."""
    label_map = {
        ("ibm", "ibm"): "IBM → IBM",
        ("elliptic2", "elliptic2"): "Elliptic2 → Elliptic2",
        ("ibm", "elliptic2"): "IBM → Elliptic2",
        ("elliptic2", "ibm"): "Elliptic2 → IBM",
    }
    unrestricted = results_raw[~results_raw["size_restricted"]].copy()
    grouped = (
        unrestricted.groupby(["train_on", "test_on", "model"], as_index=False)[["pr_auc", "f1"]]
        .mean()
        .rename(columns={"pr_auc": "archival_mean_AP", "f1": "archival_mean_F1"})
    )
    grouped["experiment"] = [
        label_map.get((str(r["train_on"]), str(r["test_on"])), f"{r['train_on']} -> {r['test_on']}")
        for _, r in grouped.iterrows()
    ]
    archival = grouped[["experiment", "model", "archival_mean_AP", "archival_mean_F1"]].round(3)
    archival.to_csv(out_dir / "unit5_archival_published_means.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    order = ["IBM → IBM", "Elliptic2 → Elliptic2", "IBM → Elliptic2", "Elliptic2 → IBM"]
    for ax, measure_col, title in (
        (axes[0], "archival_mean_AP", "Mean average precision"),
        (axes[1], "archival_mean_F1", "Mean F1 at validation threshold"),
    ):
        pivot = archival.pivot(index="experiment", columns="model", values=measure_col)
        present_order = [o for o in order if o in pivot.index]
        pivot = pivot.reindex(present_order)
        pivot.plot(kind="bar", ax=ax, rot=20, width=0.78, legend=False)
        ax.set_title(title + " — 3-seed unrestricted results")
        ax.set_xlabel("")
        ax.set_ylabel("Score (0–1)")
        ax.set_ylim(0, max(0.72, float(pivot.max().max()) * 1.1) if not pivot.empty else 0.72)
        ax.grid(axis="y", alpha=0.2)
    axes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "unit5_archival_model_comparison.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_tuned_diagnostics(
    selected_hyperparams: pd.DataFrame,
    tuned_evaluation: pd.DataFrame,
    tuned_predictions: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Generate PR curves, confusion matrices, and baseline-vs-tuned AP chart."""
    chosen = (
        selected_hyperparams.sort_values("validation_ap_selected_on", ascending=False)
        .drop_duplicates("source")
        .set_index("source")
    )
    domains = [d for d in ("ibm", "elliptic2") if d in chosen.index]
    if len(domains) == 2 and not tuned_predictions.empty:
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        fig_cm, axes_cm = plt.subplots(2, 2, figsize=(11, 9))
        plot_records: list[dict[str, Any]] = []
        for i, src in enumerate(domains):
            best_name = str(chosen.loc[src, "model"])
            for j, tgt in enumerate(domains):
                pred = tuned_predictions[
                    (tuned_predictions["source"] == src)
                    & (tuned_predictions["target"] == tgt)
                    & (tuned_predictions["model"] == best_name)
                ]
                if pred.empty:
                    continue
                y = pred["y_true"].to_numpy()
                p = pred["probability"].to_numpy()
                threshold = float(pred["threshold"].iloc[0])
                prec, rec, _ = precision_recall_curve(y, p)
                ax = axes[i, j]
                ax.step(
                    rec,
                    prec,
                    where="post",
                    label=f"{best_name} (AP={average_precision_score(y, p):.3f})",
                )
                ax.axhline(
                    y.mean(), color="gray", linestyle="--", label=f"Positive rate={y.mean():.3f}"
                )
                ax.set(
                    xlim=(0, 1),
                    ylim=(0, 1),
                    xlabel="Recall",
                    ylabel="Precision",
                    title=f"{src.upper()} → {tgt.upper()} | Tuned, source-selected",
                )
                ax.legend(fontsize=8)
                ax.grid(alpha=0.2)

                cm = confusion_matrix(y, pred["predicted_class"].to_numpy(), labels=[0, 1])
                axes_cm[i, j].imshow(cm, cmap="Blues")
                axes_cm[i, j].set_title(
                    f"{src.upper()} → {tgt.upper()} | {best_name} | thr={threshold:.3f}"
                )
                axes_cm[i, j].set_xticks([0, 1], ["Pred 0", "Pred 1"])
                axes_cm[i, j].set_yticks([0, 1], ["True 0", "True 1"])
                for row in range(2):
                    for col in range(2):
                        axes_cm[i, j].text(
                            col, row, f"{cm[row, col]:,}", va="center", ha="center", fontsize=12
                        )
                plot_records.append(
                    {
                        "source": src,
                        "target": tgt,
                        "model": best_name,
                        "threshold": threshold,
                        "tn": int(cm[0, 0]),
                        "fp": int(cm[0, 1]),
                        "fn": int(cm[1, 0]),
                        "tp": int(cm[1, 1]),
                        "positive_rate": float(y.mean()),
                    }
                )
        fig.suptitle("Held-out precision–recall curves (tuned)", y=1.01, fontsize=15)
        fig.tight_layout()
        fig.savefig(
            out_dir / "unit5_tuned_precision_recall_curves.png", dpi=160, bbox_inches="tight"
        )
        plt.close(fig)

        fig_cm.suptitle("Held-out confusion matrices", y=1.01, fontsize=15)
        fig_cm.tight_layout()
        fig_cm.savefig(out_dir / "unit5_tuned_confusion_matrices.png", dpi=160, bbox_inches="tight")
        plt.close(fig_cm)
        pd.DataFrame(plot_records).to_csv(out_dir / "unit5_selected_error_counts.csv", index=False)

    if not tuned_evaluation.empty:
        fig2, ax2 = plt.subplots(figsize=(12, 5.5))
        labels = (
            tuned_evaluation["source"].str.upper()
            + " → "
            + tuned_evaluation["target"].str.upper()
            + " | "
            + tuned_evaluation["model"]
        )
        plot_data = tuned_evaluation.assign(setting=labels).pivot_table(
            index="setting", columns="variant", values="pr_auc"
        )
        cols_avail = [c for c in ("baseline_same_split", "tuned") if c in plot_data.columns]
        if len(cols_avail) == 2:
            plot_data[cols_avail].dropna().plot(kind="barh", ax=ax2)
            ax2.set_title("Same-split baseline vs tuned AP")
            ax2.set_xlabel("Average precision")
            ax2.set_ylabel("")
            ax2.grid(axis="x", alpha=0.2)
            fig2.tight_layout()
            fig2.savefig(
                out_dir / "unit5_tuned_vs_baseline_same_split.png", dpi=160, bbox_inches="tight"
            )
        plt.close(fig2)


def plot_base_rate_sensitivity(ext_cfg: dict[str, Any], out_dir: Path) -> Path:
    """Render illustrative Bayes' rule base-rate sensitivity curve."""
    bcfg = ext_cfg.get("base_rate_illustration", {})
    tpr = float(bcfg.get("tpr", 0.80))
    fpr = float(bcfg.get("fpr", 0.02))
    prevalences = np.asarray(
        bcfg.get("prevalences", [0.001, 0.005, 0.01, 0.023, 0.05, 0.10]), dtype=float
    )
    ppv = (tpr * prevalences) / (tpr * prevalences + fpr * (1.0 - prevalences))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(100 * prevalences, 100 * ppv, marker="o")
    ax.set(
        xlabel="Hypothetical true positive prevalence (%)",
        ylabel="Hypothetical positive predictive value (%)",
        title=f"Base-rate effect at assumed {tpr:.0%} TPR / {fpr:.0%} FPR",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out_path = out_dir / "unit6_hypothetical_base_rate_sensitivity.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def generate_all_reports(out_dir: Path, ext_cfg: dict[str, Any]) -> list[str]:
    """Regenerate key evaluation and extension figures from saved CSV outputs."""
    generated: list[str] = []
    raw_csv = out_dir / "results_raw.csv"
    if raw_csv.exists():
        results_raw = pd.read_csv(raw_csv)
        generated.append(str(plot_lift_comparison(results_raw, out_dir)))
        generated.append(str(plot_archival_from_raw_csv(results_raw, out_dir)))

    sel_csv = out_dir / "unit4_selected_hyperparameters.csv"
    eval_csv = out_dir / "unit5_holdout_comparison.csv"
    pred_csv = out_dir / "unit5_test_predictions.csv"
    if sel_csv.exists() and eval_csv.exists() and pred_csv.exists():
        plot_tuned_diagnostics(
            pd.read_csv(sel_csv),
            pd.read_csv(eval_csv),
            pd.read_csv(pred_csv),
            out_dir,
        )
        generated.append(str(out_dir / "unit5_tuned_precision_recall_curves.png"))

    generated.append(str(plot_base_rate_sensitivity(ext_cfg, out_dir)))

    ext_dir = out_dir / "research_extension"
    if (ext_dir / "domain_shift_train_only.csv").exists():
        ds = pd.read_csv(ext_dir / "domain_shift_train_only.csv")
        overall = ds[ds["group"] == "overall"].sort_values("KS_statistic")
        if not overall.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(overall["feature"], overall["KS_statistic"])
            ax.set(
                xlim=(0, 1),
                xlabel="Two-sample KS statistic (0=same distributions)",
                title="IBM vs Elliptic2 feature distribution shift (train partitions)",
            )
            fig.tight_layout()
            p = ext_dir / "domain_shift_KS.png"
            fig.savefig(p, dpi=160)
            plt.close(fig)
            generated.append(str(p))

    if (ext_dir / "typology_recall_breakdown.csv").exists():
        typ = pd.read_csv(ext_dir / "typology_recall_breakdown.csv", index_col=0)
        if not typ.empty:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            typ[["recall_14feat", "recall_20feat"]].plot(kind="barh", ax=ax)
            ax.set_xlim(0, 1.05)
            ax.set_xlabel("Test recall at validation-selected threshold")
            ax.set_title("IBM outer-test recall by laundering typology (14 vs 20 features)")
            ax.grid(axis="x", alpha=0.25)
            fig.tight_layout()
            p = ext_dir / "typology_recall_breakdown.png"
            fig.savefig(p, dpi=160)
            plt.close(fig)
            generated.append(str(p))

    return generated
