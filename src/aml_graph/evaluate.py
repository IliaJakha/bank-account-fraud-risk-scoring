"""Model training, validation-only threshold selection, calibration, and extensions."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from aml_graph.features import FEATURES, SIZE_COLS, transaction_activity_features
from aml_graph.models import ValidationPlattCalibrator, baseline_models, tuned_candidates
from aml_graph.sampling import generate_common_candidates
from aml_graph.splits import (
    account_disjoint_partition,
    chronological_embargo_split,
    make_splits,
    validation_indices,
)

if TYPE_CHECKING:
    from aml_graph.sampling import EdgeIndex


def best_thr(y: np.ndarray, p: np.ndarray) -> float:
    """Choose F1-maximizing threshold from validation data ONLY."""
    pr, rc, th = precision_recall_curve(y, p)
    f = 2 * pr * rc / np.clip(pr + rc, 1e-12, None)
    return float(th[np.nanargmax(f[:-1])]) if len(th) else 0.5


def safe_ap(y: np.ndarray | pd.Series, p: np.ndarray) -> float:
    y_arr = np.asarray(y, dtype=int)
    if len(y_arr) == 0 or np.unique(y_arr).size < 2:
        return float("nan")
    return float(average_precision_score(y_arr, p))


def score(y: np.ndarray, p: np.ndarray, thr: float) -> dict[str, float]:
    """Evaluate once using a threshold selected on independent source validation."""
    y_arr = np.asarray(y)
    p_arr = np.asarray(p)
    if len(y_arr) == 0 or np.unique(y_arr).size != 2:
        raise ValueError("Evaluation requires both positive and negative test labels.")
    yh = (p_arr >= thr).astype(int)
    prev = float(y_arr.mean())
    ap = float(average_precision_score(y_arr, p_arr))
    return {
        "prevalence": prev,
        "pr_auc": ap,
        "lift": ap / prev,
        "roc_auc": float(roc_auc_score(y_arr, p_arr)),
        "precision": float(precision_score(y_arr, yh, zero_division=0)),
        "recall": float(recall_score(y_arr, yh, zero_division=0)),
        "f1": float(f1_score(y_arr, yh, zero_division=0)),
    }


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 400,
    seed: int = 0,
) -> tuple[float, float]:
    """Stratified-safe percentile bootstrap 95% CI on outer test predictions."""
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(p, dtype=float)
    b_rng = np.random.default_rng(seed)
    vals: list[float] = []
    n = len(y_arr)
    for _ in range(n_boot):
        idx_b = b_rng.integers(0, n, size=n)
        yb = y_arr[idx_b]
        if np.unique(yb).size < 2:
            continue
        vals.append(float(metric_fn(yb, p_arr[idx_b])))
    if not vals:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (float(lo), float(hi))


def labeled_metrics(
    y: np.ndarray,
    prob: np.ndarray,
    thr: float,
    n_boot: int = 400,
    seed: int = 0,
) -> dict[str, float | int]:
    """Score with a preselected validation threshold and bootstrap 95% CIs."""
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(prob, dtype=float)
    d: dict[str, float | int] = dict(score(y_arr, p_arr, float(thr)))
    tn, fp, fn, tp = confusion_matrix(y_arr, (p_arr >= thr).astype(int), labels=[0, 1]).ravel()
    ap_lo, ap_hi = bootstrap_ci(y_arr, p_arr, average_precision_score, n_boot=n_boot, seed=seed)
    auc_lo, auc_hi = bootstrap_ci(y_arr, p_arr, roc_auc_score, n_boot=n_boot, seed=seed)
    d.update(
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        false_positive_rate=float(fp / max(1, fp + tn)),
        brier=float(brier_score_loss(y_arr, p_arr)),
        pr_auc_ci95_lo=ap_lo,
        pr_auc_ci95_hi=ap_hi,
        roc_auc_ci95_lo=auc_lo,
        roc_auc_ci95_hi=auc_hi,
    )
    return d


def measure(y: np.ndarray | pd.Series, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    """Evaluate predictions at a preselected threshold; handle single-class subsets."""
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(prob, dtype=float)
    if y_arr.size == 0 or np.unique(y_arr).size < 2:
        return {
            "status": "not_estimable_both_labels_required",
            "n": int(y_arr.size),
            "positives": int(y_arr.sum()),
            "AP": np.nan,
            "ROC_AUC": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "F1": np.nan,
            "FP": np.nan,
            "FN": np.nan,
        }
    pred = (p_arr >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_arr, pred, labels=[0, 1]).ravel()
    prev = float(y_arr.mean())
    ap = safe_ap(y_arr, p_arr)
    return {
        "status": "ok",
        "n": int(y_arr.size),
        "positives": int(y_arr.sum()),
        "prevalence": prev,
        "AP": ap,
        "AP_lift": ap / prev,
        "ROC_AUC": float(roc_auc_score(y_arr, p_arr)),
        "precision": float(precision_score(y_arr, pred, zero_division=0)),
        "recall": float(recall_score(y_arr, pred, zero_division=0)),
        "F1": float(f1_score(y_arr, pred, zero_division=0)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def threshold_for_min_precision(
    y: np.ndarray,
    prob: np.ndarray,
    min_precision: float = 0.80,
    min_alerts: int = 5,
) -> float | None:
    """Choose threshold by validation data only; maximize recall under precision constraint."""
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(prob, dtype=float)
    candidates: list[tuple[float, float, float, float, int]] = []
    for threshold in np.unique(p_arr):
        predicted = p_arr >= threshold
        alerts = int(predicted.sum())
        if alerts < min_alerts:
            continue
        prec = float(precision_score(y_arr, predicted, zero_division=0))
        rec = float(recall_score(y_arr, predicted, zero_division=0))
        if prec >= min_precision:
            candidates.append((rec, prec, -float(threshold), float(threshold), alerts))
    if not candidates:
        return None
    return max(candidates)[3]


def run_legacy_benchmark(
    ibm_feat: pd.DataFrame,
    ell_feat: pd.DataFrame,
    sets: dict[str, dict[str, np.ndarray]],
    benchmark_cfg: dict[str, Any],
    out_dir: Path,
) -> pd.DataFrame:
    """Execute the Cell 17 3-seed within/cross-domain benchmark and persist CSVs."""
    data_map = {"ibm": ibm_feat, "elliptic2": ell_feat}
    seeds: list[int] = [int(s) for s in benchmark_cfg.get("seeds", [0, 1, 2])]
    test_size = float(benchmark_cfg.get("test_size", 0.30))
    val_fraction = float(benchmark_cfg.get("validation_fraction", 0.20))
    max_nodes = int(benchmark_cfg.get("max_nodes_restricted", 10))

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for size_restricted in (False, True):
            splits = make_splits(ibm_feat, ell_feat, sets, seed, test_size=test_size)
            if size_restricted:
                splits = {
                    ds: (
                        tr[data_map[ds]["n_nodes"].iloc[tr].to_numpy() <= max_nodes],
                        te[data_map[ds]["n_nodes"].iloc[te].to_numpy() <= max_nodes],
                        info,
                    )
                    for ds, (tr, te, info) in splits.items()
                }
            for src in data_map:
                tr = splits[src][0]
                fit_ids, val_ids, _ = validation_indices(
                    src, data_map, sets, tr, seed, validation_fraction=val_fraction
                )
                y_fit = data_map[src]["label"].iloc[fit_ids]
                y_val = data_map[src]["label"].iloc[val_ids]
                if y_fit.nunique() < 2 or y_val.nunique() < 2:
                    continue
                for name, (cols_kind, model) in baseline_models(seed, benchmark_cfg).items():
                    feats = SIZE_COLS if cols_kind == "size" else FEATURES
                    model.fit(data_map[src][feats].iloc[fit_ids], y_fit)
                    p_val = model.predict_proba(data_map[src][feats].iloc[val_ids])[:, 1]
                    thr = best_thr(y_val.to_numpy(), p_val)
                    for tgt in data_map:
                        te = splits[tgt][1]
                        yt = data_map[tgt]["label"].iloc[te].to_numpy()
                        if len(yt) == 0 or len(np.unique(yt)) < 2:
                            continue
                        p = model.predict_proba(data_map[tgt][feats].iloc[te])[:, 1]
                        setting = (
                            "within"
                            if src == tgt
                            else ("synthetic->real" if src == "ibm" else "real->synthetic")
                        )
                        rows.append(
                            {
                                "seed": seed,
                                "size_restricted": size_restricted,
                                "train_on": src,
                                "test_on": tgt,
                                "setting": setting,
                                "model": name,
                                "n_fit": len(fit_ids),
                                "n_validation": len(val_ids),
                                "n_test": len(yt),
                                "validation_threshold": thr,
                                **score(yt, p, thr),
                            }
                        )

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No valid evaluation folds were produced.")
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "results_raw.csv", index=False)
    summary = (
        results.groupby(["size_restricted", "train_on", "test_on", "model"])[
            ["prevalence", "pr_auc", "lift", "roc_auc", "precision", "recall", "f1"]
        ]
        .agg(["mean", "std"])
        .round(3)
    )
    summary.to_csv(out_dir / "results_summary.csv")
    return results


def run_unit4_tuning(
    ibm_feat: pd.DataFrame,
    ell_feat: pd.DataFrame,
    sets: dict[str, dict[str, np.ndarray]],
    benchmark_cfg: dict[str, Any],
    tuning_cfg: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """Execute Unit 4/5 validation tuning, Platt calibration, and outer-test evaluation."""
    data_map = {"ibm": ibm_feat, "elliptic2": ell_feat}
    seed = int(tuning_cfg.get("seed", 0))
    domains = tuple(tuning_cfg.get("domains", ["ibm", "elliptic2"]))
    model_names = tuple(
        tuning_cfg.get("models", ["LogReg", "RandomForest", "GradBoosting", "ExtraTrees"])
    )
    n_boot = int(tuning_cfg.get("bootstrap_resamples", 400))
    platt_c = float(tuning_cfg.get("platt_c", 1.0))
    test_size = float(benchmark_cfg.get("test_size", 0.30))
    val_fraction = float(benchmark_cfg.get("validation_fraction", 0.20))

    selection_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    chosen_models: dict[tuple[str, str], tuple[Any, list[str], float]] = {}
    chosen_calibrated: dict[tuple[str, str], tuple[Any, list[str], float]] = {}

    outer_splits = make_splits(ibm_feat, ell_feat, sets, seed, test_size=test_size)

    for source_ds in domains:
        outer_train, _, outer_info = outer_splits[source_ds]
        fit_idx, val_idx, inner_info = validation_indices(
            source_ds, data_map, sets, outer_train, seed, validation_fraction=val_fraction
        )
        source_df = data_map[source_ds]
        y_fit = source_df["label"].iloc[fit_idx].to_numpy()
        y_val = source_df["label"].iloc[val_idx].to_numpy()
        if np.unique(y_fit).size < 2 or np.unique(y_val).size < 2:
            raise RuntimeError(f"{source_ds}: fit/validation needs both classes")

        base_models = baseline_models(seed, benchmark_cfg)
        candidates_map = tuned_candidates(seed, tuning_cfg)

        for model_name in model_names:
            feats = list(FEATURES)
            x_fit = source_df[feats].iloc[fit_idx]
            x_val = source_df[feats].iloc[val_idx]
            winning: dict[str, Any] | None = None

            for candidate_name, candidate, params in candidates_map[model_name]:
                fitted = clone(candidate).fit(x_fit, y_fit)
                val_prob = fitted.predict_proba(x_val)[:, 1]
                validation_ap = float(average_precision_score(y_val, val_prob))
                candidate_rows.append(
                    {
                        "source": source_ds,
                        "model": model_name,
                        "candidate": candidate_name,
                        "parameters": json.dumps(params),
                        "n_fit": len(fit_idx),
                        "n_validation": len(val_idx),
                        "n_validation_positive": int(y_val.sum()),
                        "validation_ap": validation_ap,
                    }
                )
                if winning is None or validation_ap > winning["validation_ap"]:
                    winning = {
                        "validation_ap": validation_ap,
                        "fitted": fitted,
                        "threshold": best_thr(y_val, val_prob),
                        "candidate": candidate_name,
                        "params": params,
                        "raw_val_brier": float(brier_score_loss(y_val, val_prob)),
                    }

            assert winning is not None
            cal_model = ValidationPlattCalibrator(
                winning["fitted"], platt_c=platt_c
            ).fit_calibrator(x_val, y_val)
            cal_val_prob = cal_model.predict_proba(x_val)[:, 1]
            cal_thr = best_thr(y_val, cal_val_prob)
            cal_val_brier = float(brier_score_loss(y_val, cal_val_prob))

            selection_rows.append(
                {
                    "source": source_ds,
                    "model": model_name,
                    "selected_candidate": winning["candidate"],
                    "selected_params": json.dumps(winning["params"]),
                    "validation_ap_selected_on": winning["validation_ap"],
                    "validation_F1_threshold": winning["threshold"],
                    "validation_F1_threshold_calibrated": cal_thr,
                    "validation_brier_raw": winning["raw_val_brier"],
                    "validation_brier_calibrated": cal_val_brier,
                    "fit_n": len(fit_idx),
                    "validation_n": len(val_idx),
                    "outer_test_purged_n": outer_info["test_final"],
                    "inner_validation_purged_n": inner_info["test_final"],
                }
            )
            chosen_models[(source_ds, model_name)] = (
                winning["fitted"],
                feats,
                winning["threshold"],
            )
            chosen_calibrated[(source_ds, model_name)] = (cal_model, feats, cal_thr)

            variants: list[tuple[str, Any, float]] = [
                ("tuned", winning["fitted"], winning["threshold"]),
                ("tuned_calibrated", cal_model, cal_thr),
            ]
            if model_name in base_models:
                _, original_model = base_models[model_name]
                original_fit = clone(original_model).fit(x_fit, y_fit)
                original_val_prob = original_fit.predict_proba(x_val)[:, 1]
                original_threshold = best_thr(y_val, original_val_prob)
                variants.insert(0, ("baseline_same_split", original_fit, original_threshold))

            for variant, classifier, thr in variants:
                for target_ds in domains:
                    target_test = outer_splits[target_ds][1]
                    target_df = data_map[target_ds].iloc[target_test]
                    y_test = target_df["label"].to_numpy()
                    if np.unique(y_test).size < 2:
                        continue
                    prob = classifier.predict_proba(target_df[feats])[:, 1]
                    eval_rows.append(
                        {
                            "source": source_ds,
                            "target": target_ds,
                            "model": model_name,
                            "variant": variant,
                            "seed": seed,
                            "n_test": len(y_test),
                            "threshold_from_source_validation": float(thr),
                            **labeled_metrics(y_test, prob, thr, n_boot=n_boot, seed=seed),
                        }
                    )
                    if variant == "tuned":
                        prediction_frames.append(
                            pd.DataFrame(
                                {
                                    "source": source_ds,
                                    "target": target_ds,
                                    "model": model_name,
                                    "sg_id": target_df["sg_id"].to_numpy(),
                                    "y_true": y_test,
                                    "probability": prob,
                                    "threshold": float(thr),
                                    "predicted_class": (prob >= thr).astype(int),
                                }
                            )
                        )

    candidate_results = pd.DataFrame(candidate_rows)
    selected_hyperparams = pd.DataFrame(selection_rows)
    tuned_evaluation = pd.DataFrame(eval_rows)
    tuned_predictions = pd.concat(prediction_frames, ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_results.to_csv(out_dir / "unit4_tuning_candidates.csv", index=False)
    selected_hyperparams.to_csv(out_dir / "unit4_selected_hyperparameters.csv", index=False)
    tuned_evaluation.to_csv(out_dir / "unit5_holdout_comparison.csv", index=False)
    tuned_predictions.to_csv(out_dir / "unit5_test_predictions.csv", index=False)

    return {
        "candidates": candidate_results,
        "selected": selected_hyperparams,
        "evaluation": tuned_evaluation,
        "predictions": tuned_predictions,
        "chosen_models": chosen_models,
        "chosen_calibrated": chosen_calibrated,
    }


def run_extensions(
    ibm_feat: pd.DataFrame,
    ell_feat: pd.DataFrame,
    sets: dict[str, dict[str, np.ndarray]],
    idx: EdgeIndex,
    original_end: dict[str, int],
    unit4_bundle: dict[str, Any],
    sampling_cfg: dict[str, Any],
    benchmark_cfg: dict[str, Any],
    tuning_cfg: dict[str, Any],
    ext_cfg: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """Run Extensions 1-12 and write research_extension outputs."""
    ext_dir = out_dir / "research_extension"
    ext_dir.mkdir(parents=True, exist_ok=True)

    ext_seed = int(ext_cfg.get("seed", 2026))
    embargo_min = int(ext_cfg.get("embargo_minutes", 120))
    min_alerts = int(ext_cfg.get("min_predicted_positives_for_precision", 5))
    min_prec = float(ext_cfg.get("min_precision_target", 0.80))
    perm_repeats = int(ext_cfg.get("permutation_repeats", 8))

    cbfs = sampling_cfg.get("common_bfs", {})
    common_df, _, common_edges, _, _ = generate_common_candidates(
        idx,
        n_candidates=int(cbfs.get("n_candidates", 2500)),
        positive_seed_probability=float(cbfs.get("positive_seed_probability", 0.40)),
        node_sizes=tuple(cbfs.get("node_sizes", [3, 4, 5, 6, 8, 10])),
        windows_min=tuple(cbfs.get("windows_minutes", [60, 180, 360])),
        seed=int(cbfs.get("seed", ext_seed)),
    )
    common_df.to_csv(ext_dir / "common_bfs_candidates.csv", index=False)

    common_splits, common_audit = chronological_embargo_split(
        common_df, common_edges, embargo_minutes=embargo_min
    )
    common_audit.to_csv(ext_dir / "common_split_audit.csv", index=False)

    lookup_common = common_df.set_index("sg_id")
    train_common = lookup_common.loc[common_splits["train"]]
    val_common = lookup_common.loc[common_splits["validation"]]
    test_common = lookup_common.loc[common_splits["test"]]

    common_rows: list[dict[str, Any]] = []
    for model_name, (which, base_estimator) in baseline_models(ext_seed, benchmark_cfg).items():
        columns = SIZE_COLS if which == "size" else FEATURES
        fitted = clone(base_estimator).fit(train_common[columns], train_common["label"])
        val_prob = fitted.predict_proba(val_common[columns])[:, 1]
        val_ap = safe_ap(val_common["label"].to_numpy(), val_prob)
        val_threshold = best_thr(val_common["label"].to_numpy(), val_prob)
        p_test = fitted.predict_proba(test_common[columns])[:, 1]
        metric = measure(test_common["label"].to_numpy(), p_test, val_threshold)
        common_rows.append(
            {
                "experiment": "common_BFS_sensitivity",
                "model": model_name,
                "validation_AP": val_ap,
                "threshold_from_validation": val_threshold,
                "n_train": len(train_common),
                "n_validation": len(val_common),
                **metric,
            }
        )
    common_results = pd.DataFrame(common_rows).sort_values("validation_AP", ascending=False)
    common_winner = str(common_results.iloc[0]["model"])
    common_results["selected_by_validation"] = common_results["model"].eq(common_winner)
    common_results.to_csv(ext_dir / "common_bfs_model_results.csv", index=False)

    tuning_seed = int(tuning_cfg.get("seed", 0))
    test_size = float(benchmark_cfg.get("test_size", 0.30))
    val_fraction = float(benchmark_cfg.get("validation_fraction", 0.20))
    data_map = {"ibm": ibm_feat, "elliptic2": ell_feat}
    ibm_training_idx, ibm_test_idx, _ = make_splits(
        ibm_feat, ell_feat, sets, tuning_seed, test_size=test_size
    )["ibm"]
    ibm_fit_idx, ibm_val_idx, _ = validation_indices(
        "ibm", data_map, sets, ibm_training_idx, tuning_seed, validation_fraction=val_fraction
    )

    seen_train_nodes: set[int] = set()
    for ident in ibm_feat["sg_id"].iloc[np.asarray(ibm_training_idx, dtype=int)]:
        seen_train_nodes.update(sets["ibm_nodes"][ident].tolist())
    identifiers = ibm_feat["sg_id"].iloc[ibm_test_idx].tolist()
    node_shared = [
        not seen_train_nodes.isdisjoint(sets["ibm_nodes"][ident].tolist()) for ident in identifiers
    ]
    account_disjoint_test = ibm_test_idx[np.logical_not(np.asarray(node_shared))]

    all_train_val_ids = ibm_feat["sg_id"].iloc[ibm_training_idx]
    latest_train_end = max(original_end[ident] for ident in all_train_val_ids)
    time_disjoint_test = np.asarray(
        [
            pos
            for pos in ibm_test_idx
            if int(ibm_feat["t_start"].iloc[pos]) > latest_train_end + embargo_min
        ],
        dtype=int,
    )

    selected_hyperparams: pd.DataFrame = unit4_bundle["selected"]
    chosen_models = unit4_bundle["chosen_models"]
    selection_ibm = selected_hyperparams[selected_hyperparams["source"] == "ibm"].sort_values(
        "validation_ap_selected_on", ascending=False
    )
    selected_name = str(selection_ibm.iloc[0]["model"])
    selected_estimator, selected_columns, selected_threshold = chosen_models[("ibm", selected_name)]

    stress_records: list[dict[str, Any]] = []
    for subset_label, ids in (
        ("original_edge_purged", ibm_test_idx),
        ("account_disjoint_stress", account_disjoint_test),
        ("time_embargo_stress", time_disjoint_test),
    ):
        fragment = ibm_feat.iloc[np.asarray(ids, dtype=int)]
        if fragment.empty:
            metrics = measure(
                np.array([], dtype=int), np.array([], dtype=float), selected_threshold
            )
        else:
            p = selected_estimator.predict_proba(fragment[selected_columns])[:, 1]
            metrics = measure(fragment["label"].to_numpy(), p, selected_threshold)
        stress_records.append(
            {
                "subset": subset_label,
                "selected_on_validation": selected_name,
                "threshold_from_validation": selected_threshold,
                "retention_fraction_of_original_test": len(ids) / max(1, len(ibm_test_idx)),
                **metrics,
            }
        )
    stress_table = pd.DataFrame(stress_records)
    stress_table.to_csv(ext_dir / "original_split_dependence_stress.csv", index=False)

    ad_cfg = ext_cfg.get("account_disjoint_retrain", {})
    ad_fit, ad_val, ad_test = account_disjoint_partition(
        ibm_feat,
        sets["ibm_nodes"],
        train_frac=float(ad_cfg.get("train_frac", 0.60)),
        val_frac=float(ad_cfg.get("val_frac", 0.20)),
    )
    ad_fit_df, ad_val_df, ad_test_df = (
        ibm_feat.iloc[ad_fit],
        ibm_feat.iloc[ad_val],
        ibm_feat.iloc[ad_test],
    )
    if (
        ad_fit_df["label"].nunique() == 2
        and ad_val_df["label"].nunique() == 2
        and ad_test_df["label"].nunique() == 2
    ):
        ad_est = clone(selected_estimator).fit(ad_fit_df[selected_columns], ad_fit_df["label"])
        ad_pv = ad_est.predict_proba(ad_val_df[selected_columns])[:, 1]
        ad_thr = best_thr(ad_val_df["label"].to_numpy(), ad_pv)
        ad_pt = ad_est.predict_proba(ad_test_df[selected_columns])[:, 1]
        ad_metrics = measure(ad_test_df["label"].to_numpy(), ad_pt, ad_thr)
        pd.DataFrame(
            [
                {
                    "split": "account_disjoint_retrain",
                    "fit_n": len(ad_fit),
                    "val_n": len(ad_val),
                    **ad_metrics,
                }
            ]
        ).to_csv(ext_dir / "account_disjoint_retrain.csv", index=False)

    ablation_groups = {
        "size_only": ["n_nodes", "n_edges"],
        "topology_excluding_size": [f for f in FEATURES if f not in ("n_nodes", "n_edges")],
        "degree_direction_only": [
            "max_in_deg_norm",
            "max_out_deg_norm",
            "hub_share",
            "deg_gini",
            "frac_sources",
            "frac_sinks",
            "frac_passthrough",
        ],
        "cycle_related_only": ["has_cycle", "reciprocity", "longest_chain_norm"],
        "all_excluding_cycle": [f for f in FEATURES if f not in ("has_cycle", "reciprocity")],
        "all_14": list(FEATURES),
    }
    fit_df = ibm_feat.iloc[ibm_fit_idx]
    val_df = ibm_feat.iloc[ibm_val_idx]
    test_df = ibm_feat.iloc[ibm_test_idx]
    ablation_rows: list[dict[str, Any]] = []
    for group, columns in ablation_groups.items():
        est = clone(selected_estimator).fit(fit_df[columns], fit_df["label"])
        pv = est.predict_proba(val_df[columns])[:, 1]
        pt = est.predict_proba(test_df[columns])[:, 1]
        threshold = best_thr(val_df["label"].to_numpy(), pv)
        ablation_rows.append(
            {
                "group": group,
                "model": selected_name,
                "features": ", ".join(columns),
                "n_features": len(columns),
                "validation_AP": safe_ap(val_df["label"], pv),
                "validation_threshold": threshold,
                **measure(test_df["label"], pt, threshold),
            }
        )
    ablation_results = pd.DataFrame(ablation_rows)
    ablation_results.to_csv(ext_dir / "feature_ablation.csv", index=False)

    perm = permutation_importance(
        selected_estimator,
        val_df[selected_columns],
        val_df["label"],
        scoring="average_precision",
        n_repeats=perm_repeats,
        random_state=ext_seed,
        n_jobs=1,
    )
    feature_importance = pd.DataFrame(
        {
            "feature": selected_columns,
            "AP_drop_mean": perm.importances_mean,
            "AP_drop_std": perm.importances_std,
        }
    ).sort_values("AP_drop_mean", ascending=False)
    feature_importance.to_csv(ext_dir / "permutation_importance_validation.csv", index=False)

    p_val_sel = selected_estimator.predict_proba(val_df[selected_columns])[:, 1]
    p_test_sel = selected_estimator.predict_proba(test_df[selected_columns])[:, 1]
    thresholds = {
        "validation_max_F1": best_thr(val_df["label"].to_numpy(), p_val_sel),
        "validation_precision_at_least_80pct": threshold_for_min_precision(
            val_df["label"].to_numpy(), p_val_sel, min_precision=min_prec, min_alerts=min_alerts
        ),
    }
    threshold_rows: list[dict[str, Any]] = []
    for policy, thr_val in thresholds.items():
        if thr_val is None:
            threshold_rows.append(
                {"policy": policy, "status": "not_feasible_on_validation", "threshold": np.nan}
            )
            continue
        val_res = measure(val_df["label"].to_numpy(), p_val_sel, thr_val)
        test_res = measure(test_df["label"].to_numpy(), p_test_sel, thr_val)
        threshold_rows.append(
            {
                "policy": policy,
                "status": "evaluated",
                "threshold": thr_val,
                "validation_precision": val_res["precision"],
                "validation_recall": val_res["recall"],
                **test_res,
            }
        )
    threshold_table = pd.DataFrame(threshold_rows)
    threshold_table.to_csv(ext_dir / "threshold_tradeoff.csv", index=False)

    orig_splits = make_splits(ibm_feat, ell_feat, sets, tuning_seed, test_size=test_size)
    ibm_train_full = ibm_feat.iloc[orig_splits["ibm"][0]]
    ell_train_full = ell_feat.iloc[orig_splits["elliptic2"][0]]
    domain_records: list[dict[str, Any]] = []
    for group_label, subset_label in ((None, "overall"), (0, "licit_only"), (1, "suspicious_only")):
        left = (
            ibm_train_full
            if group_label is None
            else ibm_train_full[ibm_train_full["label"] == group_label]
        )
        right = (
            ell_train_full
            if group_label is None
            else ell_train_full[ell_train_full["label"] == group_label]
        )
        for feat in FEATURES:
            x = left[feat].dropna().to_numpy(dtype=float)
            y = right[feat].dropna().to_numpy(dtype=float)
            if not len(x) or not len(y):
                continue
            domain_records.append(
                {
                    "group": subset_label,
                    "feature": feat,
                    "ibm_n": len(x),
                    "elliptic2_n": len(y),
                    "ibm_mean": float(np.mean(x)),
                    "elliptic2_mean": float(np.mean(y)),
                    "KS_statistic": float(ks_2samp(x, y).statistic),
                }
            )
    domain_shift = pd.DataFrame(domain_records)
    domain_shift.to_csv(ext_dir / "domain_shift_train_only.csv", index=False)

    ibm_start_by_id = {str(k): int(v) for k, v in ibm_feat.set_index("sg_id")["t_start"].items()}
    activity = pd.DataFrame(
        [
            transaction_activity_features(
                str(sg), sets["ibm_edges"], idx, ibm_start_by_id, original_end
            )
            for sg in ibm_feat["sg_id"]
        ]
    )
    extra_columns = [c for c in activity.columns if c != "sg_id"]
    ibm_augmented = ibm_feat.merge(activity, on="sg_id", validate="one_to_one")
    ibm_augmented.to_csv(ext_dir / "ibm_activity_feature_table.csv", index=False)

    et_act_cfg = ext_cfg.get("activity_experiment", {}).get("extra_trees", {})
    activity_models: dict[str, Any] = {
        "validation_selected_original_family": clone(selected_estimator),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=int(et_act_cfg.get("n_estimators", 250)),
            min_samples_leaf=int(et_act_cfg.get("min_samples_leaf", 2)),
            class_weight=et_act_cfg.get("class_weight", "balanced"),
            n_jobs=-1,
            random_state=ext_seed,
        ),
    }
    activity_rows: list[dict[str, Any]] = []
    for family, base in activity_models.items():
        for feature_set, cols in (
            ("original_14", list(FEATURES)),
            ("augmented_20", list(FEATURES) + extra_columns),
        ):
            fitted = clone(base).fit(
                ibm_augmented.iloc[ibm_fit_idx][cols], ibm_augmented["label"].iloc[ibm_fit_idx]
            )
            pv = fitted.predict_proba(ibm_augmented.iloc[ibm_val_idx][cols])[:, 1]
            thr_act = best_thr(ibm_augmented["label"].iloc[ibm_val_idx].to_numpy(), pv)
            pt = fitted.predict_proba(ibm_augmented.iloc[ibm_test_idx][cols])[:, 1]
            activity_rows.append(
                {
                    "family": family,
                    "features": feature_set,
                    "n_features": len(cols),
                    "validation_AP": safe_ap(ibm_augmented["label"].iloc[ibm_val_idx], pv),
                    "threshold_from_validation": thr_act,
                    **measure(ibm_augmented["label"].iloc[ibm_test_idx], pt, thr_act),
                }
            )
    activity_table = pd.DataFrame(activity_rows).sort_values("validation_AP", ascending=False)
    activity_table["selected_by_validation"] = False
    activity_table.loc[activity_table.index[0], "selected_by_validation"] = True
    activity_table.to_csv(ext_dir / "IBM_activity_feature_experiment.csv", index=False)

    win_fam = str(activity_table.iloc[0]["family"])
    win_feat = str(activity_table.iloc[0]["features"])
    win_cols = list(FEATURES) if win_feat == "original_14" else list(FEATURES) + extra_columns
    win_model = clone(activity_models[win_fam]).fit(
        ibm_augmented.iloc[ibm_fit_idx][win_cols], ibm_augmented["label"].iloc[ibm_fit_idx]
    )
    win_pv = win_model.predict_proba(ibm_augmented.iloc[ibm_val_idx][win_cols])[:, 1]
    win_thr = best_thr(ibm_augmented["label"].iloc[ibm_val_idx].to_numpy(), win_pv)

    test_meta = ibm_augmented.iloc[ibm_test_idx][
        ["sg_id", "label", "pattern", "n_nodes", "n_edges"]
    ].copy()
    p_orig = selected_estimator.predict_proba(ibm_augmented.iloc[ibm_test_idx][selected_columns])[
        :, 1
    ]
    p_aug = win_model.predict_proba(ibm_augmented.iloc[ibm_test_idx][win_cols])[:, 1]
    test_meta["pred_14feat"] = (p_orig >= selected_threshold).astype(int)
    test_meta["pred_20feat"] = (p_aug >= win_thr).astype(int)
    pos_test = test_meta[test_meta["label"] == 1]
    if not pos_test.empty:
        typology_summary = (
            pos_test.groupby("pattern")
            .agg(
                test_count=("sg_id", "count"),
                median_nodes=("n_nodes", "median"),
                recall_14feat=("pred_14feat", "mean"),
                recall_20feat=("pred_20feat", "mean"),
            )
            .sort_values("recall_14feat", ascending=False)
        )
        typology_summary.to_csv(ext_dir / "typology_recall_breakdown.csv")
    else:
        typology_summary = pd.DataFrame()

    summary_payload = {
        "seed": ext_seed,
        "data_inputs": ["IBM AML HI-Small", "Elliptic2"],
        "common_candidate_count": int(len(common_df)),
        "common_candidate_positive_rate": float(common_df["label"].mean()),
        "common_seed_enrichment_fraction_requested": float(
            cbfs.get("positive_seed_probability", 0.40)
        ),
        "common_split_audit": common_audit.to_dict(orient="records"),
        "common_model_selected_by_validation": common_winner,
        "ibm_activity_candidate_selected_by_validation": activity_table.iloc[0][
            ["family", "features"]
        ].to_dict(),
        "original_model_selected_by_validation": selected_name,
        "selected_threshold_policies": {
            k: (None if v is None else float(v)) for k, v in thresholds.items()
        },
    }
    with open(ext_dir / "research_extension_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2, default=str)

    return {
        "common_results": common_results,
        "stress_table": stress_table,
        "ablation_results": ablation_results,
        "feature_importance": feature_importance,
        "domain_shift": domain_shift,
        "activity_table": activity_table,
        "typology_summary": typology_summary,
    }
