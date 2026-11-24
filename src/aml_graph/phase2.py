"""Phase 2 methodology: label-blind IPW evaluation, multi-fold splits, and paired tests."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from scipy.stats import t as student_t
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from aml_graph.evaluate import best_thr, bootstrap_ci
from aml_graph.features import ACTIVITY_COLS, AUGMENTED_COLS, FEATURES, SIZE_COLS
from aml_graph.models import ValidationPlattCalibrator, baseline_models, tuned_candidates
from aml_graph.sampling import (
    EdgeIndex,
    index_documented_attempt_edges,
    sample_label_blind_ibm,
)
from aml_graph.splits import account_component_group_splits, rolling_origin_splits


def fold_mean_ci95(values: Sequence[float]) -> tuple[float, float, float, float]:
    """Compute (mean, std, ci95_lo, ci95_hi) across K >= 2 estimable folds via Student-t."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    n = int(arr.size)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    mean = float(arr.mean())
    if n == 1:
        return (mean, 0.0, float("nan"), float("nan"))
    sd = float(arr.std(ddof=1))
    crit = float(student_t.ppf(0.975, df=n - 1))
    half = crit * sd / np.sqrt(n)
    return (mean, sd, mean - half, mean + half)


def _delong_structural_components(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute exact AUC and structural components (V10, V01) via midranks (Sun & Xu 2014)."""
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    m = len(pos)
    n = len(neg)
    pooled = np.concatenate([pos, neg])
    midranks_all = rankdata(pooled, method="average")
    midranks_pos = rankdata(pos, method="average")
    midranks_neg = rankdata(neg, method="average")

    v10 = (midranks_all[:m] - midranks_pos) / n
    v01 = 1.0 - (midranks_all[m:] - midranks_neg) / m
    auc = float(v10.mean())
    return auc, v10, v01


def delong_roc_test(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
) -> dict[str, float]:
    """Two-sided correlated DeLong test comparing ROC-AUC(prob_a) vs ROC-AUC(prob_b)."""
    y = np.asarray(y_true, dtype=int)
    pa = np.asarray(prob_a, dtype=float)
    pb = np.asarray(prob_b, dtype=float)
    m = int((y == 1).sum())
    n = int((y == 0).sum())
    if m < 2 or n < 2:
        return {
            "auc_a": np.nan,
            "auc_b": np.nan,
            "delta_auc": np.nan,
            "delong_z": np.nan,
            "delong_p": np.nan,
        }

    auc_a, v10_a, v01_a = _delong_structural_components(y, pa)
    auc_b, v10_b, v01_b = _delong_structural_components(y, pb)
    s10 = np.cov(np.vstack([v10_a, v10_b]), ddof=1)
    s01 = np.cov(np.vstack([v01_a, v01_b]), ddof=1)
    cov = s10 / m + s01 / n
    diff = auc_a - auc_b
    var_diff = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    if var_diff <= 1e-18:
        if abs(diff) <= 1e-12:
            z = 0.0
            p_val = 1.0
        else:
            z = float(np.copysign(np.inf, diff))
            p_val = 0.0
    else:
        z = float(diff / np.sqrt(var_diff))
        p_val = float(2.0 * norm.sf(abs(z)))
    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "delta_auc": float(diff),
        "delong_z": z,
        "delong_p": p_val,
    }


def paired_bootstrap_comparison(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    weights: np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 2026,
) -> dict[str, float]:
    """Paired bootstrap CI and two-sided p-value for AP(A) - AP(B) and ROC-AUC(A) - ROC-AUC(B)."""
    y = np.asarray(y_true, dtype=int)
    pa = np.asarray(prob_a, dtype=float)
    pb = np.asarray(prob_b, dtype=float)
    w = None if weights is None else np.asarray(weights, dtype=float)
    if np.unique(y).size < 2:
        return {
            "delta_ap": np.nan,
            "delta_ap_ci95_lo": np.nan,
            "delta_ap_ci95_hi": np.nan,
            "paired_p_ap": np.nan,
            "delta_ipw_ap": np.nan,
            "delta_ipw_ap_ci95_lo": np.nan,
            "delta_ipw_ap_ci95_hi": np.nan,
        }

    obs_ap_diff = float(average_precision_score(y, pa) - average_precision_score(y, pb))
    obs_ipw_diff = (
        float(
            average_precision_score(y, pa, sample_weight=w)
            - average_precision_score(y, pb, sample_weight=w)
        )
        if w is not None
        else np.nan
    )

    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    ap_diffs: list[float] = []
    ipw_diffs: list[float] = []

    for _ in range(n_boot):
        bp = rng.choice(pos_idx, size=pos_idx.size, replace=True)
        bn = rng.choice(neg_idx, size=neg_idx.size, replace=True)
        b_idx = np.concatenate([bp, bn])
        yb = y[b_idx]
        ap_diffs.append(
            float(average_precision_score(yb, pa[b_idx]) - average_precision_score(yb, pb[b_idx]))
        )
        if w is not None:
            wb = w[b_idx]
            ipw_diffs.append(
                float(
                    average_precision_score(yb, pa[b_idx], sample_weight=wb)
                    - average_precision_score(yb, pb[b_idx], sample_weight=wb)
                )
            )

    arr_ap = np.asarray(ap_diffs, dtype=float)
    lo_ap, hi_ap = np.percentile(arr_ap, [2.5, 97.5])
    # Two-sided bootstrap p-value around 0
    p_two = float(2.0 * min(np.mean(arr_ap <= 0.0), np.mean(arr_ap >= 0.0)))
    p_two = min(1.0, max(p_two, 1.0 / (n_boot + 1)))

    res = {
        "delta_ap": obs_ap_diff,
        "delta_ap_ci95_lo": float(lo_ap),
        "delta_ap_ci95_hi": float(hi_ap),
        "paired_p_ap": p_two,
    }
    if ipw_diffs:
        lo_w, hi_w = np.percentile(np.asarray(ipw_diffs, dtype=float), [2.5, 97.5])
        res.update(
            delta_ipw_ap=obs_ipw_diff,
            delta_ipw_ap_ci95_lo=float(lo_w),
            delta_ipw_ap_ci95_hi=float(hi_w),
        )
    else:
        res.update(
            delta_ipw_ap=np.nan,
            delta_ipw_ap_ci95_lo=np.nan,
            delta_ipw_ap_ci95_hi=np.nan,
        )
    res.update(delong_roc_test(y, pa, pb))
    return res


def score_unweighted_and_ipw(
    y: np.ndarray,
    prob: np.ndarray,
    thr: float,
    ipw_weights: np.ndarray,
    min_test_positives: int = 30,
) -> dict[str, Any]:
    """Compute unweighted and IPW-weighted fold metrics; enforce minimum test positives."""
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(prob, dtype=float)
    w_arr = np.asarray(ipw_weights, dtype=float)
    n_pos = int(y_arr.sum())
    n_tot = int(y_arr.size)

    if n_tot == 0 or np.unique(y_arr).size < 2:
        return {
            "status": "not_estimable_single_class",
            "n_test": n_tot,
            "n_positives": n_pos,
            "prevalence": np.nan,
            "pr_auc": np.nan,
            "lift": np.nan,
            "roc_auc": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "brier": np.nan,
            "ipw_prevalence": np.nan,
            "ipw_pr_auc": np.nan,
            "ipw_lift": np.nan,
            "ipw_roc_auc": np.nan,
        }

    if n_pos < int(min_test_positives):
        return {
            "status": "not_estimable_lt_30_positives",
            "n_test": n_tot,
            "n_positives": n_pos,
            "prevalence": float(y_arr.mean()),
            "pr_auc": np.nan,
            "lift": np.nan,
            "roc_auc": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "brier": np.nan,
            "ipw_prevalence": float(np.average(y_arr, weights=w_arr)),
            "ipw_pr_auc": np.nan,
            "ipw_lift": np.nan,
            "ipw_roc_auc": np.nan,
        }

    yh = (p_arr >= float(thr)).astype(int)
    prev = float(y_arr.mean())
    ap = float(average_precision_score(y_arr, p_arr))
    ipw_prev = float(np.average(y_arr, weights=w_arr))
    ipw_ap = float(average_precision_score(y_arr, p_arr, sample_weight=w_arr))
    ipw_auc = float(roc_auc_score(y_arr, p_arr, sample_weight=w_arr))

    return {
        "status": "ok",
        "n_test": n_tot,
        "n_positives": n_pos,
        "prevalence": prev,
        "pr_auc": ap,
        "lift": ap / prev,
        "roc_auc": float(roc_auc_score(y_arr, p_arr)),
        "precision": float(precision_score(y_arr, yh, zero_division=0)),
        "recall": float(recall_score(y_arr, yh, zero_division=0)),
        "f1": float(f1_score(y_arr, yh, zero_division=0)),
        "brier": float(brier_score_loss(y_arr, p_arr)),
        "ipw_prevalence": ipw_prev,
        "ipw_pr_auc": ipw_ap,
        "ipw_lift": ipw_ap / ipw_prev,
        "ipw_roc_auc": ipw_auc,
    }


def _fit_and_select_fold_model(
    model_name: str,
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    seed: int,
    benchmark_cfg: dict[str, Any],
    tuning_cfg: dict[str, Any],
) -> tuple[ValidationPlattCalibrator, list[str], float, float, str]:
    """Fit candidate(s) on fit_df, select strictly by validation AP, calibrate on val_df."""
    y_fit = fit_df["label"].to_numpy(dtype=int)
    y_val = val_df["label"].to_numpy(dtype=int)

    if model_name == "SizeOnly":
        cols = list(SIZE_COLS)
        _, base = baseline_models(seed, benchmark_cfg)["SizeOnly"]
        fitted = clone(base).fit(fit_df[cols], y_fit)
        val_ap = float(average_precision_score(y_val, fitted.predict_proba(val_df[cols])[:, 1]))
        cal = ValidationPlattCalibrator(fitted).fit_calibrator(val_df[cols], y_val)
        thr = best_thr(y_val, cal.predict_proba(val_df[cols])[:, 1])
        return cal, cols, thr, val_ap, "SizeOnly_default"

    if model_name in ("ActivityOnly_6", "StructPlusActivity_20"):
        cols = list(ACTIVITY_COLS) if model_name == "ActivityOnly_6" else list(AUGMENTED_COLS)
        cands: list[tuple[str, Any, dict[str, Any]]] = []
        pool = tuned_candidates(seed, tuning_cfg)
        for fam in ("LogReg", "GradBoosting"):
            for cname, est, prm in pool[fam]:
                cands.append((f"{fam}:{cname}", est, prm))
    else:
        cols = list(FEATURES)
        cands = tuned_candidates(seed, tuning_cfg)[model_name]

    best_entry: tuple[float, Any, str] | None = None
    for cname, candidate, _ in cands:
        est = clone(candidate).fit(fit_df[cols], y_fit)
        p_val = est.predict_proba(val_df[cols])[:, 1]
        vap = float(average_precision_score(y_val, p_val))
        if best_entry is None or vap > best_entry[0]:
            best_entry = (vap, est, cname)

    assert best_entry is not None
    val_ap, winning_est, winning_name = best_entry
    cal = ValidationPlattCalibrator(winning_est).fit_calibrator(val_df[cols], y_val)
    thr = best_thr(y_val, cal.predict_proba(val_df[cols])[:, 1])
    return cal, cols, thr, val_ap, winning_name


def run_phase2_methodology(
    ibm: pd.DataFrame,
    attempts: list[dict[str, Any]],
    idx: EdgeIndex,
    legacy_unit5_df: pd.DataFrame,
    legacy_results_raw_df: pd.DataFrame,
    legacy_activity_df: pd.DataFrame,
    legacy_aug_table_path: Path | None,
    benchmark_cfg: dict[str, Any],
    tuning_cfg: dict[str, Any],
    phase2_cfg: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """Execute the full Phase 2 label-blind methodology, multi-fold evaluation, and audits."""
    p2_dir = out_dir / "phase2"
    p2_dir.mkdir(parents=True, exist_ok=True)

    seed = int(phase2_cfg.get("seed", 2026))
    scfg = phase2_cfg.get("sampler", {})
    sp_cfg = phase2_cfg.get("splits", {})
    ev_cfg = phase2_cfg.get("evaluation", {})
    min_pos = int(sp_cfg.get("min_test_positives", 30))
    n_boot = int(ev_cfg.get("bootstrap_resamples", 600))
    n_paired = int(ev_cfg.get("paired_bootstrap_resamples", 1000))

    attempt_edge_map = index_documented_attempt_edges(attempts, ibm)
    lb_df, lb_nodes, lb_edges, sampler_info = sample_label_blind_ibm(
        idx=idx,
        attempt_edge_patterns=attempt_edge_map,
        n_candidates=int(scfg.get("n_candidates", 10000)),
        positive_seed_probability=float(scfg.get("positive_seed_probability", 0.25)),
        node_sizes=tuple(scfg.get("node_sizes", [4, 6, 8, 10, 12])),
        windows_min=tuple(scfg.get("windows_minutes", [120, 240, 480])),
        max_tries_multiplier=int(scfg.get("max_tries_multiplier", 15)),
        seed=seed,
    )
    lb_df.to_csv(p2_dir / "label_blind_candidates.csv", index=False)

    ro_cfg = sp_cfg.get("rolling_origin", {})
    ag_cfg = sp_cfg.get("account_group", {})
    ro_folds = rolling_origin_splits(
        lb_df,
        lb_edges,
        n_folds=int(ro_cfg.get("n_folds", 4)),
        embargo_minutes=int(ro_cfg.get("embargo_minutes", 480)),
        first_train_quantile=float(ro_cfg.get("first_train_quantile", 0.30)),
        val_quantile_width=float(ro_cfg.get("val_quantile_width", 0.12)),
    )
    ag_folds = account_component_group_splits(
        lb_df,
        lb_nodes,
        lb_edges,
        n_folds=int(ag_cfg.get("n_folds", 4)),
    )

    structural_models = list(
        ev_cfg.get("models", ["SizeOnly", "LogReg", "RandomForest", "GradBoosting", "ExtraTrees"])
    )
    all_eval_models = structural_models + ["ActivityOnly_6", "StructPlusActivity_20"]

    fold_rows: list[dict[str, Any]] = []
    oof_store: dict[tuple[str, str], list[pd.DataFrame]] = {}

    for folds in (ro_folds, ag_folds):
        for fspec in folds:
            regime = str(fspec["regime"])
            fold_k = int(fspec["fold"])
            fit_df = lb_df.iloc[fspec["fit_idx"]]
            val_df = lb_df.iloc[fspec["val_idx"]]
            test_df = lb_df.iloc[fspec["test_idx"]]

            if (
                fit_df.empty
                or val_df.empty
                or fit_df["label"].nunique() < 2
                or val_df["label"].nunique() < 2
            ):
                for mname in all_eval_models:
                    fold_rows.append(
                        {
                            "regime": regime,
                            "fold": fold_k,
                            "model": mname,
                            "feature_group": (
                                "size_2"
                                if mname == "SizeOnly"
                                else (
                                    "activity_6"
                                    if mname == "ActivityOnly_6"
                                    else (
                                        "augmented_20"
                                        if mname == "StructPlusActivity_20"
                                        else "structural_14"
                                    )
                                )
                            ),
                            "selected_candidate": "skipped",
                            "validation_ap": np.nan,
                            "validation_ipw_ap": np.nan,
                            "threshold": np.nan,
                            "n_fit": len(fit_df),
                            "n_val": len(val_df),
                            **score_unweighted_and_ipw(
                                test_df["label"].to_numpy()
                                if len(test_df)
                                else np.array([], dtype=int),
                                np.zeros(len(test_df), dtype=float),
                                0.5,
                                test_df["ipw_weight"].to_numpy()
                                if len(test_df)
                                else np.array([], dtype=float),
                                min_test_positives=min_pos,
                            ),
                        }
                    )
                continue

            for mname in all_eval_models:
                cal_est, cols, thr, val_ap, chosen_cand = _fit_and_select_fold_model(
                    mname, fit_df, val_df, seed + fold_k, benchmark_cfg, tuning_cfg
                )
                val_p = cal_est.predict_proba(val_df[cols])[:, 1]
                val_ipw_ap = float(
                    average_precision_score(
                        val_df["label"].to_numpy(),
                        val_p,
                        sample_weight=val_df["ipw_weight"].to_numpy(),
                    )
                )
                if len(test_df):
                    p_test = cal_est.predict_proba(test_df[cols])[:, 1]
                    y_test = test_df["label"].to_numpy(dtype=int)
                    w_test = test_df["ipw_weight"].to_numpy(dtype=float)
                else:
                    p_test = np.array([], dtype=float)
                    y_test = np.array([], dtype=int)
                    w_test = np.array([], dtype=float)

                m_scores = score_unweighted_and_ipw(
                    y_test, p_test, thr, w_test, min_test_positives=min_pos
                )
                fgroup = (
                    "size_2"
                    if mname == "SizeOnly"
                    else (
                        "activity_6"
                        if mname == "ActivityOnly_6"
                        else (
                            "augmented_20" if mname == "StructPlusActivity_20" else "structural_14"
                        )
                    )
                )
                fold_rows.append(
                    {
                        "regime": regime,
                        "fold": fold_k,
                        "model": mname,
                        "feature_group": fgroup,
                        "selected_candidate": chosen_cand,
                        "validation_ap": val_ap,
                        "validation_ipw_ap": val_ipw_ap,
                        "threshold": thr,
                        "n_fit": len(fit_df),
                        "n_val": len(val_df),
                        **m_scores,
                    }
                )
                if m_scores["status"] == "ok":
                    oof_store.setdefault((regime, mname), []).append(
                        pd.DataFrame(
                            {
                                "sg_id": test_df["sg_id"].to_numpy(),
                                "fold": fold_k,
                                "y_true": y_test,
                                "probability": p_test,
                                "ipw_weight": w_test,
                                "pattern": test_df["pattern"].to_numpy(),
                            }
                        )
                    )

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(p2_dir / "fold_metrics.csv", index=False)

    # Aggregate across estimable folds (mean +- 95% Student-t CI + pooled OOF bootstrap CI)
    summary_rows: list[dict[str, Any]] = []
    for regime in ("rolling_origin", "account_group"):
        sub_reg = fold_df[fold_df["regime"] == regime]
        for mname in all_eval_models:
            sub_m = sub_reg[sub_reg["model"] == mname]
            ok_m = sub_m[sub_m["status"] == "ok"]
            n_total_folds = len(sub_m)
            n_est_folds = len(ok_m)
            fgroup = str(sub_m["feature_group"].iloc[0])

            val_ap_mean = (
                float(sub_m["validation_ap"].dropna().mean())
                if sub_m["validation_ap"].notna().any()
                else np.nan
            )
            val_ipw_mean = (
                float(sub_m["validation_ipw_ap"].dropna().mean())
                if sub_m["validation_ipw_ap"].notna().any()
                else np.nan
            )

            if n_est_folds == 0:
                summary_rows.append(
                    {
                        "regime": regime,
                        "model": mname,
                        "feature_group": fgroup,
                        "status": "not_estimable",
                        "estimable_folds": f"0/{n_total_folds}",
                        "mean_validation_ap": val_ap_mean,
                        "mean_validation_ipw_ap": val_ipw_mean,
                    }
                )
                continue

            ap_m, ap_sd, ap_lo, ap_hi = fold_mean_ci95(ok_m["pr_auc"].tolist())
            lift_m, _, lift_lo, lift_hi = fold_mean_ci95(ok_m["lift"].tolist())
            auc_m, _, auc_lo, auc_hi = fold_mean_ci95(ok_m["roc_auc"].tolist())
            f1_m, _, _, _ = fold_mean_ci95(ok_m["f1"].tolist())
            prev_m = float(ok_m["prevalence"].mean())

            ipw_ap_m, _, ipw_ap_lo, ipw_ap_hi = fold_mean_ci95(ok_m["ipw_pr_auc"].tolist())
            ipw_lift_m, _, ipw_lift_lo, ipw_lift_hi = fold_mean_ci95(ok_m["ipw_lift"].tolist())
            ipw_auc_m, _, ipw_auc_lo, ipw_auc_hi = fold_mean_ci95(ok_m["ipw_roc_auc"].tolist())
            ipw_prev_m = float(ok_m["ipw_prevalence"].mean())

            # Pooled out-of-fold bootstrap 95% CI across all non-overlapping test predictions
            oof_df = pd.concat(oof_store[(regime, mname)], ignore_index=True)
            pool_ap = float(average_precision_score(oof_df["y_true"], oof_df["probability"]))
            pool_ap_lo, pool_ap_hi = bootstrap_ci(
                oof_df["y_true"].to_numpy(),
                oof_df["probability"].to_numpy(),
                average_precision_score,
                n_boot=n_boot,
                seed=seed,
            )
            pool_prev = float(oof_df["y_true"].mean())

            summary_rows.append(
                {
                    "regime": regime,
                    "model": mname,
                    "feature_group": fgroup,
                    "status": "ok",
                    "estimable_folds": f"{n_est_folds}/{n_total_folds}",
                    "total_test_positives": int(ok_m["n_positives"].sum()),
                    "mean_validation_ap": val_ap_mean,
                    "mean_validation_ipw_ap": val_ipw_mean,
                    "prevalence_mean": prev_m,
                    "ap_mean": ap_m,
                    "ap_std": ap_sd,
                    "ap_fold_ci95_lo": ap_lo,
                    "ap_fold_ci95_hi": ap_hi,
                    "pooled_ap": pool_ap,
                    "pooled_ap_boot_ci95_lo": pool_ap_lo,
                    "pooled_ap_boot_ci95_hi": pool_ap_hi,
                    "pooled_lift": pool_ap / pool_prev,
                    "lift_mean": lift_m,
                    "lift_fold_ci95_lo": lift_lo,
                    "lift_fold_ci95_hi": lift_hi,
                    "roc_auc_mean": auc_m,
                    "roc_auc_fold_ci95_lo": auc_lo,
                    "roc_auc_fold_ci95_hi": auc_hi,
                    "f1_mean": f1_m,
                    "ipw_prevalence_mean": ipw_prev_m,
                    "ipw_ap_mean": ipw_ap_m,
                    "ipw_ap_fold_ci95_lo": ipw_ap_lo,
                    "ipw_ap_fold_ci95_hi": ipw_ap_hi,
                    "ipw_lift_mean": ipw_lift_m,
                    "ipw_lift_fold_ci95_lo": ipw_lift_lo,
                    "ipw_lift_fold_ci95_hi": ipw_lift_hi,
                    "ipw_roc_auc_mean": ipw_auc_m,
                    "ipw_roc_auc_fold_ci95_lo": ipw_auc_lo,
                    "ipw_roc_auc_fold_ci95_hi": ipw_auc_hi,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    # Mark strictly validation-selected 14-feature structural model per regime!
    summary_df["selected_structural_by_validation"] = False
    for regime in ("rolling_origin", "account_group"):
        mask = (
            (summary_df["regime"] == regime)
            & (summary_df["model"].isin(structural_models))
            & (summary_df["mean_validation_ap"].notna())
        )
        if mask.any():
            best_idx = summary_df.loc[mask, "mean_validation_ap"].idxmax()
            summary_df.loc[best_idx, "selected_structural_by_validation"] = True
    summary_df.to_csv(p2_dir / "summary_by_regime.csv", index=False)

    # Paired bootstrap & DeLong comparisons on pooled non-overlapping out-of-fold test predictions
    paired_rows: list[dict[str, Any]] = []
    for regime in ("rolling_origin", "account_group"):
        sel_rows = summary_df[
            (summary_df["regime"] == regime) & summary_df["selected_structural_by_validation"]
        ]
        val_winner = str(sel_rows.iloc[0]["model"]) if not sel_rows.empty else "LogReg"

        comparisons: list[tuple[str, str]] = []
        for mname in structural_models:
            if mname != "SizeOnly":
                comparisons.append((mname, "SizeOnly"))
            if mname != val_winner:
                comparisons.append((mname, val_winner))
        comparisons.append(("StructPlusActivity_20", val_winner))
        comparisons.append(("ActivityOnly_6", val_winner))

        seen_pairs: set[tuple[str, str]] = set()
        for model_a, model_b in comparisons:
            if (model_a, model_b) in seen_pairs:
                continue
            seen_pairs.add((model_a, model_b))
            if (regime, model_a) not in oof_store or (regime, model_b) not in oof_store:
                continue
            dfa = pd.concat(oof_store[(regime, model_a)], ignore_index=True)
            dfb = pd.concat(oof_store[(regime, model_b)], ignore_index=True)
            merged = dfa.merge(dfb[["sg_id", "probability"]], on="sg_id", suffixes=("_a", "_b"))
            if merged.empty or merged["y_true"].nunique() < 2:
                continue
            pstats = paired_bootstrap_comparison(
                merged["y_true"].to_numpy(),
                merged["probability_a"].to_numpy(),
                merged["probability_b"].to_numpy(),
                weights=merged["ipw_weight"].to_numpy(),
                n_boot=n_paired,
                seed=seed,
            )
            paired_rows.append(
                {
                    "regime": regime,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_paired_test": len(merged),
                    **pstats,
                }
            )
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(p2_dir / "paired_comparisons.csv", index=False)

    # Build Before vs After table (Legacy Attempt-vs-BFS vs Label-Blind Rolling & Account)
    leg_ibm = legacy_unit5_df[
        (legacy_unit5_df["source"] == "ibm")
        & (legacy_unit5_df["target"] == "ibm")
        & (legacy_unit5_df["variant"] == "tuned_calibrated")
    ].set_index("model")
    leg_raw_size = legacy_results_raw_df[
        (legacy_results_raw_df["train_on"] == "ibm")
        & (legacy_results_raw_df["test_on"] == "ibm")
        & (~legacy_results_raw_df["size_restricted"])
        & (legacy_results_raw_df["seed"] == 0)
        & (legacy_results_raw_df["model"] == "SizeOnly")
    ]

    before_after_rows: list[dict[str, Any]] = []
    for mname in [
        "SizeOnly",
        "LogReg",
        "RandomForest",
        "GradBoosting",
        "ExtraTrees",
        "StructPlusActivity_20",
    ]:
        if mname == "SizeOnly" and not leg_raw_size.empty:
            r0 = leg_raw_size.iloc[0]
            leg_ap = float(r0["pr_auc"])
            leg_prev = float(r0["prevalence"])
            leg_lift = float(r0["lift"])
            leg_auc = float(r0["roc_auc"])
            leg_lo, leg_hi = (np.nan, np.nan)
        elif mname == "StructPlusActivity_20" and not legacy_activity_df.empty:
            ra = legacy_activity_df[legacy_activity_df["features"] == "augmented_20"].iloc[0]
            leg_ap = float(ra["AP"])
            leg_prev = float(ra["prevalence"])
            leg_lift = float(ra["AP_lift"])
            leg_auc = float(ra["ROC_AUC"])
            leg_lo, leg_hi = (np.nan, np.nan)
        elif mname in leg_ibm.index:
            rm = leg_ibm.loc[mname]
            leg_ap = float(rm["pr_auc"])
            leg_prev = float(rm["prevalence"])
            leg_lift = float(rm["lift"])
            leg_auc = float(rm["roc_auc"])
            leg_lo = float(rm["pr_auc_ci95_lo"])
            leg_hi = float(rm["pr_auc_ci95_hi"])
        else:
            leg_ap = leg_prev = leg_lift = leg_auc = leg_lo = leg_hi = np.nan

        row_ba: dict[str, Any] = {
            "model": mname,
            "legacy_prevalence": leg_prev,
            "legacy_ap": leg_ap,
            "legacy_ap_ci95_lo": leg_lo,
            "legacy_ap_ci95_hi": leg_hi,
            "legacy_lift": leg_lift,
            "legacy_roc_auc": leg_auc,
        }
        for regime, prefix in (("rolling_origin", "ro"), ("account_group", "ag")):
            sub_s = summary_df[(summary_df["regime"] == regime) & (summary_df["model"] == mname)]
            if not sub_s.empty and sub_s.iloc[0]["status"] == "ok":
                sr = sub_s.iloc[0]
                row_ba.update(
                    {
                        f"{prefix}_selected_by_val": bool(sr["selected_structural_by_validation"]),
                        f"{prefix}_val_ap": float(sr["mean_validation_ap"]),
                        f"{prefix}_unweighted_prev": float(sr["prevalence_mean"]),
                        f"{prefix}_ap_mean": float(sr["ap_mean"]),
                        f"{prefix}_ap_fold_ci95_lo": float(sr["ap_fold_ci95_lo"]),
                        f"{prefix}_ap_fold_ci95_hi": float(sr["ap_fold_ci95_hi"]),
                        f"{prefix}_pooled_ap": float(sr["pooled_ap"]),
                        f"{prefix}_pooled_ap_ci95_lo": float(sr["pooled_ap_boot_ci95_lo"]),
                        f"{prefix}_pooled_ap_ci95_hi": float(sr["pooled_ap_boot_ci95_hi"]),
                        f"{prefix}_lift_mean": float(sr["lift_mean"]),
                        f"{prefix}_roc_auc_mean": float(sr["roc_auc_mean"]),
                        f"{prefix}_ipw_prev": float(sr["ipw_prevalence_mean"]),
                        f"{prefix}_ipw_ap_mean": float(sr["ipw_ap_mean"]),
                        f"{prefix}_ipw_ap_ci95_lo": float(sr["ipw_ap_fold_ci95_lo"]),
                        f"{prefix}_ipw_ap_ci95_hi": float(sr["ipw_ap_fold_ci95_hi"]),
                        f"{prefix}_ipw_lift_mean": float(sr["ipw_lift_mean"]),
                        f"{prefix}_ipw_roc_auc_mean": float(sr["ipw_roc_auc_mean"]),
                    }
                )
        before_after_rows.append(row_ba)

    before_after_df = pd.DataFrame(before_after_rows)
    before_after_df.to_csv(p2_dir / "before_after_comparison.csv", index=False)

    # Window-leak audit table (active_span_minutes / window_minutes in legacy vs symmetric Phase 2)
    lb_span_ratio = lb_df["active_span_minutes"] / np.maximum(1.0, lb_df["window_minutes"])
    audit_rows: list[dict[str, Any]] = [
        {
            "benchmark": "phase2_symmetric_window",
            "class_label": 0,
            "n": int((lb_df["label"] == 0).sum()),
            "mean_span_ratio": float(lb_span_ratio[lb_df["label"] == 0].mean()),
            "median_span_ratio": float(lb_span_ratio[lb_df["label"] == 0].median()),
            "frac_clamped_ge_099": float((lb_span_ratio[lb_df["label"] == 0] >= 0.99).mean()),
        },
        {
            "benchmark": "phase2_symmetric_window",
            "class_label": 1,
            "n": int((lb_df["label"] == 1).sum()),
            "mean_span_ratio": float(lb_span_ratio[lb_df["label"] == 1].mean()),
            "median_span_ratio": float(lb_span_ratio[lb_df["label"] == 1].median()),
            "frac_clamped_ge_099": float((lb_span_ratio[lb_df["label"] == 1] >= 0.99).mean()),
        },
    ]
    if legacy_aug_table_path is not None and legacy_aug_table_path.exists():
        leg_aug = pd.read_csv(legacy_aug_table_path)
        # Reconstruct declared duration from n_transactions * 60 / transactions_per_hour
        leg_dur = np.where(
            leg_aug["transactions_per_hour"] > 0,
            60.0 * leg_aug["n_transactions"] / leg_aug["transactions_per_hour"],
            60.0,
        )
        leg_ratio = np.clip(leg_aug["active_span_minutes"] / np.maximum(1.0, leg_dur), 0.0, 1.0)
        for clab in (0, 1):
            sub_r = leg_ratio[leg_aug["label"] == clab]
            audit_rows.append(
                {
                    "benchmark": "legacy_clamped_attempt_window",
                    "class_label": clab,
                    "n": int(len(sub_r)),
                    "mean_span_ratio": float(np.mean(sub_r)),
                    "median_span_ratio": float(np.median(sub_r)),
                    "frac_clamped_ge_099": float(np.mean(sub_r >= 0.99)),
                }
            )
    window_audit_df = pd.DataFrame(audit_rows)
    window_audit_df.to_csv(p2_dir / "window_leak_audit_summary.csv", index=False)

    ro_sel = summary_df[
        (summary_df["regime"] == "rolling_origin") & summary_df["selected_structural_by_validation"]
    ]
    ag_sel = summary_df[
        (summary_df["regime"] == "account_group") & summary_df["selected_structural_by_validation"]
    ]
    headline = {
        "sampler": sampler_info,
        "rolling_origin_validation_selected_model": (
            ro_sel.iloc[0].to_dict() if not ro_sel.empty else None
        ),
        "account_group_validation_selected_model": (
            ag_sel.iloc[0].to_dict() if not ag_sel.empty else None
        ),
    }
    with open(p2_dir / "phase2_headline.json", "w", encoding="utf-8") as fh:
        json.dump(headline, fh, indent=2, default=str)

    return {
        "candidates": lb_df,
        "fold_metrics": fold_df,
        "summary": summary_df,
        "paired": paired_df,
        "before_after": before_after_df,
        "window_audit": window_audit_df,
        "headline": headline,
    }
