"""Phase 2 methodology: label-blind evaluation, fold-stratified tests, Kish IPW, and seed check."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from aml_graph.evaluate import best_thr
from aml_graph.features import ACTIVITY_COLS, AUGMENTED_COLS, FEATURES, SIZE_COLS
from aml_graph.models import ValidationPlattCalibrator, baseline_models, tuned_candidates
from aml_graph.sampling import (
    EdgeIndex,
    index_documented_attempt_edges,
    sample_label_blind_ibm,
)
from aml_graph.splits import account_component_group_splits, rolling_origin_splits


def kish_effective_sample_size(weights: np.ndarray) -> float:
    """Compute Kish's effective sample size (sum w)^2 / sum(w^2)."""
    w = np.asarray(weights, dtype=float)
    w = w[np.isfinite(w) & (w > 0.0)]
    if w.size == 0:
        return 0.0
    denom = float(np.sum(w * w))
    if denom <= 0.0:
        return 0.0
    s = float(np.sum(w))
    return (s * s) / denom


def _clip_interval(
    lo: float,
    hi: float,
    min_val: float | None = 0.0,
    max_val: float | None = 1.0,
) -> tuple[float, float]:
    """Clip confidence interval endpoints strictly to valid mathematical domain."""
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (float("nan"), float("nan"))
    if min_val is not None:
        lo = max(min_val, lo)
        hi = max(min_val, hi)
    if max_val is not None:
        lo = min(max_val, lo)
        hi = min(max_val, hi)
    return (float(lo), float(hi))


def fold_stratified_bootstrap_summary(
    fold_tuples: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]],
    n_boot: int = 800,
    seed: int = 2026,
) -> dict[str, float]:
    """Compute fold-mean AP, Lift, and ROC-AUC with fold-stratified clipped bootstrap 95% CIs.

    Resamples positives and negatives independently *within* each fold k, computes the metric
    inside fold k, and averages across the K folds—never pooling raw probabilities across folds
    with different prevalences, and never using normal/Student-t approximations.
    """
    valid_folds = [
        (
            np.asarray(y, dtype=int),
            np.asarray(p, dtype=float),
            None if w is None else np.asarray(w, dtype=float),
        )
        for y, p, w in fold_tuples
        if len(y) > 0 and np.unique(y).size == 2
    ]
    if not valid_folds:
        return {
            "ap_mean": np.nan,
            "ap_ci95_lo": np.nan,
            "ap_ci95_hi": np.nan,
            "lift_mean": np.nan,
            "lift_ci95_lo": np.nan,
            "lift_ci95_hi": np.nan,
            "roc_auc_mean": np.nan,
            "roc_auc_ci95_lo": np.nan,
            "roc_auc_ci95_hi": np.nan,
        }

    def eval_one(
        y_k: np.ndarray, p_k: np.ndarray, w_k: np.ndarray | None
    ) -> tuple[float, float, float]:
        if w_k is None:
            prev = float(y_k.mean())
            ap = float(average_precision_score(y_k, p_k))
            auc = float(roc_auc_score(y_k, p_k))
        else:
            prev = float(np.average(y_k, weights=w_k))
            ap = float(average_precision_score(y_k, p_k, sample_weight=w_k))
            auc = float(roc_auc_score(y_k, p_k, sample_weight=w_k))
        return ap, (ap / prev if prev > 0 else np.nan), auc

    obs = [eval_one(yk, pk, wk) for yk, pk, wk in valid_folds]
    ap_mean = float(np.mean([o[0] for o in obs]))
    lift_mean = float(np.mean([o[1] for o in obs]))
    auc_mean = float(np.mean([o[2] for o in obs]))

    rng = np.random.default_rng(seed)
    fold_pos_neg = [(np.flatnonzero(yk == 1), np.flatnonzero(yk == 0)) for yk, _, _ in valid_folds]

    boot_aps: list[float] = []
    boot_lifts: list[float] = []
    boot_aucs: list[float] = []

    for _ in range(n_boot):
        f_aps: list[float] = []
        f_lifts: list[float] = []
        f_aucs: list[float] = []
        for (yk, pk, wk), (p_idx, n_idx) in zip(valid_folds, fold_pos_neg, strict=False):
            bp = rng.choice(p_idx, size=p_idx.size, replace=True)
            bn = rng.choice(n_idx, size=n_idx.size, replace=True)
            b_idx = np.concatenate([bp, bn])
            wb = None if wk is None else wk[b_idx]
            ap_b, lift_b, auc_b = eval_one(yk[b_idx], pk[b_idx], wb)
            f_aps.append(ap_b)
            f_lifts.append(lift_b)
            f_aucs.append(auc_b)
        boot_aps.append(float(np.mean(f_aps)))
        boot_lifts.append(float(np.mean(f_lifts)))
        boot_aucs.append(float(np.mean(f_aucs)))

    ap_lo, ap_hi = _clip_interval(*np.percentile(boot_aps, [2.5, 97.5]), min_val=0.0, max_val=1.0)
    lift_lo, lift_hi = _clip_interval(
        *np.percentile(boot_lifts, [2.5, 97.5]), min_val=0.0, max_val=None
    )
    auc_lo, auc_hi = _clip_interval(
        *np.percentile(boot_aucs, [2.5, 97.5]), min_val=0.0, max_val=1.0
    )

    return {
        "ap_mean": ap_mean,
        "ap_ci95_lo": ap_lo,
        "ap_ci95_hi": ap_hi,
        "lift_mean": lift_mean,
        "lift_ci95_lo": lift_lo,
        "lift_ci95_hi": lift_hi,
        "roc_auc_mean": auc_mean,
        "roc_auc_ci95_lo": auc_lo,
        "roc_auc_ci95_hi": auc_hi,
    }


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
    """Within-fold two-sided correlated DeLong test returning delta, variance, z, and p."""
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
            "var_delta_auc": np.nan,
            "delong_z": np.nan,
            "delong_p": np.nan,
        }

    auc_a, v10_a, v01_a = _delong_structural_components(y, pa)
    auc_b, v10_b, v01_b = _delong_structural_components(y, pb)
    s10 = np.cov(np.vstack([v10_a, v10_b]), ddof=1)
    s01 = np.cov(np.vstack([v01_a, v01_b]), ddof=1)
    cov = s10 / m + s01 / n
    diff = auc_a - auc_b
    var_diff = max(0.0, float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]))
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
        "var_delta_auc": var_diff,
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
    """Single-fold paired stratified bootstrap + DeLong comparison."""
    return fold_stratified_paired_comparison(
        [(y_true, prob_a, prob_b)],
        n_boot=n_boot,
        seed=seed,
    )[0]


def fold_stratified_paired_comparison(
    fold_pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_boot: int = 1000,
    seed: int = 2026,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Fold-stratified paired bootstrap (AP & AUC) + stratified DeLong across disjoint folds.

    Computes deltas strictly inside each fold k (avoiding Simpson's-paradox ranking inversion
    from cross-fold prevalence shift) and combines across folds via:
      - Fold-mean delta_AP and delta_AUC with fold-stratified paired bootstrap 95% CIs
      - Stratified DeLong Z = mean(delta_AUC_k) / sqrt(sum(var_k) / K^2) across independent folds.
    Returns (combined_summary_dict, list_of_per_fold_dicts).
    """
    valid: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    for k, (yk, pak, pbk) in enumerate(fold_pairs):
        y_arr = np.asarray(yk, dtype=int)
        if y_arr.size > 0 and np.unique(y_arr).size == 2:
            valid.append((k, y_arr, np.asarray(pak, dtype=float), np.asarray(pbk, dtype=float)))

    if not valid:
        empty = {
            "n_folds": 0,
            "n_total_test": 0,
            "ap_a_mean": np.nan,
            "ap_b_mean": np.nan,
            "delta_ap": np.nan,
            "delta_ap_ci95_lo": np.nan,
            "delta_ap_ci95_hi": np.nan,
            "paired_p_ap": np.nan,
            "auc_a_mean": np.nan,
            "auc_b_mean": np.nan,
            "delta_auc": np.nan,
            "delta_auc_ci95_lo": np.nan,
            "delta_auc_ci95_hi": np.nan,
            "delong_z": np.nan,
            "delong_p": np.nan,
        }
        return empty, []

    per_fold_rows: list[dict[str, float]] = []
    for k, yk, pak, pbk in valid:
        ap_a = float(average_precision_score(yk, pak))
        ap_b = float(average_precision_score(yk, pbk))
        dl = delong_roc_test(yk, pak, pbk)
        per_fold_rows.append(
            {
                "fold": k,
                "n_test": int(yk.size),
                "n_positives": int(yk.sum()),
                "prevalence": float(yk.mean()),
                "ap_a": ap_a,
                "ap_b": ap_b,
                "delta_ap": ap_a - ap_b,
                "auc_a": dl["auc_a"],
                "auc_b": dl["auc_b"],
                "delta_auc": dl["delta_auc"],
                "var_delta_auc": dl["var_delta_auc"],
                "delong_z": dl["delong_z"],
                "delong_p": dl["delong_p"],
            }
        )

    k_count = len(per_fold_rows)
    ap_a_mean = float(np.mean([r["ap_a"] for r in per_fold_rows]))
    ap_b_mean = float(np.mean([r["ap_b"] for r in per_fold_rows]))
    obs_delta_ap = float(np.mean([r["delta_ap"] for r in per_fold_rows]))
    auc_a_mean = float(np.mean([r["auc_a"] for r in per_fold_rows]))
    auc_b_mean = float(np.mean([r["auc_b"] for r in per_fold_rows]))
    obs_delta_auc = float(np.mean([r["delta_auc"] for r in per_fold_rows]))

    # Stratified DeLong across K independent folds: Var(mean) = sum(var_k) / K^2
    vars_k = [r["var_delta_auc"] for r in per_fold_rows]
    if all(np.isfinite(v) for v in vars_k):
        var_mean_auc = float(np.sum(vars_k)) / (k_count * k_count)
        if var_mean_auc <= 1e-18:
            z_strat = (
                0.0 if abs(obs_delta_auc) <= 1e-12 else float(np.copysign(np.inf, obs_delta_auc))
            )
            p_strat = 1.0 if abs(obs_delta_auc) <= 1e-12 else 0.0
        else:
            z_strat = float(obs_delta_auc / np.sqrt(var_mean_auc))
            p_strat = float(2.0 * norm.sf(abs(z_strat)))
    else:
        z_strat, p_strat = np.nan, np.nan

    # Fold-stratified paired bootstrap for mean delta_AP and mean delta_AUC
    rng = np.random.default_rng(seed)
    idx_splits = [(np.flatnonzero(yk == 1), np.flatnonzero(yk == 0)) for _, yk, _, _ in valid]
    boot_dap: list[float] = []
    boot_dauc: list[float] = []

    for _ in range(n_boot):
        f_dap: list[float] = []
        f_dauc: list[float] = []
        for (_, yk, pak, pbk), (p_i, n_i) in zip(valid, idx_splits, strict=False):
            bp = rng.choice(p_i, size=p_i.size, replace=True)
            bn = rng.choice(n_i, size=n_i.size, replace=True)
            b_idx = np.concatenate([bp, bn])
            yb = yk[b_idx]
            f_dap.append(
                float(
                    average_precision_score(yb, pak[b_idx])
                    - average_precision_score(yb, pbk[b_idx])
                )
            )
            f_dauc.append(float(roc_auc_score(yb, pak[b_idx]) - roc_auc_score(yb, pbk[b_idx])))
        boot_dap.append(float(np.mean(f_dap)))
        boot_dauc.append(float(np.mean(f_dauc)))

    arr_dap = np.asarray(boot_dap, dtype=float)
    arr_dauc = np.asarray(boot_dauc, dtype=float)
    dap_lo, dap_hi = _clip_interval(*np.percentile(arr_dap, [2.5, 97.5]), min_val=-1.0, max_val=1.0)
    dauc_lo, dauc_hi = _clip_interval(
        *np.percentile(arr_dauc, [2.5, 97.5]), min_val=-1.0, max_val=1.0
    )
    p_boot_ap = float(2.0 * min(np.mean(arr_dap <= 0.0), np.mean(arr_dap >= 0.0)))
    p_boot_ap = min(1.0, max(p_boot_ap, 1.0 / (n_boot + 1)))

    combined = {
        "n_folds": k_count,
        "n_total_test": int(sum(r["n_test"] for r in per_fold_rows)),
        "ap_a_mean": ap_a_mean,
        "ap_b_mean": ap_b_mean,
        "delta_ap": obs_delta_ap,
        "delta_ap_ci95_lo": dap_lo,
        "delta_ap_ci95_hi": dap_hi,
        "paired_p_ap": p_boot_ap,
        "auc_a_mean": auc_a_mean,
        "auc_b_mean": auc_b_mean,
        "delta_auc": obs_delta_auc,
        "delta_auc_ci95_lo": dauc_lo,
        "delta_auc_ci95_hi": dauc_hi,
        "delong_z": z_strat,
        "delong_p": p_strat,
    }
    return combined, per_fold_rows


def score_unweighted_and_ipw(
    y: np.ndarray,
    prob: np.ndarray,
    thr: float,
    ipw_weights: np.ndarray,
    seed_sources: np.ndarray | None = None,
    min_test_positives: int = 30,
    min_effective_positives: float = 30.0,
) -> dict[str, Any]:
    """Compute unweighted & IPW fold metrics; enforce >=30 positives AND >=30 Kish eff positives."""
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(prob, dtype=float)
    w_arr = np.asarray(ipw_weights, dtype=float)
    n_pos = int(y_arr.sum())
    n_tot = int(y_arr.size)

    if seed_sources is not None and len(seed_sources) == n_tot:
        s_arr = np.asarray(seed_sources, dtype=str)
        n_uni_tot = int(np.sum(s_arr == "uniform"))
        n_uni_pos = int(np.sum((s_arr == "uniform") & (y_arr == 1)))
    else:
        n_uni_tot = n_tot
        n_uni_pos = n_pos

    kish_tot = kish_effective_sample_size(w_arr)
    kish_pos = kish_effective_sample_size(w_arr[y_arr == 1]) if n_pos > 0 else 0.0

    base_info = {
        "n_test": n_tot,
        "n_positives": n_pos,
        "n_uniform_test": n_uni_tot,
        "n_uniform_positives": n_uni_pos,
        "kish_n_eff_total": kish_tot,
        "kish_n_eff_positives": kish_pos,
    }

    if n_tot == 0 or np.unique(y_arr).size < 2:
        return {
            "status": "not_estimable_single_class",
            "ipw_status": "not_estimable_single_class",
            **base_info,
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
            "ipw_status": "not_estimable_lt_30_eff_positives",
            **base_info,
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

    if kish_pos < float(min_effective_positives):
        ipw_status = "not_estimable_lt_30_eff_positives"
        ipw_ap = np.nan
        ipw_lift = np.nan
        ipw_auc = np.nan
    else:
        ipw_status = "ok"
        ipw_ap = float(average_precision_score(y_arr, p_arr, sample_weight=w_arr))
        ipw_lift = ipw_ap / ipw_prev if ipw_prev > 0 else np.nan
        ipw_auc = float(roc_auc_score(y_arr, p_arr, sample_weight=w_arr))

    return {
        "status": "ok",
        "ipw_status": ipw_status,
        **base_info,
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
        "ipw_lift": ipw_lift,
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
    """Execute Phase 2 label-blind methodology, fold-stratified tests, Kish IPW, and seed check."""
    p2_dir = out_dir / "phase2"
    p2_dir.mkdir(parents=True, exist_ok=True)

    seed = int(phase2_cfg.get("seed", 2026))
    scfg = phase2_cfg.get("sampler", {})
    sp_cfg = phase2_cfg.get("splits", {})
    ev_cfg = phase2_cfg.get("evaluation", {})
    min_pos = int(sp_cfg.get("min_test_positives", 30))
    min_eff_pos = float(sp_cfg.get("min_effective_positives", 30.0))
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
    # Store per-fold test DataFrames keyed by (regime, model, fold)
    fold_preds: dict[tuple[str, str], dict[int, pd.DataFrame]] = {}

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
                continue

            for mname in all_eval_models:
                cal_est, cols, thr, val_ap, chosen_cand = _fit_and_select_fold_model(
                    mname, fit_df, val_df, seed + fold_k, benchmark_cfg, tuning_cfg
                )
                if len(test_df):
                    p_test = cal_est.predict_proba(test_df[cols])[:, 1]
                    y_test = test_df["label"].to_numpy(dtype=int)
                    w_test = test_df["ipw_weight"].to_numpy(dtype=float)
                    s_test = test_df["seed_source"].to_numpy(dtype=str)
                else:
                    p_test = np.array([], dtype=float)
                    y_test = np.array([], dtype=int)
                    w_test = np.array([], dtype=float)
                    s_test = np.array([], dtype=str)

                m_scores = score_unweighted_and_ipw(
                    y_test,
                    p_test,
                    thr,
                    w_test,
                    seed_sources=s_test,
                    min_test_positives=min_pos,
                    min_effective_positives=min_eff_pos,
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
                        "threshold": thr,
                        "n_fit": len(fit_df),
                        "n_val": len(val_df),
                        **m_scores,
                    }
                )
                fold_preds.setdefault((regime, mname), {})[fold_k] = pd.DataFrame(
                    {
                        "sg_id": test_df["sg_id"].to_numpy(),
                        "fold": fold_k,
                        "y_true": y_test,
                        "probability": p_test,
                        "ipw_weight": w_test,
                        "seed_source": s_test,
                    }
                )

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(p2_dir / "fold_metrics.csv", index=False)

    # Save Appendix IPW & Kish Effective Sample Size Audit table (one row per regime x fold)
    ipw_audit_cols = [
        "regime",
        "fold",
        "n_test",
        "n_positives",
        "prevalence",
        "n_uniform_test",
        "n_uniform_positives",
        "kish_n_eff_total",
        "kish_n_eff_positives",
        "ipw_prevalence",
        "ipw_status",
    ]
    ipw_audit_df = (
        fold_df[fold_df["model"] == "LogReg"][ipw_audit_cols]
        .sort_values(["regime", "fold"])
        .reset_index(drop=True)
    )
    ipw_audit_df.to_csv(p2_dir / "ipw_appendix_audit.csv", index=False)

    # Aggregate across estimable folds using fold-stratified clipped bootstrap CIs ONLY
    summary_rows: list[dict[str, Any]] = []
    for regime in ("rolling_origin", "account_group"):
        sub_reg = fold_df[fold_df["regime"] == regime]
        for mname in all_eval_models:
            sub_m = sub_reg[sub_reg["model"] == mname]
            ok_m = sub_m[sub_m["status"] == "ok"].sort_values("fold")
            n_total_folds = len(sub_m)
            n_est_folds = len(ok_m)
            fgroup = str(sub_m["feature_group"].iloc[0])
            val_ap_mean = (
                float(sub_m["validation_ap"].dropna().mean())
                if sub_m["validation_ap"].notna().any()
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
                    }
                )
                continue

            fmap = fold_preds[(regime, mname)]
            unw_tuples = [
                (fmap[int(fk)]["y_true"].to_numpy(), fmap[int(fk)]["probability"].to_numpy(), None)
                for fk in ok_m["fold"]
            ]
            boot_unw = fold_stratified_bootstrap_summary(unw_tuples, n_boot=n_boot, seed=seed)

            ok_ipw = ok_m[ok_m["ipw_status"] == "ok"]
            if len(ok_ipw) > 0:
                ipw_tuples = [
                    (
                        fmap[int(fk)]["y_true"].to_numpy(),
                        fmap[int(fk)]["probability"].to_numpy(),
                        fmap[int(fk)]["ipw_weight"].to_numpy(),
                    )
                    for fk in ok_ipw["fold"]
                ]
                boot_ipw = fold_stratified_bootstrap_summary(ipw_tuples, n_boot=n_boot, seed=seed)
                ipw_reg_status = f"ok ({len(ok_ipw)}/{n_total_folds} folds)"
            else:
                boot_ipw = {
                    "ap_mean": np.nan,
                    "ap_ci95_lo": np.nan,
                    "ap_ci95_hi": np.nan,
                    "lift_mean": np.nan,
                    "lift_ci95_lo": np.nan,
                    "lift_ci95_hi": np.nan,
                    "roc_auc_mean": np.nan,
                    "roc_auc_ci95_lo": np.nan,
                    "roc_auc_ci95_hi": np.nan,
                }
                ipw_reg_status = "not_estimable_lt_30_eff_positives"

            summary_rows.append(
                {
                    "regime": regime,
                    "model": mname,
                    "feature_group": fgroup,
                    "status": "ok",
                    "estimable_folds": f"{n_est_folds}/{n_total_folds}",
                    "total_test_n": int(ok_m["n_test"].sum()),
                    "total_test_positives": int(ok_m["n_positives"].sum()),
                    "mean_validation_ap": val_ap_mean,
                    "prevalence_mean": float(ok_m["prevalence"].mean()),
                    "ap_mean": boot_unw["ap_mean"],
                    "ap_ci95_lo": boot_unw["ap_ci95_lo"],
                    "ap_ci95_hi": boot_unw["ap_ci95_hi"],
                    "lift_mean": boot_unw["lift_mean"],
                    "lift_ci95_lo": boot_unw["lift_ci95_lo"],
                    "lift_ci95_hi": boot_unw["lift_ci95_hi"],
                    "roc_auc_mean": boot_unw["roc_auc_mean"],
                    "roc_auc_ci95_lo": boot_unw["roc_auc_ci95_lo"],
                    "roc_auc_ci95_hi": boot_unw["roc_auc_ci95_hi"],
                    "f1_mean": float(ok_m["f1"].mean()),
                    "ipw_status": ipw_reg_status,
                    "mean_kish_n_eff_positives": float(ok_m["kish_n_eff_positives"].mean()),
                    "ipw_prevalence_mean": float(ok_m["ipw_prevalence"].mean()),
                    "ipw_ap_mean": boot_ipw["ap_mean"],
                    "ipw_ap_ci95_lo": boot_ipw["ap_ci95_lo"],
                    "ipw_ap_ci95_hi": boot_ipw["ap_ci95_hi"],
                    "ipw_lift_mean": boot_ipw["lift_mean"],
                    "ipw_lift_ci95_lo": boot_ipw["lift_ci95_lo"],
                    "ipw_lift_ci95_hi": boot_ipw["lift_ci95_hi"],
                    "ipw_roc_auc_mean": boot_ipw["roc_auc_mean"],
                    "ipw_roc_auc_ci95_lo": boot_ipw["roc_auc_ci95_lo"],
                    "ipw_roc_auc_ci95_hi": boot_ipw["roc_auc_ci95_hi"],
                }
            )

    summary_df = pd.DataFrame(summary_rows)
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

    # Fold-stratified paired comparisons (never pooling raw probabilities across folds!)
    paired_rows: list[dict[str, Any]] = []
    paired_by_fold_rows: list[dict[str, Any]] = []
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
            if (regime, model_a) not in fold_preds or (regime, model_b) not in fold_preds:
                continue
            ok_folds_a = set(
                fold_df[
                    (fold_df["regime"] == regime)
                    & (fold_df["model"] == model_a)
                    & (fold_df["status"] == "ok")
                ]["fold"].tolist()
            )
            ok_folds_b = set(
                fold_df[
                    (fold_df["regime"] == regime)
                    & (fold_df["model"] == model_b)
                    & (fold_df["status"] == "ok")
                ]["fold"].tolist()
            )
            shared_folds = sorted(ok_folds_a & ok_folds_b)
            if not shared_folds:
                continue
            fold_pair_list: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            for fk in shared_folds:
                fa = fold_preds[(regime, model_a)][int(fk)]
                fb = fold_preds[(regime, model_b)][int(fk)]
                m_ab = fa.merge(fb[["sg_id", "probability"]], on="sg_id", suffixes=("_a", "_b"))
                fold_pair_list.append(
                    (
                        m_ab["y_true"].to_numpy(),
                        m_ab["probability_a"].to_numpy(),
                        m_ab["probability_b"].to_numpy(),
                    )
                )
            comb, pf_list = fold_stratified_paired_comparison(
                fold_pair_list, n_boot=n_paired, seed=seed
            )
            paired_rows.append({"regime": regime, "model_a": model_a, "model_b": model_b, **comb})
            for pf_entry, fk in zip(pf_list, shared_folds, strict=False):
                paired_by_fold_rows.append(
                    {
                        "regime": regime,
                        "fold": int(fk),
                        "model_a": model_a,
                        "model_b": model_b,
                        **{k: v for k, v in pf_entry.items() if k != "fold"},
                    }
                )

    paired_df = pd.DataFrame(paired_rows)
    paired_by_fold_df = pd.DataFrame(paired_by_fold_rows)
    paired_df.to_csv(p2_dir / "paired_comparisons.csv", index=False)
    paired_by_fold_df.to_csv(p2_dir / "paired_comparisons_by_fold.csv", index=False)

    # Seed-mode check: evaluate strictly on uniform-seeded test candidates (pos & neg)
    uniform_rows: list[dict[str, Any]] = []
    uniform_by_fold_rows: list[dict[str, Any]] = []
    for regime in ("rolling_origin", "account_group"):
        for mname in all_eval_models:
            if (regime, mname) not in fold_preds:
                continue
            ok_folds_m = sorted(
                fold_df[
                    (fold_df["regime"] == regime)
                    & (fold_df["model"] == mname)
                    & (fold_df["status"] == "ok")
                ]["fold"]
                .astype(int)
                .tolist()
            )
            fmap = fold_preds[(regime, mname)]
            uni_tuples: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]] = []
            tot_u_n = 0
            tot_u_pos = 0
            folds_with_both = 0
            for fk in ok_folds_m:
                fdf = fmap[fk]
                udf = fdf[fdf["seed_source"] == "uniform"]
                un = int(len(udf))
                upos = int(udf["y_true"].sum())
                tot_u_n += un
                tot_u_pos += upos
                if un > 0 and udf["y_true"].nunique() == 2:
                    folds_with_both += 1
                    yu = udf["y_true"].to_numpy()
                    pu = udf["probability"].to_numpy()
                    uni_tuples.append((yu, pu, None))
                    uniform_by_fold_rows.append(
                        {
                            "regime": regime,
                            "fold": fk,
                            "model": mname,
                            "n_uniform": un,
                            "n_uniform_positives": upos,
                            "uniform_prevalence": float(yu.mean()),
                            "ap": float(average_precision_score(yu, pu)),
                            "lift": float(average_precision_score(yu, pu) / yu.mean()),
                            "roc_auc": float(roc_auc_score(yu, pu)),
                        }
                    )
                else:
                    uniform_by_fold_rows.append(
                        {
                            "regime": regime,
                            "fold": fk,
                            "model": mname,
                            "n_uniform": un,
                            "n_uniform_positives": upos,
                            "uniform_prevalence": float(upos / un) if un > 0 else np.nan,
                            "ap": np.nan,
                            "lift": np.nan,
                            "roc_auc": np.nan,
                        }
                    )

            uboot = fold_stratified_bootstrap_summary(uni_tuples, n_boot=n_boot, seed=seed)
            uniform_rows.append(
                {
                    "regime": regime,
                    "model": mname,
                    "n_uniform_total": tot_u_n,
                    "n_uniform_positives": tot_u_pos,
                    "n_uniform_negatives": tot_u_n - tot_u_pos,
                    "uniform_prevalence": float(tot_u_pos / max(1, tot_u_n)),
                    "folds_with_positives": f"{folds_with_both}/{len(ok_folds_m)}",
                    "regime_pos_ge_30": bool(tot_u_pos >= 30),
                    "estimability_note": (
                        "ok_regime_ge_30"
                        if tot_u_pos >= 30
                        else "low_power_lt_30_uniform_positives"
                    ),
                    "ap_mean": uboot["ap_mean"],
                    "ap_ci95_lo": uboot["ap_ci95_lo"],
                    "ap_ci95_hi": uboot["ap_ci95_hi"],
                    "lift_mean": uboot["lift_mean"],
                    "lift_ci95_lo": uboot["lift_ci95_lo"],
                    "lift_ci95_hi": uboot["lift_ci95_hi"],
                    "roc_auc_mean": uboot["roc_auc_mean"],
                    "roc_auc_ci95_lo": uboot["roc_auc_ci95_lo"],
                    "roc_auc_ci95_hi": uboot["roc_auc_ci95_hi"],
                }
            )

    uniform_check_df = pd.DataFrame(uniform_rows)
    uniform_by_fold_df = pd.DataFrame(uniform_by_fold_rows)
    uniform_check_df.to_csv(p2_dir / "uniform_seed_check.csv", index=False)
    uniform_by_fold_df.to_csv(p2_dir / "uniform_seed_check_by_fold.csv", index=False)

    # Build Before vs After table (clipped fold-stratified bootstrap CIs ONLY)
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
        "ActivityOnly_6",
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
                        f"{prefix}_prevalence": float(sr["prevalence_mean"]),
                        f"{prefix}_ap_mean": float(sr["ap_mean"]),
                        f"{prefix}_ap_ci95_lo": float(sr["ap_ci95_lo"]),
                        f"{prefix}_ap_ci95_hi": float(sr["ap_ci95_hi"]),
                        f"{prefix}_lift_mean": float(sr["lift_mean"]),
                        f"{prefix}_lift_ci95_lo": float(sr["lift_ci95_lo"]),
                        f"{prefix}_lift_ci95_hi": float(sr["lift_ci95_hi"]),
                        f"{prefix}_roc_auc_mean": float(sr["roc_auc_mean"]),
                        f"{prefix}_roc_auc_ci95_lo": float(sr["roc_auc_ci95_lo"]),
                        f"{prefix}_roc_auc_ci95_hi": float(sr["roc_auc_ci95_hi"]),
                        f"{prefix}_ipw_status": str(sr["ipw_status"]),
                    }
                )
        before_after_rows.append(row_ba)

    before_after_df = pd.DataFrame(before_after_rows)
    before_after_df.to_csv(p2_dir / "before_after_comparison.csv", index=False)

    # Window-leak audit table
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
        "ipw_estimable_per_fold": bool((ipw_audit_df["ipw_status"] == "ok").any()),
    }
    with open(p2_dir / "phase2_headline.json", "w", encoding="utf-8") as fh:
        json.dump(headline, fh, indent=2, default=str)

    return {
        "candidates": lb_df,
        "fold_metrics": fold_df,
        "ipw_audit": ipw_audit_df,
        "summary": summary_df,
        "paired": paired_df,
        "paired_by_fold": paired_by_fold_df,
        "uniform_check": uniform_check_df,
        "before_after": before_after_df,
        "window_audit": window_audit_df,
        "headline": headline,
    }
