"""Classifier definitions, hyperparameter candidate pools, and Platt calibration."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1.0e-6)
    return np.log(clipped / (1.0 - clipped))


class ValidationPlattCalibrator:
    """Fit 1D logistic calibration on validation log-odds ONLY (never touches test)."""

    def __init__(self, base_estimator: Any, platt_c: float = 1.0) -> None:
        self.base_estimator = base_estimator
        self.calibrator = LogisticRegression(C=platt_c, max_iter=1000)

    def fit_calibrator(
        self, x_val: pd.DataFrame | np.ndarray, y_val: np.ndarray
    ) -> ValidationPlattCalibrator:
        p_raw = self.base_estimator.predict_proba(x_val)[:, 1]
        self.calibrator.fit(_logit(p_raw).reshape(-1, 1), np.asarray(y_val, dtype=int))
        return self

    def predict_proba(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        p_raw = self.base_estimator.predict_proba(x)[:, 1]
        return self.calibrator.predict_proba(_logit(p_raw).reshape(-1, 1))


def baseline_models(
    seed: int, benchmark_cfg: dict[str, Any] | None = None
) -> dict[str, tuple[str, Any]]:
    """Construct the four baseline classifiers from Cell 17."""
    mcfg = (benchmark_cfg or {}).get("models", {})
    rf_cfg = mcfg.get("RandomForest", {})
    gb_cfg = mcfg.get("GradBoosting", {})
    lr_iter = int(mcfg.get("LogReg", {}).get("max_iter", 2000))
    return {
        "SizeOnly": (
            "size",
            make_pipeline(
                FunctionTransformer(np.log1p),
                StandardScaler(),
                LogisticRegression(max_iter=lr_iter, class_weight="balanced"),
            ),
        ),
        "LogReg": (
            "all",
            make_pipeline(
                FunctionTransformer(np.log1p),
                StandardScaler(),
                LogisticRegression(max_iter=lr_iter, class_weight="balanced"),
            ),
        ),
        "RandomForest": (
            "all",
            RandomForestClassifier(
                int(rf_cfg.get("n_estimators", 400)),
                min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 2)),
                class_weight=rf_cfg.get("class_weight", "balanced_subsample"),
                n_jobs=-1,
                random_state=seed,
            ),
        ),
        "GradBoosting": (
            "all",
            HistGradientBoostingClassifier(
                max_iter=int(gb_cfg.get("max_iter", 300)),
                learning_rate=float(gb_cfg.get("learning_rate", 0.05)),
                class_weight=gb_cfg.get("class_weight", "balanced"),
                random_state=seed,
            ),
        ),
    }


def tuned_candidates(
    seed: int = 0,
    tuning_cfg: dict[str, Any] | None = None,
) -> dict[str, list[tuple[str, Any, dict[str, Any]]]]:
    """Construct bounded hyperparameter search space for Unit 4."""
    grids = (tuning_cfg or {}).get("grids", {})
    pool: dict[str, list[tuple[str, Any, dict[str, Any]]]] = {
        "LogReg": [],
        "RandomForest": [],
        "GradBoosting": [],
        "ExtraTrees": [],
    }

    lr_grid = grids.get("LogReg", {})
    c_values = lr_grid.get("c_values", [0.01, 0.1, 1.0, 10.0])
    lr_max_iter = int(lr_grid.get("max_iter", 2000))
    for c_val in c_values:
        model = make_pipeline(
            FunctionTransformer(np.log1p),
            StandardScaler(),
            LogisticRegression(
                C=float(c_val),
                max_iter=lr_max_iter,
                class_weight="balanced",
                random_state=seed,
            ),
        )
        pool["LogReg"].append((f"C={float(c_val):g}", model, {"C": float(c_val)}))

    rf_default = [
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2, "max_features": "sqrt"},
        {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 5, "max_features": "sqrt"},
        {"n_estimators": 400, "max_depth": 10, "min_samples_leaf": 5, "max_features": 0.5},
        {"n_estimators": 400, "max_depth": 16, "min_samples_leaf": 2, "max_features": 0.5},
    ]
    for spec in grids.get("RandomForest", {}).get("candidates", rf_default):
        n_trees = int(spec["n_estimators"])
        depth = None if spec["max_depth"] is None else int(spec["max_depth"])
        leaf = int(spec["min_samples_leaf"])
        max_feat = spec["max_features"]
        model = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=depth,
            min_samples_leaf=leaf,
            max_features=max_feat,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
        name = f"trees={n_trees}; depth={depth}; leaf={leaf}; feat={max_feat}"
        pool["RandomForest"].append(
            (
                name,
                model,
                {
                    "n_estimators": n_trees,
                    "max_depth": depth,
                    "min_samples_leaf": leaf,
                    "max_features": max_feat,
                },
            )
        )

    gb_default = [
        {
            "learning_rate": 0.03,
            "max_iter": 300,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
        },
        {
            "learning_rate": 0.03,
            "max_iter": 300,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 50,
            "l2_regularization": 1.0,
        },
        {
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
        },
        {
            "learning_rate": 0.05,
            "max_iter": 200,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 30,
            "l2_regularization": 5.0,
        },
        {
            "learning_rate": 0.08,
            "max_iter": 250,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 1.0,
        },
    ]
    for spec in grids.get("GradBoosting", {}).get("candidates", gb_default):
        lr = float(spec["learning_rate"])
        rounds = int(spec["max_iter"])
        leaves = int(spec["max_leaf_nodes"])
        min_leaf = int(spec["min_samples_leaf"])
        reg = float(spec["l2_regularization"])
        model = HistGradientBoostingClassifier(
            max_iter=rounds,
            learning_rate=lr,
            max_leaf_nodes=leaves,
            min_samples_leaf=min_leaf,
            l2_regularization=reg,
            class_weight="balanced",
            random_state=seed,
        )
        name = f"lr={lr}; rounds={rounds}; leaves={leaves}; min_leaf={min_leaf}; L2={reg}"
        pool["GradBoosting"].append(
            (
                name,
                model,
                {
                    "learning_rate": lr,
                    "max_iter": rounds,
                    "max_leaf_nodes": leaves,
                    "min_samples_leaf": min_leaf,
                    "l2_regularization": reg,
                },
            )
        )

    et_default = [
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2, "max_features": "sqrt"},
        {"n_estimators": 300, "max_depth": 14, "min_samples_leaf": 3, "max_features": 0.5},
        {"n_estimators": 400, "max_depth": 10, "min_samples_leaf": 5, "max_features": "sqrt"},
    ]
    for spec in grids.get("ExtraTrees", {}).get("candidates", et_default):
        n_trees = int(spec["n_estimators"])
        depth = None if spec["max_depth"] is None else int(spec["max_depth"])
        leaf = int(spec["min_samples_leaf"])
        max_feat = spec["max_features"]
        model = ExtraTreesClassifier(
            n_estimators=n_trees,
            max_depth=depth,
            min_samples_leaf=leaf,
            max_features=max_feat,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
        name = f"trees={n_trees}; depth={depth}; leaf={leaf}; feat={max_feat}"
        pool["ExtraTrees"].append(
            (
                name,
                model,
                {
                    "n_estimators": n_trees,
                    "max_depth": depth,
                    "min_samples_leaf": leaf,
                    "max_features": max_feat,
                },
            )
        )

    return pool
