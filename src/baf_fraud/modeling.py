"""Leakage-safe model candidates and documented sentinel-aware preprocessing."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# The BAF datasheet identifies -1 as a missing-state code for address-tenure
# and bank-tenure columns. Session/device -1 codes are also investigated here.
# IMPORTANT: do not replace every negative number. Credit-risk scores,
# velocity_6h, and intended_balcon_amount can legitimately be negative.
SENTINEL_COLUMNS = (
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
)


class SentinelMissingness(BaseEstimator, TransformerMixin):
    """Replace exact -1 codes in *selected* columns and retain their indicators.

    This transform is stateless: imputation and scaling are fitted later, only
    on the training data inside the sklearn pipeline.
    """

    def __init__(self, columns=SENTINEL_COLUMNS):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SentinelMissingness requires a pandas DataFrame")
        transformed = X.copy()
        for col in self.columns:
            if col in transformed.columns:
                is_missing = transformed[col].eq(-1)
                transformed[f"{col}__missing"] = is_missing.astype("int8")
                transformed.loc[is_missing, col] = float("nan")
        return transformed


def build_preprocessor(X_train):
    """Prepare numeric and categorical columns using train-only fitted statistics."""
    numerical = X_train.select_dtypes(include="number").columns.tolist()
    categorical = X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    if len(numerical) + len(categorical) != X_train.shape[1]:
        raise ValueError("Unexpected feature dtypes; encode them before modeling")
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    category_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numerical),
            ("categorical", category_pipeline, categorical),
        ],
        sparse_threshold=1.0,
    )


def build_models(X_train):
    """Original baselines. Keep their specification separate from experiments."""
    preprocessing = build_preprocessor(X_train)
    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", clone(preprocessing)),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=300,
                        solver="liblinear",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", clone(preprocessing)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=60,
                        max_depth=12,
                        min_samples_leaf=25,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def build_sentinel_model(X_train, c=1.0):
    """One experimental logistic model with missing flags and regularization."""
    transformer = SentinelMissingness()
    # Schema only: no medians, categories, or fit statistics are read here.
    schema = transformer.transform(X_train.iloc[:1])
    return Pipeline(
        [
            ("sentinels", transformer),
            ("preprocessor", build_preprocessor(schema)),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    C=c,
                    max_iter=300,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def build_candidates(X_train):
    """Prespecified candidates; compare only on validation month 6."""
    models = build_models(X_train)
    models["Sentinel LR (C=0.3)"] = build_sentinel_model(X_train, c=0.3)
    models["Sentinel LR (C=1.0)"] = build_sentinel_model(X_train, c=1.0)
    return models
