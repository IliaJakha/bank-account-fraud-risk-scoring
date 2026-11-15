"""Baseline classifiers matched to the original analysis notebook."""
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X_train):
    """Learn all preprocessing parameters using training features only."""
    numerical = X_train.select_dtypes(include="number").columns.tolist()
    categorical = X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    if len(numerical) + len(categorical) != X_train.shape[1]:
        raise ValueError("Unexpected feature dtypes; encode them before modeling")
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    category_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numerical),
        ("categorical", category_pipeline, categorical),
    ], sparse_threshold=1.0)


def build_models(X_train):
    """Return independently fitted pipelines; do not share fitted transformers."""
    preprocessing = build_preprocessor(X_train)
    return {
        "Logistic Regression": Pipeline([
            ("preprocessor", clone(preprocessing)),
            ("classifier", LogisticRegression(
                class_weight="balanced", max_iter=300, solver="liblinear", random_state=42
            )),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", clone(preprocessing)),
            ("classifier", RandomForestClassifier(
                n_estimators=60, max_depth=12, min_samples_leaf=25,
                class_weight="balanced_subsample", random_state=42, n_jobs=-1
            )),
        ]),
    }
