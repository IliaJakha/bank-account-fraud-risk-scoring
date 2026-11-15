import pandas as pd

from baf_fraud.modeling import build_models


def test_models_fit_categorical_and_numeric():
    X = pd.DataFrame({"amount": [1., 2., 3., 4., 5., 6., 7., 8.],
                      "source": ["web", "phone"] * 4})
    y = [0, 1, 0, 1, 0, 1, 0, 1]
    models = build_models(X)
    for model in models.values():
        model.fit(X, y)
        p = model.predict_proba(pd.DataFrame({"amount": [9.], "source": ["new"]}))
        assert p.shape == (1, 2)
        assert 0 <= p[0, 1] <= 1
