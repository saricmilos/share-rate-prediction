"""RidgeGram gives scikit-learn's Ridge, for every penalty and with any fold left out."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from sharerate.heads import RidgeGram


def _data(n=600, p=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32) + 3.0  # off-centre columns, so the intercept matters
    y = X[:, :5].astype(float) @ rng.normal(size=5) + rng.normal(size=n) + 7.0
    return X, y


@pytest.mark.parametrize("alpha", [0.1, 10.0, 1_000.0])
def test_matches_sklearn_on_all_rows(alpha):
    X, y = _data()
    rows = np.ones(len(X), dtype=bool)
    rows[::7] = False  # some rows are never fitted, only predicted
    w, b = RidgeGram(X, rows).fit(y, alpha)
    reference = Ridge(alpha=alpha).fit(X[rows].astype(float), y[rows])
    np.testing.assert_allclose(w, reference.coef_, rtol=1e-8, atol=1e-10)
    assert b == pytest.approx(reference.intercept_, rel=1e-10)


def test_leave_one_fold_out_matches_sklearn_on_the_other_folds():
    X, y = _data(seed=1)
    rows = np.arange(len(X))[::2]  # row numbers, not a mask
    folds = np.arange(len(rows)) % 5
    gram = RidgeGram(X, rows, folds, chunk=37)  # a chunk that doesn't divide the rows
    for k in range(5):
        fitted = rows[folds != k]
        w, b = gram.fit(y, 30.0, leave_out=k)
        reference = Ridge(alpha=30.0).fit(X[fitted].astype(float), y[fitted])
        np.testing.assert_allclose(gram.predict(w, b), reference.predict(X.astype(float)), rtol=1e-9, atol=1e-9)


def test_a_new_target_needs_no_new_pass_over_x():
    # The fold's X'X is reused: fitting another y gives that y's ridge
    X, y = _data(seed=2)
    rows = np.ones(len(X), dtype=bool)
    gram = RidgeGram(X, rows)
    other = y[::-1].copy()
    w, b = gram.fit(other, 5.0)
    reference = Ridge(alpha=5.0).fit(X.astype(float), other)
    np.testing.assert_allclose(gram.predict(w, b), reference.predict(X.astype(float)), rtol=1e-9, atol=1e-9)
