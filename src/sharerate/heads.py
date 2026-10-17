"""Stage 4's ridge, solved from its normal equations: one X'X for every penalty and every fold.

scikit-learn's `Ridge(alpha)` with an intercept minimises ||y - b - Xw||^2 + alpha * ||w||^2 over the rows it is
fitted on. Centring X and y on those rows removes b, and then

    w = (Xc'Xc + alpha * I)^-1 Xc'yc,    b = mean(y) - mean(X) . w

with Xc'Xc = X'X - n * m m' (m the column means of the rows fitted). X'X doesn't depend on y or on alpha, so it is
computed once per fold of rows, and the rows of any set of folds are a sum. Each penalty, each fold left out and
each new target (B3's residual changes with the fold) is then one solve of a p x p system, never another pass over
a 74,552 x 7,175 matrix: the 4B text head's columns, which sklearn would copy (2 GB each time) for every penalty.

The answer is sklearn's to rounding: tests/test_heads.py checks it against `Ridge` on held-out folds.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg


class RidgeGram:
    """Ridge with an intercept on the rows `rows` of `X`, for any penalty, with any one fold left out.

    - `X`: every row (float32 is fine; the sums are taken in float64), so `predict` can score any row.
    - `rows`: the rows that may be fitted on (a boolean mask or row numbers).
    - `folds`: one label per fitted row (in the order of `rows`), or None for a single fold.
    """

    def __init__(self, X, rows, folds=None, chunk=4_096):
        self.X = X
        self.index = np.flatnonzero(rows) if np.asarray(rows).dtype == bool else np.asarray(rows)
        self.folds = np.zeros(len(self.index), dtype=int) if folds is None else np.asarray(folds)
        self.labels = list(np.unique(self.folds))
        self.chunk = chunk
        p = X.shape[1]
        self.gram, self.sums, self.counts = {}, {}, {}
        for label in self.labels:
            gram, sums = np.zeros((p, p)), np.zeros(p)
            for block in self._blocks(self.index[self.folds == label]):
                part = np.asarray(X[block], dtype=np.float64)
                gram += part.T @ part  # X'X of this fold, a block of rows at a time
                sums += part.sum(axis=0)
            self.gram[label], self.sums[label], self.counts[label] = gram, sums, int((self.folds == label).sum())

    def _blocks(self, index):
        for start in range(0, len(index), self.chunk):
            yield index[start:start + self.chunk]

    def fit(self, y, alpha, leave_out=None):
        """(w, b) on every fitted row outside fold `leave_out`. `y` has one value per row of X."""
        keep = [label for label in self.labels if label != leave_out]
        index = self.index[np.isin(self.folds, keep)]
        n = sum(self.counts[label] for label in keep)
        mean_x = sum(self.sums[label] for label in keep) / n
        target = np.asarray(y, dtype=np.float64)[index]
        mean_y = target.mean()
        # Xc'yc = X'(y - mean y): the centring of X drops out, because y - mean y sums to 0 over these rows
        xty = np.zeros(self.X.shape[1])
        for start in range(0, len(index), self.chunk):
            block = index[start:start + self.chunk]
            xty += np.asarray(self.X[block], dtype=np.float64).T @ (target[start:start + self.chunk] - mean_y)
        a = sum(self.gram[label] for label in keep) - n * np.outer(mean_x, mean_x)  # Xc'Xc
        a[np.diag_indices_from(a)] += alpha
        w = linalg.solve(a, xty, assume_a="pos")  # symmetric positive definite: a Cholesky solve
        return w, mean_y - mean_x @ w

    def predict(self, w, b, rows=None):
        """X w + b on `rows` (every row by default), in float64."""
        index = np.arange(self.X.shape[0]) if rows is None else (
            np.flatnonzero(rows) if np.asarray(rows).dtype == bool else np.asarray(rows))
        out = np.empty(len(index))
        for start in range(0, len(index), self.chunk):
            block = index[start:start + self.chunk]
            out[start:start + self.chunk] = np.asarray(self.X[block], dtype=np.float64) @ w + b
        return out
