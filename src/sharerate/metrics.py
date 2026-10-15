"""Stage 2's metrics, written so the creator bootstrap can count each creator any number of times.

Every metric has the same signature, `metric(lift, pred, creator, weights) -> array`:

- `lift` and `pred`: the label and the prediction, one per video;
- `creator`: each video's creator, numbered 0 to C - 1;
- `weights`: shape (R, C), how many times each creator counts in each of R bootstrap replicates.

It returns one value per replicate. Counting a creator w times gives the same value as copying all of their
videos w times, which is what drawing that creator w times does, so a single row of ones gives the metric itself.
Breakout is `lift >= log10(2)` (notebook 01, cell 27), so it is derived here, not passed in.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

LOG2 = float(np.log10(2))
MIN_VIDEOS = 20  # within-creator metrics use creators with at least this many videos in the rows scored


def mae(lift, pred, creator, weights):
    """Mean absolute error of lift, in log10 units: 0.3 means off by a factor of 2 on average."""
    n_creators = weights.shape[1]
    abs_error = np.bincount(creator, np.abs(lift - pred), minlength=n_creators)  # per creator: sum of |error|
    videos = np.bincount(creator, minlength=n_creators)  # per creator: how many videos
    return weights @ abs_error / (weights @ videos)


def skill(lift, pred, creator, weights):
    """The skill score against zero lift: 1 - sum((lift - pred)^2) / sum(lift^2).

    The share of lift's squared size that the prediction removes. Predicting zero scores exactly 0, and a
    prediction worse than zero scores below it. This is not the R^2 of log share rate (stage 1's "52%").
    """
    n_creators = weights.shape[1]
    squared_error = np.bincount(creator, (lift - pred) ** 2, minlength=n_creators)
    squared_lift = np.bincount(creator, lift ** 2, minlength=n_creators)
    return 1 - (weights @ squared_error) / (weights @ squared_lift)


def spearman_within(lift, pred, creator, weights, min_videos=MIN_VIDEOS):
    """Spearman correlation of prediction and lift inside each creator, averaged over creators with equal weight.

    Only creators with at least `min_videos` videos in the rows scored. It asks the question a creator asks:
    of my own videos, does the model rank the better ones higher? A prediction that is constant within a
    creator ranks nothing and scores 0.
    """
    return _mean_over_creators(_per_creator(_spearman, lift, pred, creator, weights.shape[1], min_videos), weights)


def roc_auc(lift, pred, creator, weights):
    """ROC-AUC of the prediction as a breakout score, over all rows: the chance that a random breakout scores
    above a random non-breakout, ties counting half. A constant score gets 0.5."""
    hit = lift >= LOG2
    # Integer sample weights give the same curve as copying the rows, so this is the AUC of the replicate.
    return np.array([roc_auc_score(hit, pred, sample_weight=w[creator]) for w in weights])


def pr_auc(lift, pred, creator, weights):
    """Average precision of the prediction as a breakout score. A constant score gets the base rate."""
    hit = lift >= LOG2
    return np.array([average_precision_score(hit, pred, sample_weight=w[creator]) for w in weights])


def top_decile_share(lift, pred, creator, weights, min_videos=MIN_VIDEOS):
    """Of each creator's top 10% of videos by prediction, the share that broke out, averaged over creators.

    Only creators with at least `min_videos` videos, so the top 10% holds at least 2. The page's sentence:
    "of the videos it ranked highest for each creator, X% broke out". A constant prediction gets each
    creator's own breakout rate, the share a random pick would get.
    """
    return _mean_over_creators(
        _per_creator(_top_decile, lift, pred, creator, weights.shape[1], min_videos), weights)


# name: (metric, lower is better)
METRICS = {
    "MAE": (mae, True),
    "skill": (skill, False),
    "spearman_within": (spearman_within, False),
    "roc_auc": (roc_auc, False),
    "pr_auc": (pr_auc, False),
    "top_decile_share": (top_decile_share, False),
}


def eligible_creators(creator, min_videos=MIN_VIDEOS) -> int:
    """How many creators the within-creator metrics average over."""
    return int((np.bincount(creator) >= min_videos).sum())


def _spearman(lift, pred):
    if np.ptp(pred) == 0:  # a constant prediction ranks nothing; spearmanr would return NaN
        return 0.0
    return stats.spearmanr(lift, pred).statistic


def _top_decile(lift, pred):
    hit = lift >= LOG2
    k = int(np.ceil(len(pred) / 10))  # how many videos make the top 10%
    cutoff = np.sort(pred)[-k]  # the k-th highest prediction
    above = pred > cutoff
    tied = pred == cutoff
    # Videos above the cutoff are in. The tied ones share the places left, as a random tie-break would on average.
    taken = above + tied * (k - above.sum()) / tied.sum()
    return (taken * hit).sum() / k


def _per_creator(fn, lift, pred, creator, n_creators, min_videos):
    """fn(lift, pred) for each creator with at least `min_videos` rows; NaN for the others."""
    values = np.full(n_creators, np.nan)
    order = np.argsort(creator, kind="stable")
    bounds = np.searchsorted(creator[order], np.arange(n_creators + 1))  # each creator's rows in `order`
    for c in range(n_creators):
        rows = order[bounds[c]:bounds[c + 1]]
        if len(rows) >= min_videos:
            values[c] = fn(lift[rows], pred[rows])
    return values


def _mean_over_creators(values, weights):
    """Each replicate's mean of the per-creator values, a creator drawn w times counting w times."""
    eligible = ~np.isnan(values)
    w = weights[:, eligible]
    with np.errstate(invalid="ignore", divide="ignore"):  # a replicate with no eligible creator gives NaN
        return w @ values[eligible] / w.sum(axis=1)
