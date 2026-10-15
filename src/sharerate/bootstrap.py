"""The paired creator bootstrap: how much a model beats B0 by, with a 95% interval that respects creators.

One creator's videos are alike (the same face, style and audience), so they are not independent tests. A
bootstrap that resamples videos would treat them as independent and give intervals far too narrow. This one
resamples creators, keeping all of each drawn creator's videos (docs/why-the-rules.md §3).

It is paired: every model and the base are scored on the same replicates, and the gain is taken inside each
replicate. A replicate that happens to draw easy creators flatters every model alike, so that luck cancels in
the gain instead of widening it, which two separate intervals could not do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def paired_creator_bootstrap(metric, lift, predictions, creator, *, base="B0", lower_is_better=False,
                             n_boot=1000, seed=0, level=0.95) -> pd.DataFrame:
    """Score every prediction in `predictions` with `metric`, and its gain over `predictions[base]`.

    - `metric`: one of `sharerate.metrics`, `metric(lift, pred, creator, weights) -> array`.
    - `predictions`: {name: one prediction per video}, `base` among them.
    - `creator`: each video's creator (any labels).
    - `lower_is_better`: True for an error such as MAE, so that a positive gain always means better: the gain is
      base - model for an error and model - base otherwise.

    Each of the `n_boot` replicates draws C creators with replacement from the C in `creator`, and every model
    is scored on that same draw. The same `seed` gives the same draws, so every metric and model called with it
    shares them. Returns one row per model:

    - `value`, `value_low`, `value_high`: the metric on the rows as given, and its own interval;
    - `gain`, `gain_low`, `gain_high`: the gain over the base, and its paired interval, the middle `level` of
      the gains taken inside each replicate.
    """
    lift = np.asarray(lift, dtype=float)
    codes, creators = pd.factorize(np.asarray(creator))  # creators numbered 0 to C - 1
    n_creators = len(creators)

    rng = np.random.default_rng(seed)
    # Each replicate: draw C creators with replacement, and count how many times each one was drawn.
    draws = np.stack([np.bincount(rng.integers(0, n_creators, size=n_creators), minlength=n_creators)
                      for _ in range(n_boot)])
    everyone_once = np.ones((1, n_creators))  # the rows as given
    sign = -1.0 if lower_is_better else 1.0
    tails = [(1 - level) / 2, (1 + level) / 2]

    def score(pred):
        pred = np.asarray(pred, dtype=float)
        return metric(lift, pred, codes, everyone_once)[0], metric(lift, pred, codes, draws)

    base_value, base_replicates = score(predictions[base])
    rows = {}
    for name, pred in predictions.items():
        value, replicates = score(pred)
        gains = sign * (replicates - base_replicates)  # paired: the same creators on both sides
        rows[name] = {
            "value": value,
            "value_low": np.nanquantile(replicates, tails[0]),
            "value_high": np.nanquantile(replicates, tails[1]),
            "gain": sign * (value - base_value),
            "gain_low": np.nanquantile(gains, tails[0]),
            "gain_high": np.nanquantile(gains, tails[1]),
        }
    return pd.DataFrame.from_dict(rows, orient="index")
