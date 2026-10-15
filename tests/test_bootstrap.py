"""The paired creator bootstrap and the metrics it scores, on made-up data (no data/ needed)."""

import numpy as np
import pytest

from sharerate import data
from sharerate.bootstrap import paired_creator_bootstrap
from sharerate.metrics import LOG2, METRICS, eligible_creators


def made_up(n_creators=40, seed=1):
    """Creators with 5 to 80 videos each, lift = a creator effect + noise, roughly like the real target."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(5, 80, size=n_creators)
    creator = np.repeat(np.arange(n_creators), sizes)
    lift = rng.normal(0, 0.15, n_creators)[creator] + rng.normal(0, 0.45, len(creator))
    return lift, creator


@pytest.mark.parametrize("name", METRICS)
def test_b0_against_itself_gains_exactly_zero(name):
    lift, creator = made_up()
    metric, lower = METRICS[name]
    b0 = np.zeros_like(lift)
    table = paired_creator_bootstrap(metric, lift, {"B0": b0, "B0 again": b0.copy()}, creator,
                                     lower_is_better=lower, n_boot=200)
    gains = table[["gain", "gain_low", "gain_high"]].to_numpy()
    assert (gains == 0).all()  # exactly 0, not close to it


@pytest.mark.parametrize("name", METRICS)
def test_counting_a_creator_twice_equals_copying_their_videos(name):
    # What the bootstrap relies on: weight w for a creator gives the metric of w copies of their videos.
    lift, creator = made_up()
    metric, _ = METRICS[name]
    pred = 0.5 * lift + np.random.default_rng(2).normal(0, 0.3, len(lift))
    w = np.random.default_rng(3).integers(0, 4, size=creator.max() + 1)  # some creators 0 times, some 3
    weighted = metric(lift, pred, creator, w[None, :].astype(float))[0]

    copies, rows = [], []
    for c in np.flatnonzero(w):
        for _ in range(w[c]):
            rows.append(np.flatnonzero(creator == c))
            copies.append(np.full(len(rows[-1]), len(copies)))  # each copy is a creator of its own
    rows, copied_creator = np.concatenate(rows), np.concatenate(copies)
    copied = metric(lift[rows], pred[rows], copied_creator, np.ones((1, len(copies))))[0]
    assert weighted == pytest.approx(copied, rel=1e-12)


def test_what_zero_lift_scores():
    lift, creator = made_up()
    b0 = np.zeros_like(lift)
    once = np.ones((1, creator.max() + 1))
    score = {name: metric(lift, b0, creator, once)[0] for name, (metric, _) in METRICS.items()}
    hit = lift >= LOG2
    assert score["MAE"] == pytest.approx(np.abs(lift).mean())
    assert score["skill"] == 0
    assert score["spearman_within"] == 0
    assert score["roc_auc"] == 0.5
    assert score["pr_auc"] == pytest.approx(hit.mean())
    big = [c for c in np.unique(creator) if (creator == c).sum() >= 20]
    assert score["top_decile_share"] == pytest.approx(np.mean([hit[creator == c].mean() for c in big]))
    assert eligible_creators(creator) == len(big)


def test_a_better_model_gains_and_an_error_gain_is_oriented():
    lift, creator = made_up()
    preds = {"B0": np.zeros_like(lift), "good": 0.6 * lift}
    for name, (metric, lower) in METRICS.items():
        table = paired_creator_bootstrap(metric, lift, preds, creator, lower_is_better=lower, n_boot=300)
        assert table.loc["good", "gain_low"] > 0, name  # positive gain means better, errors included


def test_same_seed_same_interval():
    lift, creator = made_up()
    metric, lower = METRICS["MAE"]
    preds = {"B0": np.zeros_like(lift), "half": 0.5 * lift}
    one = paired_creator_bootstrap(metric, lift, preds, creator, lower_is_better=lower, n_boot=300, seed=7)
    two = paired_creator_bootstrap(metric, lift, preds, creator, lower_is_better=lower, n_boot=300, seed=7)
    other = paired_creator_bootstrap(metric, lift, preds, creator, lower_is_better=lower, n_boot=300, seed=8)
    assert one.equals(two)
    assert not one.equals(other)


def test_resampling_creators_is_wider_than_resampling_videos():
    # When a creator's videos share their error, creators are the independent units: treating each video as
    # its own creator (resampling videos) must give a narrower, overconfident interval.
    rng = np.random.default_rng(4)
    creator = np.repeat(np.arange(30), 50)
    lift = rng.normal(0, 0.5, len(creator))
    pred = rng.normal(0, 0.3, 30)[creator]  # each creator's prediction is off by its own amount
    metric, lower = METRICS["MAE"]
    preds = {"B0": np.zeros_like(lift), "model": pred}
    by_creator = paired_creator_bootstrap(metric, lift, preds, creator, lower_is_better=lower, n_boot=500)
    by_video = paired_creator_bootstrap(metric, lift, preds, np.arange(len(lift)), lower_is_better=lower,
                                        n_boot=500)
    width = lambda table: table.loc["model", "gain_high"] - table.loc["model", "gain_low"]
    assert width(by_creator) > 2 * width(by_video)


@pytest.mark.skipif(not (data.STAGE1.exists() and data.SNAPSHOT.exists()), reason="data/ is not in git")
@pytest.mark.parametrize("name", METRICS)
def test_b0_against_itself_on_validation(name):
    rows = data.load_videos()
    rows = rows[rows.time_split == "validation"]
    metric, lower = METRICS[name]
    b0 = np.zeros(len(rows))
    table = paired_creator_bootstrap(metric, rows.lift, {"B0": b0, "B0 again": b0.copy()}, rows.creator,
                                     lower_is_better=lower, n_boot=100)
    assert (table[["gain", "gain_low", "gain_high"]].to_numpy() == 0).all()


@pytest.mark.parametrize("rounding", [None, 1])  # 1 decimal: many tied predictions
def test_auc_and_average_precision_match_sklearn(rounding):
    from sklearn.metrics import average_precision_score, roc_auc_score

    from sharerate.metrics import pr_auc, roc_auc

    lift, creator = made_up()
    pred = 0.5 * lift + np.random.default_rng(5).normal(0, 0.3, len(lift))
    if rounding is not None:
        pred = pred.round(rounding)
    weights = np.random.default_rng(6).integers(0, 4, size=(5, creator.max() + 1)).astype(float)
    hit = lift >= LOG2
    for w, ours_auc, ours_ap in zip(weights, roc_auc(lift, pred, creator, weights),
                                    pr_auc(lift, pred, creator, weights)):
        assert ours_auc == pytest.approx(roc_auc_score(hit, pred, sample_weight=w[creator]), rel=1e-12)
        assert ours_ap == pytest.approx(average_precision_score(hit, pred, sample_weight=w[creator]), rel=1e-12)
