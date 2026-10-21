"""Stage 7's live test: the frozen bar, the readings and the guard, on made-up reels (no data/ or platform needed)."""

import copy
import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from sharerate import live

START = pd.Timestamp("2026-10-20T22:00:00Z")


def fingerprint(bar: dict) -> str:
    """The bar's content, whatever its spacing or line endings."""
    return hashlib.sha256(json.dumps(bar, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def test_the_bar_is_the_one_written_on_21_october():
    # Written before the first reel of the test was posted. Any edit to it fails here: a new bar is a new test.
    bar = live.load_bar()
    assert bar["set"] == {"published from": "2026-10-20T22:00:00+00:00", "reels": 30,
                          "posting deadline": "2027-01-10T23:00:00+00:00"}
    assert bar["bar"] == {"rho at most": -0.3, "interval high below": 0, "minimum reels": 20}
    assert bar["interval"] == {"method": "percentile bootstrap over reels", "draws": 2000, "seed": 2026, "level": 0.95}
    assert bar["readings"]["day 1"]["window hours"] == [18, 36] and bar["readings"]["day 14"]["window days"] == [10, 21]
    assert bar["read"] == {"days after the last reel": 21}
    assert fingerprint(bar) == "ef4f2a947cd976bfff3246452fb58f50d05c7273d2eefcea4e469eacbfe0df8e"


def quick() -> dict:
    """The real bar with 200 bootstrap draws in place of 2,000: the same code, a tenth of the time."""
    bar = copy.deepcopy(live.load_bar())
    bar["interval"]["draws"] = 200
    return bar


def made_up(n_reels=30, skip=None, views_14=None, seed=0, start=START):
    """One reel a day from the bar's start, each with snapshots at 1 hour, 24 hours and 14 days.

    By default the views at 14 days fall with the skip rate, as the bar expects; pass `views_14` to break that.
    """
    rng = np.random.default_rng(seed)
    skip = rng.uniform(25, 60, n_reels) if skip is None else np.asarray(skip, float)
    if views_14 is None:
        views_14 = 10 ** (7 - skip / 20 + rng.normal(0, 0.2, n_reels))  # about 560,000 views at 25% skip, 10,000 at 60%
    published = [start + pd.Timedelta(days=i, hours=15) for i in range(n_reels)]
    reels = pd.DataFrame({"media_id": [f"r{i}" for i in range(n_reels)], "published": published, "taken_down": False})
    snaps = []
    for i, when in enumerate(published):
        views_1 = views_14[i] / 3
        for hours, views in ((1, views_1 / 10), (24, views_1), (24 * 14, views_14[i])):
            snaps.append({"media_id": f"r{i}", "taken_at": when + pd.Timedelta(hours=hours), "skip_rate": skip[i],
                          "views": views, "shares": views / 100})
    return reels, pd.DataFrame(snaps)


def test_the_set_is_the_first_reels_from_the_start_in_the_order_posted():
    bar = live.load_bar()
    reels, _ = made_up(n_reels=35, start=START - pd.Timedelta(days=3))  # three reels from before the start
    chosen = live.test_set(reels.sample(frac=1, random_state=1), bar)  # shuffled: the order comes from the dates
    assert len(chosen) == 30
    assert chosen["published"].min() >= START
    assert list(chosen["media_id"]) == [f"r{i}" for i in range(3, 33)]


def test_the_set_closes_at_the_posting_deadline():
    bar = copy.deepcopy(live.load_bar())
    bar["set"]["posting deadline"] = str(START + pd.Timedelta(days=9))
    reels, _ = made_up(n_reels=30)
    assert len(live.test_set(reels, bar)) == 9  # reels 0 to 8 went up before the deadline


def test_a_reading_is_the_nearest_usable_snapshot_inside_its_window():
    published = START
    snaps = pd.DataFrame({
        "taken_at": [published + pd.Timedelta(hours=h) for h in (17, 20, 23, 27, 40)],
        "skip_rate": [30.0, 31.0, np.nan, 33.0, 34.0],  # the 23-hour one lacks a skip rate
        "views": [100.0] * 5,
        "shares": [1.0] * 5,
    })
    picked = live.nearest(snaps, published, 24, (18, 36), live.DAY_1)
    assert picked["skip_rate"] == 33.0  # 27 h is 3 h off; 23 h is closer but unusable, 17 h and 40 h are outside


def test_a_tie_takes_the_earlier_snapshot():
    snaps = pd.DataFrame({"taken_at": [START + pd.Timedelta(hours=h) for h in (22, 26)], "skip_rate": [1.0, 2.0],
                          "views": [10.0, 10.0], "shares": [0.0, 0.0]})
    assert live.nearest(snaps, START, 24, (18, 36), live.DAY_1)["skip_rate"] == 1.0


def test_an_empty_window_gives_no_reading():
    snaps = pd.DataFrame({"taken_at": [START + pd.Timedelta(hours=60)], "skip_rate": [30.0], "views": [10.0],
                          "shares": [0.0]})
    assert live.nearest(snaps, START, 24, (18, 36), live.DAY_1) is None


def test_read_refuses_before_the_reading_date():
    bar = quick()
    reels, snaps = made_up()
    last = reels["published"].iloc[-1]
    with pytest.raises(live.TooEarly):
        live.read(bar, reels, snaps, now=last + pd.Timedelta(days=20))
    assert live.read(bar, reels, snaps, now=last + pd.Timedelta(days=21))["verdict"] == "passed"


def test_a_short_set_waits_for_the_posting_deadline():
    bar = quick()
    reels, snaps = made_up(n_reels=25)
    with pytest.raises(live.TooEarly):
        live.read(bar, reels, snaps, now=reels["published"].iloc[-1] + pd.Timedelta(days=60))
    after = pd.Timestamp(bar["set"]["posting deadline"]) + pd.Timedelta(days=21)
    assert live.read(bar, reels, snaps, now=after)["reels in the set"] == 25


def test_too_few_readings_make_the_test_void():
    bar = quick()
    reels, snaps = made_up()
    age = snaps["taken_at"] - snaps["media_id"].map(reels.set_index("media_id")["published"])
    gone = snaps["media_id"].isin([f"r{i}" for i in range(11, 30)]) & (age > pd.Timedelta(days=5))
    result = live.read(bar, reels, snaps[~gone], now="2027-03-01")  # 19 reels lost their day-14 reading
    assert result["with the day-1 reading"] == 30
    assert result["with both"] == 11
    assert result["verdict"] == "void"
    assert not any("spearman" in key or "pearson" in key for key in result)


def test_a_strong_link_passes_and_no_link_fails():
    bar = quick()
    reels, snaps = made_up(seed=3)
    passed = live.read(bar, reels, snaps, now="2027-03-01")
    assert passed["verdict"] == "passed"
    assert passed["spearman, skip rate at day 1 against views at day 14"]["value"] < -0.8

    rng = np.random.default_rng(4)
    reels, snaps = made_up(views_14=10 ** rng.uniform(3, 5, 30), seed=3)  # views unrelated to the skip rate
    assert live.read(bar, reels, snaps, now="2027-03-01")["verdict"] == "failed"


def test_the_result_names_no_reel():
    bar = quick()
    reels, snaps = made_up(seed=5)
    result = live.read(bar, reels, snaps, now="2027-03-01")
    text = json.dumps(result)
    assert not any(f'"r{i}"' in text for i in range(30))  # no id
    pairs = result["readings, skip rate at day 1 and views at day 14"]
    assert pairs == sorted(pairs)  # sorted by skip rate, so the order says nothing about when a reel went up


def test_status_counts_and_never_correlates():
    bar = live.load_bar()
    reels, snaps = made_up(n_reels=12)
    report = live.status(bar, reels, snaps)
    assert report["reels in the set"] == 12 and report["with both"] == 12
    assert all(isinstance(v, (int, str)) for v in report.values())


def test_the_bootstrap_repeats_with_its_seed_and_brackets_the_value():
    rng = np.random.default_rng(6)
    x = rng.normal(size=40)
    y = -x + rng.normal(size=40)
    first = live.bootstrap(live.spearman, [x, y], draws=300, seed=2026, level=0.95)
    assert first == live.bootstrap(live.spearman, [x, y], draws=300, seed=2026, level=0.95)
    assert first["low"] <= first["value"] <= first["high"] < 0


def test_spearman_is_scipys_with_ties():
    from scipy import stats

    rng = np.random.default_rng(8)
    x = rng.integers(20, 30, 50).astype(float)  # skip rates repeat, so ranks tie
    y = 10 ** rng.uniform(3, 5, 50)
    assert live.spearman(x, y) == pytest.approx(stats.spearmanr(x, y).statistic, abs=1e-12)
    assert np.isnan(live.spearman(np.ones(5), np.arange(5.0)))  # a constant side has no correlation


def test_the_partial_spearman_removes_what_both_owe_to_the_third():
    rng = np.random.default_rng(7)
    z = rng.normal(size=500)
    x, y = z + 0.3 * rng.normal(size=500), z + 0.3 * rng.normal(size=500)  # x and y share only z
    assert live.spearman(x, y) > 0.8
    assert abs(live.partial_spearman(x, y, z)) < 0.1


def test_the_platform_is_opened_read_only(tmp_path):
    db = tmp_path / "content_intelligence.db"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE instagram_media (media_id TEXT, timestamp TEXT, media_product_type TEXT,"
                    " ig_status TEXT, is_archived INTEGER)")
        con.execute("CREATE TABLE instagram_media_snapshots (media_id TEXT, snapshot_at TEXT, views INTEGER,"
                    " plays INTEGER, shares INTEGER, skip_rate REAL)")
        con.execute("INSERT INTO instagram_media VALUES ('a', '2026-10-21T13:00:00+0000', 'REELS', NULL, 0)")
        con.execute("INSERT INTO instagram_media VALUES ('b', '2026-10-21T14:00:00+0000', 'FEED', NULL, 0)")
        con.execute("INSERT INTO instagram_media_snapshots VALUES ('a', '2026-10-22T13:00:00+00:00', NULL, 500, 5, 41.5)")
    con.close()
    before = db.read_bytes()
    reels, snaps = live.load_reels(db)
    assert db.read_bytes() == before
    assert list(reels["media_id"]) == ["a"]  # reels only
    assert reels["published"].iloc[0] == pd.Timestamp("2026-10-21T13:00:00Z")
    assert snaps["views"].iloc[0] == 500  # plays stand in where views are missing
