"""History features use only what was known when a video went up: strictly earlier, and settled for rates."""

import numpy as np
import pandas as pd
import pytest

from sharerate import history


def one_creator(days, lifts=None, baseline=0.0, creator=0, hour=12):
    """A frame like data.load_videos's for one creator, one video per entry in `days` (days from day 0)."""
    days = np.asarray(days, dtype=float)
    return pd.DataFrame({
        "creator": creator,
        "published_at": pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(days, unit="D")
        + pd.Timedelta(hours=hour),
        "baseline": baseline,
        "lift": np.arange(len(days), dtype=float) if lifts is None else np.asarray(lifts, dtype=float),
    })


def test_settled_means_at_least_14_days_earlier():
    frame = one_creator(range(40))  # one video a day
    n = history.settled_counts(frame)
    assert n[20] == 7  # days 0 to 6 are at least 14 days before day 20
    assert n[13] == 0 and n[14] == 1  # exactly 14 days counts, as in the baseline


def test_a_rate_younger_than_14_days_never_enters():
    frame = one_creator(range(40))
    before = {
        "median": history.recent_form(frame, 5),
        "ewm": history.ewm_form(frame, 3),
        "drift": history.shrunk_drift(frame, 10),
    }
    young = frame.copy()
    young.loc[25, "lift"] += 100  # posted 13 days before row 38: still collecting shares then
    old = frame.copy()
    old.loc[24, "lift"] -= 100  # posted 14 days before row 38: settled (down, so the median moves too)
    for name, fn in [("median", lambda f: history.recent_form(f, 5)), ("ewm", lambda f: history.ewm_form(f, 3)),
                     ("drift", lambda f: history.shrunk_drift(f, 10))]:
        assert fn(young)[38] == before[name][38], name
        assert fn(old)[38] != before[name][38], name


def test_the_video_itself_and_same_second_videos_never_enter():
    frame = one_creator([0, 0, 30, 30, 31])  # two pairs posted in the same second
    posts = frame[["creator", "published_at"]].rename(columns={"published_at": "posted_at"})
    timing = history.posting_features(frame, posts)
    assert timing.earlier_videos.tolist() == [0, 0, 2, 2, 4]
    assert np.isnan(timing.days_since_previous[0]) and np.isnan(timing.days_since_previous[1])
    assert timing.days_since_previous[2] == 30 and timing.days_since_previous[4] == 1


def test_recent_form_is_the_median_of_the_last_settled_rates_minus_the_baseline():
    frame = one_creator(range(40), baseline=0.5)
    b1 = history.recent_form(frame, 5)
    # Row 30: settled rows are 0..16; the last five are 12..16; rate = baseline + lift = 0.5 + index.
    assert b1[30] == pytest.approx(np.median(0.5 + np.arange(12, 17)) - 0.5)
    assert np.isnan(b1[22])  # 9 settled videos: fewer than 10, so no prediction
    assert not np.isnan(b1[23])


def test_ewm_runs_between_the_last_rate_and_the_mean():
    frame = one_creator(range(40))
    n = history.settled_counts(frame)
    assert history.ewm_form(frame, 1e-6)[30] == pytest.approx(n[30] - 1)  # tiny half-life: the newest settled
    assert history.ewm_form(frame, 1e9)[30] == pytest.approx(np.mean(np.arange(n[30])))  # huge: the plain mean


def test_shrinkage_runs_from_the_mean_to_zero():
    frame = one_creator(range(40))
    n = history.settled_counts(frame)
    assert history.shrunk_drift(frame, 0)[30] == pytest.approx(np.mean(np.arange(n[30])))
    assert (history.shrunk_drift(frame, np.inf) == 0).all()
    assert history.shrunk_drift(frame, 5)[30] == pytest.approx(n[30] / (n[30] + 5) * np.mean(np.arange(n[30])))
    assert history.shrunk_drift(frame, 0)[0] == 0  # no history: zero lift


def test_usual_hour_wraps_around_midnight():
    # Ten earlier posts at 23:00 and ten at 01:00 average to midnight, not to noon.
    days = np.arange(21)
    frame = one_creator(days, hour=2)
    post_hours = [23, 1] * 10 + [2]
    posts = pd.DataFrame({
        "creator": 0,
        "posted_at": [pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=int(d), hours=h)
                      for d, h in zip(days, post_hours)],
    })
    timing = history.posting_features(frame.iloc[[20]].reset_index(drop=True), posts)
    assert timing.usual_hour[0] == pytest.approx(0, abs=1e-9) or timing.usual_hour[0] == pytest.approx(24)
    assert timing.hour_from_usual[0] == pytest.approx(2)


def test_refuses_a_frame_out_of_order():
    frame = one_creator(range(5)).iloc[::-1]
    with pytest.raises(ValueError):
        history.settled_counts(frame)
