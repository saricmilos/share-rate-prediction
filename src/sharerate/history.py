"""History features: what was known about a creator at the moment each video went up.

The same as-of rules as the target (notebook 01, cell 27):

- only the creator's videos posted **strictly earlier**, never the video itself or one posted in the same second;
- for anything built from their share rates, only videos posted **at least 14 days earlier**, whose rate had
  settled by then (it moves a median 1.6% afterwards).

The rates come from the target rows themselves: a video's log share rate is its baseline + lift, so nothing here
reads a count. The cost: a creator's earliest library videos, the ones too early to get a target, are missing
from the rate history, so a creator's first few targets have less history here than their baseline had.
`settled_counts` says how much each row has.

Every function takes the frame `data.load_videos` returns, sorted by creator and then publish time, with every
split the history needs (train and validation together), and returns one value per row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SETTLED = np.timedelta64(14, "D")  # a rate is used only once the video is at least 14 days old
MIN_HISTORY = 10  # as for the baseline: fewer settled earlier videos than this, and B1 has no prediction
MIN_POSTS = 10  # earlier posts needed before a creator has a "usual" posting hour
DAY = np.timedelta64(1, "D")


def settled_counts(frame) -> np.ndarray:
    """For each row, how many of its creator's rows in `frame` were posted at least 14 days before it.

    The frame is in time order, so those are the first n rows of the creator's block.
    """
    posted = _utc(frame.published_at)
    out = np.empty(len(frame), dtype=int)
    for rows in _creator_blocks(frame):
        # side="right": a video posted exactly 14 days earlier counts as settled, as in the baseline.
        out[rows] = np.searchsorted(posted[rows], posted[rows] - SETTLED, side="right")
    return out


def recent_form(frame, window, min_history=MIN_HISTORY) -> np.ndarray:
    """B1: the median log share rate of the creator's last `window` settled videos, minus the video's baseline.

    `window=None` takes every settled video (the lifetime median). NaN where fewer than `min_history` settled
    videos exist; the notebook predicts zero lift there and counts the rows.
    """
    rate = _log_rate(frame)
    n_settled = settled_counts(frame)
    out = np.full(len(frame), np.nan)
    for rows in _creator_blocks(frame):
        creator_rates = rate[rows]
        for row, end in zip(rows, n_settled[rows]):
            if end >= min_history:
                start = 0 if window is None else max(0, end - window)
                out[row] = np.median(creator_rates[start:end])
    return out - frame.baseline.to_numpy()


def ewm_form(frame, half_life, min_history=MIN_HISTORY) -> np.ndarray:
    """B1: an exponentially weighted mean of the creator's settled log share rates, minus the baseline.

    The newest settled video weighs 1, the one before it `decay`, then `decay`^2, and so on, where `decay` is
    chosen so that a video `half_life` videos further back weighs half as much. NaN as in `recent_form`.
    """
    decay = 0.5 ** (1 / half_life)
    rate = _log_rate(frame)
    n_settled = settled_counts(frame)
    out = np.full(len(frame), np.nan)
    for rows in _creator_blocks(frame):
        creator_rates = rate[rows]
        # weighted[e] and total[e]: the decayed sum of the first e rates, and of their weights.
        weighted = np.zeros(len(rows) + 1)
        total = np.zeros(len(rows) + 1)
        for j, value in enumerate(creator_rates):
            weighted[j + 1] = decay * weighted[j] + value  # older rates fade by `decay` with each new one
            total[j + 1] = decay * total[j] + 1
        end = n_settled[rows]
        enough = end >= min_history
        out[rows[enough]] = weighted[end[enough]] / total[end[enough]]
    return out - frame.baseline.to_numpy()


def shrunk_drift(frame, k) -> np.ndarray:
    """B2: the mean lift of the creator's settled earlier videos, times n / (n + k).

    n is how many settled earlier videos there are. With few of them the mean is mostly luck, so it is pulled
    toward 0 (zero lift); with many, it is trusted. k is how many videos it takes to trust the mean halfway:
    k = 0 trusts it fully, k = infinity is B0. A creator with no settled history gets 0.
    """
    lift = frame.lift.to_numpy()
    n_settled = settled_counts(frame)
    out = np.zeros(len(frame))
    for rows in _creator_blocks(frame):
        n = n_settled[rows]
        running = np.r_[0.0, np.cumsum(lift[rows])]  # running[e]: the sum of the creator's first e lifts
        mean = np.divide(running[n], n, out=np.zeros(len(n)), where=n > 0)
        trust = np.divide(n, n + k, out=np.zeros(len(n)), where=n + k > 0)  # n / (n + k); k = inf gives 0
        out[rows] = trust * mean
    return out


def posting_features(frame, posts) -> pd.DataFrame:
    """B3's timing features for each row, from `posts` (`data.load_posting_times`): every library video of the
    creator, counted or not, posted strictly earlier.

    - `hour_utc`, `weekday_utc`: when the video went up. `published_at` is UTC and the creators live in
      different time zones, so the same UTC hour is morning for one and night for another.
    - `earlier_videos`: how many of the creator's videos the library holds from before it (the library, not
      the channel).
    - `days_since_previous`: days since the creator's previous video in the library. NaN for the first.
    - `usual_hour`: the creator's usual posting hour so far, the circular mean of their earlier posting hours
      (so 23:00 and 01:00 average to midnight, not noon). NaN until they have `MIN_POSTS` earlier videos.
    - `hour_from_usual`: hours from the usual hour, -12 to +12. The creator's own clock, whatever their zone.
    """
    when = frame.published_at
    hour = (when.dt.hour + when.dt.minute / 60).to_numpy()
    posted = _utc(when)
    post_time = _utc(posts.posted_at)
    post_angle = 2 * np.pi * (posts.posted_at.dt.hour + posts.posted_at.dt.minute / 60).to_numpy() / 24
    post_bounds = _bounds(posts.creator.to_numpy())

    earlier = np.zeros(len(frame), dtype=int)
    since = np.full(len(frame), np.nan)
    usual = np.full(len(frame), np.nan)
    for rows in _creator_blocks(frame):
        start, end = post_bounds[frame.creator.iat[rows[0]]]
        times, angles = post_time[start:end], post_angle[start:end]
        # side="left": library videos posted strictly before, so not the video itself or a same-second one.
        n = np.searchsorted(times, posted[rows], side="left")
        earlier[rows] = n
        has_previous = n > 0
        since[rows[has_previous]] = (posted[rows][has_previous] - times[n[has_previous] - 1]) / DAY
        # Running sums of each earlier post's hour as a point on a 24-hour circle; their direction is the mean.
        sin_sum = np.r_[0.0, np.cumsum(np.sin(angles))]
        cos_sum = np.r_[0.0, np.cumsum(np.cos(angles))]
        known = n >= MIN_POSTS
        mean_angle = np.arctan2(sin_sum[n[known]], cos_sum[n[known]])
        usual[rows[known]] = (mean_angle * 24 / (2 * np.pi)) % 24
    return pd.DataFrame({
        "hour_utc": when.dt.hour.to_numpy(),
        "weekday_utc": when.dt.weekday.to_numpy(),
        "earlier_videos": earlier,
        "days_since_previous": since,
        "usual_hour": usual,
        "hour_from_usual": (hour - usual + 12) % 24 - 12,
    }, index=frame.index)


def _log_rate(frame) -> np.ndarray:
    """Each row's log10 share rate, as stage 1 built it: the target is lift = log rate - baseline."""
    return (frame.baseline + frame.lift).to_numpy()


def _utc(times: pd.Series) -> np.ndarray:
    """UTC times as plain numpy datetimes, which searchsorted and subtraction need."""
    return times.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()


def _bounds(creators: np.ndarray) -> dict:
    """{creator: (start, end)} of each creator's contiguous rows in an array sorted by creator."""
    starts = np.flatnonzero(np.r_[True, creators[1:] != creators[:-1]])
    ends = np.r_[starts[1:], len(creators)]
    return {creators[s]: (s, e) for s, e in zip(starts, ends)}


def _creator_blocks(frame) -> list[np.ndarray]:
    """Each creator's rows as positions in `frame`. Refuses a frame not sorted by creator, then time."""
    creators = frame.creator.to_numpy()
    posted = _utc(frame.published_at)
    same = creators[1:] == creators[:-1]
    if (np.diff(creators) < 0).any() or (posted[1:][same] < posted[:-1][same]).any():
        raise ValueError("history features need the frame sorted by creator, then published_at")
    return [np.arange(start, end) for start, end in _bounds(creators).values()]
