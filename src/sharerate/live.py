"""Stage 7, the live test: on my own Instagram reels posted from 21 October 2026, does the share of viewers who swipe
away in the first three seconds, read a day after posting, go with how many views a reel has two weeks later?

Public counts can't show the first seconds, so stages 1 to 5 couldn't ask this. Instagram reports the skip rate only to
the account's owner, which is why the test runs on my own account. Every choice in it, from which reels count to the bar,
was frozen in `reports/live_bar.json` before the first of its reels was posted, and docs/live-test.md says why each one
is what it is. This module only reads the bar.

The readings come from the platform's Instagram tables, opened read-only (mode=ro): nothing is written in the
platform's folder. `status` reports counts only, and `read` refuses to compute anything before the bar's reading date,
so no correlation can be seen, and acted on, while the reels are still going up.

    python -m sharerate.live status    # how many reels are in so far, and how many have each reading
    python -m sharerate.live read      # the test, once, from its reading date on: writes reports/live_result.json
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from sharerate.data import ROOT

BAR = ROOT / "reports" / "live_bar.json"
RESULT = ROOT / "reports" / "live_result.json"
FLYWHEEL_DIR = Path(os.environ.get("FLYWHEEL_DIR", "C:/dev/career/prsnl-cnt-flywheel"))
INSTAGRAM_DB = FLYWHEEL_DIR / "databases" / "content_intelligence.db"
TAKEN_DOWN = {"deleted", "removed", "archived"}  # the platform's statuses for a reel that is no longer up
DAY_1 = ["skip_rate", "views", "shares"]  # what the day-1 snapshot must hold
DAY_14 = ["views", "shares"]  # and the day-14 one


class TooEarly(Exception):
    """The test was asked for before its reading date."""


def load_bar(path: Path = BAR) -> dict:
    """The frozen bar, as written before the test's first reel."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def utc(text) -> pd.Timestamp:
    """A platform timestamp as UTC. Instagram writes "+0000" without the colon; some old rows carry no offset at all."""
    stamp = pd.Timestamp(str(text).replace("+0000", "+00:00"))
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def load_reels(db: Path = INSTAGRAM_DB) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every reel the platform's collector recorded on my account, and every snapshot of them.

    Read-only: the database is opened with mode=ro, so a bug here can't write to the platform.
    """
    with closing(sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True)) as con:
        reels = pd.read_sql_query(
            "SELECT media_id, timestamp, ig_status, is_archived FROM instagram_media WHERE media_product_type = 'REELS'",
            con)
        snaps = pd.read_sql_query(
            "SELECT media_id, snapshot_at, COALESCE(views, plays) AS views, shares, skip_rate"
            " FROM instagram_media_snapshots", con)
    reels["published"] = pd.to_datetime(reels["timestamp"].map(utc), utc=True)
    reels["taken_down"] = reels["ig_status"].isin(TAKEN_DOWN) | (reels["is_archived"].fillna(0) == 1)
    snaps["taken_at"] = pd.to_datetime(snaps["snapshot_at"].map(utc), utc=True)
    return reels[["media_id", "published", "taken_down"]], snaps[["media_id", "taken_at", *DAY_1]]


def test_set(reels: pd.DataFrame, bar: dict) -> pd.DataFrame:
    """The bar's reels: the first `reels` published from its start, up to its posting deadline, in the order posted.

    Nothing about a reel decides whether it is in, only when it went up, and a reel that later loses a reading is
    excluded, never replaced by the next one.
    """
    rules = bar["set"]
    posted = reels[reels["published"].between(utc(rules["published from"]), utc(rules["posting deadline"]))]
    return posted.sort_values("published", kind="stable").head(rules["reels"]).reset_index(drop=True)


def reading_date(chosen: pd.DataFrame, bar: dict) -> pd.Timestamp:
    """When the test may be read: a fixed wait after the last reel of a full set, else after the posting deadline.

    With fewer reels than the bar asks for, the set can only close at the deadline, so that is the earliest reading.
    """
    wait = pd.Timedelta(days=bar["read"]["days after the last reel"])
    if len(chosen) == bar["set"]["reels"]:
        return chosen["published"].iloc[-1] + wait
    return utc(bar["set"]["posting deadline"]) + wait


def nearest(snaps: pd.DataFrame, published: pd.Timestamp, hours: float, window: tuple[float, float],
            needed: list[str]) -> pd.Series | None:
    """The snapshot taken nearest to `hours` after posting, inside `window` (hours), among those holding every
    `needed` field; the earlier one on a tie. None if the window holds none, because the collector wasn't running."""
    age = (snaps["taken_at"] - published).dt.total_seconds() / 3600
    usable = age.between(*window) & snaps[needed].notna().all(axis=1)
    if not usable.any():
        return None
    candidates = snaps[usable].assign(age=age[usable], gap=(age[usable] - hours).abs())
    return candidates.sort_values(["gap", "age"], kind="stable").iloc[0]


def share_rate(snapshot: pd.Series | None) -> float:
    """Shares per 1,000 views at a snapshot; NaN without one, or with no views yet."""
    if snapshot is None or not snapshot["views"]:
        return np.nan
    return 1000 * snapshot["shares"] / snapshot["views"]


def readings(chosen: pd.DataFrame, snaps: pd.DataFrame, bar: dict) -> pd.DataFrame:
    """Each chosen reel's two readings, NaN where a reading's window held no usable snapshot."""
    day_1, day_14 = bar["readings"]["day 1"], bar["readings"]["day 14"]
    rows = []
    for reel in chosen.itertuples(index=False):
        own = snaps[snaps["media_id"] == reel.media_id]
        first = nearest(own, reel.published, day_1["hours"], tuple(day_1["window hours"]), DAY_1)
        late = nearest(own, reel.published, 24 * day_14["days"], tuple(24 * d for d in day_14["window days"]), DAY_14)
        rows.append({
            "media_id": reel.media_id,
            "taken_down": bool(reel.taken_down),
            "skip_rate_1": np.nan if first is None else float(first["skip_rate"]),
            "views_1": np.nan if first is None else float(first["views"]),
            "share_rate_1": share_rate(first),
            "views_14": np.nan if late is None else float(late["views"]),
            "share_rate_14": share_rate(late),
        })
    return pd.DataFrame(rows, columns=["media_id", "taken_down", "skip_rate_1", "views_1", "share_rate_1",
                                       "views_14", "share_rate_14"])


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson's r; NaN when one side is constant, as a resample of few reels can leave it."""
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman's ρ: Pearson's r on the ranks, ties sharing their average rank, as scipy's spearmanr computes it
    (tests/test_live.py checks the two agree). Written out because the bootstrap calls it thousands of times, and
    scipy's version also works out a p-value nobody reads here."""
    return pearson(stats.rankdata(x), stats.rankdata(y))


def pearson_log10(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson's r between x and log10 y: the form the reels before the bar were looked at in."""
    if (y <= 0).any():
        return np.nan  # a reel with no views has no log
    return pearson(x, np.log10(y))


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman's ρ between x and y once z's ranks are regressed out of both: what x says about y that z doesn't."""
    ranks = [stats.rankdata(v) for v in (x, y, z)]
    design = np.column_stack([np.ones(len(ranks[2])), ranks[2]])

    def residual(r):  # what is left of r after a straight line in z's ranks
        return r - design @ np.linalg.lstsq(design, r, rcond=None)[0]

    return pearson(residual(ranks[0]), residual(ranks[1]))


def bootstrap(statistic, columns: list[np.ndarray], *, draws: int, seed: int, level: float) -> dict:
    """`statistic` on the reels as given, and the middle `level` of it over `draws` resamples of the reels.

    Each draw takes as many reels as there are, with replacement. The same seed gives the same draws, so every
    statistic computed with it is scored on the same resamples.
    """
    n = len(columns[0])
    rng = np.random.default_rng(seed)
    values = [statistic(*(c[rows] for c in columns)) for rows in (rng.integers(0, n, size=n) for _ in range(draws))]
    low, high = np.nanquantile(values, [(1 - level) / 2, (1 + level) / 2])
    return {"value": float(statistic(*columns)), "low": float(low), "high": float(high)}


def rounded(score: dict) -> dict:
    """A score as the result file shows it: four decimals, after the bar has been checked on the full values."""
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in score.items()}


def counts(table: pd.DataFrame, bar: dict) -> dict:
    """How many reels are in and how many hold each reading. Counts only: nothing here says how a reel did."""
    both = table[["skip_rate_1", "views_1", "views_14"]].notna().all(axis=1)
    return {
        "reels in the set": len(table),
        "reels the bar asks for": bar["set"]["reels"],
        "with the day-1 reading": int(table["skip_rate_1"].notna().sum()),
        "with the day-14 reading": int(table["views_14"].notna().sum()),
        "with both": int(both.sum()),
        "taken down": int(table["taken_down"].sum()),
        "taken down and missing a reading": int((table["taken_down"] & ~both).sum()),
    }


def status(bar: dict, reels: pd.DataFrame, snaps: pd.DataFrame) -> dict:
    """Where the test stands: the counts and the reading date, never a correlation."""
    chosen = test_set(reels, bar)
    return {**counts(readings(chosen, snaps, bar), bar), "reading date": f"{reading_date(chosen, bar):%Y-%m-%d %H:%M} UTC"}


def read(bar: dict, reels: pd.DataFrame, snaps: pd.DataFrame, now=None) -> dict:
    """The test, once. Refuses before the reading date; then the bar's verdict, the three extras and the counts."""
    now = pd.Timestamp.now(tz="UTC") if now is None else utc(now)
    chosen = test_set(reels, bar)
    when = reading_date(chosen, bar)
    if now < when:
        raise TooEarly(f"The live test is read from {when:%Y-%m-%d %H:%M} UTC, not before "
                       f"({len(chosen)} of {bar['set']['reels']} reels are in).")

    table = readings(chosen, snaps, bar)
    result = {"read at": f"{now:%Y-%m-%d %H:%M} UTC", **counts(table, bar)}
    both = table.dropna(subset=["skip_rate_1", "views_1", "views_14"])
    if len(both) < bar["bar"]["minimum reels"]:
        return {**result, "verdict": "void"}  # too few reels to read: neither passed nor failed

    draw = {k: bar["interval"][k] for k in ("draws", "seed", "level")}
    skip, views_1, views_14 = (both[c].to_numpy(float) for c in ("skip_rate_1", "views_1", "views_14"))
    test = bootstrap(spearman, [skip, views_14], **draw)
    passed = test["value"] <= bar["bar"]["rho at most"] and test["high"] < bar["bar"]["interval high below"]
    rated = both.dropna(subset=["share_rate_1", "share_rate_14"])
    share_rates = [rated["share_rate_1"].to_numpy(float), rated["share_rate_14"].to_numpy(float)]
    return {
        **result,
        "verdict": "passed" if passed else "failed",
        "spearman, skip rate at day 1 against views at day 14": rounded({**test, "reels": len(both)}),
        "pearson, skip rate at day 1 against log10 views at day 14": rounded(
            bootstrap(pearson_log10, [skip, views_14], **draw)),
        "partial spearman, given the views at day 1": rounded(
            bootstrap(partial_spearman, [skip, views_14, views_1], **draw)),
        "spearman, share rate at day 1 against day 14": rounded(
            {**bootstrap(spearman, share_rates, **draw), "reels": len(rated)}),
        # For the site: each reel's two readings, without its id, date or caption, sorted so their order says nothing.
        "readings, skip rate at day 1 and views at day 14": sorted(
            [round(float(s), 1), int(v)] for s, v in zip(skip, views_14)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sharerate.live", description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["status", "read"])
    args = parser.parse_args(argv)
    bar = load_bar()
    reels, snaps = load_reels()
    if args.command == "status":
        for name, value in status(bar, reels, snaps).items():
            print(f"{name}: {value}")
        return 0
    try:
        result = read(bar, reels, snaps)
    except TooEarly as early:
        print(early)
        return 1
    RESULT.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{result['verdict']}: {RESULT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
