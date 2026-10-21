"""What I saw before writing stage 7's bar: the numbers in docs/live-test.md's "Before the bar", counted again.

Reads the platform's tables for my own accounts, read-only (mode=ro), and prints three things. Only reels and Shorts
posted before the live test's first reel (reports/live_bar.json, "published from") count, so none of the test's reels
are in them.

  1. Instagram: the skip rate against log views, both at each reel's last snapshot, over the reels whose last
     snapshot is at least 3 days after posting.
  2. Instagram: shares per 1,000 views at about a day (14 to 30 hours) against the same rate 10 days or more after
     posting, both as logs. Reels with at least 300 views at the first reading count.
  3. YouTube: the audience watch ratio at 3 seconds (the share of viewers still watching, replays included) and
     YouTube's relative retention there, against log views. Shorts of up to 60 s with at least 50 views count.

    python scripts/live_exploration.py

The platform is found at FLYWHEEL_DIR if that is set, else C:/dev/career/prsnl-cnt-flywheel. Standard library only.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import statistics
from datetime import datetime, timezone
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
FLYWHEEL_DIR = Path(os.environ.get("FLYWHEEL_DIR", "C:/dev/career/prsnl-cnt-flywheel")).resolve()
DB = FLYWHEEL_DIR / "databases" / "content_intelligence.db"
START = json.loads((REPO_DIR / "reports" / "live_bar.json").read_text(encoding="utf-8"))["set"]["published from"]


def utc(text: str) -> datetime:
    """A platform timestamp as UTC. Instagram writes "+0000" without the colon; old rows carry no offset at all."""
    stamp = datetime.fromisoformat(str(text).replace("+0000", "+00:00").replace("Z", "+00:00"))
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    top = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return top / math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))


def instagram(con: sqlite3.Connection, start: datetime) -> None:
    skips, firsts = [], []
    reels = con.execute("SELECT media_id, timestamp, ig_status FROM instagram_media WHERE media_product_type = 'REELS'")
    for media_id, timestamp, status in reels.fetchall():
        published = utc(timestamp)
        if published >= start or status == "deleted":
            continue  # the test's reels, and reels deleted within minutes, are left out
        rows = con.execute(
            "SELECT snapshot_at, COALESCE(views, plays), shares, skip_rate FROM instagram_media_snapshots"
            " WHERE media_id = ? AND COALESCE(views, plays) IS NOT NULL ORDER BY snapshot_at", (media_id,)).fetchall()
        points = [((utc(at) - published).total_seconds() / 3600, views, shares, skip) for at, views, shares, skip in rows]
        points = [p for p in points if p[0] >= 0]
        if not points:
            continue
        hours, views, _, skip = points[-1]
        if skip is not None and hours >= 72 and views:
            skips.append((skip, views))
        # Share rate at about a day, against the same reel's rate 10 days or more after posting.
        with_shares = [p for p in points if p[2] is not None and p[1] > 0]
        day = [p for p in with_shares if 14 <= p[0] <= 30]
        late = [p for p in with_shares if p[0] >= 240]
        if day and late and day[-1][1] >= 300:
            firsts.append((1000 * day[-1][2] / day[-1][1], 1000 * late[-1][2] / late[-1][1]))

    r = pearson([s for s, _ in skips], [math.log(v) for _, v in skips])
    print(f"1. Instagram: {len(skips)} reels, skip rate against log views, Pearson r = {r:.2f}")
    for low, high in ((0, 40), (40, 50), (50, 101)):
        views = [v for s, v in skips if low <= s < high]
        if views:
            reels_word = "reel" if len(views) == 1 else "reels"
            print(f"   skip {low} to {min(high, 100)}%: {len(views)} {reels_word}, median views {statistics.median(views):,.0f}")
    r = pearson([math.log(a + 0.5) for a, _ in firsts], [math.log(b + 0.5) for _, b in firsts])
    print(f"2. Instagram: {len(firsts)} reels, log share rate at about a day against 10 days or more, Pearson r = {r:.2f}")


def youtube(con: sqlite3.Connection, start: datetime) -> None:
    duration = dict(con.execute("SELECT video_id, duration_seconds FROM videos").fetchall())
    published = dict(con.execute("SELECT video_id, published_at FROM videos").fetchall())
    views = dict(con.execute(
        "SELECT video_id, views FROM video_snapshots s WHERE snapshot_at = "
        "(SELECT MAX(snapshot_at) FROM video_snapshots t WHERE t.video_id = s.video_id)").fetchall())
    latest = dict(con.execute("SELECT video_id, MAX(snapshot_date) FROM video_retention_curves GROUP BY video_id").fetchall())
    curves: dict[str, dict[float, tuple[float, float]]] = {}
    for video, date, ratio, watch, relative in con.execute(
            "SELECT video_id, snapshot_date, elapsed_ratio, audience_watch_ratio, relative_retention_performance"
            " FROM video_retention_curves"):
        if date == latest[video]:  # each Short's newest curve only
            curves.setdefault(video, {})[round(ratio, 2)] = (watch, relative)

    watch_3, relative_3, log_views = [], [], []
    for video, curve in curves.items():
        seconds, count = duration.get(video), views.get(video)
        if not seconds or seconds > 60 or not count or count < 50 or not published.get(video):
            continue
        if utc(published[video]) >= start:
            continue
        target = round(min(max(3 / seconds, 0.01), 1.0), 2)  # 3 seconds as a share of the Short's length
        watch, relative = curve[min(curve, key=lambda ratio: abs(ratio - target))]
        watch_3.append(watch)
        relative_3.append(relative)
        log_views.append(math.log(count))
    print(f"3. YouTube: {len(log_views)} Shorts, against log views: audience watch ratio at 3 s, "
          f"Pearson r = {pearson(watch_3, log_views):.2f}; relative retention at 3 s, r = {pearson(relative_3, log_views):.2f}")


def main() -> None:
    start = utc(START)
    con = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)  # mode=ro: the platform is never written
    try:
        print(f"Counted {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC, before {start:%Y-%m-%d %H:%M} UTC only")
        instagram(con, start)
        youtube(con, start)
    finally:
        con.close()


if __name__ == "__main__":
    main()
