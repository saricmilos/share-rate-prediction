"""The rows every model is fitted and scored on, with only the columns a model may see.

`load_videos` returns one row per video in stage 1's `targets` table (`data/stage1.sqlite`), joined to the
snapshot's `videos` table for the columns README rule 4 allows before a video is posted. Two kinds of column
never come back:

- **Anything measured after posting** (`FORBIDDEN_COLUMNS`): the counts, `snapshots`, `fetched_at`, `age_days`,
  and every `files` column, because the platform keeps each channel's top clips by share rate (trap 3).
- **Who or what the row is** (`IDENTIFYING_COLUMNS`): usernames and video ids stay inside this module, and
  `creator` is an opaque number.

With `text=True` it also returns what the video says and shows in writing (`TEXT_COLUMNS`, stage 3). Two text
columns of the snapshot stay out: the original-language `transcript` (stored for 11% of the target videos; the
English one is there for 80%) and `description`, a language model's "why it works" note written from the other text
fields, which adds that model's opinion and nothing the fields don't hold.

The test split comes back only when it is asked for by name, so no notebook opens it by accident. Stage 5 opens
it, once.
"""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STAGE1 = ROOT / "data" / "stage1.sqlite"
SNAPSHOT = ROOT / "data" / "snapshot.sqlite"
# The splits are frozen on this export (notebook 01, cell 30): a later one would bring counts they never saw.
FROZEN_SNAPSHOT = "2026-10-14T09:56:20+00:00"

SPLITS = ("train", "validation", "test")
OPEN_SPLITS = ("train", "validation")  # the default: "test" has to be named
LOG2 = float(np.log10(2))  # breakout = lift >= log10(2), at least twice the creator's recent median

# From stage 1's targets table: the label, the baseline it is measured against, and the splits.
TARGET_COLUMNS = ("published_at", "time_split", "creator_group", "creator_seen", "has_twin",
                  "n_prior", "baseline", "lift", "breakout")
# From the snapshot's videos table: what is known before posting (rule 4).
SNAPSHOT_COLUMNS = ("duration_seconds", "is_wordless")
COLUMNS = ("creator",) + TARGET_COLUMNS + SNAPSHOT_COLUMNS
# Also from the videos table, with `text=True`: what the video says and shows in writing, all fixed when it is
# posted (rule 4). {name in the frame: column in the snapshot}. An empty field comes back as "".
TEXT_COLUMNS = {
    "caption": "title",  # the creator's caption, as the platform returned it, hashtags included
    "transcript_en": "transcript_en",  # the spoken words in English (Whisper, translated where the video isn't)
    # The script split into hook, body and payoff by a language model, from the transcript, or from the on-screen
    # text and scene description when nobody speaks
    "hook_en": "hook_en",
    "body_en": "body_en",
    "payoff_en": "payoff_en",
    "onscreen_text_en": "onscreen_text_en",  # read off sampled frames by a vision-language model (trap 5)
    "scene_description_en": "scene_description_en",  # written by the same vision-language model (trap 5)
}

FORBIDDEN_COLUMNS = frozenset({
    # Counts: outcomes of posting, read at the same moment as the shares (rule 4)
    "views", "likes", "comments", "shares", "saves",
    "view_count", "like_count", "comment_count", "share_count", "save_count",
    # Computed from the counts, or saying when they were read and whether there were any
    "share_rate", "log_sr", "fetched_at", "age_days", "has_stats",
    # The snapshots table: the same counts, measured again later
    "captured_at",
    # The files table, and the videos table's file name: whether a clip was kept depends on its share rate
    "file_name", "video_path", "has_video", "poster_path", "has_poster", "retired_at", "retire_reason",
})
IDENTIFYING_COLUMNS = frozenset({"username", "video_id"})


def load_videos(splits=OPEN_SPLITS, *, text: bool = False) -> pd.DataFrame:
    """One row per target video in `splits`, with the columns in `COLUMNS` and nothing else.

    Sorted by creator, then publish time (ties by video id, as stage 1 built the baseline), with a fresh index.
    `lift` and `breakout` are the label. `baseline` is the median log share rate of the creator's last 30
    videos posted at least 14 days earlier, and `n_prior` how many it is the median of (10 to 30).

    `text=True` adds the `TEXT_COLUMNS` after them, the same rows in the same order. They hold captions and
    transcripts, other creators' words: they are for fitting models and never for printing.
    """
    frame = _read_targets(splits, text=text)
    # breakout is lift >= log10(2) by definition (notebook 01, cell 27); the metrics rely on it.
    if not (frame.breakout == (frame.lift >= LOG2)).all():
        raise RuntimeError("breakout no longer equals lift >= log10(2) in data/stage1.sqlite")
    videos = frame[list(COLUMNS) + (list(TEXT_COLUMNS) if text else [])]
    _refuse_forbidden(videos.columns)
    return videos


def load_leak_reference(splits=OPEN_SPLITS) -> pd.Series:
    """`files.has_video` for the rows of `load_videos(splits)`, in the same order. **A leak, never a feature.**

    The platform keeps each channel's top clips by share rate, so a clip that still has its file was a winner
    (trap 3). Stage 2 shows it only as a labelled reference, to show what a leak looks like next to honest
    baselines.
    """
    return _read_targets(splits, has_video=True).has_video.rename("has_video_LEAK")


def load_creator_views(splits=OPEN_SPLITS) -> pd.Series:
    """Each creator's median view count over their target videos in `splits`: a channel's size, for cutting results
    into groups of creators. **Never a feature.**

    Views are counted after posting, so no model sees them (rule 4). Over the training and validation periods this is
    one number per creator, known before any test video went up, so it can only sort creators by size; it says nothing
    about any one video. Indexed by the opaque `creator` code, like every other frame here.
    """
    splits = tuple(splits)
    if not splits or set(splits) - set(SPLITS):
        raise ValueError(f"splits must be some of {SPLITS}, not {splits}")
    with closing(_open_snapshot()) as con:
        views = pd.read_sql(f"SELECT username, views FROM stage1.targets WHERE time_split IN ({', '.join('?' * len(splits))})",
                            con, params=splits)
        codes = _creator_codes(con)
    return views.groupby(views.username.map(codes)).views.median().rename("median_views").rename_axis("creator")


def load_posting_times() -> pd.DataFrame:
    """When each target creator posted every TikTok video in the library, counted or not: `creator`, `posted_at`.

    For timing features only (how often, and at what hour, a creator posts), never for rates. The time is
    `published_at` where the counts fetch recorded it. Library videos without counts have none, so theirs
    comes from the TikTok video id, whose top 32 bits are the upload's Unix time: on counted videos it sits a
    median 10 s before `published_at`, and within a day for 99.6% of them.
    """
    with closing(_open_snapshot()) as con:
        posts = pd.read_sql(
            """SELECT username, video_id, published_at FROM videos
               WHERE platform = 'tiktok' AND username IN (SELECT DISTINCT username FROM stage1.targets)""",
            con,
        )
        codes = _creator_codes(con)
    # Python ints, not floats: ids have 19 digits, more than a float holds exactly.
    id_time = pd.to_datetime(posts.video_id.map(lambda video_id: int(video_id) >> 32), unit="s", utc=True)
    posted_at = pd.to_datetime(posts.published_at, utc=True, format="ISO8601").fillna(id_time)
    times = pd.DataFrame({"creator": posts.username.map(codes), "posted_at": posted_at})
    return times.sort_values(["creator", "posted_at"], ignore_index=True)


def _read_targets(splits, *, has_video: bool = False, text: bool = False) -> pd.DataFrame:
    """The targets rows in `splits` joined to the snapshot, still with their usernames and video ids."""
    splits = tuple(splits)
    if not splits or set(splits) - set(SPLITS):
        raise ValueError(f"splits must be some of {SPLITS}, not {splits}")
    selected = [f"t.{c}" for c in TARGET_COLUMNS] + [f"v.{c}" for c in SNAPSHOT_COLUMNS]
    if text:
        selected += [f"v.{source} AS {name}" for name, source in TEXT_COLUMNS.items()]
    columns = ", ".join(selected)
    leak_column, leak_join = "", ""
    if has_video:
        leak_column = ", f.has_video"
        leak_join = ("JOIN files AS f "
                     "ON f.platform = v.platform AND f.username = v.username AND f.video_id = v.video_id")
    in_splits = ", ".join("?" * len(splits))
    with closing(_open_snapshot()) as con:
        frame = pd.read_sql(
            f"""SELECT t.username, t.video_id, {columns}{leak_column}
                FROM stage1.targets AS t
                JOIN videos AS v ON v.platform = 'tiktok' AND v.username = t.username AND v.video_id = t.video_id
                {leak_join}
                WHERE t.time_split IN ({in_splits})""",
            con,
            params=splits,
        )
        expected = con.execute(
            f"SELECT COUNT(*) FROM stage1.targets WHERE time_split IN ({in_splits})", splits).fetchone()[0]
        codes = _creator_codes(con)
    # An inner join drops a target whose video is missing from the snapshot; refuse rather than lose rows.
    if len(frame) != expected:
        raise RuntimeError(f"{expected - len(frame)} target rows have no row in the snapshot's videos table")
    frame["published_at"] = pd.to_datetime(frame.published_at, utc=True, format="ISO8601")
    frame["creator"] = frame.username.map(codes)
    for name in TEXT_COLUMNS if text else ():
        # 210 target videos have no notes row at all (NULL), and some fields are blank or whitespace: all become "".
        frame[name] = frame[name].fillna("").astype(str).str.strip()
    return frame.sort_values(["creator", "published_at", "video_id"], ignore_index=True)


def _open_snapshot() -> sqlite3.Connection:
    """The frozen snapshot, read-only, with stage 1's targets attached read-only as `stage1`."""
    for path in (SNAPSHOT, STAGE1):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing: run scripts/export_snapshot.py, then notebooks/01_eda_target.ipynb")
    con = sqlite3.connect(f"file:{SNAPSHOT.as_posix()}?mode=ro", uri=True)
    exported = con.execute("SELECT value FROM export_info WHERE key = 'exported_at'").fetchone()[0]
    if exported != FROZEN_SNAPSHOT:
        con.close()
        raise RuntimeError(f"The splits are frozen on the {FROZEN_SNAPSHOT} export, not {exported}")
    con.execute("ATTACH DATABASE ? AS stage1", (f"file:{STAGE1.as_posix()}?mode=ro",))
    return con


def _creator_codes(con: sqlite3.Connection) -> dict[str, int]:
    """An opaque number per target creator: the rank of the md5 of the username among all 231.

    The same in every call, whichever splits are loaded, and it says nothing about who the creator is.
    """
    names = [name for (name,) in con.execute("SELECT DISTINCT username FROM stage1.targets")]
    ranked = sorted(names, key=lambda name: hashlib.md5(name.encode()).hexdigest())
    return {name: code for code, name in enumerate(ranked)}


def _refuse_forbidden(columns) -> None:
    """Raise if a column a model must never see, or one that names a creator or a video, is about to leave."""
    bad = sorted(set(columns) & (FORBIDDEN_COLUMNS | IDENTIFYING_COLUMNS))
    if bad:
        raise RuntimeError(f"The loader must never return {bad} (README rule 4, trap 3)")
