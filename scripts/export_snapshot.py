"""Copy a read-only snapshot of the competitor video data out of the flywheel.

Reads two of the flywheel's SQLite databases in read-only mode and writes one new
SQLite file, data/snapshot.sqlite, that all the analysis in this repo works on.
Nothing in the flywheel is ever written, and no image or video is copied: the
snapshot stores paths to them, relative to the flywheel folder.

    python scripts/export_snapshot.py

The flywheel is found at FLYWHEEL_DIR if that is set, else C:/dev/career/prsnl-cnt-flywheel.
Standard library only, so it runs on any Python 3.10+ without a virtual environment.

Tables written:
  videos      one row per TikTok/YouTube video with its counts, caption and AI notes
  snapshots   repeated measurements of the same videos (counts at several dates)
  files       where each video's file and poster frame are, and whether they still exist
  export_info when the snapshot was made, from where, and the row counts
"""

from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_DIR / "data" / "snapshot.sqlite"
FLYWHEEL_DIR = Path(os.environ.get("FLYWHEEL_DIR", "C:/dev/career/prsnl-cnt-flywheel")).resolve()

STATS_DB = FLYWHEEL_DIR / "databases" / "competitor_video_stats.db"
NOTES_DB = FLYWHEEL_DIR / "databases" / "competitor_videos.db"
VIDEOS_DIR = FLYWHEEL_DIR / "competitor_videos"
POSTERS_DIR = FLYWHEEL_DIR / "Backend" / "data" / "instagram" / "competitor_thumbnails"

BATCH = 5000

STATS_COLUMNS = [
    "username", "platform", "video_id", "file_name", "title", "published_at",
    "duration_seconds", "fetched_at", "view_count", "like_count", "comment_count",
    "share_count", "save_count",
]
SNAPSHOT_COLUMNS = [
    "username", "platform", "video_id", "captured_at", "view_count", "like_count",
    "comment_count", "share_count", "save_count",
]
# Only what describes the video itself. Left out on purpose: the user's own marks
# (favourites, watched, used-as-idea) and the empty free-text `note`, because they
# were made after seeing how the video did.
NOTES_COLUMNS = [
    "username", "platform", "file_name", "transcript", "transcript_en", "hook_text_en",
    "body_text_en", "payoff_text_en", "onscreen_text_en", "scene_description_en",
    "description", "is_wordless_verified",
]
RETIRED_COLUMNS = ["username", "platform", "file_name", "retired_at", "reason"]


def open_read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        sys.exit(f"Not found: {path}")
    # mode=ro: SQLite itself refuses any write through this connection.
    return sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)


def copy_rows(src: sqlite3.Connection, query: str, dest: sqlite3.Connection, insert: str) -> int:
    cursor = src.execute(query)
    total = 0
    while rows := cursor.fetchmany(BATCH):
        dest.executemany(insert, rows)
        total += len(rows)
    return total


def safe_folder(username: str) -> str:
    # Same rule the flywheel uses to name a creator's folder.
    normalized = str(username or "").strip().lower()
    return "".join(ch for ch in normalized if ch.isalnum() or ch in "._-").strip(" .")


class Listing:
    """Lists each folder once, so checking 180k files is a few hundred directory reads."""

    def __init__(self) -> None:
        self._cache: dict[Path, set[str]] = {}

    def names(self, folder: Path) -> set[str]:
        if folder not in self._cache:
            try:
                with os.scandir(folder) as entries:
                    self._cache[folder] = {e.name for e in entries if e.is_file()}
            except FileNotFoundError:
                self._cache[folder] = set()
        return self._cache[folder]


def relative(path: Path) -> str:
    return path.relative_to(FLYWHEEL_DIR).as_posix()


def build_files_table(dest: sqlite3.Connection) -> None:
    listing = Listing()
    dest.execute(
        """CREATE TABLE files (
            platform TEXT, username TEXT, video_id TEXT,
            video_path TEXT, has_video INTEGER,
            poster_path TEXT, has_poster INTEGER,
            retired_at TEXT, retire_reason TEXT,
            PRIMARY KEY (platform, username, video_id))"""
    )
    rows = dest.execute(
        """SELECT v.platform, v.username, v.video_id, v.file_name, r.retired_at, r.reason
           FROM videos v
           LEFT JOIN retired r
             ON r.username = v.username AND r.platform = v.platform AND r.file_name = v.file_name"""
    ).fetchall()
    out = []
    for platform, username, video_id, file_name, retired_at, reason in rows:
        folder = safe_folder(username)
        # Instagram clips sit at the creator folder's root; TikTok and YouTube in a sub-folder.
        video_folder = VIDEOS_DIR / folder if platform == "instagram" else VIDEOS_DIR / folder / platform
        poster_folder = POSTERS_DIR / folder / platform
        poster_name = f"{file_name}.jpg"
        out.append((
            platform, username, video_id,
            relative(video_folder / file_name), int(file_name in listing.names(video_folder)),
            relative(poster_folder / poster_name), int(poster_name in listing.names(poster_folder)),
            retired_at, reason,
        ))
    dest.executemany("INSERT INTO files VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", out)


def main() -> None:
    out_path = OUT_PATH.resolve()
    if FLYWHEEL_DIR == out_path or FLYWHEEL_DIR in out_path.parents:
        sys.exit(f"Refusing to write inside the flywheel: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".sqlite.tmp")
    tmp_path.unlink(missing_ok=True)

    stats = open_read_only(STATS_DB)
    notes = open_read_only(NOTES_DB)
    dest = sqlite3.connect(tmp_path)
    # A throwaway file until the final rename, so no journal is needed.
    dest.execute("PRAGMA journal_mode = OFF")
    dest.execute("PRAGMA synchronous = OFF")

    dest.execute(f"CREATE TABLE stats ({', '.join(STATS_COLUMNS)})")
    n_stats = copy_rows(
        stats, f"SELECT {', '.join(STATS_COLUMNS)} FROM video_stats", dest,
        f"INSERT INTO stats VALUES ({', '.join('?' * len(STATS_COLUMNS))})",
    )
    dest.execute(f"CREATE TABLE snapshots ({', '.join(SNAPSHOT_COLUMNS)})")
    n_snapshots = copy_rows(
        stats, f"SELECT {', '.join(SNAPSHOT_COLUMNS)} FROM video_stat_snapshots ORDER BY id", dest,
        f"INSERT INTO snapshots VALUES ({', '.join('?' * len(SNAPSHOT_COLUMNS))})",
    )
    # One notes row per file; if a file has two, the later one (higher id) wins.
    dest.execute(
        f"CREATE TABLE notes ({', '.join(NOTES_COLUMNS)}, UNIQUE (username, platform, file_name))"
    )
    n_notes = copy_rows(
        notes, f"SELECT {', '.join(NOTES_COLUMNS)} FROM competitor_video_notes ORDER BY id", dest,
        f"INSERT OR REPLACE INTO notes VALUES ({', '.join('?' * len(NOTES_COLUMNS))})",
    )
    dest.execute(f"CREATE TABLE retired ({', '.join(RETIRED_COLUMNS)}, UNIQUE (username, platform, file_name))")
    copy_rows(
        notes, f"SELECT {', '.join(RETIRED_COLUMNS)} FROM retired_videos", dest,
        f"INSERT OR REPLACE INTO retired VALUES ({', '.join('?' * len(RETIRED_COLUMNS))})",
    )
    stats.close()
    notes.close()

    dest.execute(
        """CREATE TABLE videos AS
           SELECT s.platform, s.username, s.video_id, s.file_name, s.title, s.published_at,
                  s.duration_seconds, s.fetched_at, s.view_count, s.like_count, s.comment_count,
                  s.share_count, s.save_count,
                  n.transcript, n.transcript_en,
                  n.hook_text_en AS hook_en, n.body_text_en AS body_en, n.payoff_text_en AS payoff_en,
                  n.onscreen_text_en, n.scene_description_en, n.description,
                  n.is_wordless_verified AS is_wordless,
                  n.file_name IS NOT NULL AS has_notes
           FROM stats s
           LEFT JOIN notes n
             ON n.username = s.username AND n.platform = s.platform AND n.file_name = s.file_name"""
    )
    build_files_table(dest)
    dest.execute("DROP TABLE stats")
    dest.execute("DROP TABLE notes")
    dest.execute("DROP TABLE retired")
    dest.execute("CREATE UNIQUE INDEX videos_key ON videos (platform, username, video_id)")
    dest.execute("CREATE INDEX snapshots_key ON snapshots (platform, username, video_id)")

    info = {
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "flywheel_dir": FLYWHEEL_DIR.as_posix(),
        "source_stats_rows": n_stats,
        "source_snapshot_rows": n_snapshots,
        "source_notes_rows": n_notes,
    }
    dest.execute("CREATE TABLE export_info (key TEXT PRIMARY KEY, value TEXT)")
    dest.executemany("INSERT INTO export_info VALUES (?, ?)", [(k, str(v)) for k, v in info.items()])
    dest.commit()

    summary = dest.execute(
        """SELECT v.platform, COUNT(*), SUM(v.has_notes), SUM(v.transcript_en <> ''),
                  SUM(v.share_count > 0), SUM(f.has_video), SUM(f.has_poster),
                  SUM(f.has_video OR f.has_poster), COUNT(DISTINCT v.username)
           FROM videos v JOIN files f USING (platform, username, video_id)
           GROUP BY v.platform ORDER BY v.platform"""
    ).fetchall()
    dest.execute("VACUUM")
    dest.close()
    os.replace(tmp_path, out_path)

    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.0f} MB) from {FLYWHEEL_DIR}")
    print(f"{'platform':<10}{'videos':>9}{'notes':>9}{'text_en':>9}{'shares':>9}"
          f"{'video':>9}{'poster':>9}{'either':>9}{'creators':>10}")
    for row in summary:
        print(f"{row[0]:<10}" + "".join(f"{v:>9}" for v in row[1:8]) + f"{row[8]:>10}")


if __name__ == "__main__":
    main()
