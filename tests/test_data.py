"""The loader returns only what a model may see (README rule 4), one row per target video, test closed by default.

These tests read data/, which is not in git, so they skip on a fresh clone.
"""

import sqlite3

import pandas as pd
import pytest

from sharerate import data

pytestmark = pytest.mark.skipif(
    not (data.STAGE1.exists() and data.SNAPSHOT.exists()), reason="data/ is not in git")


@pytest.fixture(scope="module")
def videos():
    return data.load_videos()


@pytest.fixture(scope="module")
def every_split():
    return data.load_videos(data.SPLITS)


def snapshot_columns(table):
    with sqlite3.connect(f"file:{data.SNAPSHOT.as_posix()}?mode=ro", uri=True) as con:
        return {row[1] for row in con.execute(f"PRAGMA table_info({table})")}


def test_no_forbidden_column_comes_back(videos, every_split):
    for frame in (videos, every_split):
        assert not set(frame.columns) & data.FORBIDDEN_COLUMNS
        assert not set(frame.columns) & data.IDENTIFYING_COLUMNS


def test_forbidden_list_covers_every_count_snapshots_and_files_column():
    # Read from the snapshot's schema, so a column added to it later can't slip past the list.
    keys = {"platform", "username", "video_id"}
    after_posting = (snapshot_columns("files") | snapshot_columns("snapshots")) - keys
    counts = {c for c in snapshot_columns("videos") if c.endswith("_count")}
    assert after_posting | counts <= data.FORBIDDEN_COLUMNS
    assert {"fetched_at", "age_days", "share_rate", "log_sr", "views", "shares"} <= data.FORBIDDEN_COLUMNS


def test_only_the_allowed_columns_come_back(videos):
    assert tuple(videos.columns) == data.COLUMNS


def test_a_forbidden_column_added_to_the_loader_is_refused(monkeypatch):
    # The guard has teeth: ask the loader for a count and it raises instead of returning it.
    monkeypatch.setattr(data, "SNAPSHOT_COLUMNS", data.SNAPSHOT_COLUMNS + ("view_count",))
    monkeypatch.setattr(data, "COLUMNS", data.COLUMNS + ("view_count",))
    with pytest.raises(RuntimeError, match="view_count"):
        data.load_videos()


def test_one_row_per_target_video(every_split):
    with sqlite3.connect(f"file:{data.STAGE1.as_posix()}?mode=ro", uri=True) as con:
        targets = con.execute("SELECT COUNT(*) FROM targets").fetchone()[0]
        creators = con.execute("SELECT COUNT(DISTINCT username) FROM targets").fetchone()[0]
    assert len(every_split) == targets == 143_489
    assert every_split.creator.nunique() == creators == 231
    assert not every_split.duplicated().any()


def test_the_test_split_stays_closed_unless_named(videos, every_split):
    assert set(videos.time_split) == {"train", "validation"}
    assert "test" in set(every_split.time_split)
    with pytest.raises(ValueError):
        data.load_videos(("train", "validation", "tset"))


def test_counts_match_the_brief(videos):
    # Stage 1's splits: training 74,552 rows from 148 creators; validation 13,646 seen, 2,275 new, 5,411 held out.
    training = videos[(videos.time_split == "train") & (videos.creator_group == "train")]
    assert (len(training), training.creator.nunique()) == (74_552, 148)
    validation = videos[videos.time_split == "validation"]
    assert validation.creator_seen.value_counts().to_dict() == {"seen": 13_646, "holdout": 5_411, "new": 2_275}
    assert round(validation.breakout.mean(), 3) == 0.242
    assert validation.has_twin.sum() == 94


def test_sorted_by_creator_then_time(videos):
    assert videos.equals(videos.sort_values(["creator", "published_at"], kind="stable"))
    assert videos.index.equals(pd.RangeIndex(len(videos)))


def test_leak_reference_lines_up_with_the_rows(videos):
    leak = data.load_leak_reference()
    assert leak.index.equals(videos.index)
    assert set(leak.unique()) <= {0, 1}


def test_posting_times_hold_every_target_video(every_split):
    posts = data.load_posting_times()
    assert set(posts.columns) == {"creator", "posted_at"}
    assert posts.posted_at.notna().all()
    # Every target video is a library video, posted at its own published_at.
    found = every_split.merge(posts, left_on=["creator", "published_at"], right_on=["creator", "posted_at"],
                              how="left", indicator=True)
    assert (found._merge == "both").all()


@pytest.fixture(scope="module")
def with_text():
    return data.load_videos(data.SPLITS, text=True)


def test_text_comes_back_only_when_asked(videos, every_split, with_text):
    assert not set(videos.columns) & set(data.TEXT_COLUMNS)
    assert tuple(with_text.columns) == data.COLUMNS + tuple(data.TEXT_COLUMNS)
    # The same rows in the same order, so a text feature lines up with every stage 2 feature.
    assert with_text[list(data.COLUMNS)].equals(every_split)


def test_text_is_never_missing_only_empty(with_text):
    text = with_text[list(data.TEXT_COLUMNS)]
    assert text.notna().all().all()
    assert (text.map(lambda value: value == value.strip())).all().all()


def test_text_coverage_matches_the_brief(with_text):
    # Notebook 01, cell 27: of the 143,489 target videos, how many have each field.
    has = (with_text[list(data.TEXT_COLUMNS)] != "").sum()
    assert has[["transcript_en", "onscreen_text_en", "scene_description_en", "hook_en"]].to_dict() == {
        "transcript_en": 115_440, "onscreen_text_en": 85_261, "scene_description_en": 142_793, "hook_en": 105_086}


def test_the_left_out_text_columns_never_come_back(with_text):
    # The original-language transcript and the language model's "why it works" note stay in the snapshot.
    assert {"transcript", "description", "title"}.isdisjoint(with_text.columns)
    assert set(data.TEXT_COLUMNS.values()) <= snapshot_columns("videos")
