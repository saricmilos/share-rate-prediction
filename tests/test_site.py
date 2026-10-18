"""The site's export: nothing a creator wrote, no username or video id, and no group under 5 creators leaves.

The first tests run anywhere: they plant a username, a video id, a caption and a small group in a fake export and
check the scan finds each, and build the calculator's histogram from made-up videos to check that a sparse end bar is
merged and its creators counted again on the merged range. The last one scans the files the export wrote into the
site (SHARERATE_SITE), against every username, video id and text field in the snapshot, so it needs data/ and the
site's folder, and skips without them.
"""

import json

import numpy as np
import pandas as pd
import pytest

from sharerate import data, site

USERNAMES = {"cook.with.ana", "fitbydan"}
VIDEO_IDS = {"7301234567890123456", "dqw4w9wgxcq"}
TEXTS = ["the one trick my grandmother never told anyone about", "short"]


def _scan(tmp_path, content):
    path = tmp_path / "export.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    return site.scan([path], USERNAMES, VIDEO_IDS, TEXTS)


def test_a_clean_export_passes(tmp_path):
    clean = {"label": "metadata and the words", "examples": ["Send this to the friend who always steals your fries"],
             "group": {"videos": 120, "creators": 5}, "value": 0.3595}
    assert _scan(tmp_path, clean) == []


@pytest.mark.parametrize("planted", [
    {"label": "made by cook.with.ana"},  # a username as a word
    {"label": "Fitbydan's best"},  # in another case, with a possessive
    {"id": "7301234567890123456"},  # a TikTok video id
    {"id": 7301234567890123456},  # the same id as a number
    {"label": "watch dQw4w9WgXcQ"},  # a YouTube video id
])
def test_a_username_or_video_id_is_found(tmp_path, planted):
    assert _scan(tmp_path, planted)


def test_a_creator_text_is_found_inside_an_exported_string(tmp_path):
    assert _scan(tmp_path, {"examples": ["Look: The one trick my grandmother   never told anyone about!"]})


def test_a_quote_of_part_of_a_creator_text_is_found(tmp_path):
    assert _scan(tmp_path, {"examples": ["my grandmother never told anyone"]})


def test_a_short_creator_text_is_not_a_quote(tmp_path):
    # "short" is a caption here, and a common word: texts under 15 characters are too short to be anyone's
    assert _scan(tmp_path, {"label": "a short video"}) == []


def test_a_group_under_five_creators_is_found(tmp_path):
    assert _scan(tmp_path, {"years": [{"year": 2019, "creators": 4, "videos": 300}]})


def test_a_sparse_end_bar_is_merged_and_its_creators_counted_again():
    # Ten made-up creators with a video every 0.01 of lift from -3 to 3, so every bar holds all ten; its edges come
    # from the calculator itself
    grid = np.round(np.arange(-3, 3, 0.01), 2)
    dense = pd.DataFrame({"username": np.repeat([f"c{i}" for i in range(10)], len(grid)), "lift": np.tile(grid, 10)})
    edges = [b["to"] for b in site.calculator(dense.assign(breakout=dense.lift >= data.LOG2))["bins"][:-1]]
    # Below the second edge, three creators in the first bar and two others in the second: each is short of 5, the
    # larger count is 3, and together they are 5, so the two merge into one bar and stop there
    low = pd.DataFrame({"username": ["c0", "c1", "c2", "c3", "c4"],
                        "lift": [edges[0] - 0.5] * 3 + [(edges[0] + edges[1]) / 2] * 2})
    sparse = pd.concat([dense[dense.lift >= edges[1]], low], ignore_index=True)
    bins = site.calculator(sparse.assign(breakout=sparse.lift >= data.LOG2))["bins"]
    assert bins[0] == {"from": None, "to": edges[1], "videos": 5, "creators": 5}
    assert len(bins) == len(edges)  # one bar fewer than the dense histogram's


def test_the_site_export_holds_nothing_private():
    files = [site.SITE / site.CONTENT / name for name in site.FILES]
    if not all(path.exists() for path in files):
        pytest.skip(f"no export in {site.SITE / site.CONTENT}: run python -m sharerate.site (SHARERATE_SITE names the site)")
    if not (data.SNAPSHOT.exists() and data.STAGE1.exists()):
        pytest.skip("data/ is not here: the snapshot is not public")
    assert site.scan(files, *site.private_strings()) == []
