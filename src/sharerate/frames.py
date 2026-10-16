"""Stage 4's frames: one per target video, every one out of the same pipeline (README trap 3).

The platform keeps each channel's top clips by share rate and retires the rest. Before it deletes a clip's file it
saves a poster: the first frame it decodes after seeking 1 s into the clip, shrunk to 480 px on its longest edge, as a
JPEG at quality 82 (the flywheel's `competitors/thumbnail_service.py`). So a retired clip has only that poster, and a
kept clip has its whole video. A frame taken from a kept video any other way (sharper, another size, another JPEG
quality) would tell an image model which clips were kept, and the kept clips are the ones shared more: the target.

So this module makes kept clips' frames with the poster code itself, copied here line for line, under the same
library versions the posters were made with (PyAV 17.0.1 and Pillow 12.1.1, pinned in requirements.txt), and gives
retired clips their posters, copied byte for byte. Notebook 04_frames checks that the two can't be told apart.

Frames go to `frames/<id>.jpg`. The id is random: it says nothing about the video, its creator, its split or its
label, so a frame can go to a rented GPU without them. The map from id to video is `data/frames.sqlite`. Both are
git-ignored, and the map never leaves this disk.

The map also records where each frame came from (`source`: the video or the poster). That is the `files` table's
`has_video` under another name, the leak itself, so it is there for the checks and never comes back from `frame_paths`.
"""

from __future__ import annotations

import io
import os
import shutil
import sqlite3
import uuid
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from sharerate import data, text

FLYWHEEL_DIR = Path(os.environ.get("FLYWHEEL_DIR", "C:/dev/career/prsnl-cnt-flywheel"))
FRAMES_DIR = data.ROOT / "frames"
FRAME_MAP = data.ROOT / "data" / "frames.sqlite"
IMAGE_EMBEDDINGS = data.ROOT / "data" / "image_embeddings"  # what the GPU job's frame vectors come back into

# The flywheel's poster settings (thumbnail_service.py, unchanged since 2026-07-06)
THUMBNAIL_MAX_EDGE = 480  # longest edge, in pixels
JPEG_QUALITY = 82
POSTER_SEEK_SECONDS = 1.0  # frame 0 of a reel is often a black encoder warm-up frame
# The versions every poster was made with: installed in the flywheel's venv in March and April 2026, before its
# poster code existed. Another PyAV can convert colours differently, another Pillow can resample or encode differently.
PIPELINE_VERSIONS = {"av": "17.0.1", "PIL": "12.1.1"}


def decode_poster_frame(video_path):
    """The flywheel's `_decode_video_poster`, line for line: the first frame decoded after seeking to 1 s.

    `container.seek` lands on the keyframe at or before 1 s, and the first frame decoded from there is returned,
    so "about 1 s in" is really "the last keyframe before 1 s", often the very first frame. Copied, not improved:
    a better frame would be a different pipeline. The file is opened for reading only.
    """
    import av

    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError("File has no video stream.")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        duration_seconds = 0.0
        if container.duration:
            duration_seconds = float(container.duration) / av.time_base
        # Only seek when the clip is comfortably longer than the seek point;
        # a failed/over-shot seek falls back to whatever decodes first.
        if duration_seconds > POSTER_SEEK_SECONDS * 2:
            try:
                container.seek(int(POSTER_SEEK_SECONDS * av.time_base))
            except Exception:
                container.seek(0)
        for frame in container.decode(stream):
            return frame.to_image()
    raise ValueError("Could not decode a poster frame from the video.")


def encode_poster(image) -> bytes:
    """The flywheel's `_write_thumbnail`, line for line, into bytes instead of a file: the same JPEG bytes."""
    from PIL import Image

    image = image.convert("RGB")
    image.thumbnail((THUMBNAIL_MAX_EDGE, THUMBNAIL_MAX_EDGE), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buffer.getvalue()


def frame_from_video(video_path) -> bytes:
    """A kept clip's frame: the poster the platform would save if it retired the clip now."""
    return encode_poster(decode_poster_frame(video_path))


def check_versions() -> None:
    """Refuse to make frames under library versions the posters weren't made with."""
    import av
    import PIL

    found = {"av": av.__version__, "PIL": PIL.__version__}
    if found != PIPELINE_VERSIONS:
        raise RuntimeError(f"The posters were made with {PIPELINE_VERSIONS}, this environment has {found}: "
                           "pip install -r requirements.txt")


def target_files(splits=data.SPLITS) -> pd.DataFrame:
    """Every target video in `splits`: its split and creator group, and where its video file and poster are.

    The paths are relative to the flywheel, from the snapshot's `files` table, which is for finding pixels and never
    a feature (trap 3). No label comes back, so the test's rows can be counted without opening the test. Holds
    usernames and video ids (in the paths too): for this module and the checks, never for printing.
    """
    splits = tuple(splits)
    in_splits = ", ".join("?" * len(splits))
    with closing(data._open_snapshot()) as con:
        files = pd.read_sql(
            f"""SELECT t.username, t.video_id, t.time_split, t.creator_group, t.creator_seen,
                       f.video_path, f.poster_path
                FROM stage1.targets AS t
                JOIN files AS f ON f.platform = 'tiktok' AND f.username = t.username AND f.video_id = t.video_id
                WHERE t.time_split IN ({in_splits})""",
            con, params=splits)
    return files


def naive_frame(video_path) -> bytes:
    """The obvious way to take a frame, for notebook 04_frames' positive control only, never a model's frame.

    The frame shown at exactly 1 s (decoded on from the keyframe before it), resized to 480 px on its longest edge
    with bicubic, saved at JPEG quality 95. A check that can't tell this from a poster can't tell anything.
    """
    import av
    from PIL import Image

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        container.seek(int(POSTER_SEEK_SECONDS * av.time_base))  # lands on the keyframe at or before 1 s
        chosen = None
        for frame in container.decode(stream):
            chosen = frame
            if frame.time is not None and frame.time >= POSTER_SEEK_SECONDS - 1e-3:
                break  # the first frame at or after 1 s: the one a player shows at 1 s
        if chosen is None:
            raise ValueError("Could not decode a frame from the video.")
        image = chosen.to_image()
    scale = THUMBNAIL_MAX_EDGE / max(image.size)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.BICUBIC)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _make_one(job):
    """Make one video's frame in a worker process. Returns (username, video_id, source, error)."""
    username, video_id, video_path, poster_path, out_path = job
    video = FLYWHEEL_DIR / video_path if video_path else None
    poster = FLYWHEEL_DIR / poster_path if poster_path else None
    errors = []
    if video is not None and video.is_file():
        try:
            jpeg = frame_from_video(video)
            _write_atomically(out_path, jpeg)
            return username, video_id, "video", None
        except Exception as exc:  # an unreadable clip falls back to its poster, made by the same code earlier
            errors.append(f"video: {type(exc).__name__}: {exc}")
    if poster is not None and poster.is_file():
        tmp = Path(f"{out_path}.tmp")
        # copyfile, not copy2: the poster's file date (the day its clip was retired) must not travel with it
        shutil.copyfile(poster, tmp)
        os.replace(tmp, out_path)
        return username, video_id, "poster", "; ".join(errors) or None
    return username, video_id, None, "; ".join(errors) or "neither a video file nor a poster"


def _write_atomically(path, payload: bytes) -> None:
    tmp = Path(f"{path}.tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)  # an interrupted run never leaves half a JPEG under a frame's name


def build(splits=data.SPLITS, *, workers=None, limit=None, progress=None) -> pd.DataFrame:
    """Make every missing frame for the target videos in `splits`, and return the map's rows for them.

    Frames already in `frames/` are kept, so a run that stops resumes. A video whose file exists gets a frame from
    it (`frame_from_video`); one whose file is gone gets its poster. `limit` makes only the first `limit` missing
    ones (in random-id order, so a sample spans creators and splits). `progress(done, total)` after each frame.
    """
    check_versions()
    FRAMES_DIR.mkdir(exist_ok=True)
    files = target_files(splits)
    with closing(_open_map()) as con:
        known = pd.read_sql("SELECT username, video_id, frame_id, source FROM frames", con)
        files = files.merge(known, on=["username", "video_id"], how="left")
        # A new video gets a random id once; the id is kept in the map, so a rerun reuses it.
        new = files.frame_id.isna()
        files.loc[new, "frame_id"] = [uuid.uuid4().hex for _ in range(int(new.sum()))]
        con.executemany("INSERT OR IGNORE INTO frames (username, video_id, frame_id) VALUES (?, ?, ?)",
                        files.loc[new, ["username", "video_id", "frame_id"]].itertuples(index=False))
        con.commit()
        done = files.source.notna() & files.frame_id.map(lambda i: (FRAMES_DIR / f"{i}.jpg").is_file())
        todo = files[~done].sort_values("frame_id")
        if limit is not None:
            todo = todo.head(limit)
        jobs = [(r.username, r.video_id, r.video_path, r.poster_path, FRAMES_DIR / f"{r.frame_id}.jpg")
                for r in todo.itertuples()]
        with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
            for n, (username, video_id, source, error) in enumerate(pool.map(_make_one, jobs, chunksize=16), 1):
                con.execute("UPDATE frames SET source = ?, error = ? WHERE username = ? AND video_id = ?",
                            (source, error, username, video_id))
                if n % 500 == 0 or n == len(jobs):
                    con.commit()
                    if progress:
                        progress(n, len(jobs))
        con.commit()
        rows = pd.read_sql("SELECT username, video_id, frame_id, source, error FROM frames", con)
    return files[["username", "video_id"]].merge(rows, on=["username", "video_id"], how="left")


def frame_paths(splits=data.OPEN_SPLITS) -> pd.Series:
    """The frame of each row of `data.load_videos(splits)`, in the same order: a path, or None if it has none.

    The one way a model reaches the pixels. It returns where the frame is, never where it came from.
    """
    targets = data._read_targets(splits)[["username", "video_id"]]  # load_videos' rows, in its order
    with closing(_open_map()) as con:
        frames = pd.read_sql("SELECT username, video_id, frame_id FROM frames WHERE source IS NOT NULL", con)
    ids = targets.merge(frames, on=["username", "video_id"], how="left").frame_id
    paths = ids.map(lambda i: str(FRAMES_DIR / f"{i}.jpg") if isinstance(i, str) else None)
    return paths.rename("frame_path")


def frame_ids(splits=data.OPEN_SPLITS) -> pd.Series:
    """The opaque frame id of each row of `data.load_videos(splits)`, in the same order (None if it has no frame).

    The key the image embeddings come back under from the rented GPU.
    """
    paths = frame_paths(splits)
    return paths.map(lambda p: Path(p).stem if isinstance(p, str) else None).rename("frame_id")


def frame_vectors(model_name: str, splits=data.OPEN_SPLITS, folder=IMAGE_EMBEDDINGS):
    """The image embedding of each row of `data.load_videos(splits)`, from the GPU job: (vectors, has_vector).

    Rows without a frame, or whose frame has no vector yet, are zeros with `has_vector` False.
    """
    ids = frame_ids(splits)
    cache = text._read_cache(Path(folder) / model_name.replace("/", "--"))  # {frame id: vector}
    if not cache:
        raise FileNotFoundError(f"No vectors for {model_name} in {folder}: run the GPU job (docs/gpu.md)")
    width = len(next(iter(cache.values())))
    vectors = np.zeros((len(ids), width), dtype=np.float32)
    has_vector = np.zeros(len(ids), dtype=bool)
    for row, key in enumerate(ids):
        if isinstance(key, str) and key in cache:
            vectors[row], has_vector[row] = cache[key], True
    return vectors, has_vector


def _open_map() -> sqlite3.Connection:
    FRAME_MAP.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(FRAME_MAP)
    con.execute("""CREATE TABLE IF NOT EXISTS frames (
                       username TEXT NOT NULL, video_id TEXT NOT NULL,
                       frame_id TEXT NOT NULL UNIQUE,
                       source TEXT,  -- 'video' or 'poster': the leak itself, for the checks only
                       error TEXT,
                       PRIMARY KEY (username, video_id))""")
    return con


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Make the frames of the target videos (README trap 3).")
    parser.add_argument("--limit", type=int, default=None, help="make at most this many missing frames")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    start = time.time()
    rows = build(workers=args.workers, limit=args.limit,
                 progress=lambda n, total: print(f"{n:,} / {total:,} frames, {time.time() - start:,.0f} s",
                                                 flush=True))
    print(rows.source.value_counts(dropna=False).to_string())
