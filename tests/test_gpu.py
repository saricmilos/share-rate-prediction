"""The GPU kit: what goes up is named by opaque ids only, arrives whole, and comes back in the cache's format."""

import gzip
import importlib.util
import io
import json
import tarfile

import numpy as np
import pytest

from sharerate import data, gpu, text

# The box's script is not part of the package (nothing of the package goes to the box): load it from its file.
_spec = importlib.util.spec_from_file_location("embed_job", data.ROOT / "gpu" / "embed_job.py")
embed_job = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(embed_job)

HAS_DATA = data.STAGE1.exists() and data.SNAPSHOT.exists()


def test_the_shards_split_the_inputs_without_overlap(tmp_path):
    rows = [(text._md5(str(i)), str(i)) for i in range(1_000)]
    first = embed_job.remaining(rows, tmp_path, "0/2", None)
    second = embed_job.remaining(rows, tmp_path, "1/2", None)
    assert not {k for k, _ in first} & {k for k, _ in second}
    assert len(first) + len(second) == 1_000 and 400 < len(first) < 600


def test_the_box_parts_are_the_laptop_cache_format(tmp_path):
    # A text embedded on the box is read back by text.embed with no encoder: the kit only adds vectors to its cache.
    texts = ["first text", "second text", "a third"]
    vectors = np.random.default_rng(0).normal(size=(3, 8)).astype(np.float32)
    folder = tmp_path / "some--model"
    folder.mkdir()
    embed_job.write_part(folder, [text._md5(t) for t in texts], vectors)

    def no_encoder(batch):
        raise AssertionError("everything should come from the cache")

    found = text.embed(["a third", "", "first text"], "some/model", encoder=no_encoder, cache_dir=tmp_path)
    assert np.array_equal(found, np.stack([vectors[2], np.zeros(8, np.float32), vectors[0]]))


def test_a_run_resumes_where_the_parts_end(tmp_path):
    rows = [(text._md5(str(i)), str(i)) for i in range(10)]
    embed_job.write_part(tmp_path, [k for k, _ in rows[:4]], np.zeros((4, 2), np.float32))
    assert embed_job.remaining(rows, tmp_path, "0/1", None) == rows[4:]


def test_a_damaged_upload_is_refused(tmp_path):
    (tmp_path / "texts.jsonl.gz").write_bytes(b"the packed bytes")
    manifest = {"files": {"texts.jsonl.gz": {"bytes": 16, "sha256": embed_job.sha256(tmp_path / "texts.jsonl.gz")}}}
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
    embed_job.check_manifest(tmp_path)  # whole: no error
    (tmp_path / "texts.jsonl.gz").write_bytes(b"the packed byteS")  # same size, one byte off
    with pytest.raises(ValueError, match="upload again"):
        embed_job.check_manifest(tmp_path)


def test_the_frame_tars_say_nothing_but_the_id(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu.frames, "FRAMES_DIR", tmp_path)
    ids = ["0123456789abcdef0123456789abcdef", "fedcba9876543210fedcba9876543210"]
    for i in ids:
        (tmp_path / f"{i}.jpg").write_bytes(b"\xff\xd8 not really a jpeg")
    gpu._write_tar(tmp_path / "frames-00.tar", ids)
    with tarfile.open(tmp_path / "frames-00.tar") as tar:
        members = tar.getmembers()
    assert [m.name for m in members] == [f"{i}.jpg" for i in ids]
    # Every entry has the same date, owner and permissions, so none says whether it was a poster or when.
    assert {(m.mtime, m.uid, m.gid, m.uname, m.gname, m.mode) for m in members} == {(0, 0, 0, "", "", 0o644)}
    unpacked = embed_job.unpack_frames(tmp_path)
    assert sorted(p.stem for p in unpacked.glob("*.jpg")) == ids


def test_an_unexpected_entry_in_a_tar_is_refused(tmp_path):
    with tarfile.open(tmp_path / "frames-00.tar", "w") as tar:
        info = tarfile.TarInfo("../outside.jpg")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"abc"))
    with pytest.raises(ValueError, match="unexpected entry"):
        embed_job.unpack_frames(tmp_path)


@pytest.mark.skipif(not HAS_DATA, reason="data/ is not in git")
def test_the_packed_texts_are_the_ones_text_embed_will_look_up():
    # Packing reads no label or split, yet must clean the texts exactly as the loader does, or the md5 keys miss.
    packed = set(gpu.target_texts())
    videos = data.load_videos(data.SPLITS, text=True)
    loaded = {t for field in text.FIELDS for t in videos[field] if t}
    assert packed == loaded


def test_a_packed_text_line_holds_only_the_id_and_the_text(tmp_path):
    path = tmp_path / "texts.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as out:
        out.write(json.dumps({"id": text._md5("hello"), "text": "hello"}) + "\n")
    assert embed_job.read_texts(path) == [(text._md5("hello"), "hello")]


def test_a_round_trip_through_a_local_box(tmp_path, monkeypatch):
    # The laptop's side, without a model: texts go up first, the box's parts come back into the cache folders.
    monkeypatch.setattr(gpu, "INBOX", tmp_path / "inbox")
    monkeypatch.setattr(gpu.frames, "FRAMES_DIR", tmp_path / "frames")
    (tmp_path / "frames").mkdir()
    frame_id = "0123456789abcdef0123456789abcdef"
    (tmp_path / "frames" / f"{frame_id}.jpg").write_bytes(b"\xff\xd8 a frame")
    outbox = tmp_path / "outbox"
    outbox.mkdir()
    texts = ["one text", "another"]
    with gzip.open(outbox / "texts.jsonl.gz", "wt", encoding="utf-8") as out:
        for t in texts:
            out.write(json.dumps({"id": text._md5(t), "text": t}) + "\n")
    gpu._write_tar(outbox / "frames-00.tar", [frame_id])
    files = {p.name: {"bytes": p.stat().st_size, "sha256": gpu._sha256(p)} for p in outbox.iterdir()}
    (outbox / "MANIFEST.json").write_text(json.dumps({"files": files}))

    box = gpu.LocalBox(tmp_path / "box")
    gpu.upload(box, outbox, what="texts")
    assert sorted(p.name for p in (tmp_path / "box").iterdir()) == ["MANIFEST.json", "embed_job.py",
                                                                    "requirements.txt", "texts.jsonl.gz"]
    embed_job.check_manifest(tmp_path / "box", "text")  # the text job can start before the frames are up

    done = tmp_path / "box" / "out" / "text" / "some--model"
    done.mkdir(parents=True)
    embed_job.write_part(done, [text._md5(t) for t in texts], np.ones((2, 4), np.float32))
    back = gpu.fetch(box, into=tmp_path / "into", outbox=outbox)
    assert back.to_dict("records") == [{"job": "text", "model": "some/model", "vectors": 2, "inputs": 2, "missing": 0}]
    assert len(list((tmp_path / "into" / "embeddings" / "some--model").glob("part-*.npz"))) == 1
