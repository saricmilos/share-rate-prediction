"""The laptop's side of the rented-GPU jobs: pack the inputs, send them up, run, bring the vectors back.

The box runs `gpu/embed_job.py` and nothing else of this repo. What goes up is named by opaque ids only:

- **texts**: every distinct non-empty text of the seven stage 3 fields over all target videos, train, validation
  and test, each named by the md5 of its text (the key of `text.embed`'s cache), sorted by that md5;
- **frames**: every target video's frame (`sharerate.frames`), named by its random id, in tars with every file
  date, owner and permission set to the same values, so not even a file's date says whether it was a poster.

No username, video id, creator, split or label goes up, and neither does `data/frames.sqlite`, the map from frame id
to video. Test inputs go up blind with the rest: their vectors are read only when stage 5 opens the test.

    python -m sharerate.gpu pack                                  # writes data/gpu/outbox/
    python -m sharerate.gpu upload --host HOST --port PORT        # the kit and the outbox, to /root/job on the box
    python -m sharerate.gpu run    --host HOST --port PORT text --model Qwen/Qwen3-Embedding-4B --dims 1024
    python -m sharerate.gpu status --host HOST --port PORT
    python -m sharerate.gpu fetch  --host HOST --port PORT        # the vectors, into data/embeddings and
                                                                  # data/image_embeddings
    python -m sharerate.gpu wipe   --host HOST --port PORT        # then destroy the instance

`--host local` runs the same steps against a folder on this disk (data/gpu/box-local), with the job on the CPU:
the end-to-end test before anything is rented. docs/gpu.md is the runbook.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from sharerate import data, frames, text

KIT = data.ROOT / "gpu"  # what runs on the box: embed_job.py and its requirements.txt
WORK = data.ROOT / "data" / "gpu"  # git-ignored, like all of data/
OUTBOX = WORK / "outbox"
INBOX = WORK / "inbox"
LOCAL_BOX = WORK / "box-local"
FRAMES_PER_TAR = 20_000  # about 450 MB a tar: a dropped upload costs one tar, not all of them
REMOTE_FOLDER = "/root/job"


# ---------------------------------------------------------------------------------------------------------------
# Packing: what goes up, on this disk first


def target_texts() -> list[str]:
    """Every distinct non-empty text of the seven fields over all target videos, sorted by md5.

    Read straight from the snapshot without the label or the split, so packing the test's texts opens nothing of
    the test: the texts are cleaned exactly as `data.load_videos(text=True)` cleans them (missing becomes "", then
    stripped), so their md5s are the keys `text.embed` will look up.
    """
    columns = ", ".join(f"v.{source} AS {name}" for name, source in data.TEXT_COLUMNS.items())
    with closing(data._open_snapshot()) as con:
        fields = pd.read_sql(
            f"""SELECT {columns} FROM stage1.targets AS t
                JOIN videos AS v ON v.platform = 'tiktok' AND v.username = t.username AND v.video_id = t.video_id""",
            con)
    found = set()
    for name in data.TEXT_COLUMNS:
        found.update(fields[name].fillna("").astype(str).str.strip())
    found.discard("")  # an empty field is a row of zeros in text.embed, never embedded
    return sorted(found, key=text._md5)


def pack(outbox: Path = OUTBOX, *, sample: int | None = None, seed: int = 2026) -> dict:
    """Write the inputs for the box into `outbox`: texts.jsonl.gz, frames-NN.tar and MANIFEST.json.

    `sample=N` is for the laptop test: N texts and N frames drawn at random from the training and validation
    videos only (whose texts the bge-small cache already holds), instead of everything.
    """
    if outbox.exists():
        shutil.rmtree(outbox)  # an outbox is rebuilt whole, never topped up
    outbox.mkdir(parents=True)
    texts = target_texts() if sample is None else _sample_texts(sample, seed)
    ids = frame_ids_to_send() if sample is None else _sample_frames(sample, seed)
    with gzip.open(outbox / "texts.jsonl.gz", "wt", encoding="utf-8", compresslevel=6) as out:
        for t in sorted(texts, key=text._md5):
            out.write(json.dumps({"id": text._md5(t), "text": t}, ensure_ascii=False) + "\n")
    ids = sorted(ids)  # random ids, so their order says nothing
    for n, start in enumerate(range(0, len(ids), FRAMES_PER_TAR)):
        _write_tar(outbox / f"frames-{n:02d}.tar", ids[start:start + FRAMES_PER_TAR])
    manifest = {
        "packed": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "texts": len(texts), "frames": len(ids), "sample": sample,
        "files": {path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
                  for path in sorted(outbox.iterdir())},
    }
    (outbox / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def frame_ids_to_send() -> list[str]:
    """The id of every frame made (all three splits), and nothing else from the map."""
    with closing(frames._open_map()) as con:
        return [i for (i,) in con.execute("SELECT frame_id FROM frames WHERE source IS NOT NULL")]


def _write_tar(path: Path, ids) -> None:
    """A tar of frames/<id>.jpg as <id>.jpg, every entry with the same date, owner and permissions."""
    with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as tar:
        for frame_id in ids:
            payload = (frames.FRAMES_DIR / f"{frame_id}.jpg").read_bytes()
            info = tarfile.TarInfo(f"{frame_id}.jpg")
            info.size, info.mtime, info.mode = len(payload), 0, 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(payload))


def _sample_texts(n: int, seed: int) -> list[str]:
    videos = data.load_videos(text=True)  # training and validation only
    pool = sorted({t for field in text.FIELDS for t in videos[field] if t})
    rng = np.random.default_rng(seed)
    return [pool[i] for i in rng.choice(len(pool), size=min(n, len(pool)), replace=False)]


def _sample_frames(n: int, seed: int) -> list[str]:
    ids = frames.frame_ids().dropna().tolist()  # training and validation only
    rng = np.random.default_rng(seed)
    return [ids[i] for i in rng.choice(len(ids), size=min(n, len(ids)), replace=False)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# The box: a rented machine over SSH, or a folder on this disk for the test


class Remote:
    """A rented box reached with the ssh and scp that ship with Windows, as root, working in /root/job."""

    def __init__(self, host: str, port: int, user: str = "root", folder: str = REMOTE_FOLDER):
        self.host, self.port, self.user, self.folder = host, port, user, folder
        # accept-new: a fresh box's key is trusted the first time and checked after; keepalives survive idle links
        self.options = ["-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30"]

    def shell(self, command: str, capture: bool = False) -> str:
        # The vast.ai PyTorch template keeps its Python in /venv/main; a non-interactive ssh must activate it.
        script = f"source /venv/main/bin/activate 2>/dev/null; mkdir -p {self.folder} && cd {self.folder} && {command}"
        done = subprocess.run(["ssh", "-p", str(self.port), *self.options, f"{self.user}@{self.host}", script],
                              check=True, text=True, capture_output=capture)
        return done.stdout if capture else ""

    def ready(self) -> None:
        self.shell("true")  # shell() makes the job folder first, so scp has somewhere to write

    def size(self, name: str) -> int | None:
        out = self.shell(f"stat -c %s {shlex.quote(name)} 2>/dev/null || true", capture=True).strip()
        return int(out) if out.isdigit() else None

    def put(self, local: Path, name: str) -> None:
        subprocess.run(["scp", "-P", str(self.port), *self.options, str(local),
                        f"{self.user}@{self.host}:{self.folder}/{name}"], check=True)

    def list(self, pattern_dir: str) -> list[str]:
        out = self.shell(f"find {pattern_dir} -type f \\( -name 'part-*.npz' -o -name runs.jsonl \\) "
                         "! -name '*.tmp.npz' 2>/dev/null || true", capture=True)
        return [line.strip() for line in out.splitlines() if line.strip()]

    def get(self, name: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["scp", "-P", str(self.port), *self.options,
                        f"{self.user}@{self.host}:{self.folder}/{name}", str(local)], check=True)

    def start(self, job_args: list[str], log: str) -> None:
        # setsid nohup: the job keeps running when the ssh link drops (the flywheel runbook's lesson). The job's own
        # ( ... &) matters: a bare `a && b && job &` sends the whole chain to the background, and that background
        # shell holds the ssh link open until the job ends, so `run` would wait hours instead of returning.
        command = (f"mkdir -p logs && python -m pip install -q -r requirements.txt && "
                   f"(setsid nohup python embed_job.py {shlex.join(job_args)} > logs/{log} 2>&1 < /dev/null &)")
        self.shell(command)
        print(f"Started on {self.host}: python embed_job.py {shlex.join(job_args)}  (log: {self.folder}/logs/{log})")

    def status(self) -> None:
        print(self.shell("tail -n 3 logs/*.log 2>/dev/null; echo; "
                         "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu "
                         "--format=csv,noheader 2>/dev/null; echo; "
                         "pgrep -af '[e]mbed_job.py' || echo 'no job running'", capture=True))

    def wipe(self) -> None:
        # Everything uploaded and downloaded: the job folder (texts, frames, vectors) and the model cache
        subprocess.run(["ssh", "-p", str(self.port), *self.options, f"{self.user}@{self.host}",
                        f"rm -rf {self.folder} /root/.cache/huggingface && echo wiped"], check=True)


class LocalBox(Remote):
    """The same steps against a folder on this disk, with the job run on this CPU in the foreground."""

    def __init__(self, folder: Path = LOCAL_BOX):
        self.host, self.root = "local", Path(folder)

    def ready(self):
        self.root.mkdir(parents=True, exist_ok=True)

    def size(self, name):
        path = self.root / name
        return path.stat().st_size if path.exists() else None

    def put(self, local, name):
        self.root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, self.root / name)

    def list(self, pattern_dir):
        base = self.root / pattern_dir
        found = [p for p in base.rglob("*") if p.is_file() and (p.name == "runs.jsonl" or (
            p.name.startswith("part-") and p.name.endswith(".npz") and not p.name.endswith(".tmp.npz")))]
        return [p.relative_to(self.root).as_posix() for p in found]

    def get(self, name, local):
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.root / name, local)

    def start(self, job_args, log):
        (self.root / "logs").mkdir(parents=True, exist_ok=True)
        with open(self.root / "logs" / log, "w", encoding="utf-8") as out:
            subprocess.run([sys.executable, str(self.root / "embed_job.py"), *job_args, "--root", str(self.root)],
                           check=True, stdout=out, stderr=subprocess.STDOUT)
        print((self.root / "logs" / log).read_text(encoding="utf-8").strip().splitlines()[-1])

    def status(self):
        for log in sorted((self.root / "logs").glob("*.log")):
            print(log.name, "|", log.read_text(encoding="utf-8").strip().splitlines()[-1:])

    def wipe(self):
        shutil.rmtree(self.root, ignore_errors=True)
        shutil.rmtree(INBOX / "local", ignore_errors=True)  # the test's parts, so no later fetch can file them
        print("wiped", self.root, "and", INBOX / "local")


def upload(box: Remote, outbox: Path = OUTBOX, what: str = "all") -> None:
    """The kit and the outbox, to the box. A file already there whole is skipped, so a dropped upload resumes.

    `what="texts"` sends the kit and the texts only (51 MB), so the text job can start while the frames (3.3 GB) go up.
    """
    manifest = json.loads((outbox / "MANIFEST.json").read_text())
    sending = [(KIT / "embed_job.py", "embed_job.py"), (KIT / "requirements.txt", "requirements.txt"),
               (outbox / "MANIFEST.json", "MANIFEST.json")]
    sending += [(outbox / name, name) for name in manifest["files"]
                if what == "all" or name.startswith(what)]
    box.ready()
    for local, name in sending:
        always = name in ("embed_job.py", "requirements.txt", "MANIFEST.json")
        if not always and box.size(name) == local.stat().st_size:
            print(f"already there: {name}")
            continue
        print(f"sending {name} ({local.stat().st_size / 1e6:,.1f} MB)", flush=True)
        box.put(local, name)
    # embed_job.py checks every file's sha256 against MANIFEST.json before it starts, so a damaged copy is caught.


def fetch(box: Remote, into: Path = data.ROOT / "data", outbox: Path = OUTBOX) -> pd.DataFrame:
    """Bring every finished part back into this box's inbox, file them where the laptop reads them, count what is back.

    Text parts go to `into`/embeddings/<model>/ (`text.embed`'s cache), image parts to `into`/image_embeddings/<model>/.
    Each box has its own inbox (data/gpu/inbox/<host>/), so the laptop test's parts never reach the real caches.
    Returns one row per job and model: how many vectors came back and how many of the outbox's inputs still have none.
    """
    inbox = INBOX / box.host.replace(":", "_")
    for name in box.list("out"):
        local = inbox / name
        if not local.exists() or name.endswith("runs.jsonl"):  # parts never change once written; the log grows
            box.get(name, local)
    expected = {"text": {text._md5(t) for t in _outbox_texts(outbox)}, "frames": _outbox_frames(outbox)}
    rows = []
    for job, target in (("text", into / "embeddings"), ("frames", into / "image_embeddings")):
        for folder in sorted((inbox / "out" / job).glob("*")):
            destination = target / folder.name
            destination.mkdir(parents=True, exist_ok=True)
            for part in folder.glob("part-*.npz"):
                if not (destination / part.name).exists():
                    shutil.copyfile(part, destination / part.name)
            back = set(text._read_cache(destination))
            rows.append({"job": job, "model": folder.name.replace("--", "/"), "vectors": len(back),
                         "inputs": len(expected[job]), "missing": len(expected[job] - back)})
    return pd.DataFrame(rows)


def _outbox_texts(outbox: Path) -> list[str]:
    with gzip.open(outbox / "texts.jsonl.gz", "rt", encoding="utf-8") as lines:
        return [json.loads(line)["text"] for line in lines]


def _outbox_frames(outbox: Path) -> set[str]:
    ids = set()
    for shard in outbox.glob("frames-*.tar"):
        with tarfile.open(shard) as tar:
            ids.update(Path(name).stem for name in tar.getnames())
    return ids


def _box(args) -> Remote:
    if args.host == "local":
        return LocalBox()
    if args.port is None:
        raise SystemExit("--port is needed for a rented box (the instance card's ssh -p PORT)")
    return Remote(args.host, args.port)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m sharerate.gpu", description=__doc__.split("\n\n")[0])
    steps = parser.add_subparsers(dest="step", required=True)
    packing = steps.add_parser("pack", help="write the inputs into data/gpu/outbox")
    packing.add_argument("--sample", type=int, default=None, help="N texts and N frames, for the laptop test")
    for name in ("upload", "run", "status", "fetch", "wipe"):
        step = steps.add_parser(name)
        step.add_argument("--host", required=True, help="the box's address, or local for the laptop test")
        step.add_argument("--port", type=int, default=None, help="its ssh port")
        if name == "run":
            step.add_argument("job_args", nargs=argparse.REMAINDER, help="embed_job.py's arguments: text|frames ...")
        if name == "upload":
            step.add_argument("--what", choices=["all", "texts", "frames"], default="all")
        if name == "fetch":
            step.add_argument("--into", type=Path, default=data.ROOT / "data",
                              help="where embeddings/ and image_embeddings/ are (default: data/)")
    args = parser.parse_args(argv)
    if args.step == "pack":
        manifest = pack(sample=args.sample)
        size = sum(f["bytes"] for f in manifest["files"].values())
        print(f"{manifest['texts']:,} texts and {manifest['frames']:,} frames, {size / 1e6:,.1f} MB in {OUTBOX}")
    elif args.step == "upload":
        upload(_box(args), what=args.what)
    elif args.step == "run":
        job_args = [a for a in args.job_args if a != "--"]
        value = lambda flag, default: next((job_args[i + 1] for i, a in enumerate(job_args[:-1]) if a == flag), default)
        model = value("--model", "model").replace("/", "--")
        # One log per job, model and shard, so a later run never overwrites an earlier one's log
        _box(args).start(job_args, f"{job_args[0]}-{model}-{value('--shard', '0/1').replace('/', 'of')}.log")
    elif args.step == "status":
        _box(args).status()
    elif args.step == "fetch":
        print(fetch(_box(args), into=args.into).to_string(index=False))
    elif args.step == "wipe":
        _box(args).wipe()
        if args.host != "local":
            print("Now destroy the instance on vast.ai (the trash icon), not stop it: docs/gpu.md, step 7.")


if __name__ == "__main__":
    main()
