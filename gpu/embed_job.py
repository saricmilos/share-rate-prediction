"""Embeddings on a rented GPU box, for inputs named by opaque ids only.

This file runs on the box (and on the laptop's CPU, to test it before anything is rented). It runs pretrained
encoders and trains nothing. It needs nothing from the project: what reaches the box is texts named by the md5 of
their text and frames named by a random id, with no creator, video, split or label, so nothing on the box knows
which inputs are the test's.

Two jobs:

    python embed_job.py text   --model Qwen/Qwen3-Embedding-4B --dims 1024
    python embed_job.py frames --model google/siglip2-base-patch16-224

Inputs, in the job folder (--root, by default this file's folder), checked against MANIFEST.json first:

    texts.jsonl.gz      one {"id": <md5 of the text>, "text": ...} per line
    frames-NN.tar       frames/<random id>.jpg, unpacked into frames/ on the first run

Output, under out/<job>/<model, with "/" written "--">/:

    part-<random>.npz   `keys` (the 32-character ids, as bytes) and `vectors` (float32), one row per input
    runs.jsonl          one line per run: the settings, the library versions, the card, how many rows, how long

The parts are in the format of the laptop's `sharerate.text.embed` cache, so the text vectors drop into
data/embeddings/ and `text.embed(..., model_name=...)` reads them with no change. A run that stops resumes: ids
already in a part are skipped, and parts are named at random, so two runs (two cards, `--shard`) never collide.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import tarfile
import time
import uuid
from pathlib import Path

import numpy as np

PART_ROWS = 10_000  # rows per part: a part lost to a crash costs at most this many


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("job", choices=["text", "frames"])
    parser.add_argument("--model", required=True, help="a Hugging Face model id")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help="the job folder")
    parser.add_argument("--device", default=None, help="cuda, cuda:1 or cpu (default: cuda if there is one)")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"],
                        help="float32, the precision stage 3 ran in; bfloat16 is faster on a GPU, and untested")
    parser.add_argument("--batch", type=int, default=None, help="inputs per forward pass")
    parser.add_argument("--shard", default="0/1", help="i/n: this run takes only the ids whose hash mod n is i")
    parser.add_argument("--limit", type=int, default=None, help="at most this many inputs (for tests)")
    # The text job's settings. The defaults are stage 3's: only the encoder changes.
    parser.add_argument("--dims", type=int, default=None, help="keep the first DIMS numbers (Matryoshka cut)")
    parser.add_argument("--max-tokens", type=int, default=256, help="each text is cut to its first MAX_TOKENS")
    # The frames job's settings
    parser.add_argument("--workers", type=int, default=None, help="processes decoding JPEGs (default: up to 8)")
    args = parser.parse_args(argv)

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    if device == "cpu":
        torch.set_num_threads(os.cpu_count())  # every logical core, as the laptop's text.embed does
    check_manifest(args.root, args.job)
    out = args.root / "out" / args.job / args.model.replace("/", "--")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.job == "text":
        rows = run_text(args, out, device, dtype)
    else:
        rows = run_frames(args, out, device, dtype)
    seconds = time.time() - started
    record_run(out, args, device, dtype, rows, seconds)
    print(f"done: {rows:,} new vectors in {seconds:,.0f} s ({rows / max(seconds, 1e-9):,.1f} a second)", flush=True)


def run_text(args, out, device, dtype) -> int:
    """Embed every text not embedded yet: the first `max_tokens` tokens, no instruction, cut to `dims`, unit length."""
    import torch
    from sentence_transformers import SentenceTransformer

    todo = remaining(read_texts(args.root / "texts.jsonl.gz"), out, args.shard, args.limit)
    print(f"text: {len(todo):,} texts to embed with {args.model}", flush=True)
    if not todo:
        return 0
    # The weights come from the Hugging Face hub (public, no token). No remote code is run.
    model = SentenceTransformer(args.model, device=device, model_kwargs={"dtype": dtype})
    model.max_seq_length = args.max_tokens
    # No instruction: Qwen3's are for search queries, and stage 3's bge-small had none either.
    if model.default_prompt_name is not None:
        raise RuntimeError(f"{args.model} adds the prompt {model.default_prompt_name!r} by default")
    width = model.get_embedding_dimension()
    if args.dims is not None and args.dims > width:
        raise ValueError(f"{args.model} gives {width} numbers, fewer than --dims {args.dims}")
    batch = args.batch or (16 if device == "cpu" else 64)
    # Shortest first: the batches hold texts of about one length, so little is spent on padding. A run that
    # resumes takes up where the order left off.
    todo.sort(key=lambda row: (len(row[1]), row[0]))
    written = 0
    for start in range(0, len(todo), PART_ROWS):
        chunk = todo[start:start + PART_ROWS]
        vectors = model.encode([text for _, text in chunk], batch_size=batch, truncate_dim=args.dims,
                               normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        vectors = np.asarray(vectors, dtype=np.float32)
        if dtype != torch.float32:
            # Normalised in 16 bits, a vector's length is off by up to about 0.4%; normalise again in 32.
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        write_part(out, [key for key, _ in chunk], vectors)
        written += len(chunk)
        print(f"text: {written:,} / {len(todo):,}", flush=True)
    return written


def run_frames(args, out, device, dtype) -> int:
    """One vector per frame from a pretrained image encoder: its image features, as the model gives them (raw)."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoImageProcessor, AutoModel

    folder = unpack_frames(args.root)
    every = [(path.stem, path) for path in sorted(folder.glob("*.jpg"))]
    todo = remaining(every, out, args.shard, args.limit)
    print(f"frames: {len(todo):,} frames to embed with {args.model}", flush=True)
    if not todo:
        return 0
    model = AutoModel.from_pretrained(args.model, dtype=dtype).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(args.model)
    batch = args.batch or (32 if device == "cpu" else 256)
    workers = args.workers if args.workers is not None else min(8, os.cpu_count() or 1)
    loader = DataLoader(Frames(todo), batch_size=batch, num_workers=workers, collate_fn=Collate(processor))
    keys, found, written = [], [], 0
    with torch.inference_mode():
        for batch_keys, inputs in loader:
            inputs = {name: (value.to(device, dtype) if value.is_floating_point() else value.to(device))
                      for name, value in inputs.items()}
            keys.extend(batch_keys)
            found.append(image_features(model, inputs).float().cpu().numpy())
            if (written + len(keys)) // 1_000 > (written + len(keys) - len(batch_keys)) // 1_000:
                print(f"frames: {written + len(keys):,} / {len(todo):,}", flush=True)
            if len(keys) >= PART_ROWS:
                write_part(out, keys, np.concatenate(found))
                written += len(keys)
                keys, found = [], []
    if keys:
        write_part(out, keys, np.concatenate(found))
        written += len(keys)
    return written


def image_features(model, inputs):
    """The encoder's image vector: `get_image_features` for CLIP and SigLIP, the pooled output for the rest."""
    import torch

    if hasattr(model, "get_image_features"):
        output = model.get_image_features(**inputs)
    else:  # an image-only model, such as DINOv2
        output = model(**inputs)
    if isinstance(output, torch.Tensor):
        return output
    if getattr(output, "pooler_output", None) is not None:
        return output.pooler_output
    return output.last_hidden_state[:, 0]  # the class token


class Frames:
    """The frames to embed, as (id, image) pairs. Decoded in the loader's worker processes."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from PIL import Image

        key, path = self.rows[i]
        with Image.open(path) as image:
            return key, image.convert("RGB")


class Collate:
    """Turns a list of (id, image) into the ids and the encoder's own preprocessing of the images."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        keys = [key for key, _ in batch]
        return keys, dict(self.processor(images=[image for _, image in batch], return_tensors="pt"))


def read_texts(path: Path) -> list[tuple[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8") as lines:
        return [(row["id"], row["text"]) for row in map(json.loads, lines)]


def remaining(rows, out: Path, shard: str, limit) -> list:
    """The rows of this shard whose id has no vector in `out` yet, at most `limit` of them."""
    index, count = (int(n) for n in shard.split("/"))
    done = done_keys(out)
    # The shard is decided by the id's own hash, so the same id always goes to the same shard.
    todo = [row for row in rows
            if row[0] not in done and int(hashlib.md5(row[0].encode()).hexdigest()[:8], 16) % count == index]
    return todo[:limit] if limit is not None else todo


def done_keys(out: Path) -> set[str]:
    keys = set()
    for path in out.glob("part-*.npz"):
        if path.name.endswith(".tmp.npz"):  # left by an interrupted write; its rows are redone
            continue
        with np.load(path) as part:
            keys.update(key.decode() for key in part["keys"])
    return keys


def write_part(out: Path, keys, vectors: np.ndarray) -> None:
    """One part, written under a temporary name and renamed, so a crash never leaves half a part."""
    if len(keys) != len(vectors):
        raise ValueError(f"{len(keys)} keys for {len(vectors)} vectors")
    for path in out.glob("part-*.npz"):  # one folder, one width: a run with other --dims must not mix into it
        if not path.name.endswith(".tmp.npz"):
            with np.load(path) as part:
                if part["vectors"].shape[1] != vectors.shape[1]:
                    raise ValueError(f"{out} holds {part['vectors'].shape[1]}-number vectors, not "
                                     f"{vectors.shape[1]}: use another --root for another --dims")
            break
    name = f"part-{uuid.uuid4().hex}"
    np.savez(out / f"{name}.tmp.npz", keys=np.array(keys, dtype="S32"), vectors=vectors.astype(np.float32))
    os.replace(out / f"{name}.tmp.npz", out / f"{name}.npz")


def unpack_frames(root: Path) -> Path:
    """Unpack every frames-NN.tar into frames/, once."""
    folder = root / "frames"
    folder.mkdir(exist_ok=True)
    for shard in sorted(root.glob("frames-*.tar")):
        marker = folder / f".{shard.stem}.unpacked"
        if marker.exists():
            continue
        with tarfile.open(shard) as tar:
            for member in tar.getmembers():
                # Only plain files named <32 hex characters>.jpg, written flat into frames/: nothing else.
                name = Path(member.name).name
                if not (member.isfile() and len(name) == 36 and name.endswith(".jpg")):
                    raise ValueError(f"{shard.name} holds an unexpected entry: {member.name!r}")
                with tar.extractfile(member) as source:
                    (folder / name).write_bytes(source.read())
        marker.touch()
    return folder


def check_manifest(root: Path, job: str | None = None) -> None:
    """Refuse to start on an input that didn't arrive whole: each file's size and sha256 against MANIFEST.json.

    With `job`, only that job's inputs are checked (texts.jsonl.gz, or the frames-NN.tar), so the text job can start
    before the frames have finished uploading.
    """
    manifest = json.loads((root / "MANIFEST.json").read_text())
    prefix = {"text": "texts", "frames": "frames", None: ""}[job]
    for name, expected in manifest["files"].items():
        if not name.startswith(prefix):
            continue
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"{name} is in MANIFEST.json but not in {root}: upload again")
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"{name} differs from the one packed on the laptop: upload again")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record_run(out: Path, args, device, dtype, rows: int, seconds: float) -> None:
    import torch
    import transformers

    try:
        import sentence_transformers
        st_version = sentence_transformers.__version__
    except ImportError:
        st_version = None
    run = {
        "job": args.job, "model": args.model, "dims": args.dims, "max_tokens": args.max_tokens if args.job == "text"
        else None, "device": device, "dtype": str(dtype).replace("torch.", ""), "shard": args.shard,
        "limit": args.limit, "rows": rows, "seconds": round(seconds, 1),
        "card": torch.cuda.get_device_name(device) if device.startswith("cuda") else platform.processor(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__, "sentence_transformers": st_version},
        "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(out / "runs.jsonl", "a", encoding="utf-8") as log:
        log.write(json.dumps(run) + "\n")


if __name__ == "__main__":
    main()
