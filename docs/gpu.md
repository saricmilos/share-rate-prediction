# The rented GPU: stage 4's embedding jobs

Stage 4 needs two sets of embeddings that this laptop can't make in reasonable time:

- **the frames**: one vector per target video's frame, from the pretrained image encoder stage 4 names;
- **stage 3's text job**: Qwen3-Embedding-4B over every distinct text of the seven fields, as notebook 03's hand-off
  writes it (the first 1,024 numbers, the first 256 tokens, unit length, no instruction).

Both run pretrained encoders and train nothing. The kit is `gpu/embed_job.py` (what runs on the box) and
`python -m sharerate.gpu` (the laptop's side). Every step below was run end to end on the laptop's CPU first
(`--host local`, section 6), so no GPU hour pays for a bug.

## 1. What goes up, and what never does

| Goes up | Named by | Size |
|---|---|---|
| `embed_job.py` and `gpu/requirements.txt` | | 16 KB |
| `texts.jsonl.gz`: every distinct non-empty text of the seven fields, over all target videos (train, validation and test) | the md5 of the text | 51 MB (644,955 texts) |
| `frames-00.tar` to `frames-NN.tar`: every target video's frame | a random id | 3.3 GB (143,441 frames in 8 tars) |
| `MANIFEST.json`: each file's size and sha256, and the counts | | 1 KB |

Never: a username, a video id, a creator number, a split, a label, a file date, or `data/frames.sqlite` (the map
from frame id to video). The texts are sorted by their md5 and the frames by their random id, so even the order says
nothing, and every entry in a tar has the same date and owner. The test's texts and frames go up blind with the
rest: nothing on the box knows which they are, and their vectors are read only when stage 5 opens the test.

The texts themselves have to go up to be embedded, and they are other creators' words: transcripts, on-screen text
and captions, and 20,039 of them (3.1%) hold an @mention of an account. They go up unlinked to any video or creator,
and that is why the last step destroys the box (section 5, step 7).

## 2. The card, the disk and the template

From the flywheel's runbooks (`Backend/scripts/remote_gpu/README.md` and `EMBEDDING_README.md`, read-only):

- **GPU: 2× RTX 3090, 24 GB each** (about $0.13 an hour a card in July 2026). The 4B needs 24 GB or more (notebook
  03): its weights take about 16 GB in float32, and the flywheel ran it at batch 64 on this pair, 22 to 24 GB a card
  with texts up to 1,024 tokens. Ours stop at 256 tokens, so they fit with room. One card works too, at twice the time.
- **Template:** the vast.ai PyTorch template (CUDA build of torch already in `/venv/main`).
- **Disk: 60 GB**, set before renting, because it can't grow later: the 4B's weights (8 GB), the image encoder (up to
  4 GB), the frames twice (packed and unpacked, 3.3 GB each), the vectors (3 GB), pip's packages.
- **Offer:** on demand, internet at least 500 Mbps down, reliability at least 99%.
- **SSH:** your public key in vast.ai → Account → SSH Keys. The instance card shows `ssh -p PORT root@HOST`.

## 3. Before renting, on the laptop

```bash
python -m sharerate.frames            # every target video's frame into frames/ (about 80 minutes the first time)
jupyter nbconvert --to notebook --execute --inplace notebooks/04_frames.ipynb   # the frames' checks (about 30 minutes)
python -m sharerate.gpu pack          # data/gpu/outbox/: texts.jsonl.gz, frames-NN.tar, MANIFEST.json
```

`pack` rebuilds the outbox whole. Stage 4 names the image encoder before renting; any Hugging Face image model that
`transformers.AutoModel` loads with a fixed input size works (CLIP, SigLIP and SigLIP 2, DINOv2 were tested).

## 4. The time and the cost

Measured where it says so; the rest is estimated from a measurement named beside it.

| Step | Time | How it was worked out |
|---|---|---|
| Upload the texts | under a minute | 51 MB |
| Upload the frames | about 45 minutes at 10 Mbit/s up, 22 at 20 (not measured) | 3.3 GB, during the text job, so it adds no box time |
| **Text job**, Qwen3-Embedding-4B, float32, both cards | **about 2 hours, at most 5** | The flywheel measured this model on 2× RTX 3090: 126,952 TikTok transcripts in about 58 minutes, 36 texts a second (its server sets no precision, so it ran at the library's default; if that was bfloat16, float32 here is slower, and the upper bound allows for it). Transcripts of that kind average 78 tokens (the snapshot's 169,218 distinct TikTok transcripts, counted with Qwen3's tokenizer); our 644,955 texts average 32 tokens after the cut at 256, 20.8 million in all. At the flywheel's rate in tokens (about 2,800 a second) that is 2.0 hours; at its rate in texts, 4.9 hours, an upper bound since short texts cost less. One card: twice as long. |
| **Frame job**, a base-size encoder (CLIP, SigLIP 2 base) | minutes (not measured on a GPU) | On this laptop's CPU, CLIP ViT-B/32 embedded 1,000 frames in 131 s, 7.6 a second with the model's loading; a GPU runs such an encoder at hundreds of frames a second. A large encoder (SigLIP so400m at 384 px) might take half an hour (a guess). |
| Fetch the vectors | about 5 minutes at 100 Mbit/s down | 2.6 GB of text vectors (644,955 × 1,024 × 4 bytes) and 0.3 to 0.6 GB of frame vectors |
| **The whole rental** | **about 3 to 6 hours** | setup, the text job, the frame job after it, the fetch |
| **Cost** | **about $1 to $2** | 2× RTX 3090 at about $0.13 a card an hour (the flywheel's price sheet, July 2026), plus a few cents of disk and egress (about $1.33 a TB) |

Two things this table says plainly:

- **The text job is the reason to rent.** The 4B is 4.02 billion parameters, 16.1 GB in float32, more than this
  laptop's free memory (31.8 GB in all, 13.4 GB free while it was checked), and the 0.6B, a seventh of its size, ran at
  2.1 texts a second here: the 4B would take weeks.
- **A base-size image encoder doesn't need the GPU.** At 7.6 frames a second, CLIP ViT-B/32 would embed all 143,441
  frames on this laptop in about 5 hours, overnight. The rental is for the 4B, for a large image encoder, and for
  fine-tuning if stage 4's frozen embeddings earn it.

bfloat16 (`--dtype bfloat16`) would run faster on the cards, but it is untested here, and in float32 the whole job
costs about a dollar; stage 3 ran in float32, so only the encoder changes.

## 5. The commands, start to finish

`HOST` and `PORT` are from the instance card. Every command runs in this repo's folder on the laptop.

**Step 1. The texts and the kit go up** (51 MB, a minute or less):

```bash
python -m sharerate.gpu upload --host HOST --port PORT --what texts
```

**Step 2. Check the box's torch.** It must be 2.5 or newer, with CUDA:

```bash
ssh -p PORT root@HOST "source /venv/main/bin/activate; nvidia-smi --query-gpu=name,memory.total --format=csv; python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
```

If it is older, install the build the flywheel validated on 2026-10-03:
`ssh -p PORT root@HOST "source /venv/main/bin/activate; pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130"`.

**Step 3. Start the text job, one half on each card.** `run` installs `gpu/requirements.txt`, then starts the job
detached (`setsid nohup`), so a dropped SSH link doesn't stop it:

```bash
python -m sharerate.gpu run --host HOST --port PORT text --model Qwen/Qwen3-Embedding-4B --dims 1024 --device cuda:0 --shard 0/2
python -m sharerate.gpu run --host HOST --port PORT text --model Qwen/Qwen3-Embedding-4B --dims 1024 --device cuda:1 --shard 1/2
```

On one card: a single `run` with `--device cuda` and no `--shard`. Each text goes to the half its md5 picks, so the
halves never overlap. The job checks `texts.jsonl.gz` against `MANIFEST.json` before it starts, and refuses a damaged
upload.

**Step 4. While it runs, the frames go up** (3.3 GB: about 45 minutes at 10 Mbit/s up, 22 at 20):

```bash
python -m sharerate.gpu upload --host HOST --port PORT --what frames
python -m sharerate.gpu status --host HOST --port PORT      # the jobs' last lines, the cards, what is running
```

An upload that drops is run again: files already there whole are skipped.

**Step 5. The frame job, on the first card that is free** (when its text half is done; `status` shows which).
`ENCODER` is the model stage 4 names, and `cuda:0` becomes `cuda:1` if that card finished first:

```bash
python -m sharerate.gpu run --host HOST --port PORT frames --model ENCODER --device cuda:0
```

**Step 6. The vectors come back:**

```bash
python -m sharerate.gpu fetch --host HOST --port PORT
```

It copies every finished part into `data/gpu/inbox/HOST/`, files the text vectors under
`data/embeddings/Qwen--Qwen3-Embedding-4B/` (where `text.embed(..., model_name="Qwen/Qwen3-Embedding-4B")` reads them
with no change) and the frame vectors under `data/image_embeddings/<encoder>/` (where `frames.frame_vectors` reads
them), and prints, for each job, how many inputs still have no vector. Both must read `missing 0`. If one doesn't,
run that job again (it resumes from its parts) and fetch again. Also copy the run logs:
`data/gpu/inbox/HOST/out/*/*/runs.jsonl` holds each run's card, rows and seconds.

**Step 7. Wipe the box, then destroy it:**

```bash
python -m sharerate.gpu wipe --host HOST --port PORT        # deletes /root/job and the model cache
```

Then, on vast.ai → Instances, **destroy** the instance (the trash icon), never just stop it. A stopped instance keeps
its disk, and with it every text and frame of the creators that went up, and it bills for that disk every hour.
Destroying deletes the disk. The wipe is the belt; destroying is the braces, and the one that counts. Write down the
hours and the cost from the billing page for the README.

## 6. The laptop test (done before any rental)

The same commands with `--host local` run against `data/gpu/box-local/`, with the job on the CPU in the foreground:

```bash
python -m sharerate.gpu pack --sample 1000              # 1,000 texts and 1,000 frames, training and validation only
python -m sharerate.gpu upload --host local
python -m sharerate.gpu run --host local text --model BAAI/bge-small-en-v1.5
python -m sharerate.gpu run --host local frames --model laion/CLIP-ViT-B-32-laion2B-s34B-b79K
python -m sharerate.gpu run --host local text --model Qwen/Qwen3-Embedding-0.6B --dims 1024
python -m sharerate.gpu fetch --host local --into data/gpu/test
python -m sharerate.gpu wipe --host local
```

What it showed (2026-10-16, on this laptop's i7-6820HQ, every input back with `missing 0`):

- **bge-small through the kit gives stage 3's vectors.** The same 1,000 texts, embedded by `embed_job.py` with stage
  3's settings, against `data/embeddings/BAAI--bge-small-en-v1.5/`: the largest difference in any number is
  2.1 × 10⁻⁷ (float32 rounding), 693 of the 1,000 vectors are bit-identical, and the smallest cosine is 0.9999998. So
  the kit changes nothing but the encoder. 61 s with the model's loading.
- **The frame job:** CLIP ViT-B/32 on 1,000 frames, 512 numbers each, all finite, read back through
  `frames.frame_vectors` onto their videos' rows; 131 s. SigLIP 2 (base, 768 numbers) and DINOv2 (small, 384) ran on
  50 frames each, so all three ways the job reads an image vector are tested (CLIP's and SigLIP's
  `get_image_features`, DINOv2's pooled output).
- **The 4B didn't fit, so the 0.6B ran the same code.** Qwen3-Embedding-0.6B with `--dims 1024` on the 1,000 texts:
  1,024 numbers each, every vector of length 1 within 1.2 × 10⁻⁷; 466 s. Its own width is 1,024, so that run cuts
  nothing; to test the cut itself, the same job with `--dims 512` on 100 of the texts gave unit vectors equal, within
  4.5 × 10⁻⁷, to the 1,024-number vectors cut to 512 and normalised again by hand. Before that second normalisation
  the first 512 numbers had length 0.70 to 0.78, so it is what makes them unit length. The 4B's config gives 2,560
  numbers, cut to 1,024 the same way.

Afterwards, `wipe --host local` deletes `data/gpu/box-local/` and the test's inbox, and `data/gpu/test/` can be
deleted: nothing else reads it.

## 7. When something goes wrong

- **`differs from the one packed on the laptop: upload again`**: an upload was cut or damaged. `upload` again; whole
  files are skipped.
- **A job died** (`status` shows no job running and its log ends in an error): fix the cause and `run` the same
  command again. It skips every id already in a part, and loses at most the part it was writing (10,000 rows).
- **CUDA out of memory** on the text job: add `--batch 32` (the default on a GPU is 64).
- **`holds 384-number vectors, not 1024`**: a folder holds one model at one width; a run with other `--dims` needs
  another `--root`.
- **`fetch` shows `missing` above 0 after every job finished**: a part was being written when the job stopped. Run
  the job again; it embeds only what is missing.
