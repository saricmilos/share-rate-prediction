# Will it be shared?

Can you tell from a short video alone, before it is posted, whether people will share it?

## The data

About 247,000 TikTok videos from 320 creators, plus 26,000 YouTube videos from 43, collected for my own
content platform. 156,000 of the TikTok videos have counts, fetched between July and October 2026; the other
91,000 have everything else but no counts (trap 6). For each video:

- **Counts:** views, likes, comments, shares and saves, measured on one or more dates. TikTok reports shares;
  YouTube's API returns none, so the share-rate work is TikTok only.
- **Pixels:** the video file, or for clips the platform has since retired, one poster frame taken 1 s in.
- **Text from my labelling pipeline:** the spoken transcript and its English translation (Whisper), on-screen
  text and a scene description (Qwen2.5-VL), and a hook / body / payoff split of the script.

The videos and frames are other creators' work, so they stay on my disk. This repo publishes code, metrics and
plots only.

## Run it

```bash
python scripts/export_snapshot.py               # standard library only
py -3.11 -m venv .venv                          # then, for the analysis:
.venv/Scripts/activate                          # Windows; on macOS or Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
jupyter lab
```

It reads the platform's databases read-only and writes `data/snapshot.sqlite` (about 690 MB, not in git):

| Table | One row per | Holds |
|---|---|---|
| `videos` | video | counts, caption, published and fetched dates, the text fields; `has_stats = 0` when there are no counts |
| `snapshots` | video and date | the same counts measured again later |
| `files` | video | where the clip and its poster frame are, and whether they still exist |
| `stat_runs` | creator | the last counts fetch: when, how many videos it saw and how many it matched |
| `export_info` | key | when and from where the snapshot was made |

Notebook 03 needs two things the others don't. The CPU build of torch goes in before the requirements, so pip doesn't
fetch the much larger CUDA build. bge-small, the sentence encoder, is downloaded once by the `python -c` line (about
130 MB), because the notebook only ever reads it from disk. The notebook's first run then embeds every distinct text of
the training and validation videos once, 550,823 of them, which takes about 3.5 hours on a laptop CPU. The vectors go
into `data/embeddings/` (about 820 MB, not in git). A run that stops keeps what it finished, and later runs only read
the cache.

Notebook 04_image reads embeddings made on a rented GPU: [docs/gpu.md](docs/gpu.md) has the kit, the commands and what
they cost (about 3 hours and $1 on 2× RTX 3090). The vectors go into `data/embeddings/` and `data/image_embeddings/`
(about 3 GB, not in git).

## Traps to handle before any model

1. **Channel size.** Big channels get big numbers. The target is share rate (shares per 1,000 views) compared
   with the creator's own median, not raw shares.
2. **Video age.** A clip fetched two days after posting hasn't finished collecting shares. `published_at` and
   `fetched_at` give its age; `snapshots` shows how much the counts still move.
3. **Survivorship.** The platform retires old clips but keeps each channel's top clips by share rate, so whether
   a file still exists depends on the target (stage 1 measured it: kept clips beat their creator's median 64% of
   the time, retired ones 43%). The `files` columns are for finding pixels, never features, and a model trained
   only on clips that still have their file learns from winners. Every image must also come out of one pipeline:
   retired clips have only a poster (taken about 1 s in, 480 px, JPEG quality 82), so kept clips' frames must be
   made the same way, or compression artefacts give away which clips were retired.
4. **Leaky splits.** A random split puts one creator's near-identical videos on both sides. Split by time (train
   on older, test on newer), and test separately on creators the model never saw.
5. **Model-made labels.** The scene descriptions and on-screen text came from a vision-language model, not from
   people. They are features, not ground truth.
6. **Missing counts.** 90,889 TikTok videos from 188 creators have no counts. Almost all of it is per creator,
   not per video: 33 creators were never fetched (27,373 videos), the last fetch for 107 returned nothing (47,420),
   and for 48 it returned only part of the channel (16,096). The counts were fetched again for the 14 October
   2026 export, and none of the missing ones came back.
   - **The 140 creators without any counts stay out,** so every result holds for creators like the 231 that have
     a target, not for the whole library.
   - **The splits are final.** They are frozen on that snapshot, and counts that arrive later never enter them.

[docs/why-the-rules.md](docs/why-the-rules.md) explains why each of these rules exists, with the splits'
counts, and what would give a better result.

## Plan

Every model has to beat the one above it, on the same split.

| Stage | What | Status |
|---|---|---|
| 0. Data | Read-only snapshot out of the platform | Done |
| 1. EDA and target | Share-rate distribution, per-creator normalisation, the age effect, the survivorship check | Done: [notebook](notebooks/01_eda_target.ipynb) |
| 2. Baselines | The creator's own history; length and posting hour | Done: [notebook](notebooks/02_baselines.ipynb). Only metadata (in effect, length) beats zero lift, by 0.2% |
| 3. Text | The caption, transcript, on-screen text and scene description, as counted words and as sentence embeddings | Done: [notebook](notebooks/03_text.ipynb). The words add to the baseline: sentence embeddings cut its error by 2.2% |
| 4. Image | The first frame through a pretrained image model (SigLIP 2), and a larger text encoder (Qwen3-Embedding-4B), both on a rented GPU; fine-tuning only if the frozen frame model earns it | Done: [frames](notebooks/04_frames.ipynb), [models](notebooks/04_image.ipynb). The frame adds nothing beyond the creator and the words; the larger text encoder cuts the error by 2.6% |
| 5. Combined | The baseline, the words and the creator's recent form together, then the test split, once | |
| 6. Demo | A FastAPI `/predict` endpoint in Docker | |

## What stage 1 found

- **The creator matters more than the video.** The creator alone explains 38% of the variance in log share
  rate, so every video is compared with its own creator.
- **Share rate drifted.** The same creators' median share rate rose about 4.7× between their 2022 and 2026
  videos. The baseline is therefore the median of the creator's last 30 videos, not their lifetime.
- **The rate settles fast.** Young videos keep gaining views, but their share rate barely moves: a video first read
  at 15 to 30 days old moved a median 1.6% afterwards (932 videos from 64 creators). So the baseline uses only
  videos posted at least 14 days earlier, whose rate had settled when the video went up.
- **The target:** `lift` = log share rate minus that baseline, for 143,489 videos from 231 creators. `breakout`
  (at least twice the baseline) is true for 24%. The target's distribution is the same in train, validation and
  test.
- **The bar.** Predicting zero lift ("every video does as well as its creator's recent median") misses by 0.371
  log10 on validation, a factor of 2.35 either way. Every model has to shrink that error, and is reported for
  creators it trained on, creators who started later, and creators it never saw.

## What stage 2 found

All on validation, each gain with a 95% interval from 2,000 resamples of creators. The test stays closed until
stage 5.

- **Only one baseline beats the bar, barely.** Zero lift misses by 0.369 log10 for creators the model trained on.
  Ridge regression on metadata cuts that by 0.0007 [0.0002, 0.0011], an error 0.2% smaller: real, and too small to
  help anyone choose what to post.
- **That gain is the video's length.** Leave duration out and it's gone. Shorter videos do slightly better against
  their creator's median: a video twice as long does 3.6% worse. The posting hour adds nothing, in UTC or measured
  from the creator's usual hour, and neither do the weekday or the gap since the previous video.
- **Recent form ranks breakouts but misses the bar.** A weighted mean of the creator's recent settled videos doesn't
  shrink the typical error, but of each creator's top tenth by it, 32% broke out, against 25%. The medians of the
  last 5 and 10 videos do worse than zero lift.
- **A creator's past lift doesn't carry over.** Any trust in a creator's average lift made the next prediction
  worse.
- **For creators it never saw,** nothing passes in both unseen groups: the gains are under 0.002 and depend on the
  model.
- **What a leak looks like:** whether a clip's file still exists cuts the error by 0.019, against 0.0007 for the
  honest winner, because the platform keeps each channel's top clips.

## What stage 3 found

All on validation, against stage 2's ridge on metadata, each gain with a 95% interval from 2,000 resamples of
creators. The test stays closed until stage 5.

- **The words add to the baseline, and the gain holds.** For creators the model trained on, the metadata ridge misses
  by 0.3681 log10. Sentence embeddings (bge-small) of seven text fields (the caption, the transcript and its hook,
  body and payoff, on-screen text, the scene description), on top of it, cut that by 0.0082 [0.0051, 0.0117]: an
  error 2.2% smaller. TF-IDF of the words at least 20 creators use cuts it by 0.0060 [0.0033, 0.0085]. Both hold
  without the 73 reposts, and both clear 0 for new and for held-out creators too, which nothing in stage 2 did.
  Small, and about twelve times what metadata added over zero lift.
- **Meaning beats counted words, by a little.** The embeddings beat TF-IDF by 0.0023 [0.0004, 0.0043] in the error,
  and in every field alone, but not at ranking breakouts (ROC-AUC). Under the bar written before the run, that earns
  a larger encoder a run on stage 4's rented GPU.
- **The words rank videos too.** Inside a creator, the Spearman correlation rises from the baseline's 0.071 to 0.218
  with the embeddings, above recent form's 0.146. Of each creator's top tenth by the TF-IDF model, 36.8% broke out,
  against 29.0% for the baseline and 24.7% for zero lift. Recent form still ranks breakouts best (ROC-AUC 0.588), so
  the two signals differ.
- **Where the signal is.** Alone, on-screen text (+0.0051) and the scene description (+0.0049) are the strongest
  fields, and a vision-language model wrote both (trap 5): the creator's words read off the frames, and the model's
  reading of the picture. The six word fields without the scene description get most of the gain (+0.0071 of
  +0.0082), and the scene description adds about 0.001 on top. Inside a creator, a video with text on screen beats
  one without by 15%.
- **What the words say.** Share prompts and friendship push the prediction up ("send", "this to", "bestie",
  "friend"). Ads push it down most ("ad", −27%; "code"), and so do borrowed formats: duets, challenges, trends,
  filters. Two strong terms mark a language (Serbian), not a subject.
- **Not a leak.** Given a clip's lift, neither model's correction knows whether the platform kept the clip (TF-IDF
  −0.0025 [−0.0097, +0.0047]; the embeddings slightly below 0, the opposite of a detector). Both also rank videos
  among kept clips alone and among retired clips alone.

## What stage 4 found

All on validation, each gain with a 95% interval from 2,000 resamples of creators. The test stays closed until
stage 5. One rented GPU (2× RTX 3090) made the embeddings in about 3.2 hours, for about $1 (estimated; the bill wasn't
read).

- **The first frame adds nothing beyond who made the video and what it says.** On top of the metadata ridge and the
  text, the frame's embedding (SigLIP 2, frozen) changes the error by +0.0003 [−0.0002, +0.0008] in ridge and +0.0001
  [−0.0007, +0.0010] in a small boosted model, and doesn't rank a creator's videos better either. Under the bar written
  before the rental that's a miss, so the image model wasn't fine-tuned and the combined model leaves the pixels out.
- **The frame knows whose video it is.** From the first frame alone, a linear model names the creator of a later video
  65% of the time (top five: 88%), against 9% for always guessing the biggest creator. Lift is already measured
  against the creator, so that knowledge is worth nothing here. Stage 3's picture signal, the scene description, was
  mostly in the words already.
- **A larger text encoder helps, by a little.** Qwen3-Embedding-4B in place of bge-small cuts the metadata ridge's
  error by 0.0097 [0.0062, 0.0133], 2.6% (bge-small: 2.2%). Its edge over bge-small, 0.0015 [0.0002, 0.0027], holds
  without the reposts, so under the bar written in stage 3 it replaces bge-small: over a hundred times the parameters,
  for 0.4% of the error and no better ranking.
- **Not a leak.** Every frame came out of the platform's own poster code (a classifier can't tell a frame from a
  poster: ROC-AUC 0.500), and given a clip's lift, no image model's correction knows whether the platform kept the
  clip.
