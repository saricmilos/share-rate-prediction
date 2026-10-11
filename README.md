# Will it be shared?

Can you tell from a short video alone, before it is posted, whether people will share it?

## The data

About 156,000 TikTok videos from 246 creators, plus 25,000 YouTube videos from 43, collected for my own
content platform. The counts were fetched between July and October 2026. For each video:

- **Counts:** views, likes, comments, shares and saves, measured on one or more dates. TikTok reports shares;
  YouTube's API returns none, so the share-rate work is TikTok only.
- **Pixels:** the video file, or for clips the platform has since retired, one poster frame taken 1 s in.
- **Text from my labelling pipeline:** the spoken transcript and its English translation (Whisper), on-screen
  text and a scene description (Qwen2.5-VL), and a hook / body / payoff split of the script.

The videos and frames are other creators' work, so they stay on my disk. This repo publishes code, metrics and
plots only.

## Run it

```bash
python scripts/export_snapshot.py
```

It reads the platform's databases read-only and writes `data/snapshot.sqlite` (about 460 MB, not in git):

| Table | One row per | Holds |
|---|---|---|
| `videos` | video | counts, caption, published and fetched dates, the text fields |
| `snapshots` | video and date | the same counts measured again later |
| `files` | video | where the clip and its poster frame are, and whether they still exist |
| `export_info` | key | when and from where the snapshot was made |

## Traps to handle before any model

1. **Channel size.** Big channels get big numbers. The target is share rate (shares per 1,000 views) compared
   with the creator's own median, not raw shares.
2. **Video age.** A clip fetched two days after posting hasn't finished collecting shares. `published_at` and
   `fetched_at` give its age; `snapshots` shows how much the counts still move.
3. **Survivorship.** The platform retires old clips but keeps each channel's top clips by share rate, so whether
   a file still exists depends on the target. The `files` columns are for finding pixels, never features, and a
   model trained only on clips that still have their file learns from winners.
4. **Leaky splits.** A random split puts one creator's near-identical videos on both sides. Split by time (train
   on older, test on newer), and test separately on creators the model never saw.
5. **Model-made labels.** The scene descriptions and on-screen text came from a vision-language model, not from
   people. They are features, not ground truth.

## Plan

Every model has to beat the one above it, on the same split.

| Stage | What | Status |
|---|---|---|
| 0. Data | Read-only snapshot out of the platform | Done |
| 1. EDA and target | Share-rate distribution, per-creator normalisation, the age effect, the survivorship check | |
| 2. Baselines | The creator's own history; length and posting hour | |
| 3. Text | Transcript and on-screen text | |
| 4. Image | The poster frame through a pretrained image model, then fine-tuned on a rented GPU | |
| 5. Combined | Frames and text together, every run logged in MLflow | |
| 6. Demo | A FastAPI `/predict` endpoint in Docker | |
