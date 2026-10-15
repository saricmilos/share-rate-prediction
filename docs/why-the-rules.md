# Why the rules are what they are

The rules in the README ("Traps to handle before any model") and in each stage are what make the
published number true: the score a model gets here should be the score it would get on next
month's videos. The best result is the highest number that survives the test, not the highest
number a search can find.

## 1. The splits

Stage 1 cut the 143,489 videos of 231 creators in `data/stage1.sqlite`'s `targets` table two
ways: by time (`time_split`) and by creator (`creator_group`). The splits are frozen on the
14 October 2026 snapshot, so counts that arrive later never enter them.

| Period | Training group | Holdout group |
|---|---|---|
| Train, to 2025-10-10 | 74,552 videos, 148 creators | 25,865 videos, 35 creators |
| Validation, to 2026-04-01 | 15,921 videos, 165 creators | 5,411 videos, 38 creators |
| Test, to 2026-09-18 | 17,356 videos, 177 creators | 4,384 videos, 42 creators |

Counted from `targets` on 2026-10-15.

- **Models train on one cell only:** the training group's videos from the training period. The
  holdout group's training-period videos are never trained on. They only feed each video's
  baseline, the median of the creator's last 30 videos posted at least 14 days earlier.
- **Why split by time.** A model is meant to predict a video before it is posted, so it may learn
  only from the past. Share rates drifted: the same creators' median rose about 4.7× between their
  2022 and 2026 videos (stage 1). A model trained on 2026 videos and tested on 2025 ones would
  already know that trend, and its score would mean nothing.
- **Why a holdout group.** It answers whether the model works for a creator it has never seen,
  which is how it would be used on a new channel.
- **"Known creators" needs care.** Only 138 of the 177 training-group creators with test videos
  have videos in the training period. The other 39 started posting later, or reached ten settled
  videos only after it, so the model never saw them either. In validation it is 138 of 165.
  "Creators the model trained on" therefore means creators with training-period videos, not the
  whole training group. The `creator_seen` column holds the three groups: `seen`, `new` and
  `holdout`.
- **Reposts.** Accounts repost each other's clips, so the same clip can sit in training and in
  validation under two names. `has_twin` marks validation and test videos whose clip (the same
  English transcript of at least 8 words, on-screen text and duration) is among the training
  videos: 94 in validation. A text or image model can memorise those, so stages 3 and 4 report
  with and without them.
- **Validation decides everything.** Every choice (the model, the features, the settings) is a
  look at data, so all of them are made on validation.
- **The test is opened once.** Try 30 models and report the best test score, and part of that
  score is the luck of whichever model happened to fit those videos. Opened once, after every
  choice is frozen, the test is an honest estimate of how the model does on videos it has never
  met.

## 2. Leaks

A leak is a feature that would not exist when the prediction is made, or that carries the answer
in disguise. A leaky model looks brilliant and is useless.

- **Counts.** Views are share rate's denominator, and likes, comments and saves arrive along with
  the shares, after posting. Given likes, a model "predicts" shares almost perfectly, but no video
  has likes before it is posted.
- **Age.** `age_days` and `fetched_at` say when the counts were read, not anything about the video,
  and they are tied to the drift.
- **The `files` table.** The platform keeps each channel's top clips by share rate and retires the
  rest, so "this clip still has its video file" means "this clip was a winner". Kept clips beat
  their creator's median 64% of the time, retired ones 43% (stage 1). So `has_video`, `has_poster`,
  `retired_at` and `retire_reason` are never features.
- **The same leak, through the pixels.** Retired clips have only the platform's poster: one frame
  about 1 s in, 480 px, JPEG quality 82. Kept clips have the whole video. Take sharp frames from the
  videos and the posters for the rest, and an image model learns "sharp = kept = shared more". It
  would score by spotting image quality, not content. So every frame comes out of one pipeline,
  made exactly like the posters, and the pipeline is checked against the posters of clips that have
  both.
- **The baseline's own look-ahead.** Stage 1's first baseline used the previous 30 videos, however
  recent. These creators post often, so the median video had 15 of its 30 baseline videos posted
  less than two weeks before it, still collecting shares, with rates read months later. A week in
  which the account got pushed would lift the video and its baseline together, and hide the
  breakout. The baseline now uses only videos posted at least 14 days earlier, whose rate had
  settled: after two weeks it moves a median 1.6%.
- **What a model may see:** only what exists before posting. That is the creator's earlier videos
  (strictly earlier), duration, posting hour and weekday, the caption, the transcript, the on-screen
  text, the hook / body / payoff split, the scene description, and the pixels. The scene
  description and on-screen text were written by a vision-language model, so they are features,
  not facts (trap 5).

## 3. Intervals that resample creators

There are 143,489 videos but only 231 creators, and one creator's videos are alike: the same face,
the same style, the same audience. They carry far less information than 143,489 independent tests
would.

A bootstrap that resamples videos treats them as independent, so its intervals come out far too
narrow and noise looks significant. Resampling creators gives the honest width: draw creators with
replacement and keep all of each one's videos. A gain whose interval still clears 0 is real.

## 4. What the site may show

The videos, frames, captions and transcripts are other creators' work, so they stay on my disk,
whatever the project is for. The site shows numbers, charts and findings, each an aggregate over at
least 5 creators, so no creator can be picked out. That limits the demo, not the result.

Whether a model trained on their videos may run in a visitor's browser is a separate decision. Its
weights reveal little, and a small linear head almost nothing, but they are still made from the
creators' work.

## 5. The GPU

Only the image stage needs one.
- **Time:** running a pretrained image model over about 143,000 frames takes minutes to an hour on
  an RTX 3090, and fine-tuning takes a few hours.
- **Cost:** my GPU runbook's price sheet (July 2026) puts a 3090 box at about $0.13 an hour and a
  5090 at about $0.40, so the whole stage costs a few dollars.
- **No paid debugging:** every job is first tested end to end on 1,000 videos on the laptop's CPU,
  so no GPU hour pays for fixing bugs.
- **Worth paying for:** the 5090 with a larger encoder, for the strongest image model. Money isn't
  the limit here.

## 6. Why the result may be small, and what "best" means

The target, lift, compares each video with the median of its own creator's last 30 videos posted
at least 14 days earlier. The bar is predicting zero lift, which misses by 0.371 log10 on
validation, a factor of 2.35 either way. Whatever a model gains is a share of that error: why one
of a creator's videos does better than the next, the hardest part to predict, and partly luck. So
modest numbers are expected, and a small gain that holds up is a good result.

The baseline's R² of log share rate (0.44 on validation) is not the bar. It depends on how many
years and creators a split spans, which the baseline absorbs by construction. On lift itself,
predicting zero scores an R² of 0.

Writing the bar first protects against fooling myself. Try enough variants and one of them beats
the stage before on validation by chance, then fails on test. So each stage writes down, before its
first run, what counts as beating the stage before (for example, the validation gain's interval
lies above 0). A model that misses the bar is reported as missing it, never tuned until it passes.
"The video adds nothing beyond who made it" is a result worth publishing.

## 7. What would give a better real result

In order:

1. **More creators.** 90,889 videos from 188 creators have no counts (trap 6), and the fetch for
   the 14 October 2026 export brought none back. Their counts could add more than any change of
   model, because the creators are the independent units, not the videos. The splits are frozen,
   so they would enter only a new version of the project, with its own splits and its own test.
2. **Words and pixels together**, combined in stage 5 on top of the best baseline.
3. **Bigger encoders**, on the bigger GPU.

What would not help: models that watch several frames or listen to the audio. Only kept clips have
the whole video, and kept clips are the winners, so such a model would learn from survivors and
bring trap 3 back. One frame per video, made the same way for every clip, is the best honest image
input this data allows.
