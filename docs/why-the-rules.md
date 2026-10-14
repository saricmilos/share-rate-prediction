# Why the rules are what they are

The rules in the README ("Traps to handle before any model") and in each stage are what make the
published number true: the score a model gets here should be the score it would get on next
month's videos. The best result is the highest number that survives the test, not the highest
number a search can find.

## 1. The splits

Stage 1 cut the 146,431 videos of 234 creators in `data/stage1.sqlite`'s `targets` table two
ways: by time (`time_split`) and by creator (`creator_group`).

| Period | Training group | Holdout group |
|---|---|---|
| Train, to 2025-10-10 | 76,044 videos, 151 creators | 26,457 videos, 35 creators |
| Validation, to 2026-04-01 | 16,461 videos, 166 creators | 5,504 videos, 38 creators |
| Test, to 2026-09-18 | 17,544 videos, 178 creators | 4,421 videos, 43 creators |

Counted from `targets` on 2026-10-14.

- **Models train on one cell only:** the training group's videos from the training period. The
  holdout group's training-period videos are never trained on. They only feed each video's
  baseline, the median of the creator's previous 30 videos.
- **Why split by time.** A model is meant to predict a video before it is posted, so it may learn
  only from the past. Share rates drifted: the same creators' median rose about 4.7× between their
  2022 and 2026 videos (stage 1). A model trained on 2026 videos and tested on 2025 ones would
  already know that trend, and its score would mean nothing.
- **Why a holdout group.** It answers whether the model works for a creator it has never seen,
  which is how it would be used on a new channel.
- **"Known creators" needs care.** Only 141 of the 178 training-group creators with test videos
  have videos in the training period. The other 37 started posting later, so the model never saw
  them either. In validation it is 141 of 166. "Creators the model trained on" therefore means
  creators with training-period videos, not the whole training group.
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
  their creator's median 55% of the time, retired ones 43% (stage 1). So `has_video`, `has_poster`,
  `retired_at` and `retire_reason` are never features.
- **The same leak, through the pixels.** Retired clips have only the platform's poster: one frame
  about 1 s in, 480 px, JPEG quality 82. Kept clips have the whole video. Take sharp frames from the
  videos and the posters for the rest, and an image model learns "sharp = kept = shared more". It
  would score by spotting image quality, not content. So every frame comes out of one pipeline,
  made exactly like the posters, and the pipeline is checked against the posters of clips that have
  both.
- **What a model may see:** only what exists before posting. That is the creator's earlier videos
  (strictly earlier), duration, posting hour and weekday, the caption, the transcript, the on-screen
  text, the hook / body / payoff split, the scene description, and the pixels. The scene
  description and on-screen text were written by a vision-language model, so they are features,
  not facts (trap 5).

## 3. Intervals that resample creators

There are 146,431 videos but only 234 creators, and one creator's videos are alike: the same face,
the same style, the same audience. They carry far less information than 146,431 independent tests
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
- **Time:** running a pretrained image model over about 146,000 frames takes minutes to an hour on
  an RTX 3090, and fine-tuning takes a few hours.
- **Cost:** my GPU runbook's price sheet (July 2026) puts a 3090 box at about $0.13 an hour and a
  5090 at about $0.40, so the whole stage costs a few dollars.
- **No paid debugging:** every job is first tested end to end on 1,000 videos on the laptop's CPU,
  so no GPU hour pays for fixing bugs.
- **Worth paying for:** the 5090 with a larger encoder, for the strongest image model. Money isn't
  the limit here.

## 6. Why the result may be small, and what "best" means

The target, lift, compares each video with its own creator's previous 30 videos. Stage 1 reports
that this baseline alone explains 52% of the variance. What is left is why one of a creator's
videos does better than the next: the hardest part to predict, and partly luck. So modest numbers
are expected, and a small gain that holds up is a good result.

Writing the bar first protects against fooling myself. Try enough variants and one of them beats
the stage before on validation by chance, then fails on test. So each stage writes down, before its
first run, what counts as beating the stage before (for example, the validation gain's interval
lies above 0). A model that misses the bar is reported as missing it, never tuned until it passes.
"The video adds nothing beyond who made it" is a result worth publishing.

## 7. What would give a better real result

In order:

1. **More creators.** 90,889 videos from 188 creators have no counts yet (trap 6). Fetching their
   counts could add more than any change of model. More creators help more than more videos,
   because the creators are the independent units.
2. **Words and pixels together**, combined in stage 5 on top of the best baseline.
3. **Bigger encoders**, on the bigger GPU.

What would not help: models that watch several frames or listen to the audio. Only kept clips have
the whole video, and kept clips are the winners, so such a model would learn from survivors and
bring trap 3 back. One frame per video, made the same way for every clip, is the best honest image
input this data allows.
