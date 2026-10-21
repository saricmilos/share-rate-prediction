# Stage 7: the live test, on my own reels

Stages 1 to 5 read public counts: views, likes, comments, shares and saves. Those can't show how many viewers leave in
the first seconds. Instagram reports that as the skip rate, the share of a reel's plays swiped away within the first
three seconds, and only to the account's owner. So this test runs on my own Instagram account, and it asks one thing:
does a reel's skip rate a day after posting go with how many views it has two weeks later?

Finding 4 was that the first frame adds nothing to the share rate once the creator and the words are known. This is a
different outcome: not how many viewers pass a reel on, but how many get to see it.

Everything below was written on 21 October 2026, before the first reel of the test was posted, and frozen in
[`reports/live_bar.json`](../reports/live_bar.json). `sharerate.live` reads the bar and changes none of it, and
`tests/test_live.py` fails if the bar is edited.

## Before the bar

Counted with `scripts/live_exploration.py`, on reels and Shorts posted before the test's first reel, so none of the
test's reels are in it:

- **27 Instagram reels** whose last snapshot is at least 3 days after posting: Pearson's r = −0.72 between the skip
  rate and log views, both at that snapshot. The 8 reels under 40% skip have a median of 56,924 views; the 18 at 40% to
  50%, 5,884.
- **55 Instagram reels:** shares per 1,000 views at about a day (14 to 30 hours) against the same rate 10 days or more
  after posting, r = 0.97 on logs. A reel's share rate is nearly settled after one day.
- **399 YouTube Shorts:** the share of viewers still watching at 3 seconds goes the same way, far more weakly: r = 0.17
  with log views (YouTube's relative retention there, 0.30). YouTube's ratio counts replays as well as viewers who stay,
  so it isn't the same measure.

A reel pushed to strangers should be skipped more, not less, since strangers chose nothing about it. Yet the most-viewed
reels were skipped least, so the link isn't just a big reel reaching a colder audience. That is the reason to test it,
and the reason it is tested on reels nobody has seen yet: 27 reels looked at before the question was chosen can't
confirm what they suggested.

## The test

- **The reels.** The first 30 reels on my Instagram account published from 21 October 2026, 00:00 Belgrade time
  (2026-10-20 22:00 UTC), in the order posted, as the platform's collector records them. None are chosen, and none are
  replaced.
- **Two readings per reel,** from the collector's snapshots (one every 10 minutes while it runs):
  - **day 1:** the snapshot nearest to 24 hours after posting, between 18 and 36 hours, that holds views, shares and a
    skip rate;
  - **day 14:** the snapshot nearest to 14 days, between 10 and 21 days, that holds views and shares.
- **The statistic.** Spearman's ρ between the skip rate at day 1 and the views at day 14, over the reels with both
  readings. Ranks, because views are fat-tailed: one reel can have a hundred times another's.
- **The interval.** 95%, the middle of 2,000 bootstrap draws of those reels with replacement, seed 2026. The reels are
  the units: they are all one creator's.
- **The bar.** It passes if ρ is −0.30 or lower and the interval's upper end is below 0. If fewer than 20 reels have both
  readings, the test is void, neither passed nor failed, and a new bar comes before any new reel is used.
- **When.** Once, 21 days after the 30th reel is posted, when its last reading window has closed. If fewer than 30 are
  posted by 10 January 2027, the set is every reel posted by then, read 21 days later. Until then,
  `python -m sharerate.live status` gives counts only, and `python -m sharerate.live read` refuses to compute anything.

## While it runs

- Every reel goes up the normal way. No trial reels: Instagram shows those to non-followers first, so their first
  audience isn't the same.
- None of the 30 comes down before its day-14 reading.
- A reel missing a reading, because the collector wasn't running in its window or because the reel came down, is
  excluded and counted, never replaced. The collector has to run the day after each post and again in its second week.

## Reported with no bar

With the same draws:

1. Pearson's r between the skip rate at day 1 and log10 views at day 14, the form the 27 reels were looked at in.
2. The partial Spearman between the skip rate at day 1 and the views at day 14, given the views at day 1: what the skip
   rate says that the first day's views don't.
3. Spearman's ρ between shares per 1,000 views at day 1 and at day 14: whether a reel's share rate is settled after a
   day.

## What it can't say

- It is one account, and a correlation. A pass says that the skip rate a day after posting goes with the reach two
  weeks later, on reels posted after the bar. It doesn't say that a better opening causes more reach.
- If every reel opens the same way, the skip rates bunch together and ρ has less range to show itself in. That is a
  reason a true link could miss the bar, not a reason to vary the reels for the test.
- The opening itself comes next, as an A/B: two openings of the same video posted as trial reels, compared on skip
  rate, with its own bar written before the first trial.

`read` writes `reports/live_result.json`: the test, the three extras, the counts and the exclusions, and each reel's two
readings without its id, date or caption, sorted by skip rate so their order says nothing. The data is my own account's,
so the rule that each published number covers at least 5 creators
([docs/why-the-rules.md §4](why-the-rules.md#4-what-the-site-may-show)), which protects other creators' work, doesn't
bind it. Nothing that names a reel or the account is published.
