"""Stage 6: the data behind the page on saricmilos.com, which runs with no server.

    python -m sharerate.site        # SHARERATE_SITE names the site's folder, if not C:\\dev\\sites\\saricmilos.com

It writes three JSON files into the site's `src/content/share-rate/`, and nothing else:

- `results.json`: the numbers of `reports/results.json` the page shows, copied as notebook 05 wrote them. Nothing in
  it is recomputed: the test was opened once, and its numbers come from that run alone.
- `findings.json`: what the page's findings and its calculator draw, as aggregates of the data, each over at least
  `MIN_CREATORS` creators (rule 9), and two findings read from notebooks 03 and 04's printed outputs, the way
  `results.json` reads stage 1's from notebook 01: the words the text model learned, and how well a first frame names
  its creator.
- `hook.json`: the hook demo's model. Stage 3 scored bge-small on each text field alone, on top of B3; the hook's head
  is refitted here exactly as notebook 03 fitted it, and its validation gain must equal the one that notebook printed,
  so the page runs a model that was scored, with its number beside it.

Before it finishes, it scans what it wrote for any creator's username, video id, caption, transcript or other text
field (`scan`), and deletes the files if it finds one. tests/test_site.py runs the same scan on the site's copy.
"""

from __future__ import annotations

import bisect
import json
import os
import re
import tempfile
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sharerate import data, history, report, text
from sharerate.bootstrap import paired_creator_bootstrap
from sharerate.metrics import METRICS

SITE = Path(os.environ.get("SHARERATE_SITE", r"C:\dev\sites\saricmilos.com"))
CONTENT = Path("src") / "content" / "share-rate"
FILES = ("results.json", "findings.json", "hook.json")
MIN_CREATORS = 5  # rule 9: every number on the page covers at least this many creators
SEED, N_BOOT = 2026, 2_000  # every stage's seed and draws, so the hook's interval is notebook 03's
NOTEBOOK_03 = data.ROOT / "notebooks" / "03_text.ipynb"
NOTEBOOK_04 = data.ROOT / "notebooks" / "04_image.ipynb"

# Notebook 01's definitions, repeated for the aggregates (cells 7, 13 and 27)
MIN_VIEWS = 1_000  # below this one share moves the rate by a whole point
HALF_SHARE = 0.5  # added before the log, so a video with no shares has a finite rate
WINDOW, MIN_PRIOR, MIN_AGE_DAYS = 30, 10, 14  # the baseline: the last 30 videos posted at least 14 days earlier
YEARS = range(2019, 2027)  # notebook 01's chart of the yearly median starts in 2019

HOOK_ALPHA = 100  # notebook 03, cell 27: the penalty ridge_head chose for bge on the hook alone
# Notebook 03, cell 49: two of the strongest terms are Serbian words, which mark a language, not a subject
NOT_ENGLISH = {"da", "na"}
# The browser's copy of bge-small: Xenova's ONNX export at half precision (about 67 MB). The 8-bit file is half the
# size, but moved the examples' pushes by up to 0.009 log10 (the site's scripts/share-rate/parity.mjs); this one by
# under 0.0001, so the page runs the model that was scored.
BROWSER_DTYPE = "fp16"
# Hooks for the demo's buttons, written for the page before any was scored, never taken from a video. The export
# scores each with the same encoder the model was fitted on, and the site checks its browser encoder against these.
EXAMPLES = [
    "Send this to the friend who always steals your fries",
    "POV: your bestie finally texts back",
    "Use my code for 20% off, link in bio",
    "Duet this with your best impression",
    "Nobody tells you this about moving abroad",
    "Three-ingredient pasta, ranked from worst to best",  # replaced one in which the scan found a creator's text
    "Day 12 of learning to dance in my kitchen",
    "Rating every airport snack I can find",
]


def export(site: Path = SITE) -> dict:
    """Write the three files into `site`'s content folder, scan them, and return their sizes in bytes."""
    out = Path(site) / CONTENT
    if not (Path(site) / "package.json").exists():
        raise FileNotFoundError(f"{site} is not the site's folder: set SHARERATE_SITE")
    results = json.loads((report.REPORTS / "results.json").read_text(encoding="utf-8"))
    files = {
        "results.json": page_results(results),
        "findings.json": findings(results),
        "hook.json": hook_model(),
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (out / name).write_text(json.dumps(_rounded(content), ensure_ascii=False, separators=(",", ":")) + "\n",
                                encoding="utf-8")
    problems = scan([out / name for name in files], *private_strings())
    if problems:
        for name in files:  # nothing private stays on disk in the site's folder, not even for a look
            (out / name).unlink()
        raise RuntimeError("The export holds private text, so it was deleted:\n" + "\n".join(problems))
    return {name: (out / name).stat().st_size for name in files}


# ---------------------------------------------------------------------------------------------------------------
# results.json: the report's numbers, as they are


def page_results(results: dict) -> dict:
    """The parts of reports/results.json the page shows. The test's two groups are rule 3's: creators the model
    trained on, and creators it never saw ('new' and 'holdout' together)."""
    keep = ["about", "data", "final_model", "models", "claims", "ablation", "slices", "stage1"]
    return {
        **{key: results[key] for key in keep},
        "test": {group: results["test"][group] for group in ("seen", "unseen")},
        "validation": {"seen": results["validation"]["seen"]},
        "combined": results["frozen"]["combined"],  # the stack's weights, frozen before the test
    }


# ---------------------------------------------------------------------------------------------------------------
# findings.json: aggregates of the data, each over at least 5 creators


def findings(results: dict) -> dict:
    measurable = load_measurable()
    targets = load_targets()
    return {
        "min_creators": MIN_CREATORS,
        "calculator": calculator(targets),
        "drift": drift(measurable, targets, results["stage1"]["drift_factor"]),
        "creators": creator_deciles(measurable),
        "survivorship": survivorship(),
        "words": learned_words(),
        "frame": frame_names_creator(),
    }


def load_measurable() -> pd.DataFrame:
    """Notebook 01's measurable videos: TikTok videos with counts and at least 1,000 views, with their share rate.
    The usernames stay in this module: they group videos by creator and are never written."""
    with closing(data._open_snapshot()) as con:
        videos = pd.read_sql(
            """SELECT username, video_id, published_at, fetched_at, view_count AS views, share_count AS shares
               FROM videos WHERE platform = 'tiktok' AND has_stats = 1""", con)
    for column in ("published_at", "fetched_at"):
        videos[column] = pd.to_datetime(videos[column], utc=True, format="ISO8601")
    videos["age_days"] = (videos.fetched_at - videos.published_at).dt.total_seconds() / 86_400
    measurable = videos[videos.views >= MIN_VIEWS].copy()
    measurable["share_rate"] = measurable.shares / measurable.views * 1_000
    measurable["log_sr"] = np.log10((measurable.shares + HALF_SHARE) / measurable.views * 1_000)
    return measurable


def load_targets() -> pd.DataFrame:
    """Every target row, all three splits, as stage 1 stored it (only these columns)."""
    with closing(data._open_snapshot()) as con:
        targets = pd.read_sql("SELECT username, video_id, published_at, log_sr, baseline, lift, breakout "
                              "FROM stage1.targets", con)
    targets["published_at"] = pd.to_datetime(targets.published_at, utc=True, format="ISO8601")
    return targets


def _group(values, creators, keys) -> dict:
    """How many videos and creators a group holds, refusing one under MIN_CREATORS."""
    n = int(pd.Series(creators).nunique())
    if n < MIN_CREATORS:
        raise ValueError(f"{keys} covers {n} creators, fewer than {MIN_CREATORS} (rule 9)")
    return {"videos": int(len(values)), "creators": n}


def calculator(targets: pd.DataFrame) -> dict:
    """What the calculator places a visitor's video against: lift over all 143,489 target videos, as its 1st to 99th
    percentiles and as a histogram whose every bar covers at least 5 creators (sparse bars at the ends are merged
    inward). No 0th or 100th percentile: each would be one video's own number."""
    lift = targets.lift.to_numpy()
    creators = targets.username.to_numpy()

    def bar(lo, hi):
        """The videos and creators with lift from `lo` up to `hi`; None is an open end."""
        rows = (lift >= (-np.inf if lo is None else lo)) & (lift < (np.inf if hi is None else hi))
        return {"from": lo, "to": hi, "videos": int(rows.sum()), "creators": int(pd.Series(creators[rows]).nunique())}

    # Bars a tenth wide, one edge on the breakout line, so every bar is wholly a breakout or wholly not
    edges = [float(e) for e in data.LOG2 + np.arange(-18, 13) * 0.1]  # about -1.5 to +1.5
    bins = _merge_sparse_ends([bar(lo, hi) for lo, hi in zip([None, *edges], [*edges, None])], bar)
    for b in bins:
        _group(range(b["videos"]), range(b["creators"]), f"lift from {b['from']} to {b['to']}")
    return {
        **_group(lift, creators, "every target video"),
        "min_views": MIN_VIEWS,
        "half_share": HALF_SHARE,
        "breakout_line": float(data.LOG2),
        "breakout_rate": float(targets.breakout.mean()),
        "percentiles": [float(q) for q in np.quantile(lift, np.arange(1, 100) / 100)],
        "bins": bins,
    }


def _merge_sparse_ends(bins: list[dict], bar) -> list[dict]:
    """Merge each end bar into its neighbour until it covers MIN_CREATORS creators. `bar(lo, hi)` counts the merged
    range's videos and creators again from the data, since a creator in both bars would be counted twice by a sum;
    the caller's check then refuses any bar still short, so a merge that can't reach 5 creators stops the export."""
    while len(bins) > 2 and bins[0]["creators"] < MIN_CREATORS:
        bins = [bar(bins[0]["from"], bins[1]["to"])] + bins[2:]
    while len(bins) > 2 and bins[-1]["creators"] < MIN_CREATORS:
        bins = bins[:-2] + [bar(bins[-2]["from"], bins[-1]["to"])]
    return bins


def drift(measurable: pd.DataFrame, targets: pd.DataFrame, drift_factor: float) -> dict:
    """The yardstick moved. By year: the library's median share rate (notebook 01, cell 13), and the target videos'
    mean lift and breakout rate measured two ways on the same rows: against the creator's last 30 videos posted at
    least 14 days earlier (the target), and against every earlier video of theirs (a lifetime median, also only
    videos posted at least 14 days earlier). And the same creators' own change from 2022 to 2026 (cell 15)."""
    pool = measurable[measurable.age_days >= MIN_AGE_DAYS].sort_values(["username", "published_at", "video_id"])
    recent, lifetime = _as_of_baselines(pool)
    pool = pool.assign(recent=recent, lifetime=lifetime)
    rows = targets.merge(pool[["username", "video_id", "recent", "lifetime"]], on=["username", "video_id"],
                         how="left", validate="one_to_one")
    # The pool rebuilds stage 1's baseline exactly, so the lifetime one is built from the same videos
    if not np.allclose(rows.recent, rows.baseline, atol=1e-9):
        raise RuntimeError("The pool no longer rebuilds stage 1's baseline (notebook 01, cell 27)")
    rows["lifetime_lift"] = rows.log_sr - rows.lifetime

    years = []
    library_year = measurable.published_at.dt.year
    target_year = rows.published_at.dt.year
    for year in YEARS:
        lib = measurable[library_year == year]
        tgt = rows[target_year == year]
        years.append({
            "year": year,
            "library": {**_group(lib, lib.username, f"library {year}"), "median_rate": float(lib.share_rate.median())},
            "targets": _group(tgt, tgt.username, f"targets {year}"),
            "recent": {"mean_lift": float(tgt.lift.mean()), "breakout": float(tgt.breakout.mean())},
            "lifetime": {"mean_lift": float(tgt.lifetime_lift.mean()),
                         "breakout": float((tgt.lifetime_lift >= data.LOG2).mean())},
        })

    both = measurable[library_year.isin([2022, 2026])]
    medians = both.groupby(["username", both.published_at.dt.year]).log_sr.median().unstack().dropna()
    change = medians[2026] - medians[2022]
    factor = float(10 ** change.median())
    if round(factor, 1) != drift_factor:  # results.json's number, read from notebook 01's printed output
        raise RuntimeError(f"The same creators' change is {factor:.2f}x here, {drift_factor}x in results.json")
    return {
        "years": years,
        "same_creators": {**_group(change, change.index, "creators in 2022 and 2026"), "factor": factor,
                          "rose": float((change > 0).mean())},
        "window": WINDOW,
        "min_prior": MIN_PRIOR,
        "min_age_days": MIN_AGE_DAYS,
    }


def _as_of_baselines(pool: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """For each pool video (sorted by creator, then publish time), the median log share rate of the creator's pool
    videos posted at least 14 days before it: the last 30 of them (stage 1's baseline) and all of them (a lifetime
    median). NaN with fewer than 10."""
    recent = np.full(len(pool), np.nan)
    lifetime = np.full(len(pool), np.nan)
    settled = np.timedelta64(MIN_AGE_DAYS, "D")
    offset = 0
    for _, creator in pool.groupby("username", sort=False):
        posted = creator.published_at.dt.tz_localize(None).to_numpy()
        rates = creator.log_sr.to_numpy()
        ends = np.searchsorted(posted, posted - settled, side="right")  # rows [0, end) were posted <= t - 14 days
        # The median of every prefix rates[:e], kept sorted as it grows: one insertion per video
        prefix_median, ordered = np.full(len(rates) + 1, np.nan), []
        for e, rate in enumerate(rates, start=1):
            bisect.insort(ordered, rate)
            mid = e // 2
            prefix_median[e] = ordered[mid] if e % 2 else (ordered[mid - 1] + ordered[mid]) / 2
        for i, end in enumerate(ends):
            if end >= MIN_PRIOR:
                recent[offset + i] = np.median(rates[max(end - WINDOW, 0):end])
                lifetime[offset + i] = prefix_median[end]
        offset += len(creator)
    return recent, lifetime


def creator_deciles(measurable: pd.DataFrame) -> dict:
    """Who made it: each creator's median share rate, cut into ten groups of 24 or 25 creators, with each group's
    median (notebook 01, cell 10). One point per creator would break rule 9; a decile is an aggregate."""
    medians = measurable.groupby("username").share_rate.median()
    decile = pd.qcut(medians.rank(method="first"), 10, labels=range(1, 11))
    groups = []
    for d, part in medians.groupby(decile, observed=True):
        groups.append({"decile": int(d), "median_rate": float(part.median()),
                       **_group(part, part.index, f"decile {d}")})
    return {"deciles": groups, "creators": int(len(medians)), "videos": int(len(measurable))}


def survivorship() -> dict:
    """The data remembers its winners: the breakout rate of target videos whose clip the platform kept, against those
    it retired. `has_video` is a leak and never a feature (trap 3); here it is only counted."""
    videos = data.load_videos(data.SPLITS)
    kept = data.load_leak_reference(data.SPLITS).to_numpy().astype(bool)
    out = {"base_rate": float(videos.breakout.mean())}
    for name, rows in (("kept", kept), ("retired", ~kept)):
        part = videos[rows]
        out[name] = {**_group(part, part.creator, name), "breakout": float(part.breakout.mean())}
    return out


def learned_words() -> dict:
    """The 12 terms that push the TF-IDF model's prediction up most and the 12 that push it down most, as notebook
    03 printed them (cell 48): each used by at least 20 training creators, so a pattern, never a quote. `effect` is
    in log10 of lift; `share_rate` is the notebook's own rounding of it as a change in share rate."""
    nb = json.loads(NOTEBOOK_03.read_text(encoding="utf-8"))
    cell = next(c for c in nb["cells"] if "learned.nlargest" in "".join(c["source"]))
    tables, heading = {}, None
    row = re.compile(r"^\s*\d+\s+(Caption|Transcript|Hook|Body|Payoff|On-screen text|Scene description)\s+(.+?)\s+"
                     r"(\d+)\s+(\d+)\s+(-?\d+\.\d+)\s+([+-]?\d+\.\d)%\s*$")
    for output in cell["outputs"]:
        printed = "".join(output.get("text", "")) + "".join(output.get("data", {}).get("text/plain", ""))
        if "up most" in printed:
            heading = "up"
        elif "down most" in printed:
            heading = "down"
        elif heading and heading not in tables:
            found = [row.match(line) for line in printed.splitlines()]
            tables[heading] = [{"field": m[1], "term": m[2], "creators": int(m[3]), "videos": int(m[4]),
                                "effect": float(m[5]), "share_rate": float(m[6]) / 100} for m in found if m]
    if [len(tables.get(k, [])) for k in ("up", "down")] != [12, 12]:
        raise LookupError(f"{NOTEBOOK_03} no longer prints the 12 terms each way (cell 48)")
    for term in tables["up"] + tables["down"]:
        if term["creators"] < 20:
            raise ValueError(f"A term used by {term['creators']} creators: the vocabulary rule needs 20")
        term["english"] = term["term"] not in NOT_ENGLISH
    return {
        **tables,
        "terms": report.notebook_number(NOTEBOOK_03, r"Terms in the model: ([\d,]+)", lambda s: int(s.replace(",", ""))),
        "min_creators": text.MIN_CREATORS,
    }


def frame_names_creator() -> dict:
    """How well a linear probe on a first frame names the creator of a later video (notebook 04, cell 38)."""
    number = lambda pattern, cast=float: report.notebook_number(NOTEBOOK_04, pattern, cast)  # noqa: E731
    percent = lambda pattern: number(pattern) / 100  # noqa: E731
    return {
        "top1": percent(r"Top-1: ([\d.]+)%"),
        "top1_low": percent(r"Top-1: [\d.]+% \[([\d.]+)%"),
        "top1_high": percent(r"Top-1: [\d.]+% \[[\d.]+%, ([\d.]+)%\]"),
        "top5": percent(r"top-5: ([\d.]+)%"),
        "biggest_creator": percent(r"always guessing the biggest creator: ([\d.]+)%"),
        "creators": number(r"Probe: \d+ iterations \(at most [\d,]+\), (\d+) creators", int),
        "frames": number(r"([\d,]+) validation frames", lambda s: int(s.replace(",", ""))),
    }


# ---------------------------------------------------------------------------------------------------------------
# hook.json: stage 3's hook-only head, for the browser


def hook_model() -> dict:
    """B3 and bge-small's head on the hook alone, refitted as notebook 03 fits them (cells 12, 18 and 27), and what the
    page needs to run them for a visitor who gives a hook and a length and nothing else:

    - the head's 384 weights, its 'there is a hook' weight and its intercept;
    - B3 for a typical training video of a given length: B3 is linear in log duration, so its prediction averaged
      over the training rows with every duration set to L is `at_1s + per_log10_second * log10(L)`;
    - the head's validation gain over B3 for seen creators, which must equal notebook 03's (cell 33);
    - where hooks sit: the head's push (what the hook adds against the same video with no hook) over the validation
      videos' hooks, as percentiles;
    - EXAMPLES with their pushes, for the demo's buttons and the site's check of its browser encoder.
    """
    videos = data.load_videos(text=True)  # train and validation; the texts are never written anywhere
    posts = data.load_posting_times()
    training = ((videos.time_split == "train") & (videos.creator_group == "train")).to_numpy()
    validation = (videos.time_split == "validation").to_numpy()
    seen = validation & (videos.creator_seen == "seen").to_numpy()
    lift = videos.lift.to_numpy()

    timing = history.posting_features(videos, posts)
    features = pd.DataFrame({
        "log_duration": np.log10(videos.duration_seconds.clip(lower=1)),
        "wordless": videos.is_wordless,
        "log_earlier_videos": np.log10(1 + timing.earlier_videos),
        "log_days_since_previous": np.log10(1 + timing.days_since_previous),
        "weekday_utc": timing.weekday_utc,
    })
    numeric = ["log_duration", "wordless", "log_earlier_videos", "log_days_since_previous"]
    b3 = make_pipeline(
        ColumnTransformer([
            ("numbers", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), numeric),
            ("categories", OneHotEncoder(handle_unknown="ignore"), ["weekday_utc"]),
        ]),
        Ridge(alpha=0.1),  # notebook 02, cell 24
    ).fit(features[training], lift[training])
    b3_pred = b3.predict(features)
    if round(np.abs(lift[seen] - b3_pred[seen]).mean(), 4) != 0.3681:  # notebook 02, cell 36
        raise RuntimeError("B3 no longer misses by 0.3681 for seen creators on validation")

    def typical(seconds):  # B3's prediction averaged over the training rows, every one set to this length
        return b3.predict(features[training].assign(log_duration=np.log10(seconds))).mean()
    at_1s = typical(1.0)
    per_log10_second = typical(10.0) - at_1s
    if abs(typical(40.0) - (at_1s + per_log10_second * np.log10(40.0))) > 1e-9:
        raise RuntimeError("B3 is no longer linear in log duration")

    has_hook = (videos.hook_en != "").to_numpy()
    X = np.hstack([has_hook[:, None].astype(np.float32), text.embed(videos.hook_en)])
    head = Ridge(alpha=HOOK_ALPHA).fit(X[training], (lift - b3_pred)[training])
    pred = b3_pred + head.predict(X)

    mae, lower_is_better = METRICS["MAE"]
    table = paired_creator_bootstrap(mae, lift[seen], {"B3": b3_pred[seen], "hook": pred[seen]},
                                     videos.creator.to_numpy()[seen], base="B3", lower_is_better=lower_is_better,
                                     n_boot=N_BOOT, seed=SEED)
    scored = report.score(table.loc["hook"], seen.sum(), videos.creator[seen].nunique())
    printed = _printed_hook_gain()
    if [round(scored[k], 4) for k in ("gain", "gain_low", "gain_high")] != printed:
        raise RuntimeError(f"The hook head gains {report.gain(scored)} here, {printed} in notebook 03 (cell 33)")

    flag, weights = float(head.coef_[0]), head.coef_[1:]
    push = flag + X[:, 1:] @ weights  # what a hook adds against the same video with no hook
    with_hook = validation & has_hook
    with tempfile.TemporaryDirectory() as cache:  # the examples' vectors go nowhere near the project's cache
        example_vectors = text.embed(EXAMPLES, cache_dir=cache)
    seconds = videos.duration_seconds[training]
    return {
        "encoder": {"model": text.EMBEDDING_MODEL, "browser": "Xenova/bge-small-en-v1.5", "dtype": BROWSER_DTYPE,
                    "pooling": "cls", "normalize": True, "max_tokens": text.MAX_TOKENS},
        "alpha": HOOK_ALPHA,
        "intercept": float(head.intercept_),
        "flag": flag,
        "weights": [float(w) for w in weights],
        "b3": {"at_1s": float(at_1s), "per_log10_second": float(per_log10_second),
               "seconds": {"p01": float(seconds.quantile(0.01)), "median": float(seconds.median()),
                           "p99": float(seconds.quantile(0.99))}},
        "validation": {"hook": scored, "b3_mae": float(table.loc["B3", "value"])},
        "pushes": {**_group(push[with_hook], videos.creator[with_hook], "validation hooks"),
                   "percentiles": [float(q) for q in np.quantile(push[with_hook], np.arange(1, 100) / 100)]},
        "examples": [{"text": t, "push": float(flag + v @ weights)} for t, v in zip(EXAMPLES, example_vectors)],
    }


def _printed_hook_gain() -> list[float]:
    """bge's MAE gain over B3 for the hook alone and its interval, from notebook 03's field table (cell 33), whose
    row reads 'Hook', TF-IDF's gain [interval], bge's gain [interval], then the two ROC-AUC columns."""
    number = r"[+-]\d\.\d{4}"
    pattern = rf"\nHook\s+{number} \[{number}, {number}\]\s+({number}) \[({number}), ({number})\]"
    nb = json.loads(NOTEBOOK_03.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        for output in cell.get("outputs", []):
            found = re.search(pattern, "\n" + "".join(output.get("data", {}).get("text/plain", "")))
            if found:
                return [float(g) for g in found.groups()]
    raise LookupError(f"{NOTEBOOK_03} no longer prints the hook's row of the field table (cell 33)")


# ---------------------------------------------------------------------------------------------------------------
# The scan: nothing a creator wrote, and nothing that names a creator or a video, leaves in the export


def private_strings() -> tuple[set[str], set[str], list[str]]:
    """Every username and video id in the snapshot, and every text field of every video (captions, transcripts in
    both languages, the hook, body and payoff, the on-screen text, the scene description and the notes' summary),
    lowercased with runs of whitespace as one space."""
    fields = ["title", "transcript", "transcript_en", "hook_en", "body_en", "payoff_en", "onscreen_text_en",
              "scene_description_en", "description"]
    with closing(data._open_snapshot()) as con:
        rows = con.execute(f"SELECT username, video_id, {', '.join(fields)} FROM videos").fetchall()
    usernames = {r[0].lower() for r in rows}
    video_ids = {str(r[1]).lower() for r in rows}  # YouTube's ids hold letters; the scan compares lowercased words
    texts = [_normal(t) for r in rows for t in r[2:] if t and t.strip()]
    return usernames, video_ids, texts


def _normal(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _strings(value) -> list[str]:
    """Every string in a JSON value, keys included."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in [k, *_strings(v)]]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def scan(paths, usernames: set[str], video_ids: set[str], texts: list[str], *, min_text=15, min_quote=25) -> list[str]:
    """What in the exported files is private, as one line per problem (empty when nothing is):

    - a username or a video id as a word of any string, or any run of 15 or more digits (a TikTok id has 19);
    - any creator text of at least `min_text` characters inside the exported strings;
    - any exported string of at least `min_quote` characters inside a creator's text (a quote of part of one);
    - a group of 1 to 4 creators: every "creators" count in the files is 0 (an empty group) or at least 5.
    """
    problems = []
    long_texts = [t for t in texts if len(t) >= min_text]
    corpus = "\n".join(texts)  # one string, so each quote check is one fast search
    for path in map(Path, paths):
        raw = path.read_text(encoding="utf-8")
        content = json.loads(raw)
        strings = sorted({_normal(s) for s in _strings(content)})  # each distinct string once (keys repeat)
        if re.search(r"\d{15,}", raw):
            problems.append(f"{path.name}: a run of 15 or more digits, the shape of a video id")
        for s in strings:
            if any(word in usernames or word in video_ids for word in re.findall(r"[\w.]+", s)):
                problems.append(f"{path.name}: {s[:60]!r} holds a username or a video id")
        blob = "\n".join(strings)
        if any(t in blob for t in long_texts):  # one fast pass; only on a hit, which string holds it
            for s in strings:
                if len(s) >= min_text and any(t in s for t in long_texts):
                    problems.append(f"{path.name}: {s[:60]!r} holds a creator's text of {min_text} or more characters")
        for s in strings:
            if len(s) >= min_quote and s in corpus:
                problems.append(f"{path.name}: {s[:60]!r} is part of a creator's text")
        # 0 is an empty group, which shows nothing (the training period has no 'new' creators, by definition)
        problems += [f"{path.name}: a group of {n} creators" for n in _creator_counts(content) if 0 < n < MIN_CREATORS]
    return problems


def _creator_counts(value) -> list[int]:
    """Every number stored under a "creators" key, at any depth."""
    counts = []
    if isinstance(value, dict):
        for key, v in value.items():
            if key == "creators" and isinstance(v, int):
                counts.append(v)
            else:
                counts += _creator_counts(v)
    elif isinstance(value, list):
        for v in value:
            counts += _creator_counts(v)
    return counts


def _rounded(value):
    """Floats to 6 significant digits: enough for every number the page shows, and a third of the bytes."""
    if isinstance(value, float):
        return float(f"{value:.6g}")
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_rounded(v) for v in value]
    return value


def main() -> None:
    sizes = export()
    print(f"Wrote {SITE / CONTENT}: " + ", ".join(f"{name} {size / 1024:.0f} kB" for name, size in sizes.items()))


if __name__ == "__main__":
    main()
