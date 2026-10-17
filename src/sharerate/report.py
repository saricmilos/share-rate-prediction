"""Stage 5's report: `reports/results.json` holds every number, and `reports/results.md` is rendered from it alone.

Rule 8: every number in the README, a report or the site comes from a run or from a named file, and reports are written
by code. Notebook 05 builds one dict of results from its own runs (and from notebook 01's printed outputs for stage 1's
numbers), and `write` saves it as JSON and renders the Markdown from that JSON, so the two files can't disagree:
`render(json.load(...))` gives `results.md` again, byte for byte (tests/test_report.py).

A score is a dict with `value`, `value_low`, `value_high` (the metric and its own 95% interval) and `gain`, `gain_low`,
`gain_high` (the gain over the score's base and its paired 95% interval, positive meaning better, for MAE too), with
`videos` and `creators` beside them. Nothing in the report is about one creator: every number covers at least 5.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from sharerate.data import ROOT

REPORTS = ROOT / "reports"
DIGITS = {"MAE": 4, "skill": 4}  # lift errors move in the fourth decimal; the other metrics show three
METRIC_NAMES = {"MAE": "MAE", "skill": "skill", "spearman_within": "Spearman inside a creator", "roc_auc": "ROC-AUC",
                "pr_auc": "PR-AUC", "top_decile_share": "top-10% share"}


def write(results: dict, folder: Path = REPORTS) -> tuple[Path, Path]:
    """Save `results` as `results.json` and render `results.md` from what was saved."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    json_path, md_path = folder / "results.json", folder / "results.md"
    json_path.write_text(json.dumps(_rounded(results), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    saved = json.loads(json_path.read_text(encoding="utf-8"))  # render what was saved, not what was in memory
    md_path.write_text(render(saved), encoding="utf-8")
    return json_path, md_path


def score(row, videos: int, creators: int) -> dict:
    """One row of `paired_creator_bootstrap`'s table as a plain dict, with the rows it was scored on."""
    keys = ["value", "value_low", "value_high", "gain", "gain_low", "gain_high"]
    return {k: float(row[k]) + 0.0 for k in keys} | {"videos": int(videos), "creators": int(creators)}


def gain(s: dict, digits: int = 4) -> str:
    """'+0.0097 [+0.0062, +0.0133]': a gain and its interval."""
    return f"{s['gain']:+.{digits}f} [{s['gain_low']:+.{digits}f}, {s['gain_high']:+.{digits}f}]"


def value(s: dict, digits: int = 4) -> str:
    """'0.3583 [0.3324, 0.3851]': a metric and its own interval."""
    return f"{s['value']:.{digits}f} [{s['value_low']:.{digits}f}, {s['value_high']:.{digits}f}]"


def above(s: dict) -> bool:
    """The bar's test, everywhere in this project: the lower end of the gain's interval lies above 0."""
    return s["gain_low"] > 0


def notebook_number(path: Path, pattern: str, cast=float):
    """The first match of `pattern`'s group 1 in a notebook's printed outputs: a number from a file named in the report.

    Raises if nothing matches, so a changed notebook stops the report instead of leaving a stale number in it.
    """
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        for output in cell.get("outputs", []):
            found = re.search(pattern, "".join(output.get("text", "")) + "".join(output.get("data", {}).get("text/plain", "")))
            if found:
                return cast(found.group(1))
    raise LookupError(f"{pattern!r} matches no output of {path}")


def render(r: dict) -> str:
    """The Markdown report, from the results dict (as saved in results.json) and nothing else."""
    final = r["final_model"]
    names = r["models"]
    test, claims = r["test"], r["claims"]
    seen_final = test["seen"][final]["MAE"]
    unseen_final = test["unseen"][final]["MAE"]
    b0_seen, b0_unseen = test["seen"]["B0 zero lift"]["MAE"], test["unseen"]["B0 zero lift"]["MAE"]
    lines = [
        "# Results",
        "",
        f"Written by `{r['about']['notebook']}` from its own runs; every number below is in `results.json` beside this "
        f"file. Intervals are 95%, from {r['about']['draws']:,} draws of creators (seed {r['about']['seed']}), paired "
        "where they are gains. Lift is log10 of the share rate minus the creator's recent median, so an MAE of 0.37 is a "
        "factor of 10^0.37 = 2.3 either way. A gain is the base's error minus the model's: positive is better.",
        "",
        "## The result, on the test split (opened once)",
        "",
        f"The final model is **{names[final]}**. On the newest videos ({r['data']['test_period']}), for creators it trained "
        f"on ({seen_final['videos']:,} videos from {seen_final['creators']} creators), predicting zero lift misses by "
        f"{b0_seen['value']:.4f} log10, a factor of {10 ** b0_seen['value']:.2f} either way. The final model misses by "
        f"{seen_final['value']:.4f}: a gain of {gain(seen_final)}, an error {100 * seen_final['gain'] / b0_seen['value']:.1f}% "
        f"smaller. For creators it never saw ({unseen_final['videos']:,} videos from {unseen_final['creators']} creators), "
        f"zero lift misses by {b0_unseen['value']:.4f} and the final model gains {gain(unseen_final)} "
        f"({100 * unseen_final['gain'] / b0_unseen['value']:.1f}%).",
        "",
        "### What each part adds, on validation and on the test",
        "",
        "Each row is one model's gain over the model before it, in MAE, seen creators unless the column says otherwise.",
        "",
        "| Claim | Model over base | Validation, seen | Test, seen | Test, seen, no reposts | Test, unseen | Holds on test (seen / unseen) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, c in claims.items():
        t = c["test"]
        lines.append(
            f"| {name} | {c['model']} over {c['base']}{' (rows with a frame)' if c.get('rows') == 'with a frame' else ''} | "
            f"{gain(c['validation']['seen'])} | {gain(t['seen'])} | {gain(t['seen, no reposts'])} | {gain(t['unseen'])} | "
            f"{'yes' if c['holds'] else 'no'} / {'yes' if c['holds unseen'] else 'no'} |")
    lines += ["", "### Every model on the test", ""]
    for g, label in [("seen", "Creators the model trained on"), ("unseen", "Creators it never saw")]:
        rows = test[g]
        any_model = next(iter(rows.values()))
        some = next(iter(any_model.values()))
        lines += [f"**{label}** ({some['videos']:,} videos, {some['creators']} creators). Each cell: the value, then its "
                  "gain over zero lift [interval].", "",
                  "| Model | " + " | ".join(METRIC_NAMES[m] for m in r["metrics"]) + " |",
                  "|---|" + "---|" * len(r["metrics"])]
        for model in r["test_models"]:
            cells = []
            for m in r["metrics"]:
                s = rows[model].get(m)
                d = DIGITS.get(m, 3)
                cells.append("—" if s is None else f"{s['value']:.{d}f} {gain(s, d)}")
            lines.append(f"| {names[model]} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Validation: every model, three ways", "",
              "The numbers every choice was made on, against zero lift, MAE and Spearman inside a creator.", "",
              "| Model | Seen: MAE gain | Seen: Spearman | New: MAE gain | Held out: MAE gain |", "|---|---|---|---|---|"]
    val = r["validation"]
    for model in r["validation_models"]:
        lines.append(f"| {names[model]} | {gain(val['seen'][model]['MAE'])} | "
                     f"{val['seen'][model]['spearman_within']['value']:.3f} | {gain(val['new'][model]['MAE'])} | "
                     f"{gain(val['holdout'][model]['MAE'])} |")
    a = r["ablation"]
    lines += ["", f"## The ablation (validation, seen creators, the {a['combiner']} combiner refitted on each subset)", "",
              "| Parts | MAE | Against the combined model |", "|---|---|---|"]
    for subset, s in a["subsets"].items():
        lines.append(f"| {subset} | {s['value']:.4f} | {gain(s)} |")
    lines += ["", "## Where it works (test, seen creators)", "",
              f"The final model's MAE gain over zero lift in each group, as a share of zero lift's error there. Groups of "
              f"fewer than {r['about']['min_creators']} creators are not shown.", ""]
    for kind, bins in r["slices"].items():
        lines += [f"**{kind}**", "", "| Group | Videos | Creators | Zero lift's MAE | Gain | Share of the error |",
                  "|---|---|---|---|---|---|"]
        for label, s in bins.items():
            base = s["value"] + s["gain"]  # zero lift's MAE: the gain is its error minus the model's
            lines.append(f"| {label} | {s['videos']:,} | {s['creators']} | {base:.4f} | {gain(s)} | "
                         f"{100 * s['gain'] / base:.1f}% |" if s["creators"] >= r["about"]["min_creators"] else
                         f"| {label} | {s['videos']:,} | {s['creators']} | — | — | — |")
        lines.append("")
    one = r["stage1"]
    lines += ["## Stage 1, for the page", "",
              f"From `{one['source']}`'s printed outputs: the creator explains {one['creator_variance_share']:.0%} of the "
              f"variance in log share rate; the same creators' median share rate rose {one['drift_factor']:.1f}× between "
              f"their 2022 and 2026 videos; kept clips beat their creator's median {one['kept_above_median']:.0%} of the "
              f"time, retired ones {one['retired_above_median']:.0%}. {r['data']['videos']:,} videos from "
              f"{r['data']['creators']} creators have a target, and {r['data']['breakout_rate']:.1%} broke out.", ""]
    return "\n".join(lines)


def _rounded(x):
    """Floats to 6 decimals, everywhere in the structure: enough for every number shown, and a stable file."""
    if isinstance(x, float):
        return round(x, 6) + 0.0
    if isinstance(x, dict):
        return {k: _rounded(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_rounded(v) for v in x]
    return x
