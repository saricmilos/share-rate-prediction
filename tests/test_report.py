"""The report's Markdown comes from its JSON alone, and a number taken from a notebook is found or refused."""

import json

import pytest

from sharerate import report


def _score(value, gain, videos=1_000, creators=20):
    return {"value": value, "value_low": value - 0.02, "value_high": value + 0.02, "gain": gain,
            "gain_low": gain - 0.003, "gain_high": gain + 0.003, "videos": videos, "creators": creators}


def _results():
    metrics = ["MAE", "spearman_within"]
    models = {"B0 zero lift": "zero lift", "4B ridge": "B3 + the words", "combined": "everything together"}
    by_model = {m: {"MAE": _score(0.37, 0.01 * i), "spearman_within": _score(0.2, 0.1)} for i, m in enumerate(models)}
    claim = {"base": "4B ridge", "model": "combined", "validation": {"seen": _score(0.36, 0.002)},
             "test": {"seen": _score(0.36, 0.0021), "seen, no reposts": _score(0.36, 0.0022),
                      "unseen": _score(0.37, 0.001)}, "holds": True, "holds unseen": False}
    return {
        "about": {"notebook": "notebooks/05_results.ipynb", "draws": 2_000, "seed": 2026, "min_creators": 5},
        "data": {"videos": 143_489, "creators": 231, "breakout_rate": 0.2412, "test_period": "the test period"},
        "final_model": "combined", "models": models, "metrics": metrics, "test_models": list(models),
        "validation_models": list(models),
        "test": {"seen": by_model, "unseen": by_model},
        "validation": {g: by_model for g in ["seen", "new", "holdout"]},
        "claims": {"Recent form adds to the words": claim},
        "ablation": {"combiner": "stacked", "subsets": {"B3": _score(0.368, -0.01), "B3 + the words": _score(0.358, 0)}},
        "slices": {"Video length": {"under 10 s": _score(0.38, 0.01, creators=30), "over 60 s": _score(0.4, 0.0, creators=3)}},
        "stage1": {"source": "notebooks/01_eda_target.ipynb", "creator_variance_share": 0.38, "drift_factor": 4.7,
                   "kept_above_median": 0.638, "retired_above_median": 0.433},
    }


def test_the_markdown_is_rendered_from_the_saved_json(tmp_path):
    json_path, md_path = report.write(_results(), tmp_path)
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert md_path.read_text(encoding="utf-8") == report.render(saved)
    text = md_path.read_text(encoding="utf-8")
    assert "everything together" in text and "+0.0021 [-0.0009, +0.0051]" in text
    # A group of fewer than 5 creators shows its counts and no number
    assert "| over 60 s | 1,000 | 3 | — | — | — |" in text


def test_floats_are_rounded_once_everywhere():
    assert report._rounded({"a": [0.12345678, {"b": -0.0}], "c": 3}) == {"a": [0.123457, {"b": 0.0}], "c": 3}


def test_a_notebook_number_is_found_or_refused(tmp_path):
    nb = {"cells": [{"cell_type": "code", "outputs": [{"output_type": "stream", "text": ["Knowing only the creator "
                                                                                           "explains 38% of it\n"]}]}]}
    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps(nb), encoding="utf-8")
    assert report.notebook_number(path, r"explains (\d+)%", int) == 38
    with pytest.raises(LookupError):
        report.notebook_number(path, r"changed by ([\d.]+)x")
