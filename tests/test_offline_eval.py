"""オフライン解析レポート読み込み."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from boatrace.prediction import offline_eval as oe


@pytest.fixture
def sample_report(tmp_path: Path, monkeypatch):
    report = {
        "as_of": "2026-08-03",
        "model": "rank_v3",
        "holdout_7d": {
            "n": 1054,
            "trio_top3": 0.524,
            "trifecta_top3": 0.202,
            "win_top1": 0.566,
        },
        "train_meta": {"start": "2026-01-01", "end": "2026-08-02", "samples": 32973},
        "ticket_ranks": {
            "label": "テストベースライン",
            "period": {"start": "2026-07-30", "end": "2026-08-05"},
            "n_races": 1020,
            "sanrenpuku": {"any_rate": 0.62, "ranks": [{"rank": 1, "hit_rate": 0.22, "label": "本命"}]},
            "sanrentan": {"any_rate": 0.21, "ranks": [{"rank": 1, "hit_rate": 0.06, "label": "本命"}]},
        },
    }
    path = tmp_path / "trifecta_eval_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(oe, "DEFAULT_REPORT_PATH", path)
    oe.load_offline_eval_report(reload=True)
    return report


def test_load_offline_eval_report(sample_report):
    loaded = oe.load_offline_eval_report()
    assert loaded["as_of"] == "2026-08-03"
    assert loaded["holdout_7d"]["n"] == 1054


def test_baseline_merges_holdout_any_rates(sample_report):
    base = oe.get_baseline_ticket_rank_stats()
    assert base["sanrenpuku"]["any_rate"] == pytest.approx(0.524)
    assert base["sanrentan"]["any_rate"] == pytest.approx(0.202)
    assert base["n_races"] == 1054
    assert base["source"] == "offline_eval_report"


def test_offline_accuracy_summary(sample_report):
    summary = oe.get_offline_accuracy_summary()
    assert summary["n_races"] == 1054
    assert summary["trio_rate"] == pytest.approx(0.524)
    assert summary["trifecta_rate"] == pytest.approx(0.202)
    assert summary["train_period"]["samples"] == 32973


def test_holdout_reference(sample_report):
    ref = oe.get_holdout_reference()
    assert ref["holdout_7d"]["n"] == 1054
    assert ref["train_period"]["start"] == "2026-01-01"
