"""GitHub 同梱のオフライン解析レポート（trifecta_eval_report.json）を読み込む."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from boatrace.config import ROOT_DIR
from boatrace.logging_setup import get_logger

logger = get_logger(__name__)

DEFAULT_REPORT_PATH = ROOT_DIR / "data" / "models" / "trifecta_eval_report.json"

# ticket_ranks 未同梱時の順位別フォールバック（2026-07-30〜08-05 検証）
_FALLBACK_TICKET_RANKS: dict[str, Any] = {
    "source": "fallback",
    "label": "検証ベースライン（5点カバー）",
    "period": {"start": "2026-07-30", "end": "2026-08-05"},
    "n_races": 1020,
    "note": "候補のうち当該順位の点が的中した割合。全体の「いずれか的中」とは別指標。",
    "sanrenpuku": {
        "any_rate": 0.626,
        "ranks": [
            {"rank": 1, "hit_rate": 0.228, "label": "本命"},
            {"rank": 2, "hit_rate": 0.153, "label": "2番手"},
            {"rank": 3, "hit_rate": 0.124, "label": "3番手"},
            {"rank": 4, "hit_rate": 0.070, "label": "4番手"},
            {"rank": 5, "hit_rate": 0.052, "label": "5番手"},
        ],
    },
    "sanrentan": {
        "any_rate": 0.217,
        "ranks": [
            {"rank": 1, "hit_rate": 0.065, "label": "本命"},
            {"rank": 2, "hit_rate": 0.045, "label": "2番手"},
            {"rank": 3, "hit_rate": 0.048, "label": "3番手"},
            {"rank": 4, "hit_rate": 0.027, "label": "4番手"},
            {"rank": 5, "hit_rate": 0.031, "label": "5番手"},
        ],
    },
    "pre_exhibition": {
        "n_races": 867,
        "note": "同じ期間で展示を消した展示前モード",
        "sanrenpuku": {
            "any_rate": 0.627,
            "ranks": [
                {"rank": 1, "hit_rate": 0.232},
                {"rank": 2, "hit_rate": 0.158},
                {"rank": 3, "hit_rate": 0.127},
                {"rank": 4, "hit_rate": 0.053},
                {"rank": 5, "hit_rate": 0.058},
            ],
        },
        "sanrentan": {
            "any_rate": 0.218,
            "ranks": [
                {"rank": 1, "hit_rate": 0.077},
                {"rank": 2, "hit_rate": 0.053},
                {"rank": 3, "hit_rate": 0.045},
                {"rank": 4, "hit_rate": 0.023},
                {"rank": 5, "hit_rate": 0.020},
            ],
        },
    },
}

_report_cache: dict[str, Any] | None = None


def _report_path_label() -> str:
    try:
        return str(DEFAULT_REPORT_PATH.relative_to(ROOT_DIR))
    except ValueError:
        return str(DEFAULT_REPORT_PATH)


def load_offline_eval_report(*, reload: bool = False) -> dict[str, Any]:
    """trifecta_eval_report.json を読み込む（起動時・再解析 commit 後に reload 可）."""
    global _report_cache
    if _report_cache is not None and not reload:
        return _report_cache
    if DEFAULT_REPORT_PATH.exists():
        try:
            raw = json.loads(DEFAULT_REPORT_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                _report_cache = raw
                logger.info(
                    "offline_eval_report_loaded",
                    path=str(DEFAULT_REPORT_PATH),
                    as_of=raw.get("as_of"),
                    holdout_n=(raw.get("holdout_7d") or {}).get("n"),
                )
                return _report_cache
        except Exception as e:  # noqa: BLE001
            logger.warning("offline_eval_report_load_failed", path=str(DEFAULT_REPORT_PATH), error=str(e))
    _report_cache = {}
    return _report_cache


def report_status() -> dict[str, Any]:
    report = load_offline_eval_report()
    meta = report.get("train_meta") or {}
    holdout = report.get("holdout_7d") or {}
    return {
        "loaded": bool(report),
        "path": _report_path_label(),
        "as_of": report.get("as_of"),
        "model": report.get("model"),
        "train_period": {"start": meta.get("start"), "end": meta.get("end")},
        "train_samples": meta.get("samples"),
        "holdout_n": holdout.get("n"),
        "has_ticket_ranks": bool(report.get("ticket_ranks")),
    }


def _merge_holdout_summary(base: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    holdout = report.get("holdout_7d") or {}
    meta = report.get("train_meta") or {}
    if holdout.get("n"):
        out["n_races"] = int(holdout["n"])
    if holdout.get("trio_top3") is not None:
        sp = dict(out.get("sanrenpuku") or {})
        sp["any_rate"] = float(holdout["trio_top3"])
        out["sanrenpuku"] = sp
    if holdout.get("trifecta_top3") is not None:
        st = dict(out.get("sanrentan") or {})
        st["any_rate"] = float(holdout["trifecta_top3"])
        out["sanrentan"] = st
    if meta.get("start") and meta.get("end"):
        out["train_period"] = {"start": meta["start"], "end": meta["end"]}
    out["source"] = "offline_eval_report"
    out["report_path"] = _report_path_label()
    out["as_of"] = report.get("as_of")
    if holdout.get("n"):
        out["label"] = f"オフライン検証（holdout 7日・n={int(holdout['n'])}）"
    return out


def get_baseline_ticket_rank_stats() -> dict[str, Any]:
    """UI/API 向けベースライン（JSON 優先、なければ holdout + フォールバック順位）."""
    report = load_offline_eval_report()
    if not report:
        return copy.deepcopy(_FALLBACK_TICKET_RANKS)
    if report.get("ticket_ranks"):
        return _merge_holdout_summary(report["ticket_ranks"], report)
    return _merge_holdout_summary(_FALLBACK_TICKET_RANKS, report)


def get_holdout_reference() -> dict[str, Any]:
    report = load_offline_eval_report()
    holdout = dict(report.get("holdout_7d") or {})
    meta = report.get("train_meta") or {}
    return {
        "source": _report_path_label(),
        "as_of": report.get("as_of"),
        "model": report.get("model"),
        "train_period": {"start": meta.get("start"), "end": meta.get("end"), "samples": meta.get("samples")},
        "holdout_7d": holdout,
        "note": "学習ホールドアウト7日。ライブ追跡とは別の検証値です。",
    }


def fallback_ticket_rank_template() -> dict[str, Any]:
    """eval レポート生成時に ticket_ranks へ埋め込むテンプレート."""
    return copy.deepcopy(_FALLBACK_TICKET_RANKS)


def get_offline_accuracy_summary() -> dict[str, Any]:
    """DB 集計が空のときに /accuracy へ載せる参考精度."""
    report = load_offline_eval_report()
    holdout = report.get("holdout_7d") or {}
    meta = report.get("train_meta") or {}
    return {
        "source": "offline_eval_report",
        "path": _report_path_label(),
        "as_of": report.get("as_of"),
        "model": report.get("model"),
        "label": "GitHub同梱オフライン検証（holdout 7日）",
        "n_races": int(holdout.get("n") or 0),
        "win_rate": float(holdout.get("win_top1") or 0),
        "win_top3": float(holdout.get("win_top3") or 0),
        "quinella_rate": 0.0,
        "trio_rate": float(holdout.get("trio_top3") or 0),
        "trio_top1": float(holdout.get("trio_top1") or 0),
        "trifecta_rate": float(holdout.get("trifecta_top3") or 0),
        "trifecta_top1": float(holdout.get("trifecta_top1") or 0),
        "confident_n": int(holdout.get("confident_n") or 0),
        "trifecta_top3_confident": float(holdout.get("trifecta_top3_confident") or 0),
        "train_period": {"start": meta.get("start"), "end": meta.get("end"), "samples": meta.get("samples")},
        "note": (
            "Render 無料枠など DB が空のときの参考値。"
            "再学習後は eval_trifecta_report.py の出力を commit すると自動反映されます。"
        ),
    }
