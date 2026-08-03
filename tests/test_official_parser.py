"""公式HTMLパーサの回帰テスト."""

from __future__ import annotations

from pathlib import Path

from boatrace.collectors.official import OfficialCollector

FIX = Path(__file__).parent / "fixtures"


def test_parse_racelist_six_entries():
    html = (FIX / "racelist_sample.html").read_text(encoding="utf-8")
    data = OfficialCollector()._parse_racelist(html, "01", 1)
    assert len(data["entries"]) == 6
    e1 = data["entries"][0]
    assert e1["waku"] == 1
    assert e1["racer_id"] == "3778"
    assert e1["national_win_rate"] == 2.87
    assert e1["motor_no"] == 70
    assert e1["motor_quinella_rate"] == 34.38


def test_parse_beforeinfo_exhibition_and_weather():
    html = (FIX / "beforeinfo_sample.html").read_text(encoding="utf-8")
    data = OfficialCollector()._parse_beforeinfo(html)
    assert data["weather"]["temperature"] == 24.0
    assert data["weather"]["wind_speed"] == 2.0
    assert data["exhibition"][1]["exhibition_time"] == 6.6
    assert data["exhibition"][1]["tilt"] == -0.5
