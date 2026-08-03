"""CLIエントリ."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import typer
from rich import print as rprint

from boatrace.collectors.pipeline import collect_daily
from boatrace.config import get_settings
from boatrace.db.models import Base, ModelWeights, Venue
from boatrace.db.session import get_engine, session_scope
from boatrace.features.builder import DEFAULT_WEIGHTS
from boatrace.config import get_venues
from boatrace.learning.service import LearningService
from boatrace.logging_setup import setup_logging
from boatrace.prediction.service import PredictionService

app = typer.Typer(help="ボートレース予想システム CLI")


def _parse_date(value: Optional[str]) -> date:
    if not value:
        return date.today()
    return datetime.strptime(value, "%Y-%m-%d").date()


@app.command("init-db")
def init_db() -> None:
    """テーブル作成と会場・初期重みを投入."""
    setup_logging()
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    with session_scope() as session:
        for v in get_venues():
            existing = session.get(Venue, v.id)
            if existing is None:
                session.add(
                    Venue(
                        id=v.id,
                        name=v.name,
                        name_kana=v.name_kana,
                        prefecture=v.prefecture,
                        tide_sensitive=v.tide_sensitive,
                        water_type=v.water_type,
                        tide_station=v.tide_station,
                        typical_in_advantage=v.typical_in_advantage,
                    )
                )
            else:
                existing.name = v.name
                existing.tide_sensitive = v.tide_sensitive
                existing.water_type = v.water_type
                existing.tide_station = v.tide_station
                existing.typical_in_advantage = v.typical_in_advantage

        model_name = get_settings().prediction.model_name
        for k, w in DEFAULT_WEIGHTS.items():
            row = (
                session.query(ModelWeights)
                .filter_by(model_name=model_name, feature_key=k)
                .one_or_none()
            )
            if row is None:
                session.add(ModelWeights(model_name=model_name, feature_key=k, weight=w))
    rprint("[green]DB initialized[/green]", get_settings().database.url)


@app.command("collect")
def collect(
    day: Optional[str] = typer.Option(None, help="YYYY-MM-DD"),
    venue: Optional[str] = typer.Option(None, help="場コード 例: 01"),
    no_lookback: bool = typer.Option(False, help="直近過去日を取得しない"),
) -> None:
    setup_logging()
    venues = [venue] if venue else None
    result = collect_daily(_parse_date(day), venue_ids=venues, include_lookback=not no_lookback)
    rprint(result)


@app.command("predict")
def predict(
    day: Optional[str] = typer.Option(None, help="YYYY-MM-DD"),
    venue: Optional[str] = typer.Option(None),
) -> None:
    setup_logging()
    with session_scope() as session:
        svc = PredictionService(session)
        outputs = svc.predict_day(_parse_date(day), venue_id=venue)
    rprint(f"[cyan]predictions[/cyan]: {len(outputs)}")
    for o in outputs[:5]:
        rprint(
            f"{o['venue_name']} {o['race_no']}R → {o['rankings']} "
            f"win={o['candidates_win']} upset={o['has_upset']}"
        )


@app.command("learn")
def learn(day: Optional[str] = typer.Option(None, help="YYYY-MM-DD")) -> None:
    setup_logging()
    with session_scope() as session:
        svc = LearningService(session)
        result = svc.learn_day(_parse_date(day))
    rprint(result)


if __name__ == "__main__":
    app()
