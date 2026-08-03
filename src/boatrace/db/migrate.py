"""SQLite向けの簡易マイグレーション（不足カラム追加）."""

from __future__ import annotations

from sqlalchemy import inspect, text

from boatrace.db.session import get_engine
from boatrace.logging_setup import get_logger

logger = get_logger(__name__)

RACE_ENTRY_COLS = {
    "exhibition_st": "FLOAT",
    "weight_adjustment": "FLOAT",
    "parts_changed_flag": "BOOLEAN DEFAULT 0",
    "motor_trio_rate": "FLOAT",
    "boat_trio_rate": "FLOAT",
    "grade_code": "VARCHAR(4)",
    "win_odds": "FLOAT",
}

RACE_CARD_COLS = {
    "grade_number": "INTEGER",
    "day_number": "INTEGER",
    "distance_m": "INTEGER",
}


ACCURACY_DAILY_COLS = {
    "hit_trifecta": "INTEGER DEFAULT 0",
    "trifecta_rate": "FLOAT DEFAULT 0",
}


def migrate() -> dict[str, list[str]]:
    engine = get_engine()
    added: dict[str, list[str]] = {
        "race_entry": [],
        "race_card": [],
        "accuracy_daily": [],
    }
    insp = inspect(engine)
    with engine.begin() as conn:
        entry_cols = {c["name"] for c in insp.get_columns("race_entry")}
        for name, typ in RACE_ENTRY_COLS.items():
            if name not in entry_cols:
                conn.execute(text(f"ALTER TABLE race_entry ADD COLUMN {name} {typ}"))
                added["race_entry"].append(name)
        card_cols = {c["name"] for c in insp.get_columns("race_card")}
        for name, typ in RACE_CARD_COLS.items():
            if name not in card_cols:
                conn.execute(text(f"ALTER TABLE race_card ADD COLUMN {name} {typ}"))
                added["race_card"].append(name)
        if "accuracy_daily" in insp.get_table_names():
            acc_cols = {c["name"] for c in insp.get_columns("accuracy_daily")}
            for name, typ in ACCURACY_DAILY_COLS.items():
                if name not in acc_cols:
                    conn.execute(text(f"ALTER TABLE accuracy_daily ADD COLUMN {name} {typ}"))
                    added["accuracy_daily"].append(name)
    logger.info("migrate_done", added=added)
    return added


if __name__ == "__main__":
    print(migrate())
