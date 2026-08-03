"""ORMモデル定義."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Venue(Base):
    __tablename__ = "venue"

    id: Mapped[str] = mapped_column(String(2), primary_key=True)
    name: Mapped[str] = mapped_column(String(32), nullable=False)
    name_kana: Mapped[str] = mapped_column(String(64), default="")
    prefecture: Mapped[str] = mapped_column(String(32), default="")
    tide_sensitive: Mapped[bool] = mapped_column(Boolean, default=False)
    water_type: Mapped[str] = mapped_column(String(32), default="unknown")
    tide_station: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    typical_in_advantage: Mapped[float] = mapped_column(Float, default=0.52)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    race_cards: Mapped[list["RaceCard"]] = relationship(back_populates="venue")


class Racer(Base):
    __tablename__ = "racer"

    id: Mapped[str] = mapped_column(String(4), primary_key=True)  # 登録番号
    name: Mapped[str] = mapped_column(String(64), default="")
    branch: Mapped[str] = mapped_column(String(16), default="")
    grade: Mapped[str] = mapped_column(String(4), default="")  # A1/A2/B1/B2
    birth_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class RaceCard(Base):
    __tablename__ = "race_card"
    __table_args__ = (
        UniqueConstraint("venue_id", "race_date", "race_no", name="uq_race_card"),
        Index("ix_race_card_date", "race_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    venue_id: Mapped[str] = mapped_column(String(2), ForeignKey("venue.id"), nullable=False)
    race_date: Mapped[date] = mapped_column(Date, nullable=False)
    race_no: Mapped[int] = mapped_column(Integer, nullable=False)
    race_title: Mapped[str] = mapped_column(String(128), default="")
    race_grade: Mapped[str] = mapped_column(String(32), default="")
    grade_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    day_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    distance_m: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    deadline_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    is_fixed_entry: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(16), default="scheduled")  # scheduled/finished
    raw_payload: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    fetched_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    venue: Mapped["Venue"] = relationship(back_populates="race_cards")
    entries: Mapped[list["RaceEntry"]] = relationship(
        back_populates="race_card", cascade="all, delete-orphan"
    )
    weather: Mapped[Optional["WeatherSnapshot"]] = relationship(
        back_populates="race_card", uselist=False, cascade="all, delete-orphan"
    )
    tide: Mapped[Optional["TideSnapshot"]] = relationship(
        back_populates="race_card", uselist=False, cascade="all, delete-orphan"
    )
    result: Mapped[Optional["RaceResult"]] = relationship(
        back_populates="race_card", uselist=False, cascade="all, delete-orphan"
    )


class RaceEntry(Base):
    __tablename__ = "race_entry"
    __table_args__ = (
        UniqueConstraint("race_card_id", "waku", name="uq_race_entry_waku"),
        Index("ix_race_entry_racer", "racer_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_card_id: Mapped[int] = mapped_column(ForeignKey("race_card.id"), nullable=False)
    waku: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-6
    racer_id: Mapped[str] = mapped_column(String(4), ForeignKey("racer.id"), nullable=False)
    age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    weight: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    f_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    l_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    avg_st: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    national_win_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    national_quinella_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    national_trio_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    local_win_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    local_quinella_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    local_trio_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    motor_no: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    motor_quinella_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    boat_no: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    boat_quinella_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    previous_course: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    previous_rank: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    previous_st: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    exhibition_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    exhibition_st: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # スタート展示ST
    tilt: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    weight_adjustment: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    parts_changed: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    parts_changed_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    estimated_course: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    motor_trio_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    boat_trio_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    grade_code: Mapped[Optional[str]] = mapped_column(String(4), nullable=True)  # A1/A2/B1/B2
    win_odds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 単勝オッズ
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    race_card: Mapped["RaceCard"] = relationship(back_populates="entries")
    racer: Mapped["Racer"] = relationship()


class WeatherSnapshot(Base):
    __tablename__ = "weather_snapshot"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_card_id: Mapped[int] = mapped_column(
        ForeignKey("race_card.id"), unique=True, nullable=False
    )
    temperature: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    weather: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    wind_speed: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    wind_direction: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    water_temperature: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    wave_height: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="official")
    fetched_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    race_card: Mapped["RaceCard"] = relationship(back_populates="weather")


class TideSnapshot(Base):
    __tablename__ = "tide_snapshot"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_card_id: Mapped[int] = mapped_column(
        ForeignKey("race_card.id"), unique=True, nullable=False
    )
    station_name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    tide_level_cm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tide_delta_cm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    near_high_tide: Mapped[bool] = mapped_column(Boolean, default=False)
    near_low_tide: Mapped[bool] = mapped_column(Boolean, default=False)
    high_tide_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    low_tide_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="jma")
    fetched_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    race_card: Mapped["RaceCard"] = relationship(back_populates="tide")


class RaceResult(Base):
    __tablename__ = "race_result"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_card_id: Mapped[int] = mapped_column(
        ForeignKey("race_card.id"), unique=True, nullable=False
    )
    rank1_waku: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rank2_waku: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rank3_waku: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    kimarite: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    payouts: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    entry_results: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(JSON, nullable=True)
    # entry_results: [{waku, rank, st, course}, ...]
    fetched_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    race_card: Mapped["RaceCard"] = relationship(back_populates="result")


class MotorStats(Base):
    __tablename__ = "motor_stats"
    __table_args__ = (UniqueConstraint("venue_id", "motor_no", "as_of_date", name="uq_motor"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    venue_id: Mapped[str] = mapped_column(String(2), ForeignKey("venue.id"), nullable=False)
    motor_no: Mapped[int] = mapped_column(Integer, nullable=False)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    quinella_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    trio_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    starts: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class BoatStats(Base):
    __tablename__ = "boat_stats"
    __table_args__ = (UniqueConstraint("venue_id", "boat_no", "as_of_date", name="uq_boat"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    venue_id: Mapped[str] = mapped_column(String(2), ForeignKey("venue.id"), nullable=False)
    boat_no: Mapped[int] = mapped_column(Integer, nullable=False)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    quinella_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    trio_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    starts: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class VenueCourseStats(Base):
    """場×コース別入着傾向."""

    __tablename__ = "venue_course_stats"
    __table_args__ = (
        UniqueConstraint("venue_id", "course", "condition_key", name="uq_venue_course"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    venue_id: Mapped[str] = mapped_column(String(2), ForeignKey("venue.id"), nullable=False)
    course: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-6
    condition_key: Mapped[str] = mapped_column(String(64), default="all")
    # condition_key例: all / wind_head_ge3 / wind_tail / tide_high / wave_ge5
    starts: Mapped[int] = mapped_column(Integer, default=0)
    wins: Mapped[int] = mapped_column(Integer, default=0)
    quinellas: Mapped[int] = mapped_column(Integer, default=0)
    trios: Mapped[int] = mapped_column(Integer, default=0)
    win_rate: Mapped[float] = mapped_column(Float, default=0.0)
    quinella_rate: Mapped[float] = mapped_column(Float, default=0.0)
    trio_rate: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class VenueBias(Base):
    """場別補正係数."""

    __tablename__ = "venue_bias"
    __table_args__ = (UniqueConstraint("venue_id", "feature_key", name="uq_venue_bias"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    venue_id: Mapped[str] = mapped_column(String(2), ForeignKey("venue.id"), nullable=False)
    feature_key: Mapped[str] = mapped_column(String(64), nullable=False)
    coefficient: Mapped[float] = mapped_column(Float, default=1.0)
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class ModelWeights(Base):
    __tablename__ = "model_weights"
    __table_args__ = (UniqueConstraint("model_name", "feature_key", name="uq_model_weight"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_key: Mapped[str] = mapped_column(String(64), nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class PredictHistory(Base):
    __tablename__ = "predict_history"
    __table_args__ = (
        UniqueConstraint("race_card_id", "model_name", name="uq_predict"),
        Index("ix_predict_date", "predicted_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_card_id: Mapped[int] = mapped_column(ForeignKey("race_card.id"), nullable=False)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    # per-waku probabilities and reasons
    rankings: Mapped[list[int]] = mapped_column(JSON, nullable=False)  # [waku,...] expected order
    win_probs: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)
    quinella_probs: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)
    trio_probs: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)
    candidates_win: Mapped[list[int]] = mapped_column(JSON, nullable=False)
    candidates_quinella: Mapped[list[int]] = mapped_column(JSON, nullable=False)
    candidates_trio: Mapped[list[int]] = mapped_column(JSON, nullable=False)
    upset_candidates: Mapped[list[int]] = mapped_column(JSON, default=list)
    reasons: Mapped[dict[str, list[str]]] = mapped_column(JSON, nullable=False)
    feature_snapshot: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    scores: Mapped[Optional[dict[str, float]]] = mapped_column(JSON, nullable=True)
    has_upset: Mapped[bool] = mapped_column(Boolean, default=False)
    predicted_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class AccuracyDaily(Base):
    __tablename__ = "accuracy_daily"
    __table_args__ = (
        UniqueConstraint("stat_date", "venue_id", "slice_key", "model_name", name="uq_acc"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stat_date: Mapped[date] = mapped_column(Date, nullable=False)
    venue_id: Mapped[Optional[str]] = mapped_column(String(2), nullable=True)  # null=全体
    slice_key: Mapped[str] = mapped_column(String(64), default="all")
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    n_races: Mapped[int] = mapped_column(Integer, default=0)
    hit_win: Mapped[int] = mapped_column(Integer, default=0)
    hit_quinella: Mapped[int] = mapped_column(Integer, default=0)
    hit_trio: Mapped[int] = mapped_column(Integer, default=0)
    hit_trifecta: Mapped[int] = mapped_column(Integer, default=0)
    win_rate: Mapped[float] = mapped_column(Float, default=0.0)
    quinella_rate: Mapped[float] = mapped_column(Float, default=0.0)
    trio_rate: Mapped[float] = mapped_column(Float, default=0.0)
    trifecta_rate: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class FetchLog(Base):
    __tablename__ = "fetch_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    target_key: Mapped[str] = mapped_column(String(128), nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, default=False)
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    payload_summary: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
