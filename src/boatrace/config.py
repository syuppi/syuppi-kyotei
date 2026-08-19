"""設定読み込み."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"


class DatabaseConfig(BaseModel):
    url: str = "sqlite:///./data/boatrace.db"
    echo: bool = False


class CollectConfig(BaseModel):
    base_url: str = "https://www.boatrace.jp"
    timeout_sec: float = 20.0
    max_retries: int = 3
    retry_backoff_sec: float = 2.0
    user_agent: str = "Mozilla/5.0 (compatible; BoatRacePredict/1.0)"
    lookback_days: int = 7
    request_interval_sec: float = 0.4


class RuntimeConfig(BaseModel):
    """デプロイ形態（Render無料の当日予想など）."""

    # True: 場統計・重み学習をスキップ。予想とデータ取得のみ。
    predict_only: bool = False
    # 起動時に直近結果を裏で取得して特徴を暖める日数（0で無効）
    warm_lookback_days: int = 0


class JobsConfig(BaseModel):
    """バックグラウンドジョブ."""

    preclose_predict_interval_sec: int = 300
    preclose_lead_minutes: int = 45
    result_refresh_interval_sec: int = 600


class TideConfig(BaseModel):
    jma_base_url: str = "https://www.data.jma.go.jp"
    kaiho_base_url: str = "https://www1.kaiho.mlit.go.jp"
    timeout_sec: float = 20.0
    max_retries: int = 3


class PredictionConfig(BaseModel):
    model_name: str = "scoring_v1"
    temperature: float = 0.85
    upset_margin_threshold: float = 0.08
    recent_n_races: int = 10
    recent_decay: float = 0.85
    win_candidates: int = 3
    sanrenpuku_candidates: int = 5
    sanrentan_candidates: int = 5
    # 自信ありレース選別（3連複的中メタモデル）
    confidence_threshold: float = 0.75
    confidence_target_coverage: float = 0.02
    confidence_max_per_day: int = 3
    confidence_enabled: bool = True


class LearningConfig(BaseModel):
    min_samples_for_bias: int = 20
    weight_lr: float = 0.05
    accuracy_window_days: int = 30


class LoggingConfig(BaseModel):
    level: str = "INFO"
    json_logs: bool = False


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    title: str = "BoatRace Prediction System"


class AppSettings(BaseModel):
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    collect: CollectConfig = Field(default_factory=CollectConfig)
    tide: TideConfig = Field(default_factory=TideConfig)
    prediction: PredictionConfig = Field(default_factory=PredictionConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    jobs: JobsConfig = Field(default_factory=JobsConfig)


class EnvOverrides(BaseSettings):
    """環境変数による上書き."""

    database_url: str | None = None
    log_level: str | None = None
    predict_only: str | None = None
    lookback_days: int | None = None
    warm_lookback_days: int | None = None

    model_config = {"env_prefix": "BOATRACE_", "extra": "ignore"}


class VenueConfig(BaseModel):
    id: str
    name: str
    name_kana: str = ""
    prefecture: str = ""
    tide_sensitive: bool = False
    water_type: str = "unknown"
    tide_station: str | None = None
    typical_in_advantage: float = 0.52
    # ホームストレッチ進行方位（度, 北=0）。絶対風向→相対変換用
    course_heading_deg: float | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML root in {path}")
    return data


def _truthy(v: bool | str | None) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


@lru_cache
def get_settings() -> AppSettings:
    raw = _load_yaml(CONFIG_DIR / "settings.yaml")
    settings = AppSettings.model_validate(raw)
    env = EnvOverrides()
    if env.database_url:
        settings.database.url = env.database_url
    if env.log_level:
        settings.logging.level = env.log_level
    if env.predict_only is not None and str(env.predict_only).strip() != "":
        settings.runtime.predict_only = _truthy(env.predict_only)
    if env.lookback_days is not None:
        settings.collect.lookback_days = max(0, int(env.lookback_days))
    if env.warm_lookback_days is not None:
        settings.runtime.warm_lookback_days = max(0, int(env.warm_lookback_days))
    elif settings.runtime.predict_only and settings.runtime.warm_lookback_days <= 0:
        # Render当日予想モードの既定: 特徴用に直近を暖機
        settings.runtime.warm_lookback_days = max(7, int(settings.collect.lookback_days or 7))
    # sqlite相対パスをプロジェクトルート基準に
    if settings.database.url.startswith("sqlite:///./"):
        rel = settings.database.url.replace("sqlite:///./", "", 1)
        abs_path = (ROOT_DIR / rel).resolve()
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        settings.database.url = f"sqlite:///{abs_path}"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return settings


def is_predict_only() -> bool:
    return bool(get_settings().runtime.predict_only)


@lru_cache
def get_venues() -> list[VenueConfig]:
    raw = _load_yaml(CONFIG_DIR / "venues.yaml")
    items = raw.get("venues", [])
    return [VenueConfig.model_validate(v) for v in items]


def get_venue_map() -> dict[str, VenueConfig]:
    return {v.id: v for v in get_venues()}
