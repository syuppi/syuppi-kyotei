"""的中率改善のための特徴量設計・ロジック仕様（実装と同期）。

短期=既存DBで実装済み / 中期=追加収集が必要 / 長期=高度モデル
"""

from __future__ import annotations

# 優先度: P0=短期実装, P1=中期, P2=長期
DATA_PRIORITY: dict[str, str] = {
    # 展示
    "exhibition_st": "P0",
    "exhibition_time": "P0",
    "exhibition_rank_gap": "P0",
    "st_gap_1_vs_2": "P0",
    "stretch_turn_quality": "P2",  # 映像/周回展示詳細が必要
    "nobí_ashi_type": "P2",
    # モーター
    "motor_quinella_card": "P0",
    "motor_recent_venue_form": "P0",
    "parts_changed_history": "P1",  # Official HTML埋込が必要
    "motor_feel_shift": "P2",
    # 選手
    "racer_course_win_rate": "P0",
    "racer_course_avg_st": "P0",
    "racer_recent_form": "P0",
    "avg_st_3m": "P0",  # 履歴STで近似
    # 場
    "venue_in_win_rate": "P0",
    "venue_kimarite_rates": "P0",
    "venue_kado_strength": "P0",
    # 気象
    "wind_speed_wave": "P0",
    "temperature": "P0",
    "relative_wind_bucket": "P1",  # 方位→相対風の場別変換
    # 飛び
    "course1_fly_risk": "P0",
}

# MLに追加する実力系特徴（枠リークにならないもの）
HITRATE_EXTRA_FEATURES: list[str] = [
    "racer_course_st",
    "racer_recent_form",
    "motor_recent_q",
    "venue_nige_rate",
    "venue_makuri_rate",
    "venue_sashi_rate",
    "venue_kado_strength",
    "venue_in_win_rate",
    "temperature_norm",
    "ex_time_gap",
    "ex_st_gap_vs_best",
    # st_gap_vs_inner / course1_upset_pressure は枠を代理しやすいのでML除外
    "course1_fly_risk",
]

# 評価指標
EVAL_METRICS: list[str] = [
    "win_hit",  # 本命1着的中
    "favorite_in_top3",  # 本命が3着以内
    "always1_baseline",  # 常時1号艇ベースラインとの差分
    "course1_fly_precision",  # 飛び予測が当たった率
    "course1_fly_recall",
    "fav1_rate",  # 本命多様性
    "venue_slice_win",
    "weather_slice_win",
]
