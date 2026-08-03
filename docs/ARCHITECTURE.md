# ボートレース予想システム — 設計書

## 1. 全体アーキテクチャ

```
┌─────────────────────────────────────────────────────────────────┐
│                        Presentation                              │
│   FastAPI (REST)  +  Jinja2 Web UI (予測一覧/場傾向/精度検証)     │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     Prediction Service                           │
│  FeatureBuilder → ScoringModel → (optional ML) → Explainer       │
└───────┬─────────────────────────────┬───────────────────────────┘
        │                             │
┌───────▼──────────┐         ┌────────▼───────────────────────────┐
│ Learning Pipeline│         │         SQLite / PostgreSQL        │
│ ・結果取込        │◄───────►│ venue / race_card / race_result    │
│ ・場別補正更新    │         │ weather / tide / motor / boat      │
│ ・重み自動調整    │         │ predict_history / venue_bias       │
└───────┬──────────┘         └────────────────────────────────────┘
        │
┌───────▼──────────────────────────────────────────────────────────┐
│                      Collectors (再試行付き)                      │
│  Official(BOAT RACE) │ Weather(直前情報) │ Tide(JMA/海上保安庁)   │
└──────────────────────────────────────────────────────────────────┘
```

### 処理フロー（日次）

1. **collect**: 当日プログラム・直前情報・潮位を取得し正規化保存
2. **predict**: 特徴量生成 → スコアリング → 予測理由付きで保存
3. **ingest_results**: 確定結果を取り込み
4. **learn**: 場別・条件別統計と補正係数を更新（翌日予測に反映）
5. **evaluate**: 予測履歴と結果を突合し精度を集計

### 設計原則

- **場ごとのクセ最優先**: `venue_course_stats` / `venue_bias` を中核に置く
- **説明可能性**: ルールベーススコアを第一モデルとし、寄与度を理由文に変換
- **段階的高度化**: Scoring → ML(LightGBM等) → モデル比較 の差し替え可能な `BasePredictor`
- **正規化 + 履歴**: マスター正規化テーブルと `*_snapshot` / `predict_history` を両立

---

## 2. DB設計

### 主要テーブル

| テーブル | 役割 |
|---------|------|
| `venue` | 開催場マスター（潮汐影響フラグ含む固定設定） |
| `racer` | 選手マスター |
| `race_card` | レースカード（場・日付・R・締切等） |
| `race_entry` | 出走表（枠番・選手・モーター・ボート） |
| `race_result` | 着順・ST・決まり手 |
| `weather_snapshot` | 気温・天候・風速・風向・水温・波高 |
| `tide_snapshot` | 潮位・満干潮時刻・潮位変化量 |
| `exhibition` | 展示タイム・チルト・部品交換 |
| `motor_stats` / `boat_stats` | モーター/ボート2連対率等 |
| `racer_stats` | 当地/全国/季節/直近成績 |
| `venue_course_stats` | 場×コース別入着傾向 |
| `predict_history` | 予測結果・理由・モデル版 |
| `model_weights` | スコア重み（自動調整対象） |
| `venue_bias` | 場別補正係数 |
| `accuracy_daily` | 日次精度集計 |
| `fetch_log` | 取得成否・再試行・更新日時 |

### 正規化キー

- レース: `(venue_id, race_date, race_no)`
- 出走: `(venue_id, race_date, race_no, waku)` ※枠番1〜6
- 選手: `racer_id`（登録番号）
- 条件帯: 風速帯 / 潮位帯 / 展示タイム帯 を集計時に離散化

---

## 3. 取得データ項目一覧

### BOAT RACE公式

- 開催日・場コード・R番号・レース種別・締切時刻
- 枠番・選手登録番号・級別・年齢・体重
- 当地成績（勝率/2連対/3連対）・全国成績
- 平均ST・モーター番号・モーター2連対率・ボート番号・ボート2連対率
- 前走成績（着順・ST）
- 直前: 展示タイム・チルト・部品交換・進入予想
- 結果: 着順・ST・決まり手・払戻

### 水面条件

- 気温 / 天候 / 風速 / 風向 / 水温 / 波高

### 潮位（JMA / 海上保安庁系）

- 観測点コード・潮位(cm)・満潮/干潮時刻・潮位変化量(Δ)
- 場ごとの `tide_sensitive` 固定フラグで影響の強弱を判定

### 過去結果

- 開催日ごとの直近N日分（デフォルト7日）の結果をバッチ取得

---

## 4. 学習用特徴量設計

| カテゴリ | 特徴量 | 備考 |
|---------|--------|------|
| 号艇 | win/quinella/trio rate | 全国・場別 |
| 選手 | local_win_rate, recent_form_wavg | 直近N走指数減衰 |
| 展示 | exhibition_time, time_rank, time_gap_to_best | |
| ST | avg_st, st_gap_to_best | |
| 機材 | motor_quinella_rate, boat_quinella_rate | |
| 環境 | wind_speed, wind_dir_bucket, wave, water_temp | |
| 潮汐 | tide_level, tide_delta, near_high_tide | tide_sensitive場のみ強く効かせる |
| 進入 | is_fixed_entry | |
| 場クセ | venue_course_win_rate, in_advantage_score | 場別補正の核 |
| 季節 | season_bucket 別成績 | |

---

## 5. 予測スコア初期式

号艇 \(i\) の生スコア:

\[
S_i = \sum_k w_k \cdot f_{k,i} \cdot c_{venue,k} \cdot e_{env,k}
\]

初期重み \(w\)（合計1.0に正規化前の相対値）:

| 特徴 | 初期重み |
|------|---------|
| venue_course_win_rate | 0.22 |
| local_win_rate | 0.14 |
| exhibition_advantage | 0.14 |
| motor_quinella_rate | 0.10 |
| national_win_rate | 0.08 |
| recent_form | 0.08 |
| boat_quinella_rate | 0.06 |
| st_advantage | 0.06 |
| tide_adjustment | 0.06 |
| wind_course_bias | 0.06 |

環境補正の例:

- 向かい風 ≥ 3m: 差し・まくり寄り（アウトコース加点、イン減点）
- 追い風: イン信頼度やや上昇
- 満潮付近 & tide_sensitive: 1号艇信頼度を減衰
- 波高高め: イン有利をやや強化

確率化:

\[
P_i^{1st} = \mathrm{softmax}(S_i / T)
\]

2連対・3連対は上位スコア組み合わせの近似（独立仮定 + 場別共起補正）。

穴判定: \(P^{1st}_{favorite} - P^{1st}_{second} < \theta\) または アウトの展示最上位かつイン当地低迷。

---

## 6. ファイル構成

```
boatrace_predict/          # パッケージルート (src layout)
config/
  settings.yaml
  venues.yaml
docs/ARCHITECTURE.md
scripts/
  init_db.py
  collect_daily.py
  predict_today.py
  learn_results.py
  seed_demo.py
src/boatrace/
  ...
tests/
requirements.txt
README.md
```
