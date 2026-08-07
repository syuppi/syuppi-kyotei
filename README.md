# ボートレース予想システム

場ごとのクセを最優先に学習し、説明可能なスコアリングで着順予測を行うシステムです。

## 特徴

- BOAT RACE公式・水面条件・潮位データの自動収集
- 場別・条件別の継続学習（当日結果 → 翌日予測へ反映）
- 説明可能なルールベーススコア + ML差し替え可能な設計
- 予測一覧 / 場傾向 / 精度検証の Web UI

## データ取得

実データは次の順で取得します。

1. **Boatrace Open API**（非公式JSON・全24場・出走/直前/結果）  
   `https://boatraceopenapi.github.io/api/v1/today.json`
2. 失敗時は **BOAT RACE公式HTML** へフォールバック
3. 潮位は JMA到達確認＋天文近似（場別 `tide_sensitive` 設定あり）

## 機械学習

```bash
# 90日学習済みDB上で LightGBM を学習し、本日を再予想
python scripts/train_lgbm.py
```

モデルは `data/models/lgbm_win_v1.joblib` に保存され、以降の予測は自動で LightGBM（ルールベースとブレンド）を使います。

各場の公式ホームページは広報・ライブ映像中心で、構造化出走表としては公式ポータル／Open APIの方が安定です。

## Render で当日予想だけ公開

無料枠向けに **学習スキップ＋当日予想** モードがあります。

```bash
# 環境変数
BOATRACE_PREDICT_ONLY=1
BOATRACE_WARM_LOOKBACK_DAYS=14
```

手順は [docs/RENDER.md](docs/RENDER.md) を参照（`Dockerfile` / `render.yaml` 同梱）。

## 設計

詳細は [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) を参照。

## フェーズ

1. **ルールベース + 統計**（現行）
2. **機械学習モデル追加**（`boatrace.models.ml_model`）
3. **モデル比較・自動重み調整**（`boatrace.learning.weight_tuner`）
