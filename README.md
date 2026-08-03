# ボートレース予想システム

場ごとのクセを最優先に学習し、説明可能なスコアリングで着順予測を行うシステムです。

## 特徴

- BOAT RACE公式・水面条件・潮位データの自動収集
- 場別・条件別の継続学習（当日結果 → 翌日予測へ反映）
- 説明可能なルールベーススコア + ML差し替え可能な設計
- 予測一覧 / 場傾向 / 精度検証の Web UI

## クイックスタート

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# DB初期化 + デモデータ
python scripts/init_db.py
python scripts/seed_demo.py

# 当日データ収集 → 予測 → 結果学習
python scripts/collect_daily.py
python scripts/predict_today.py
python scripts/learn_results.py

# Web / API
uvicorn boatrace.api.main:app --reload --host 0.0.0.0 --port 8000
```

ブラウザで http://localhost:8000 を開いてください。

## 設計

詳細は [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) を参照。

## フェーズ

1. **ルールベース + 統計**（現行）
2. **機械学習モデル追加**（`boatrace.models.ml_model`）
3. **モデル比較・自動重み調整**（`boatrace.learning.weight_tuner`）
