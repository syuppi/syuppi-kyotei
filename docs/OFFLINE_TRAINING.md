# オフライン再学習ガイド

本番（Render）では学習を行わず予想のみ実行します。モデルを更新する場合はローカルまたは専用マシンで以下を実行してください。

## 前提

- Python 3.11+
- 依存関係インストール済み（`pip install -e .` または `requirements.txt`）
- 公式サイトからのデータ収集が可能なネットワーク

## 一括実行

```bash
chmod +x scripts/offline_retrain_all.sh
./scripts/offline_retrain_all.sh
```

### オプション

| オプション | 説明 |
|-----------|------|
| `--skip-collect` | DB に履歴がある場合、収集をスキップして学習のみ |
| `--smoke` | LightGBM を軽量検証のみ（本番モデル非更新） |
| `--skip-holdout` | ホールドアウト検証を省略し、自信度学習は `--auto-split` 相当で日付自動分割 |

環境変数 `BOATRACE_LOOKBACK_DAYS` で収集日数を変更できます（既定 60）。

直近数日しか DB にない場合:

```bash
./scripts/offline_retrain_all.sh --skip-collect --skip-holdout
```

## 個別スクリプト

| 順序 | スクリプト | 内容 |
|------|-----------|------|
| 1 | `scripts/init_db.py` | DB 初期化 |
| 2 | `scripts/collect_real.py` | 実データ収集 |
| 3 | `scripts/train_lgbm.py` | LightGBM 本命モデル学習 → `data/models/lgbm_win_v1.joblib` |
| 4 | `scripts/train_confidence.py` | 自信度メタモデル → `data/models/race_confidence_v1.joblib` |
| 5 | `scripts/eval_trifecta_report.py` | 3連複/3連単カバー率レポート |

## 設定との関係

`config/settings.yaml` の自信あり関連:

- `confidence_threshold`: 0.75（閾値）
- `confidence_max_per_day`: 3（1日最大3R）
- `confidence_target_coverage`: 0.02（学習時の選別率目標）

学習後は `data/models/*.joblib` と `data/models/trifecta_eval_report.json` を Git にコミットし、Render にデプロイしてください。API 起動時に JSON を自動読み込みし、DB が空でも `/api/accuracy/*` に参考精度を表示します。

## 参考精度（オフライン検証）

- 3連複5点 any: 約 60% 前後（期間・データ量依存）
- 自信あり 3連単 top3: 約 20% 前後（5回に1回程度）

本番の `/api/accuracy/summary` は **締切前予想** のみ集計対象です。結果後の再予想は的中表示されません。
