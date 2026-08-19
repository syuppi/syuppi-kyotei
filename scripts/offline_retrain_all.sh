#!/usr/bin/env bash
# オフライン一括再学習: データ収集 → LightGBM → 自信度モデル → 評価レポート
#
# 使い方:
#   ./scripts/offline_retrain_all.sh
#   ./scripts/offline_retrain_all.sh --skip-collect
#   ./scripts/offline_retrain_all.sh --smoke
#   ./scripts/offline_retrain_all.sh --skip-holdout
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"

SKIP_COLLECT=0
SMOKE=0
SKIP_HOLDOUT=0
LOOKBACK_DAYS="${BOATRACE_LOOKBACK_DAYS:-60}"

for arg in "$@"; do
  case "$arg" in
    --skip-collect) SKIP_COLLECT=1 ;;
    --smoke) SMOKE=1 ;;
    --skip-holdout) SKIP_HOLDOUT=1 ;;
    --help|-h)
      echo "Usage: $0 [--skip-collect] [--smoke] [--skip-holdout]"
      echo "  --skip-collect  DBにデータがある前提で学習のみ実行"
      echo "  --smoke         LightGBMを軽量検証モード（本番モデル非更新）"
      echo "  --skip-holdout  ホールドアウト検証を省略（直近数日のデータのみの場合）"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 1
      ;;
  esac
done

echo "==> offline retrain (root=$ROOT)"

if [[ "$SKIP_COLLECT" -eq 0 ]]; then
  echo "==> [1/4] collect historical data (lookback=${LOOKBACK_DAYS} days)"
  "$PYTHON" scripts/init_db.py
  "$PYTHON" scripts/collect_real.py
else
  echo "==> [1/4] skip collect"
fi

echo "==> [2/4] train LightGBM"
HOLDOUT_ARGS=()
if [[ "$SKIP_HOLDOUT" -eq 1 ]]; then
  HOLDOUT_ARGS=(--skip-holdout)
fi
if [[ "$SMOKE" -eq 1 ]]; then
  "$PYTHON" scripts/train_lgbm.py --smoke --smoke-days 7 "${HOLDOUT_ARGS[@]}"
else
  "$PYTHON" scripts/train_lgbm.py "${HOLDOUT_ARGS[@]}"
fi

if [[ "$SMOKE" -eq 1 ]]; then
  echo "==> smoke mode: skip confidence train and eval"
  exit 0
fi

echo "==> [3/4] train confidence meta-model"
CONF_ARGS=()
if [[ "$SKIP_HOLDOUT" -eq 1 ]]; then
  CONF_ARGS=(--auto-split)
fi
"$PYTHON" scripts/train_confidence.py "${CONF_ARGS[@]}"

echo "==> [4/4] evaluation report"
"$PYTHON" scripts/eval_trifecta_report.py

echo "==> done. Models:"
ls -la data/models/*.joblib 2>/dev/null || true
