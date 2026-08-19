#!/usr/bin/env bash
# Render / コンテナ起動: 空DBでも当日予想できるように初期化して起動する。
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-src}"
export BOATRACE_PREDICT_ONLY="${BOATRACE_PREDICT_ONLY:-1}"

echo "[start_render] init-db"
python scripts/init_db.py

echo "[start_render] bootstrap today cards (JST)"
python scripts/bootstrap_today.py || echo "[start_render] bootstrap_today failed (continuing)"

PORT="${PORT:-8000}"
echo "[start_render] uvicorn :${PORT} predict_only=${BOATRACE_PREDICT_ONLY}"
exec python -m uvicorn boatrace.api.main:app --host 0.0.0.0 --port "${PORT}"
