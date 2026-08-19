#!/usr/bin/env bash
# GitHub push → Render 自動デプロイ（初回は PAT または gh auth が必要）
set -euo pipefail
cd "$(dirname "$0")/.."

BRANCH="${1:-cursor/render-predict-only-1822}"
TOKEN="${GITHUB_TOKEN:-${GH_TOKEN:-}}"

if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  echo "[deploy] push via gh (${BRANCH})"
  git push -u origin "${BRANCH}"
elif [[ -n "${TOKEN}" ]]; then
  echo "[deploy] push via GITHUB_TOKEN (${BRANCH})"
  git push "https://x-access-token:${TOKEN}@github.com/syuppi/syuppi-kyotei.git" "${BRANCH}"
else
  echo "ERROR: GitHub 認証がありません。" >&2
  echo "  export GITHUB_TOKEN=ghp_xxxx  && $0" >&2
  echo "  または: gh auth login" >&2
  exit 1
fi

if [[ -n "${RENDER_DEPLOY_HOOK:-}" ]]; then
  echo "[deploy] trigger Render hook"
  curl -fsS -X POST "${RENDER_DEPLOY_HOOK}"
fi

echo "[deploy] done. Render が GitHub 連携なら push 後に自動ビルドされます。"
