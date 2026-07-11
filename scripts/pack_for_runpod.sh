#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUT="${OUT:-../qwen3.5-4b-runpod.tar.gz}"
REPO_DIR="$(basename "$PWD")"
INCLUDE_MODEL="${INCLUDE_MODEL:-0}"

EXCLUDES=(
  "--exclude=$REPO_DIR/.git"
  "--exclude=$REPO_DIR/.venv"
  "--exclude=$REPO_DIR/artifacts"
  "--exclude=$REPO_DIR/__pycache__"
  "--exclude=$REPO_DIR/.pytest_cache"
  "--exclude=$REPO_DIR/.ruff_cache"
)

if [[ "$INCLUDE_MODEL" != "1" ]]; then
  EXCLUDES+=("--exclude=$REPO_DIR/models")
fi

tar \
  -C .. \
  "${EXCLUDES[@]}" \
  -czf "$OUT" \
  "$REPO_DIR"

echo "Wrote $OUT"
