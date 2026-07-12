#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="${CLOUD_CONFIG:-$SCRIPT_DIR/config.json}"
REPORT="${1:-*.ncu-rep}"
LOCAL_DIR="${2:-$PROJECT_ROOT/artifacts/profiles}"

mapfile -t CONFIG < <(
  python3 - "$CONFIG_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as file:
    config = json.load(file)

print(config["target"])
print(config["remote_root"])
PY
)

TARGET="${CLOUD_TARGET:-${CONFIG[0]}}"
REMOTE_ROOT="${CLOUD_REMOTE_ROOT:-${CONFIG[1]}}"

if [[ "$REPORT" = /* ]]; then
  REMOTE_REPORT="$REPORT"
else
  REMOTE_REPORT="${REMOTE_ROOT%/}/qwen3.5-4b/artifacts/profiles/$REPORT"
fi

mkdir -p "$LOCAL_DIR"
scp "$TARGET:$REMOTE_REPORT" "$LOCAL_DIR/"

if [[ "$REPORT" == "*.ncu-rep" ]]; then
  echo "Downloaded all profiles from $TARGET to $LOCAL_DIR/"
else
  echo "Downloaded $REMOTE_REPORT to $LOCAL_DIR/$(basename "$REMOTE_REPORT")"
fi
