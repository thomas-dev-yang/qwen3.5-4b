#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="${CLOUD_CONFIG:-$SCRIPT_DIR/config.json}"
ARCHIVE_DIR="$PROJECT_ROOT/build/cloud"
ARCHIVE="$ARCHIVE_DIR/qwen3.5-4b-cloud.tar.gz"

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
REMOTE_ARCHIVE="${REMOTE_ROOT%/}/$(basename "$ARCHIVE")"

printf -v REMOTE_ROOT_Q '%q' "$REMOTE_ROOT"
printf -v REMOTE_ARCHIVE_Q '%q' "$REMOTE_ARCHIVE"

mkdir -p "$ARCHIVE_DIR"
OUT="$ARCHIVE" bash "$PROJECT_ROOT/scripts/pack_for_cloud.sh"

ssh "$TARGET" "mkdir -p $REMOTE_ROOT_Q"
scp "$ARCHIVE" "$TARGET:$REMOTE_ROOT/"
ssh "$TARGET" "tar -xzf $REMOTE_ARCHIVE_Q -C $REMOTE_ROOT_Q"

echo "Uploaded and extracted $ARCHIVE to $TARGET:$REMOTE_ROOT"
