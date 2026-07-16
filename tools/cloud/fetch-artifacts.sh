#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="${CLOUD_CONFIG:-$SCRIPT_DIR/config.json}"

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
REMOTE_PROJECT="${REMOTE_ROOT%/}/qwen3.5-4b"

printf -v REMOTE_PROJECT_Q '%q' "$REMOTE_PROJECT"

mkdir -p "$PROJECT_ROOT/artifacts"
ssh "$TARGET" "test -d $REMOTE_PROJECT_Q/artifacts && tar -C $REMOTE_PROJECT_Q -czf - artifacts" \
  | tar -C "$PROJECT_ROOT" -xzf -

echo "Downloaded all artifacts from $TARGET:$REMOTE_PROJECT/artifacts to $PROJECT_ROOT/artifacts/"
