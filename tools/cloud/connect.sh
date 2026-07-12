#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_FILE="${CLOUD_CONFIG:-$SCRIPT_DIR/config.json}"

TARGET="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["target"])' "$CONFIG_FILE")"

exec ssh "${CLOUD_TARGET:-$TARGET}"
