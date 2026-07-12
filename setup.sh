#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

bash scripts/setup_cloud.sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv run --locked qwen35 download

echo
echo "Setup complete. Run: ./run.sh"
