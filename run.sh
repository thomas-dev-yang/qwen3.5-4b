#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv >/dev/null 2>&1 || [[ ! -d .venv ]]; then
  echo "Missing environment. Run ./setup.sh first." >&2
  exit 1
fi

bash scripts/check_correctness.sh
