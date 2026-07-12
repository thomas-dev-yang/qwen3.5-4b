#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v ninja >/dev/null 2>&1; then
  if [[ $EUID -eq 0 ]]; then
    APT=(apt-get)
  elif command -v sudo >/dev/null 2>&1; then
    APT=(sudo apt-get)
  else
    echo "ninja is missing and installing it requires root or sudo" >&2
    exit 1
  fi
  "${APT[@]}" update
  "${APT[@]}" install -y --no-install-recommends ninja-build
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

uv sync --locked
uv run --locked qwen35 doctor
