#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -d .git && -f .gitmodules ]]; then
  git submodule update --init --recursive
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "CUDA 12.8+ is required for the ThunderKittens v5 build" >&2
  exit 1
fi
CUDA_VERSION="$(nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' | head -1)"
if [[ -z "$CUDA_VERSION" ]] || [[ "$(printf '%s\n' "12.8" "$CUDA_VERSION" | sort -V | head -1)" != "12.8" ]]; then
  echo "CUDA 12.8+ is required for ThunderKittens v5; found ${CUDA_VERSION:-unknown}" >&2
  exit 1
fi

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
