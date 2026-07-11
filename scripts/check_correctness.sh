#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

rm -rf artifacts/correctness

uv run --locked qwen35 reference
uv run --locked qwen35 candidate
uv run --locked qwen35 compare
