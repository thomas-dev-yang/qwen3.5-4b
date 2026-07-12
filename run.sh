#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv >/dev/null 2>&1 || [[ ! -d .venv ]]; then
  echo "Missing environment. Run ./setup.sh first." >&2
  exit 1
fi

run_kernel_correctness() {
  local version="${1:-}"
  if [[ -n "$version" ]]; then
    QWEN35_TEST_CUDA_KERNEL=1 QWEN35_ATTENTION_VERSION="$version" \
      .venv/bin/pytest -q tests/attention/test_cuda_attention.py
  else
    env -u QWEN35_ATTENTION_VERSION QWEN35_TEST_CUDA_KERNEL=1 \
      .venv/bin/pytest -q tests/attention/test_cuda_attention.py
  fi
}

if (( $# == 0 )); then
  bash scripts/check_correctness.sh
  run_kernel_correctness
elif [[ "$1" == "v1" || "$1" == "v2" || "$1" == "v3" ]]; then
  run_kernel_correctness "$1"
else
  echo "usage: ./run.sh [v1|v2|v3]" >&2
  exit 1
fi
