#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

MODE="${1:-decode}"
TOKENS="${2:-}"
WARMUP=5
NCU="${NCU:-$(command -v ncu || true)}"
NCU_SET="${NCU_SET:-full}"

if [[ -z "$NCU" ]]; then
  echo "ncu is required; use a CUDA environment containing Nsight Compute" >&2
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  echo "Missing environment. Run ./setup.sh first." >&2
  exit 1
fi

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/build/torch_extensions}"
mkdir -p "$TORCH_EXTENSIONS_DIR" artifacts/profiles

# Compile outside NCU so compilation and extension loading do not pollute the capture.
.venv/bin/python -c 'from cuda_impl.attention import _load_extension; _load_extension()'

PROFILE_ARGS=(--mode "$MODE" --warmup "$WARMUP")
if [[ -n "$TOKENS" ]]; then
  PROFILE_ARGS+=(--tokens "$TOKENS")
fi

exec "$NCU" \
  --set "$NCU_SET" \
  --force-overwrite \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*qwen35_attention_kernel.*' \
  --launch-skip "$WARMUP" \
  --launch-count 1 \
  --export "artifacts/profiles/attention-$MODE" \
  .venv/bin/python scripts/profile_attention.py "${PROFILE_ARGS[@]}"
