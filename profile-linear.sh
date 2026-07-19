#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

NCU="${NCU:-$(command -v ncu || true)}"
NCU_SET="${NCU_SET:-full}"
WARMUP="${WARMUP:-5}"

if [[ -z "$NCU" ]]; then
  echo "ncu is required; use a CUDA environment containing Nsight Compute" >&2
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  echo "Missing environment. Run ./setup.sh first." >&2
  exit 1
fi

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0a}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/build/torch_extensions}"
mkdir -p "$TORCH_EXTENSIONS_DIR" artifacts/profiles

.venv/bin/python -c \
  'from cuda_impl.linear_attention import _load_linear_attention_extension; _load_linear_attention_extension()'

"$NCU" \
  --set "$NCU_SET" \
  --force-overwrite \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*qwen35_gated_delta_decode_kernel.*' \
  --launch-skip "$WARMUP" \
  --launch-count 1 \
  --export artifacts/profiles/gated-delta-decode \
  .venv/bin/python scripts/profile_gated_delta.py --warmup "$WARMUP"
