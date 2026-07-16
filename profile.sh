#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

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

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0a}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/build/torch_extensions}"
mkdir -p "$TORCH_EXTENSIONS_DIR" artifacts/profiles

# Compile once outside NCU so extension loading does not pollute any capture.
.venv/bin/python -c 'from cuda_impl.attention import _load_extension; _load_extension()'

profile_one() {
  local version="$1"
  local mode="$2"
  local tokens="$3"
  local kernel_pattern="regex:.*qwen35_attention_${version}_kernel.*"
  local launch_skip="$WARMUP"
  local launch_count=1

  if [[ "$version" != "v1" && "$version" != "v2" && "$version" != "v3" && "$version" != "v4" && "$version" != "v5" ]]; then
    echo "attention version must be v1, v2, v3, v4, or v5" >&2
    return 1
  fi
  if [[ "$mode" != "decode" && "$mode" != "prefill" ]]; then
    echo "mode must be decode or prefill" >&2
    return 1
  fi
  if [[ "$version" == "v3" || "$version" == "v4" ]]; then
    kernel_pattern="regex:.*qwen35_attention_${version}_.*kernel.*"
    launch_skip=$((WARMUP * 2))
    launch_count=2
  fi

  echo "Profiling $version $mode with $tokens tokens"
  "$NCU" \
    --set "$NCU_SET" \
    --force-overwrite \
    --kernel-name-base demangled \
    --kernel-name "$kernel_pattern" \
    --launch-skip "$launch_skip" \
    --launch-count "$launch_count" \
    --export "artifacts/profiles/attention-$version-$mode-k$tokens" \
    .venv/bin/python scripts/profile_attention.py \
      --version "$version" \
      --mode "$mode" \
      --tokens "$tokens" \
      --warmup "$WARMUP"
}

if (( $# == 0 )); then
  for version in v1 v2 v3 v4 v5; do
    profile_one "$version" decode 1024
    profile_one "$version" prefill 64
  done
else
  VERSION="$1"
  MODE="${2:-decode}"
  if [[ -n "${3:-}" ]]; then
    TOKENS="$3"
  elif [[ "$MODE" == "decode" ]]; then
    TOKENS=1024
  else
    TOKENS=64
  fi
  profile_one "$VERSION" "$MODE" "$TOKENS"
fi
