#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

BACKEND="${1:-cuda-attention}"
PROMPT_LENGTH="${2:-1024}"
NSYS="${NSYS:-$(command -v nsys || true)}"

if [[ "$BACKEND" != "hf" && "$BACKEND" != "cuda-attention" && "$BACKEND" != "cuda-all" ]]; then
  echo "usage: ./profile-model.sh [hf|cuda-attention|cuda-all] [prompt-length]" >&2
  exit 1
fi
if [[ -z "$NSYS" ]]; then
  echo "nsys is required; use a CUDA environment containing Nsight Systems" >&2
  exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
  echo "Missing environment. Run ./setup.sh first." >&2
  exit 1
fi

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0a}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PWD/build/torch_extensions}"
mkdir -p "$TORCH_EXTENSIONS_DIR" artifacts/profiles

OUTPUT="artifacts/profiles/model-${BACKEND}-decode-k${PROMPT_LENGTH}"
"$NSYS" profile \
  --trace=cuda,nvtx,cublas,osrt \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  --output "$OUTPUT" \
  .venv/bin/python scripts/profile_model_decode.py \
    --backend "$BACKEND" \
    --prompt-length "$PROMPT_LENGTH"

"$NSYS" stats \
  --report nvtx_gpu_proj_sum,cuda_gpu_kern_sum \
  "${OUTPUT}.nsys-rep" >"${OUTPUT}.txt" || true

echo "Report: ${OUTPUT}.nsys-rep"
echo "Summary: ${OUTPUT}.txt"
