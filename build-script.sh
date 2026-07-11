#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${QWEN35_BUILD_IMAGE:-qwen35-cuda-build}"
BUILD_ROOT="$ROOT/build"
EXTENSION_ROOT="$BUILD_ROOT/torch_extensions"
UV_CACHE="$BUILD_ROOT/uv-cache"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required" >&2
  exit 1
fi

mkdir -p "$EXTENSION_ROOT" "$UV_CACHE"

docker build \
  --file "$ROOT/Dockerfile.build" \
  --tag "$IMAGE" \
  "$ROOT"

docker run --rm \
  --volume "$ROOT:$ROOT" \
  --workdir "$ROOT" \
  --env "HOST_UID=$(id -u)" \
  --env "HOST_GID=$(id -g)" \
  --env "TORCH_CUDA_ARCH_LIST=9.0" \
  --env "TORCH_EXTENSIONS_DIR=$EXTENSION_ROOT" \
  --env "UV_CACHE_DIR=$UV_CACHE" \
  --env "UV_PROJECT_ENVIRONMENT=$BUILD_ROOT/venv" \
  "$IMAGE" \
  bash -lc '
    set -euo pipefail
    fix_ownership() {
      for path in build compile_commands.json; do
        if [[ -e "$path" ]]; then
          chown -R "$HOST_UID:$HOST_GID" "$path"
        fi
      done
    }
    trap fix_ownership EXIT

    uv sync --locked
    rm -rf "$TORCH_EXTENSIONS_DIR/qwen35_attention_cuda"
    rm -f compile_commands.json
    bear --output compile_commands.json -- \
      uv run --locked python -c \
      "from cuda_impl.attention import _load_extension; _load_extension()"
  '

echo "Built H100 CUDA extension"
echo "Compilation database: $ROOT/compile_commands.json"
echo "Extension output: $EXTENSION_ROOT/qwen35_attention_cuda"
