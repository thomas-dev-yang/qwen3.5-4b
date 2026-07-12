#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${QWEN35_BUILD_IMAGE:-qwen35-cuda-build}"
BASE_IMAGE="${QWEN35_BUILD_BASE_IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
BUILD_ROOT="$ROOT/build"
EXTENSION_ROOT="$BUILD_ROOT/torch_extensions"
UV_CACHE="$BUILD_ROOT/uv-cache"
TOOLCHAIN_ROOT="$BUILD_ROOT/toolchain"
CUDA_ROOT="$TOOLCHAIN_ROOT/cuda"
PYTHON_INCLUDE="$TOOLCHAIN_ROOT/python/include/python3.11"
IMAGE_MARKER="$TOOLCHAIN_ROOT/image-id"
TOOLCHAIN_LAYOUT_VERSION=4

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required" >&2
  exit 1
fi

mkdir -p "$EXTENSION_ROOT" "$UV_CACHE"

docker build \
  --file "$ROOT/Dockerfile.build" \
  --build-arg "BUILD_BASE_IMAGE=$BASE_IMAGE" \
  --tag "$IMAGE" \
  "$ROOT"

IMAGE_ID="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
TOOLCHAIN_ID="$IMAGE_ID:$TOOLCHAIN_LAYOUT_VERSION"
if [[ ! -f "$IMAGE_MARKER" ]] || [[ "$(<"$IMAGE_MARKER")" != "$TOOLCHAIN_ID" ]]; then
  rm -rf "$TOOLCHAIN_ROOT"
  mkdir -p "$CUDA_ROOT" "$TOOLCHAIN_ROOT/python/include"

  docker run --rm --entrypoint /bin/tar "$IMAGE" \
    --dereference -C /usr/local/cuda -cf - bin/nvcc include nvvm/libdevice \
    | tar -C "$CUDA_ROOT" -xf -
  printf 'CUDA Version 12.4.1\n' >"$CUDA_ROOT/version.txt"

  docker run --rm --entrypoint /bin/tar "$IMAGE" \
    -C /usr/include -cf - python3.11 x86_64-linux-gnu/python3.11 \
    | tar -C "$TOOLCHAIN_ROOT/python/include" -xf -

  printf '%s\n' "$TOOLCHAIN_ID" >"$IMAGE_MARKER"
fi

set +e
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
BUILD_STATUS=$?
set -e

if [[ -s "$ROOT/compile_commands.json" ]]; then
  python3 "$ROOT/scripts/normalize_compile_commands.py" \
    "$ROOT/compile_commands.json" \
    --project-root "$ROOT" \
    --cuda-root "$CUDA_ROOT" \
    --python-include "$PYTHON_INCLUDE"
fi

if ((BUILD_STATUS != 0)); then
  echo "CUDA extension build failed; compile_commands.json was retained for clangd" >&2
  exit "$BUILD_STATUS"
fi

echo "Built H100 CUDA extension"
echo "Compilation database: $ROOT/compile_commands.json"
echo "Extension output: $EXTENSION_ROOT/qwen35_attention_cuda"
