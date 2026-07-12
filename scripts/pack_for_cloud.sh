#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

OUT="${OUT:-../qwen3.5-4b-cloud.tar.gz}"
REPO_DIR="$(basename "$PWD")"
INCLUDE_MODEL="${INCLUDE_MODEL:-0}"
MANIFEST="$(mktemp)"
trap 'rm -f "$MANIFEST"' EXIT

while IFS= read -r -d '' path; do
  if [[ -e "$path" || -L "$path" ]]; then
    printf '%s\0' "$path"
  fi
done < <(git ls-files --cached --others --exclude-standard -z) >"$MANIFEST"

if [[ "$INCLUDE_MODEL" == "1" && -d models ]]; then
  find models -type f -print0 >>"$MANIFEST"
fi

tar \
  -C "$REPO_ROOT" \
  --null \
  --verbatim-files-from \
  --files-from="$MANIFEST" \
  --transform="s|^|$REPO_DIR/|" \
  -czf "$OUT"

echo "Wrote $OUT"
