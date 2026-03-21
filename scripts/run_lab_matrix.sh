#!/usr/bin/env bash
set -eu

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ "$#" -eq 0 ]; then
  profiles=(
    dev-small
    single-h100-hbm-max
  )
else
  profiles=("$@")
fi

for profile in "${profiles[@]}"; do
  uv run pose bench run --profile "$profile"
done
