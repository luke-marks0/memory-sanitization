#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "expected Python virtualenv at $VENV_DIR" >&2
  exit 1
fi

if [[ -f "$HOME/.cargo/env" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
fi

unset CONDA_PREFIX
export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"

cd "$ROOT_DIR"
python -m maturin develop --release --manifest-path native/pose_native_label_engine/Cargo.toml
