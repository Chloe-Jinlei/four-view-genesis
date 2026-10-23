#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
GENESIS_PYTHON="${GENESIS_PYTHON:-/home/chloe/Desktop/genesis-local/.venv/bin/python}"
exec "$GENESIS_PYTHON" simulate.py "$@"
