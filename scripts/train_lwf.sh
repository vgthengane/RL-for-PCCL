#!/usr/bin/env bash
set -euo pipefail

# Lightweight wrapper to run the LwF continual training script.

PYTHON=${PYTHON:-python3}
SCRIPT="train_lwf.py"

if [ ! -f "$SCRIPT" ]; then
  echo "Error: $SCRIPT not found."
  exit 1
fi

echo "Starting continual LwF training..."
exec "$PYTHON" "$SCRIPT" "$@"
