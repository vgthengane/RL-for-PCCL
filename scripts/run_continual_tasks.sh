#!/usr/bin/env bash
set -euo pipefail

# Simple helper to run the toy continual learning experiment twice: once with
# supervised fine-tuning and once with the RL policy-gradient objective.

PYTHON=${PYTHON:-python3}
COMMON_ARGS=("$@")

echo "Running supervised fine-tuning path..."
"${PYTHON}" toy_rl_razor.py --methods sft "${COMMON_ARGS[@]}"

echo
echo "Running RL path..."
"${PYTHON}" toy_rl_razor.py --methods rl "${COMMON_ARGS[@]}"
