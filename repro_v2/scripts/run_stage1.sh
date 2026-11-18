#!/usr/bin/env bash
set -euo pipefail
MANIFEST=$1
TARGET_POOL=$2
OUTPUT_DIR=$3
python -m aide_repro.cli pgd --manifest "$MANIFEST" --target-root "$TARGET_POOL" --output-dir "$OUTPUT_DIR" --model-name audioldm-s-full-v2 --epsilon 0.01 --step-size 0.001 --steps 40
