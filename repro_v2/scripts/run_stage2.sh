#!/usr/bin/env bash
set -euo pipefail
TRAIN=$1
DEV=$2
LATENTS=$3
OUTPUT=$4
python -m aide_repro.cli train-psm --train-jsonl "$TRAIN" --dev-jsonl "$DEV" --latent-dir "$LATENTS" --output-dir "$OUTPUT" --model-name audioldm-s-full-v2 --epochs 10 --batch-size 4 --generation-ddim-steps 8 --generation-t-start 4 --generation-every 1 --owm
