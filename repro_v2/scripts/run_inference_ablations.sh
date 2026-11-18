#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
PROMPT_JSONL=$(printenv PROMPT_JSONL || true)
PGD_AUDIO_DIR=$(printenv PGD_AUDIO_DIR || true)
FULL_PSM_CHECKPOINT=$(printenv FULL_PSM_CHECKPOINT || true)
RUN_ROOT=$(printenv RUN_ROOT || true)
for name in PROMPT_JSONL PGD_AUDIO_DIR FULL_PSM_CHECKPOINT RUN_ROOT; do
  value=$(printenv "$name" || true)
  if [ -z "$value" ]; then echo "Missing required environment variable: $name" >&2; exit 2; fi
done

PYTHON=$(printenv PYTHON || true)
if [ -z "$PYTHON" ]; then PYTHON=python; fi
MODEL_NAME=$(printenv MODEL_NAME || true)
if [ -z "$MODEL_NAME" ]; then MODEL_NAME=audioldm-s-full-v2; fi

run_suite() {
  mode=$1
  output=$2
  shift 2
  "$PYTHON" -m aide_repro.cli suite --prompt-jsonl "$PROMPT_JSONL" --pgd-audio-dir "$PGD_AUDIO_DIR" --psm-checkpoint "$FULL_PSM_CHECKPOINT" --model-name "$MODEL_NAME" --mode "$mode" --no-paired --output-dir "$output" "$@"
}

run_suite clean "$RUN_ROOT/no_perturbation"
run_suite protected "$RUN_ROOT/static_protection_no_psm"
run_suite aide "$RUN_ROOT/full_aide"
run_suite aide "$RUN_ROOT/hard_gate_no_soft_gate" --hard-gate
run_suite aide "$RUN_ROOT/no_channel_gate" --disable-channel-gate

BINARY_PSM_CHECKPOINT=$(printenv BINARY_PSM_CHECKPOINT || true)
NO_GENERATION_PSM_CHECKPOINT=$(printenv NO_GENERATION_PSM_CHECKPOINT || true)
NO_OWM_PSM_CHECKPOINT=$(printenv NO_OWM_PSM_CHECKPOINT || true)

if [ -n "$BINARY_PSM_CHECKPOINT" ]; then
  "$PYTHON" -m aide_repro.cli suite --prompt-jsonl "$PROMPT_JSONL" --pgd-audio-dir "$PGD_AUDIO_DIR" --psm-checkpoint "$BINARY_PSM_CHECKPOINT" --model-name "$MODEL_NAME" --mode aide --no-paired --output-dir "$RUN_ROOT/no_multiclass_risk"
fi
if [ -n "$NO_GENERATION_PSM_CHECKPOINT" ]; then
  "$PYTHON" -m aide_repro.cli suite --prompt-jsonl "$PROMPT_JSONL" --pgd-audio-dir "$PGD_AUDIO_DIR" --psm-checkpoint "$NO_GENERATION_PSM_CHECKPOINT" --model-name "$MODEL_NAME" --mode aide --no-paired --output-dir "$RUN_ROOT/no_generation_coupled_loss"
fi
if [ -n "$NO_OWM_PSM_CHECKPOINT" ]; then
  "$PYTHON" -m aide_repro.cli suite --prompt-jsonl "$PROMPT_JSONL" --pgd-audio-dir "$PGD_AUDIO_DIR" --psm-checkpoint "$NO_OWM_PSM_CHECKPOINT" --model-name "$MODEL_NAME" --mode aide --no-paired --output-dir "$RUN_ROOT/no_owm"
fi
