#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

TRAIN_AUDIO_ROOT=$(printenv TRAIN_AUDIO_ROOT || true)
TEST_AUDIO_ROOT=$(printenv TEST_AUDIO_ROOT || true)
TARGET_POOL=$(printenv TARGET_POOL || true)
AUDIOCAPS_CSV=$(printenv AUDIOCAPS_CSV || true)
AART_CSV=$(printenv AART_CSV || true)
RED_ROOT=$(printenv RED_ROOT || true)
RUN_ROOT=$(printenv RUN_ROOT || true)
for name in TRAIN_AUDIO_ROOT TEST_AUDIO_ROOT TARGET_POOL AUDIOCAPS_CSV AART_CSV RED_ROOT RUN_ROOT; do
  value=$(printenv "$name" || true)
  if [ -z "$value" ]; then echo "Missing required environment variable: $name" >&2; exit 2; fi
done

PYTHON=$(printenv PYTHON || true)
if [ -z "$PYTHON" ]; then PYTHON=python; fi
TAXONOMY=$(printenv TAXONOMY || true)
if [ -z "$TAXONOMY" ]; then TAXONOMY=configs/taxonomy.json; fi
MODEL_NAME=$(printenv MODEL_NAME || true)
if [ -z "$MODEL_NAME" ]; then MODEL_NAME=audioldm-s-full-v2; fi
AUDIO_CHECKPOINT=$(printenv AUDIO_CHECKPOINT || true)
mkdir -p "$RUN_ROOT"

"$PYTHON" -m aide_repro.cli check-env
"$PYTHON" -m aide_repro.cli manifest --input-root "$TRAIN_AUDIO_ROOT" --output "$RUN_ROOT/train_audio_manifest.jsonl"
"$PYTHON" -m aide_repro.cli manifest --input-root "$TEST_AUDIO_ROOT" --output "$RUN_ROOT/test_audio_manifest.jsonl"
"$PYTHON" -m aide_repro.cli build-prompts --audiocaps-csv "$AUDIOCAPS_CSV" --aart-csv "$AART_CSV" --red-root "$RED_ROOT" --taxonomy "$TAXONOMY" --output-dir "$RUN_ROOT/prompts"

if [ -n "$AUDIO_CHECKPOINT" ]; then
  "$PYTHON" -m aide_repro.cli pgd --manifest "$RUN_ROOT/train_audio_manifest.jsonl" --target-root "$TARGET_POOL" --output-dir "$RUN_ROOT/stage1_train" --model-name "$MODEL_NAME" --audio-checkpoint "$AUDIO_CHECKPOINT"
  "$PYTHON" -m aide_repro.cli pgd --manifest "$RUN_ROOT/test_audio_manifest.jsonl" --target-root "$TARGET_POOL" --output-dir "$RUN_ROOT/stage1_test" --model-name "$MODEL_NAME" --audio-checkpoint "$AUDIO_CHECKPOINT"
else
  "$PYTHON" -m aide_repro.cli pgd --manifest "$RUN_ROOT/train_audio_manifest.jsonl" --target-root "$TARGET_POOL" --output-dir "$RUN_ROOT/stage1_train" --model-name "$MODEL_NAME"
  "$PYTHON" -m aide_repro.cli pgd --manifest "$RUN_ROOT/test_audio_manifest.jsonl" --target-root "$TARGET_POOL" --output-dir "$RUN_ROOT/stage1_test" --model-name "$MODEL_NAME"
fi

FIRST_AUDIO=$(find "$RUN_ROOT/stage1_train/audio" -type f -name '*_raw.wav' | sort | head -n 1)
if [ -z "$FIRST_AUDIO" ]; then
  echo "No Stage-I raw audio was produced." >&2
  exit 2
fi
if [ -n "$AUDIO_CHECKPOINT" ]; then
  "$PYTHON" -m aide_repro.cli gradient-check --audio "$FIRST_AUDIO" --model-name "$MODEL_NAME" --audio-checkpoint "$AUDIO_CHECKPOINT"
  "$PYTHON" -m aide_repro.cli train-psm --train-jsonl "$RUN_ROOT/prompts/psm_train.jsonl" --dev-jsonl "$RUN_ROOT/prompts/psm_dev.jsonl" --latent-dir "$RUN_ROOT/stage1_train/latents" --output-dir "$RUN_ROOT/stage2_psm" --model-name "$MODEL_NAME" --audio-checkpoint "$AUDIO_CHECKPOINT" --epochs 10 --batch-size 4 --generation-ddim-steps 8 --generation-t-start 4 --generation-every 1 --owm
else
  "$PYTHON" -m aide_repro.cli gradient-check --audio "$FIRST_AUDIO" --model-name "$MODEL_NAME"
  "$PYTHON" -m aide_repro.cli train-psm --train-jsonl "$RUN_ROOT/prompts/psm_train.jsonl" --dev-jsonl "$RUN_ROOT/prompts/psm_dev.jsonl" --latent-dir "$RUN_ROOT/stage1_train/latents" --output-dir "$RUN_ROOT/stage2_psm" --model-name "$MODEL_NAME" --epochs 10 --batch-size 4 --generation-ddim-steps 8 --generation-t-start 4 --generation-every 1 --owm
fi

"$PYTHON" -m aide_repro.cli attach-audio-ids --prompt-jsonl "$RUN_ROOT/prompts/psm_test.jsonl" --audio-manifest "$RUN_ROOT/test_audio_manifest.jsonl" --output "$RUN_ROOT/prompts/psm_test_with_audio_ids.jsonl"
if [ -n "$AUDIO_CHECKPOINT" ]; then
  "$PYTHON" -m aide_repro.cli suite --prompt-jsonl "$RUN_ROOT/prompts/psm_test_with_audio_ids.jsonl" --pgd-audio-dir "$RUN_ROOT/stage1_test/audio" --psm-checkpoint "$RUN_ROOT/stage2_psm/best.pt" --output-dir "$RUN_ROOT/test_suite" --model-name "$MODEL_NAME" --audio-checkpoint "$AUDIO_CHECKPOINT" --ddim-steps 50 --guidance-scale 2.5
  "$PYTHON" -m aide_repro.cli eval-clap --meta-root "$RUN_ROOT/test_suite" --output "$RUN_ROOT/test_suite/clap_metrics.json" --model-name "$MODEL_NAME" --audio-checkpoint "$AUDIO_CHECKPOINT"
else
  "$PYTHON" -m aide_repro.cli suite --prompt-jsonl "$RUN_ROOT/prompts/psm_test_with_audio_ids.jsonl" --pgd-audio-dir "$RUN_ROOT/stage1_test/audio" --psm-checkpoint "$RUN_ROOT/stage2_psm/best.pt" --output-dir "$RUN_ROOT/test_suite" --model-name "$MODEL_NAME" --ddim-steps 50 --guidance-scale 2.5
  "$PYTHON" -m aide_repro.cli eval-clap --meta-root "$RUN_ROOT/test_suite" --output "$RUN_ROOT/test_suite/clap_metrics.json" --model-name "$MODEL_NAME"
fi

"$PYTHON" -m aide_repro.cli eval-perturbation --raw-dir "$RUN_ROOT/stage1_test/audio" --adv-dir "$RUN_ROOT/stage1_test/audio" --output "$RUN_ROOT/stage1_test/perturbation_metrics.json"

REFERENCE_BENIGN_DIR=$(printenv REFERENCE_BENIGN_DIR || true)
REFERENCE_HARMFUL_DIR=$(printenv REFERENCE_HARMFUL_DIR || true)
if [ -n "$REFERENCE_BENIGN_DIR" ]; then
  "$PYTHON" -m aide_repro.cli eval-distribution --generated-dir "$RUN_ROOT/test_suite/benign" --reference-dir "$REFERENCE_BENIGN_DIR" --output "$RUN_ROOT/test_suite/benign_distribution.json"
fi
if [ -n "$REFERENCE_HARMFUL_DIR" ]; then
  "$PYTHON" -m aide_repro.cli eval-distribution --generated-dir "$RUN_ROOT/test_suite/harmful" --reference-dir "$REFERENCE_HARMFUL_DIR" --output "$RUN_ROOT/test_suite/harmful_distribution.json"
fi

echo "AIDE pipeline finished. Review logs, hashes, taxonomy mapping and metrics before reporting results."
