#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

PYTHON=$(printenv PYTHON || true)
if [ -z "$PYTHON" ]; then PYTHON=python; fi
TORCH_INDEX_URL=$(printenv TORCH_INDEX_URL || true)
if [ -z "$TORCH_INDEX_URL" ]; then TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121; fi

"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 --index-url "$TORCH_INDEX_URL"
"$PYTHON" -m pip install -r requirements.txt

AUDIO_LDM_DIR=../external/AudioLDM
if [ ! -d "$AUDIO_LDM_DIR/.git" ]; then
  mkdir -p ../external
  git clone https://github.com/haoheliu/AudioLDM.git "$AUDIO_LDM_DIR"
fi
git -C "$AUDIO_LDM_DIR" fetch --all
git -C "$AUDIO_LDM_DIR" checkout 4054cb418ba3d947d94f6ad302f1d6a320e2313c
"$PYTHON" -m pip install -e "$AUDIO_LDM_DIR" --no-deps
"$PYTHON" -m pip install -e . --no-deps

"$PYTHON" -m aide_repro.cli check-env
echo "Environment setup finished. AudioLDM checkpoint will be downloaded on first model initialization."
