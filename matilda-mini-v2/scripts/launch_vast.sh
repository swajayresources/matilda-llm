#!/usr/bin/env bash
# Spot-safe launch wrapper for a Vast.ai A100 instance.
#
# The trainer itself traps SIGTERM and checkpoints before exit (train.py), so
# this wrapper's job is environment setup + syncing checkpoints to durable
# storage so an instance death doesn't lose them.
#
#   bash scripts/launch_vast.sh configs/calibration.json   # MFU smoke test
#   bash scripts/launch_vast.sh configs/base_124m.json      # v1: 124M / FineWeb-Edu
#   bash scripts/launch_vast.sh configs/base_350m.json      # v2: 350M / SmolLM mix
#
# Set REMOTE to an rclone/s3 target to enable checkpoint upload on exit.
# Set DATA_DIR to override the default per-config data path.

set -euo pipefail

CONFIG="${1:-configs/base_124m.json}"
REMOTE="${REMOTE:-}"                       # e.g. s3://my-bucket/matilda or gdrive:matilda
CKPT_DIR="$(python -c "import json;print(json.load(open('$CONFIG'))['train']['ckpt_dir'])")"
USE_LIGER="$(python -c "import json;print(json.load(open('$CONFIG'))['model'].get('use_liger', False))")"

# Default DATA_DIR depends on which config is being run.
if [ -z "${DATA_DIR:-}" ]; then
  case "$CONFIG" in
    *base_152m_v2*|*base_350m*) DATA_DIR="data/smollm_mix" ;;
    *)                           DATA_DIR="data/fwedu"      ;;
  esac
fi

sync_checkpoints() {
  [ -z "$REMOTE" ] && { echo "[sync] REMOTE unset, skipping upload"; return; }
  echo "[sync] uploading $CKPT_DIR -> $REMOTE"
  if command -v aws >/dev/null;     then aws s3 sync "$CKPT_DIR" "$REMOTE" || true
  elif command -v rclone >/dev/null; then rclone copy "$CKPT_DIR" "$REMOTE" || true
  fi
}
trap sync_checkpoints EXIT               # runs on normal exit AND on spot kill

echo "[setup] installing deps"
pip install -q -r requirements.txt
if [ "$USE_LIGER" = "True" ]; then
  echo "[setup] config has use_liger=True; ensuring liger-kernel is importable"
  python -c "import liger_kernel" 2>/dev/null || pip install -q liger-kernel
fi

# Pull checkpoints back first so a relaunched instance resumes (no-op if absent).
if [ -n "$REMOTE" ]; then
  mkdir -p "$CKPT_DIR"
  (command -v aws >/dev/null && aws s3 sync "$REMOTE" "$CKPT_DIR") || \
  (command -v rclone >/dev/null && rclone copy "$REMOTE" "$CKPT_DIR") || true
fi

if [ ! -f "$DATA_DIR/manifest.json" ]; then
  case "$CONFIG" in
    *base_152m_v2*|*base_350m*)
      echo "[data] no manifest in $DATA_DIR; building SmolLM mix (~15B tokens)"
      python scripts/prepare_smollm_data.py --out-dir "$DATA_DIR" \
             --target-tokens 15000000000
      ;;
    *)
      echo "[data] no manifest in $DATA_DIR; tokenizing FineWeb-Edu (~3B tokens)"
      python scripts/prepare_data.py --out-dir "$DATA_DIR" \
             --target-tokens 3000000000
      ;;
  esac
fi

echo "[train] launching $CONFIG"
python run.py --config "$CONFIG" --data-dir "$DATA_DIR"
