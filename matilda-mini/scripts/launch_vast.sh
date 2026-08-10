#!/usr/bin/env bash
# Spot-safe launch wrapper for a Vast.ai A100 instance.
#
# The trainer itself traps SIGTERM and checkpoints before exit (train.py), so
# this wrapper's job is environment setup + syncing checkpoints to durable
# storage so an instance death doesn't lose them.
#
#   bash scripts/launch_vast.sh configs/calibration.json   # MFU smoke test
#   bash scripts/launch_vast.sh configs/base_124m.json      # the long run
#
# Set REMOTE to an rclone/s3 target to enable checkpoint upload on exit.

set -euo pipefail

CONFIG="${1:-configs/base_124m.json}"
DATA_DIR="${DATA_DIR:-data/fwedu}"
REMOTE="${REMOTE:-}"                       # e.g. s3://my-bucket/matilda or gdrive:matilda
CKPT_DIR="$(python -c "import json,sys;print(json.load(open('$CONFIG'))['train']['ckpt_dir'])")"

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

# Pull checkpoints back first so a relaunched instance resumes (no-op if absent).
if [ -n "$REMOTE" ]; then
  mkdir -p "$CKPT_DIR"
  (command -v aws >/dev/null && aws s3 sync "$REMOTE" "$CKPT_DIR") || \
  (command -v rclone >/dev/null && rclone copy "$REMOTE" "$CKPT_DIR") || true
fi

if [ ! -f "$DATA_DIR/manifest.json" ]; then
  echo "[data] no manifest in $DATA_DIR; tokenizing FineWeb-Edu (~3B tokens)"
  python scripts/prepare_data.py --out-dir "$DATA_DIR" --target-tokens 3000000000
fi

echo "[train] launching $CONFIG"
python run.py --config "$CONFIG" --data-dir "$DATA_DIR"
