#!/usr/bin/env bash
# Build a minimal backend/ tree for Colab Qwen3-VL + MediaPipe training.
#
# Always includes:
#   - backend/pipelines/vlm/common/   (JSONL helpers, pose bridge, shared deps list)
#   - backend/pipelines/vlm/qwen-4b/  (4B config, train/infer, notebook)
#   - backend/pipelines/vlm/qwen-8b/  (8B config, train/infer, notebook)
#   - backend/pipelines/vlm/package_colab_backend.sh
#   - backend/core/pose_utils.py, split.py, __init__.py  (pose + vlm_jsonl_video_level_split)
#
# Optional (env vars):
#   INCLUDE_FINEBADMINTON=1       copy labels + jsonl (default: 1); json/jsonl may be
#                                 filled from FineBadminton-master/dataset when 20K has images only
#   INCLUDE_FINEBADMINTON_IMAGES=1   copy dataset/image/ — LARGE (~10k files); default: 0
#   INCLUDE_POSE_TASK=1         copy backend/models/pose_landmarker_lite.task if present
#
# Usage (from anywhere):
#   bash /path/to/IsoCourt/backend/pipelines/vlm/package_colab_backend.sh
#   INCLUDE_FINEBADMINTON_IMAGES=1 bash .../package_colab_backend.sh
#
# Output: backend_vlm_colab.zip next to repo root (override with OUTPUT=/tmp/foo.zip)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$BACKEND_ROOT/.." && pwd)"

INCLUDE_FINEBADMINTON="${INCLUDE_FINEBADMINTON:-1}"
INCLUDE_FINEBADMINTON_IMAGES="${INCLUDE_FINEBADMINTON_IMAGES:-0}"
INCLUDE_POSE_TASK="${INCLUDE_POSE_TASK:-0}"
OUTPUT="${OUTPUT:-$REPO_ROOT/backend_vlm_colab.zip}"

STAGE="$(mktemp -d)"
cleanup() { rm -rf "$STAGE"; }
trap cleanup EXIT

mkdir -p "$STAGE/backend/pipelines/vlm/common"
mkdir -p "$STAGE/backend/pipelines/vlm/qwen-4b"
mkdir -p "$STAGE/backend/pipelines/vlm/qwen-8b"
mkdir -p "$STAGE/backend/core"

echo "Backend root: $BACKEND_ROOT"
echo "Output zip:   $OUTPUT"

rsync -a \
  --exclude 'outputs/' \
  --exclude '.ipynb_checkpoints/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  "$BACKEND_ROOT/pipelines/vlm/common/" "$STAGE/backend/pipelines/vlm/common/"

rsync -a \
  --exclude 'outputs/' \
  --exclude '.ipynb_checkpoints/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  "$BACKEND_ROOT/pipelines/vlm/qwen-4b/" "$STAGE/backend/pipelines/vlm/qwen-4b/"

rsync -a \
  --exclude 'outputs/' \
  --exclude '.ipynb_checkpoints/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  "$BACKEND_ROOT/pipelines/vlm/qwen-8b/" "$STAGE/backend/pipelines/vlm/qwen-8b/"

cp "$BACKEND_ROOT/pipelines/vlm/package_colab_backend.sh" "$STAGE/backend/pipelines/vlm/"

cp "$BACKEND_ROOT/core/pose_utils.py" "$STAGE/backend/core/pose_utils.py"
cp "$BACKEND_ROOT/core/split.py" "$STAGE/backend/core/split.py"
touch "$STAGE/backend/core/__init__.py"

if [[ "$INCLUDE_FINEBADMINTON" == "1" ]]; then
  DS20="$BACKEND_ROOT/data/FineBadminton-20K/dataset"
  DS_LEGACY="$BACKEND_ROOT/data/FineBadminton-master/dataset"
  if [[ -d "$DS20" ]]; then
    DS="$DS20"
    DS_STAGE="FineBadminton-20K/dataset"
    DS_ALT=""
    [[ -d "$DS_LEGACY" ]] && DS_ALT="$DS_LEGACY"
  elif [[ -d "$DS_LEGACY" ]]; then
    DS="$DS_LEGACY"
    DS_STAGE="FineBadminton-master/dataset"
    DS_ALT=""
    [[ -d "$DS20" ]] && DS_ALT="$DS20"
  else
    DS=""
    DS_STAGE=""
    DS_ALT=""
  fi
  TF="$BACKEND_ROOT/data/transformed_combined_rounds_output_en_evals_translated.json"
  if [[ -f "$TF" ]]; then
    mkdir -p "$STAGE/backend/data"
    cp "$TF" "$STAGE/backend/data/"
    echo "  + data/transformed_combined_rounds_output_en_evals_translated.json"
  fi
  if [[ -n "$DS" ]]; then
    mkdir -p "$STAGE/backend/data/$DS_STAGE"
    for f in \
      transformed_combined_rounds_output_en_evals_translated.json \
      transformed_combined_rounds_zh.json \
      finebadminton_vlm_train.jsonl; do
      if [[ -f "$DS/$f" ]]; then
        cp "$DS/$f" "$STAGE/backend/data/$DS_STAGE/"
        echo "  + data/$DS_STAGE/$f"
      elif [[ -n "${DS_ALT:-}" && -f "$DS_ALT/$f" ]]; then
        cp "$DS_ALT/$f" "$STAGE/backend/data/$DS_STAGE/"
        echo "  + data/$DS_STAGE/$f (from alternate tree)"
      fi
    done
    if [[ "$INCLUDE_FINEBADMINTON_IMAGES" == "1" ]]; then
      if [[ -d "$DS/image" ]]; then
        echo "  + data/$DS_STAGE/image/  (this may take a while)"
        mkdir -p "$STAGE/backend/data/$DS_STAGE/image"
        rsync -a "$DS/image/" "$STAGE/backend/data/$DS_STAGE/image/"
      else
        echo "  ! FineBadminton image/ not found at $DS/image" >&2
      fi
    else
      echo "  (skipped dataset/image — set INCLUDE_FINEBADMINTON_IMAGES=1 to include training frames)"
    fi
  else
    echo "  ! No dataset dir (expected $DS20 or $DS_LEGACY)" >&2
  fi
fi

if [[ "$INCLUDE_POSE_TASK" == "1" ]]; then
  TASK="$BACKEND_ROOT/models/pose_landmarker_lite.task"
  if [[ -f "$TASK" ]]; then
    mkdir -p "$STAGE/backend/models"
    cp "$TASK" "$STAGE/backend/models/"
    echo "  + models/pose_landmarker_lite.task"
  else
    echo "  ! pose task not found (Colab notebook can download it). $TASK" >&2
  fi
fi

rm -f "$OUTPUT"
(
  cd "$STAGE"
  zip -rq "$OUTPUT" backend
)

echo "Done. Zip size: $(du -h "$OUTPUT" | cut -f1)"
echo "Colab: upload then  !unzip -q -o backend_vlm_colab.zip -d /content"
