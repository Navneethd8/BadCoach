#!/usr/bin/env bash
# Package IsoCourt stroke-model training for Colab.
#
# Creates one zip per architecture (7 total). Each zip is self-contained:
#   - All backend Python modules needed by that trainer
#   - The Colab notebook
#   - MediaPipe pose_landmarker_lite.task
#   - Merged FineBadminton-20K JSON (~22 MB)
#   - model_registry.json (for checkpoint registration)
#   - prepare_finebadminton_20k.py (so the notebook can download + extract)
#
# Images (~8 GB) are NOT bundled by default. Each Colab runtime downloads
# FineBadminton-20K from HuggingFace and extracts contact frames on its own.
# To include pre-extracted images (skips HF download but makes each zip ~8 GB):
#   INCLUDE_IMAGES=1 bash package_colab_training.sh
#
# Usage:
#   cd /path/to/IsoCourt
#   bash backend/pipelines/training/colab/package_colab_training.sh
#
# Output: backend/pipelines/training/colab/zips/  (7 zip files)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO="$(cd "$BACKEND/.." && pwd)"
OUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/zips}"
INCLUDE_IMAGES="${INCLUDE_IMAGES:-0}"

mkdir -p "$OUT_DIR"

MERGED_JSON="$BACKEND/data/transformed_combined_rounds_output_en_evals_translated.json"
POSE_TASK="$BACKEND/models/pose_landmarker_lite.task"
REGISTRY="$BACKEND/models/model_registry.json"
IMAGE_DIR="$BACKEND/data/FineBadminton-20K/dataset/image"
PREP_SCRIPT="$BACKEND/pipelines/vlm/common/prepare_finebadminton_20k.py"

# Sanity checks
for f in "$MERGED_JSON" "$POSE_TASK" "$REGISTRY" "$PREP_SCRIPT"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f"; exit 1; }
done

# Shared core modules (every model needs these)
CORE_FILES=(
  core/__init__.py
  core/dataset.py
  core/split.py
  core/seed_utils.py
  core/training_progress.py
  core/model_registry.py
  core/finebadminton_dataset_spec.py
  core/pose_utils.py
  core/model.py
  core/conv3d_pose.py
  core/staeformer.py
  core/timesformer.py
  core/videomae_pose.py
  core/videomae_timesformer.py
  core/vit_gcn.py
  core/st_tr.py
)

# All training scripts (conv3d/videomae/videomae_tf dynamically import train_timesformer)
TRAIN_SCRIPTS=(
  pipelines/training/train_full.py
  pipelines/training/train_conv3d.py
  pipelines/training/train_staeformer.py
  pipelines/training/train_timesformer.py
  pipelines/training/train_videomae.py
  pipelines/training/train_videomae_timesformer.py
  pipelines/training/train_vit_gcn.py
  pipelines/training/train_st_tr.py
)

notebook_for_model() {
  case "$1" in
    cnn_lstm)              echo "Train_CNN_LSTM.ipynb" ;;
    conv3d_pose)           echo "Train_Conv3D_Pose.ipynb" ;;
    staeformer)            echo "Train_STAEformer.ipynb" ;;
    timesformer)           echo "Train_TimeSformer.ipynb" ;;
    videomae_pose)         echo "Train_VideoMAE_Pose.ipynb" ;;
    videomae_timesformer)  echo "Train_VideoMAE_TimeSformer.ipynb" ;;
    vit_gcn)               echo "Train_ViT_GCN.ipynb" ;;
    st_tr)                 echo "Train_ST_TR.ipynb" ;;
    *) echo "UNKNOWN"; return 1 ;;
  esac
}

build_zip() {
  local model="$1"
  local notebook
  notebook="$(notebook_for_model "$model")"
  local zipname="isocourt_train_${model}.zip"
  local stage
  stage="$(mktemp -d)"

  echo "--- Packaging $model -> $zipname ---"

  # Backend code tree
  for f in "${CORE_FILES[@]}"; do
    mkdir -p "$stage/IsoCourt/backend/$(dirname "$f")"
    cp "$BACKEND/$f" "$stage/IsoCourt/backend/$f"
  done

  for f in "${TRAIN_SCRIPTS[@]}"; do
    mkdir -p "$stage/IsoCourt/backend/$(dirname "$f")"
    cp "$BACKEND/$f" "$stage/IsoCourt/backend/$f"
  done

  # Notebook
  mkdir -p "$stage/IsoCourt/backend/pipelines/training/colab"
  cp "$SCRIPT_DIR/$notebook" "$stage/IsoCourt/backend/pipelines/training/colab/"

  # prepare_finebadminton_20k.py (data download/extract)
  mkdir -p "$stage/IsoCourt/backend/pipelines/vlm/common"
  cp "$PREP_SCRIPT" "$stage/IsoCourt/backend/pipelines/vlm/common/"

  # MediaPipe task + registry
  mkdir -p "$stage/IsoCourt/backend/models"
  cp "$POSE_TASK" "$stage/IsoCourt/backend/models/"
  cp "$REGISTRY" "$stage/IsoCourt/backend/models/"

  # Merged JSON
  mkdir -p "$stage/IsoCourt/backend/data"
  cp "$MERGED_JSON" "$stage/IsoCourt/backend/data/"

  # Optional: pre-extracted JPEG images
  if [[ "$INCLUDE_IMAGES" == "1" ]] && [[ -d "$IMAGE_DIR" ]]; then
    echo "  Including pre-extracted images (~8 GB) ..."
    mkdir -p "$stage/IsoCourt/backend/data/FineBadminton-20K/dataset/image"
    rsync -a --exclude '._*' "$IMAGE_DIR/" \
      "$stage/IsoCourt/backend/data/FineBadminton-20K/dataset/image/"
  fi

  # Zip
  rm -f "$OUT_DIR/$zipname"
  (cd "$stage" && zip -rq "$OUT_DIR/$zipname" IsoCourt)
  local sz
  sz="$(du -h "$OUT_DIR/$zipname" | cut -f1)"
  echo "  -> $OUT_DIR/$zipname  ($sz)"

  rm -rf "$stage"
}

echo "Backend root: $BACKEND"
echo "Output dir:   $OUT_DIR"
echo "Include images: $INCLUDE_IMAGES"
echo ""

for model in cnn_lstm conv3d_pose staeformer timesformer videomae_pose st_tr; do
  build_zip "$model"
done

echo ""
echo "Done. Zips:"
ls -lh "$OUT_DIR"/isocourt_train_*.zip
echo ""
echo "Colab usage (per runtime):"
echo "  1. Upload the zip for the model you want"
echo "  2. In Colab: !unzip -q -o isocourt_train_<model>.zip -d /content"
echo "  3. Open the notebook from /content/IsoCourt/backend/pipelines/training/colab/"
echo "  4. Run all cells (skip 'Clone repo' cell — code is already there from the zip)"
