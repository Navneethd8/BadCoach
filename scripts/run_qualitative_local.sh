#!/usr/bin/env bash
# Render qualitative figures locally (slower on CPU — use cluster script when possible).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${ISOCOURT_PYTHON:-python3}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/backend:${REPO_ROOT}/docs/figures"

FIG="${REPO_ROOT}/docs/figures/generate_qualitative_predictions.py"
POSE="${REPO_ROOT}/backend/models/pose_cache_span_linspace.pt"
if [[ ! -f "${POSE}" ]]; then
  POSE="${REPO_ROOT}/backend/models/pose_cache_mediapipe_lite_224.pt"
  echo "Using ${POSE} (span_linspace cache not found)" >&2
fi

COMMON=(
  --pose-cache "${POSE}"
  --contact-frames 16
  --frame-stride 4
  --cell-size 220
  --figure-scale 0.68
  --faithful
  --num-correct 2
  --num-errors 2
  --batch-size 8
  --no-progress
)

_run() {
  echo "--- $* ---" >&2
  "${PYTHON}" "${FIG}" "${COMMON[@]}" "$@"
}

echo "Step 1: pick panels (JVC)..." >&2
_run \
  --checkpoint backend/models/badminton_model_k_st_vit.pth \
  --model-label "JVC (K-STViT, 80.6%)" \
  --out docs/figures/qualitative_jvc_val.png

INDICES="$("${PYTHON}" - <<'PY'
import json
from pathlib import Path
rows = json.loads(Path("docs/figures/qualitative_jvc_val.meta.json").read_text())["rows"]
print(",".join(str(r["dataset_idx"]) for r in rows))
PY
)"
echo "Shared indices: ${INDICES}" >&2

_run \
  --checkpoint backend/models/badminton_model_jvc_no_xattn_20260705T041121Z.pth \
  --model-label "JVC no-xattn four-stream (79.9%)" \
  --out docs/figures/qualitative_jvc_no_xattn_4stream.png \
  --curated-idx "${INDICES}"

_run \
  --checkpoint backend/models/badminton_model_jvc_no_xattn_20260706T044917Z.pth \
  --model-label "JVC no-xattn single-stream (80.5%)" \
  --out docs/figures/qualitative_jvc_no_xattn_1stream.png \
  --curated-idx "${INDICES}"

"${PYTHON}" "${FIG}" --compose-latex docs/figures/qualitative_figures.tex \
  --latex-block "qualitative_jvc_val:JVC (K-STViT, 80.6%)" \
  --latex-block "qualitative_jvc_no_xattn_4stream:JVC no-xattn four-stream (79.9%)" \
  --latex-block "qualitative_jvc_no_xattn_1stream:JVC no-xattn single-stream (80.5%)"

echo "Done." >&2
