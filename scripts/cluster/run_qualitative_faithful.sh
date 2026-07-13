#!/usr/bin/env bash
# Faithful qualitative figures: 2 correct + 2 failures, 4 frames/row, LaTeX row labels.
#
# 1) Pick shared panel indices on JVC (val scan on GPU)
# 2) Re-render all 3 checkpoints on those same clips (each model's own preds)
# 3) Write *_rows.tex, *_labeled.png, qualitative_figures.tex
#
# Usage (repo root on cluster):
#   conda activate isocourt
#   export ISOCOURT_PYTHON="$(which python)"
#   ./scripts/cluster/run_qualitative_faithful.sh
#
# Optional: ISOCOURT_GPUS="0 1" to run models 2–3 in parallel after pick (default: sequential).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${ISOCOURT_PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON="${CONDA_PREFIX}/bin/python"
  elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON="${REPO_ROOT}/.venv/bin/python"
  else
    PYTHON="$(command -v python)"
  fi
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}/backend:${REPO_ROOT}/docs/figures:${PYTHONPATH:-}"

FIG="${REPO_ROOT}/docs/figures/generate_qualitative_predictions.py"
POSE="${REPO_ROOT}/backend/models/pose_cache_span_linspace.pt"
BATCH="${ISOCOURT_QUAL_BATCH_SIZE:-16}"
COMMON=(
  --pose-cache "${POSE}"
  --contact-frames 16
  --frame-stride 4
  --cell-size 220
  --figure-scale 0.68
  --faithful
  --num-correct 2
  --num-errors 2
  --batch-size "${BATCH}"
)

_run() {
  echo "--- $* ---" >&2
  "${PYTHON}" "${FIG}" "${COMMON[@]}" "$@"
}

echo "Step 1/4: pick shared panels on JVC (K-STViT)..." >&2
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
echo "Shared panel indices: ${INDICES}" >&2

echo "Step 2/4: JVC no-xattn four-stream..." >&2
_run \
  --checkpoint backend/models/badminton_model_jvc_no_xattn_20260705T041121Z.pth \
  --model-label "JVC no-xattn four-stream (79.9%)" \
  --out docs/figures/qualitative_jvc_no_xattn_4stream.png \
  --curated-idx "${INDICES}"

echo "Step 3/4: JVC no-xattn single-stream..." >&2
_run \
  --checkpoint backend/models/badminton_model_jvc_no_xattn_20260706T044917Z.pth \
  --model-label "JVC no-xattn single-stream (80.5%)" \
  --out docs/figures/qualitative_jvc_no_xattn_1stream.png \
  --curated-idx "${INDICES}"

echo "Step 4/4: compose LaTeX..." >&2
"${PYTHON}" "${FIG}" \
  --compose-latex docs/figures/qualitative_figures.tex \
  --latex-block "qualitative_jvc_val:JVC (K-STViT, 80.6%)" \
  --latex-block "qualitative_jvc_no_xattn_4stream:JVC no-xattn four-stream (79.9%)" \
  --latex-block "qualitative_jvc_no_xattn_1stream:JVC no-xattn single-stream (80.5%)"

ls -lh docs/figures/qualitative_*_labeled.png docs/figures/qualitative_*_rows.tex docs/figures/qualitative_figures.tex 2>/dev/null | head -20
echo "Done. Pull: rsync -avz host:~/IsoCourt/docs/figures/qualitative_* docs/figures/" >&2
