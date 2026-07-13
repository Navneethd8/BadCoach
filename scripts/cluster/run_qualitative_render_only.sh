#!/usr/bin/env bash
# Fast re-render after panel indices are locked (no val inference).
#
# Set PANELS from a prior faithful pick (see qualitative_jvc_val.meta.json rows).
# Or run ./scripts/cluster/run_qualitative_faithful.sh to pick + render on cluster.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

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
COMMON=(
  --pose-cache "${POSE}"
  --contact-frames 16
  --frame-stride 4
  --cell-size 220
  --figure-scale 0.68
  --faithful
  --no-progress
)

_run() {
  echo "--- $* ---" >&2
  "${PYTHON}" "${FIG}" "${COMMON[@]}" "$@"
}

cd "${REPO_ROOT}"

_run \
  --checkpoint backend/models/badminton_model_k_st_vit.pth \
  --model-label "JVC (K-STViT, 80.6%)" \
  --out docs/figures/qualitative_jvc_val.png \
  --render-panels "13717:Serve:Serve:0.87,13474:Clear:Clear:0.91,986:Smash:Drop:0.94,12085:Drop:Smash:0.90"

_run \
  --checkpoint backend/models/badminton_model_jvc_no_xattn_20260705T041121Z.pth \
  --model-label "JVC no-xattn four-stream (79.9%)" \
  --out docs/figures/qualitative_jvc_no_xattn_4stream.png \
  --render-panels "13717:Serve:Serve:0.88,13474:Clear:Clear:0.92,986:Smash:Drop:0.95,12085:Drop:Smash:0.90"

_run \
  --checkpoint backend/models/badminton_model_jvc_no_xattn_20260706T044917Z.pth \
  --model-label "JVC no-xattn single-stream (80.5%)" \
  --out docs/figures/qualitative_jvc_no_xattn_1stream.png \
  --render-panels "13717:Serve:Serve:0.87,13474:Clear:Clear:0.92,986:Smash:Drop:0.93,12085:Drop:Smash:0.90"

"${PYTHON}" "${FIG}" \
  --compose-latex docs/figures/qualitative_figures.tex \
  --latex-block "qualitative_jvc_val:JVC (K-STViT, 80.6%)" \
  --latex-block "qualitative_jvc_no_xattn_4stream:JVC no-xattn four-stream (79.9%)" \
  --latex-block "qualitative_jvc_no_xattn_1stream:JVC no-xattn single-stream (80.5%)"

ls -lh docs/figures/qualitative_*.{png,pdf,tex,meta.json} 2>/dev/null | head -40
echo "Done. Row strips + *_rows.tex for LaTeX; \\input{figures/qualitative_figures.tex}" >&2
