#!/usr/bin/env bash
# Launch one qualitative figure job per GPU (parallel tmux sessions).
#
# Usage (repo root on cluster):
#   ./scripts/cluster/run_qualitative_parallel.sh
#   ISOCOURT_GPUS="0 1 2" ./scripts/cluster/run_qualitative_parallel.sh
#
# Default mapping (3 GPUs):
#   GPU 0 -> jvc
#   GPU 1 -> jvc_no_xattn_4stream
#   GPU 2 -> jvc_no_xattn_1stream
#
# Env:
#   ISOCOURT_GPUS           space-separated GPU indices (default: 0 1 2)
#   ISOCOURT_TMUX_REPLACE   if 1, kill each target session before launch
#   ISOCOURT_VENV / ISOCOURT_PYTHON / ISOCOURT_RSYNC_DEST  passed through

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LAUNCHER="${SCRIPT_DIR}/run_qualitative_tmux.sh"

# Use active conda env when launching from `(isocourt)` and .venv is absent.
if [[ -z "${ISOCOURT_PYTHON:-}" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  export ISOCOURT_PYTHON="${CONDA_PREFIX}/bin/python"
fi

read -r -a GPUS <<< "${ISOCOURT_GPUS:-0 1 2}"
MODELS=(jvc jvc_no_xattn_4stream jvc_no_xattn_1stream)

if [[ ${#GPUS[@]} -ne ${#MODELS[@]} ]]; then
  echo "ISOCOURT_GPUS has ${#GPUS[@]} GPU(s) but expected ${#MODELS[@]} (one per model)." >&2
  echo "Example: ISOCOURT_GPUS=\"0 1 2\" $0" >&2
  exit 2
fi

for i in "${!MODELS[@]}"; do
  gpu="${GPUS[$i]}"
  model="${MODELS[$i]}"
  echo "Launching ${model} on GPU ${gpu}..."
  "${LAUNCHER}" --gpu "${gpu}" "${model}"
done

echo
echo "Started ${#MODELS[@]} tmux sessions:"
for i in "${!MODELS[@]}"; do
  gpu="${GPUS[$i]}"
  model="${MODELS[$i]}"
  echo "  tmux attach -t isocourt-qual-${model}-gpu${gpu}"
done
