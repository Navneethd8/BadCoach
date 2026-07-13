#!/usr/bin/env bash
# Push only qualitative-figure scripts to the cluster (not the full repo).
#
# Usage (repo root):
#   ./scripts/cluster/rsync_push_qualitative.sh
#   ./scripts/cluster/rsync_push_qualitative.sh user@host:~/IsoCourt
#
# Loads scripts/cluster.env for CLUSTER_HOST / KEY_FILE / REMOTE_REPO.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CLUSTER_ENV="${REPO_ROOT}/scripts/cluster.env"

if [[ -f "${CLUSTER_ENV}" ]]; then
  unset CLUSTER_HOST KEY_FILE REMOTE_REPO REMOTE_DATA
  set -a
  # shellcheck disable=SC1090
  source "${CLUSTER_ENV}"
  set +a
  echo "Using ${CLUSTER_ENV} (CLUSTER_HOST=${CLUSTER_HOST:-})" >&2
fi

RSYNC_OPTS=( -avz --partial )
if [[ -n "${KEY_FILE:-}" ]]; then
  RSYNC_OPTS+=( -e "ssh -i ${KEY_FILE}" )
fi

DEST="${1:-}"
if [[ -z "${DEST}" ]]; then
  if [[ -z "${CLUSTER_HOST:-}" ]]; then
    echo "Set CLUSTER_HOST in scripts/cluster.env or pass user@host:path" >&2
    exit 1
  fi
  REMOTE_REPO="${REMOTE_REPO:-~/IsoCourt}"
  DEST="${CLUSTER_HOST}:${REMOTE_REPO}"
fi

FIG_FILES=(
  docs/figures/generate_qualitative_predictions.py
  docs/figures/teaser_pose_utils.py
)

CLUSTER_FILES=(
  scripts/cluster/run_qualitative_tmux.sh
  scripts/cluster/run_qualitative_parallel.sh
  scripts/cluster/run_qualitative_render_only.sh
  scripts/cluster/run_qualitative_faithful.sh
  scripts/cluster/setup_mediapipe_gl.sh
)

for f in "${FIG_FILES[@]}" "${CLUSTER_FILES[@]}"; do
  if [[ ! -f "${REPO_ROOT}/${f}" ]]; then
    echo "Missing ${REPO_ROOT}/${f}" >&2
    exit 1
  fi
done

echo "rsync_push_qualitative → ${DEST}" >&2
# shellcheck disable=SC2086
rsync "${RSYNC_OPTS[@]}" \
  "${FIG_FILES[@]/#/${REPO_ROOT}/}" \
  "${DEST%/}/docs/figures/"

# shellcheck disable=SC2086
rsync "${RSYNC_OPTS[@]}" \
  "${CLUSTER_FILES[@]/#/${REPO_ROOT}/}" \
  "${DEST%/}/scripts/cluster/"

echo "Done. On cluster:" >&2
echo "  chmod +x scripts/cluster/run_qualitative_*.sh scripts/cluster/setup_mediapipe_gl.sh" >&2
echo "  ./scripts/cluster/setup_mediapipe_gl.sh && source scripts/cluster/mediapipe_gl.env" >&2
echo "  ./scripts/cluster/run_qualitative_faithful.sh      # 2 correct + 2 failure pick+render" >&2
