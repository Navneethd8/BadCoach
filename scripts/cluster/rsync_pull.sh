#!/usr/bin/env bash
# Pull checkpoints + registry + mlruns from the cluster to local repo.
#
# Loads scripts/cluster.env from the repo root when present.
#
# Usage:
#   ./scripts/cluster/rsync_pull.sh
#   ./scripts/cluster/rsync_pull.sh user@host:~/IsoCourt   # explicit override
#
# Env (optional; often set in scripts/cluster.env):
#   CLUSTER_HOST   default remote when no argument is passed
#   REMOTE_REPO    remote clone root (default: ~/IsoCourt)
#   KEY_FILE       PEM path for ssh -i
#   RSYNC_EXTRA    additional rsync options

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

SRC="${1:-}"
if [[ -z "${SRC}" ]]; then
  if [[ -z "${CLUSTER_HOST:-}" ]]; then
    echo "Usage: $0 [user@host:path/to/IsoCourt]" >&2
    echo "Or create ${CLUSTER_ENV} with CLUSTER_HOST=nd17@host (and optional KEY_FILE)." >&2
    exit 1
  fi
  REMOTE_REPO="${REMOTE_REPO:-~/IsoCourt}"
  SRC="${CLUSTER_HOST}:${REMOTE_REPO}"
fi

mkdir -p "${REPO_ROOT}/backend/models" "${REPO_ROOT}/backend/mlruns"

# shellcheck disable=SC2086
rsync "${RSYNC_OPTS[@]}" ${RSYNC_EXTRA:-} \
  "${SRC%/}/backend/models/" "${REPO_ROOT}/backend/models/"

# shellcheck disable=SC2086
rsync "${RSYNC_OPTS[@]}" ${RSYNC_EXTRA:-} \
  "${SRC%/}/backend/mlruns/" "${REPO_ROOT}/backend/mlruns/" || true
rsync "${RSYNC_OPTS[@]}" ${RSYNC_EXTRA:-} \
  "${SRC%/}/mlruns/" "${REPO_ROOT}/mlruns/" || true
