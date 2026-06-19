#!/usr/bin/env bash
# On the cluster: symlink REMOTE_DATA → REMOTE_REPO/backend/data so trainers see backend/data.
#
# Run from the repo root on nd17 (after rsync_push_code and rsync_push_data):
#   ./scripts/cluster/setup_data_symlink.sh
#
# Env (optional; defaults match scripts/cluster.env.example):
#   REMOTE_REPO   clone root in home (default: ~/IsoCourt)
#   REMOTE_DATA   shared data volume (default: /data/models/navneeth)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CLUSTER_ENV="${REPO_ROOT}/scripts/cluster.env"

if [[ -f "${CLUSTER_ENV}" ]]; then
  unset REMOTE_REPO REMOTE_DATA
  set -a
  # shellcheck disable=SC1090
  source "${CLUSTER_ENV}"
  set +a
fi

REMOTE_REPO="${REMOTE_REPO:-~/IsoCourt}"
REMOTE_DATA="${REMOTE_DATA:-/data/models/navneeth}"

expand_path() {
  local p="$1"
  if [[ "${p}" == "~" ]]; then
    printf '%s' "${HOME}"
  elif [[ "${p}" == "~/"* ]]; then
    printf '%s/%s' "${HOME}" "${p#~/}"
  else
    printf '%s' "${p}"
  fi
}

REPO_ABS="$(expand_path "${REMOTE_REPO}")"
DATA_ABS="$(expand_path "${REMOTE_DATA}")"
LINK_PATH="${REPO_ABS}/backend/data"

mkdir -p "${DATA_ABS}" "${REPO_ABS}/backend"

if [[ -L "${LINK_PATH}" ]]; then
  current="$(readlink -f "${LINK_PATH}" 2>/dev/null || readlink "${LINK_PATH}")"
  target="$(readlink -f "${DATA_ABS}" 2>/dev/null || printf '%s' "${DATA_ABS}")"
  if [[ "${current}" == "${target}" ]]; then
    echo "Symlink already correct: ${LINK_PATH} -> ${DATA_ABS}" >&2
    exit 0
  fi
  echo "Replacing symlink ${LINK_PATH} (was -> ${current})" >&2
  rm -f "${LINK_PATH}"
elif [[ -e "${LINK_PATH}" ]]; then
  backup="${LINK_PATH}.bak.$(date -u +%Y%m%dT%H%M%SZ)"
  echo "Moving existing ${LINK_PATH} to ${backup}" >&2
  mv "${LINK_PATH}" "${backup}"
fi

ln -sfn "${DATA_ABS}" "${LINK_PATH}"
echo "Linked ${LINK_PATH} -> ${DATA_ABS}" >&2

if [[ -d "${LINK_PATH}/FineBadminton-20K/videos" ]]; then
  echo "OK: FineBadminton-20K/videos present" >&2
else
  echo "WARN: ${LINK_PATH}/FineBadminton-20K/videos not found; run rsync_push_data or prepare_finebadminton_20k.py" >&2
fi
