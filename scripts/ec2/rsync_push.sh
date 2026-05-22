#!/usr/bin/env bash
# Push local IsoCourt toward EC2. Loads scripts/ec2.env when present.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EC2_ENV="${REPO_ROOT}/scripts/ec2.env"
if [[ -f "${EC2_ENV}" ]]; then
  unset EC2_HOST KEY_FILE
  set -a
  # shellcheck disable=SC1090
  source "${EC2_ENV}"
  set +a
  echo "Using ${EC2_ENV} (EC2_HOST=${EC2_HOST})" >&2
fi
RSYNC_SSH=()
if [[ -n "${KEY_FILE:-}" ]]; then
  RSYNC_SSH=( -e "ssh -i ${KEY_FILE}" )
fi
DEST="${1:-}"
if [[ -z "${DEST}" ]]; then
  if [[ -z "${EC2_HOST:-}" ]]; then
    echo "Set EC2_HOST in scripts/ec2.env or pass user@host:path" >&2
    exit 1
  fi
  DEST="${EC2_HOST}:~/IsoCourt"
fi
echo "rsync_push → ${DEST}" >&2
# shellcheck disable=SC2086
rsync -avz --partial \
  "${RSYNC_SSH[@]}" \
  --exclude '.git/' \
  --exclude '.DS_Store' \
  --exclude '._*' \
  --exclude '.AppleDouble' \
  --exclude '.LSOverride' \
  --exclude '.Spotlight-V100/' \
  --exclude '.Trashes/' \
  --exclude '.tmp_mpl/' \
  --exclude '.hf_hub/' \
  --exclude '.venv/' \
  --exclude 'venv/' \
  --exclude 'env/' \
  --exclude '**/.venv/' \
  --exclude '**/venv/' \
  --exclude '**/env/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache/' \
  --exclude '.mypy_cache/' \
  --exclude 'mlruns/' \
  --exclude 'mlflow.db' \
  --exclude 'logs/' \
  --exclude '*.log' \
  --exclude 'training_output.txt' \
  --exclude '**/Dockerfile' \
  --exclude '**/docker-compose.yml' \
  --exclude '**/docker-compose.yaml' \
  --exclude '**/docker-compose*.yml' \
  --exclude '**/docker-compose*.yaml' \
  --exclude 'backend/tests/' \
  --exclude '**/tests/' \
  --exclude '**/test_*.py' \
  --exclude '**/*_test.py' \
  --exclude 'backend/pipelines/eda_and_data/' \
  --exclude 'backend/pipelines/evaluation/' \
  --exclude 'backend/data/FineBadminton-master/' \
  --exclude '.github/' \
  --exclude 'frontend/' \
  --exclude 'remotion/' \
  --exclude 'backend/api/' \
  --exclude 'backend/deploy/' \
  --exclude 'backend/docker-compose.yml' \
  --exclude 'backend/Dockerfile' \
  --exclude 'backend/requirements-inference.txt' \
  ${RSYNC_EXTRA:-} \
  "${REPO_ROOT}/" "${DEST%/}/"
