#!/usr/bin/env bash
# Portable convenience wrapper for setting up the project environment.
# Creates a Python venv and installs the dependencies pinned in
# `requirements.txt`. The dependency spec itself is the source of truth;
# this script just mirrors:
#
#   python3 -m venv .env
#   source .env/bin/activate
#   python -m pip install --upgrade pip setuptools wheel
#   python -m pip install -r requirements.txt
#
# Overridable environment variables:
#   PYTHON            Python interpreter to use (default: python3)
#   VENV_DIR          Path for the virtual environment (default: $REPO_ROOT/.env)
#   REQUIREMENTS_FILE Path to requirements file (default: $REPO_ROOT/requirements.txt)
#
# For HPC clusters, see `scripts/slurm/prepare_env_helios.sbatch` as an example.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.env}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$REPO_ROOT/requirements.txt}"

if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "ERROR: requirements file not found: $REQUIREMENTS_FILE" >&2
  exit 1
fi

echo "Using Python:        $("$PYTHON" --version)"
echo "Using repo root:     $REPO_ROOT"
echo "Using virtualenv:    $VENV_DIR"
echo "Using requirements:  $REQUIREMENTS_FILE"

"$PYTHON" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r "$REQUIREMENTS_FILE"

echo
echo "Environment ready."
echo "Activate it with:"
echo "  source \"$VENV_DIR/bin/activate\""
