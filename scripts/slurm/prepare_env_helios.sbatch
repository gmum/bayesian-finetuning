#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --account=plgbloraxs-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=prepare_env-%j.out
#SBATCH --error=prepare_env-%j.err

# Example environment-preparation script for the Polish HPC Helios cluster.
# Not a general-purpose installer: it relies on the Helios `ML-bundle` module
# and Helios-local wheel paths. For a portable install, use
# `scripts/prepare_env.sh` together with `requirements.txt`.
#
# Submit from the repository root, for example:
#   sbatch scripts/slurm/prepare_env_helios.sbatch
# or pass an explicit working directory:
#   sbatch --chdir=/path/to/bay_loraxs scripts/slurm/prepare_env_helios.sbatch

REPO_ROOT="${REPO_ROOT:-$PWD}"
if [ ! -f "$REPO_ROOT/requirements.txt" ]; then
  echo "ERROR: requirements.txt not found in REPO_ROOT=$REPO_ROOT" >&2
  echo "Submit this job from the repo root or pass --chdir=/path/to/bay_loraxs." >&2
  exit 1
fi

# IMPORTANT: load the modules for machine learning tasks and libraries
ml ML-bundle/24.06a

echo "USING REPO_ROOT=$REPO_ROOT"
echo "USING WORKSPACE_DIR=$WORKSPACE_DIR"
cp "$REPO_ROOT/requirements.txt" "$WORKSPACE_DIR/"
cd "$WORKSPACE_DIR"

# create and activate the virtual environment
python -m venv .env/
source .env/bin/activate

# install one of torch versions available at Helios wheel repo
pip install --no-cache-dir torch==2.5.1+cu124.post3

pip install --no-cache-dir /net/software/aarch64/el8/wheels/ML-bundle/24.06a/accelerate-1.1.0-py3-none-any.whl
pip install --no-cache-dir /net/software/aarch64/el8/wheels/ML-bundle/24.06a/torchmetrics-1.4.0.post0-py3-none-any.whl


# install the rest of requirements, for example via requirements file
pip install --no-cache-dir -r requirements.txt
