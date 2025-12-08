#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --account=plgbloraxs-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=prepare_env-%j.out
#SBATCH --error=prepare_env-%j.err

# IMPORTANT: load the modules for machine learning tasks and libraries
ml ML-bundle/24.06a

# cd $SCRATCH
echo "USING WORKSPACE_DIR=$WORKSPACE_DIR"
cp requirements.txt $WORKSPACE_DIR/
cd $WORKSPACE_DIR

# create and activate the virtual environment 
python -m venv .env/
source .env/bin/activate

# # install one of torch versions available at Helios wheel repo
# pip install --no-cache-dir torch==2.5.1+cu124.post3

# pip install --no-cache-dir /net/software/aarch64/el8/wheels/ML-bundle/24.06a/accelerate-1.1.0-py3-none-any.whl 
# pip install --no-cache-dir /net/software/aarch64/el8/wheels/ML-bundle/24.06a/torchaudio-2.4.0+cu124-cp311-cp311-linux_aarch64.whl
# pip install --no-cache-dir /net/software/aarch64/el8/wheels/ML-bundle/24.06a/torchvision-0.20.1+cu124torch251-cp311-cp311-linux_aarch64.whl
# pip install --no-cache-dir /net/software/aarch64/el8/wheels/ML-bundle/24.06a/torchmetrics-1.4.0.post0-py3-none-any.whl


# pip install /net/software/aarch64/el8/wheels/ML-bundle/24.06a/bitsandbytes-0.44.1-cp311-cp311-linux_aarch64.whl

# pip install /net/software/aarch64/el8/wheels/ML-bundle/24.06a/triton-3.0.0-cp311-cp311-linux_aarch64.whl

# install the rest of requirements, for example via requirements file
pip install --no-cache-dir -r requirements.txt

# pip install --no-cache-dir wandb==0.16.1

# pip list