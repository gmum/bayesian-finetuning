#!/bin/bash -l
#SBATCH --job-name=laplace
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --account=plgbloraxs-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --mem=120G
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err

# Set envs for distributed training
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=29637  # Or another free port
# Set environment variables with defaults if not already set
LORA_R="${LORA_R:-8}"
# TASK="${TASK:-cola}"
TASK="${TASK:-obqa}" # obqa, arc-e
SEED="${SEED:-0}"
MODEL="${MODEL:-llama_7b_hf}"

ml ML-bundle/24.06a

source $SCRATCH/bloraxs/bin/activate

# Set up Hugging Face cache directory
export HF_HOME="$SCRATCH/.cache/huggingface"
export TRANSFORMERS_CACHE="$SCRATCH/.cache/huggingface/transformers"

# Set your Hugging Face token here
# export HUGGING_FACE_HUB_TOKEN="your_token_here"

# export PYTHONPATH="/home/z1192950/bay_loraxs:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"
echo "Pip packages:"
pip list

accelerate launch launch_exp_hydra.py \
  model=$MODEL \
  experiment.task=$TASK \
  method.force_save=-1 \
  experiment.learning_rate=1e-3 \
  experiment.cls_learning_rate=5e-3 \
  experiment.num_epochs=5 \
  experiment.use_loraxs=True \
  experiment.lora_r=$LORA_R \
  experiment.seed=$SEED \
  experiment.skip_training=False \
  experiment.overwrite=True
