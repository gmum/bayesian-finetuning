#!/bin/bash -l
#SBATCH --job-name=test1gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00
#SBATCH --account=plgbloraxs-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --mem=120G
#SBATCH --output=test1gpu-%j.out
#SBATCH --error=test1gpu-%j.err


# WORKSPACE_DIR is set and not empty, ensure it ends with /
if [ -n "${WORKSPACE_DIR}" ]; then
  [[ "${WORKSPACE_DIR}" != */ ]] && WORKSPACE_DIR="${WORKSPACE_DIR}/"
fi

# Specific for the cluster
ml ML-bundle/24.06a
echo "USING WORKSPACE_DIR=$WORKSPACE_DIR"
source $WORKSPACE_DIR.env/bin/activate


# Set envs for distributed training
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
# export MASTER_PORT=29637  # Or another free port
export MASTER_PORT=$((12000 + RANDOM % 20000))
export HYDRA_FULL_ERROR=1

echo "A single-GPU test script..."

# CONFIGURATION PARAMETERS:
LORA_R=4
TASK="obqa" # obqa, arc-e
SEED=3407

# MODEL="${MODEL:-llama_7b_hf}"
MODEL="roberta-base"

# optional parameter EPOCHS
EPOCHS=6

LEARNING_RATE=1e-3
CLS_LEARNING_RATE=1e-3
LORA_DROPOUT=-0.0
LORA_WEIGHT_DECAY=0.01
CLASSIFIER_WEIGHT_DECAY=0.01
DO_LAPLACE=True
LORA_ALPHA=16
UNFREEZE_A=False
UNFREEZE_B=False
ADD_LM_HEAD=False
EXTEND_TARGET_MODULES=False

RECONSTRUCT_CONFIG="reconstruct_config.yaml"
# RECONSTRUCT_CONFIG="reconstruct_config_halfdct.yaml"

RECONSTRUCT_TYPE="dct-1/4_svd-1/4_random"  # overrides the value from the reconstruct_config


# RUN ON A SINGLE GPU:
torchrun --standalone --nnodes=1 --nproc_per_node=1 launch_exp_hydra.py \
 +reconstruct_config=$RECONSTRUCT_CONFIG \
 +reconstruction_type=$RECONSTRUCT_TYPE \
 model=$MODEL \
 experiment.task=$TASK \
 method.do_laplace=$DO_LAPLACE \
 experiment.learning_rate=$LEARNING_RATE \
 experiment.cls_learning_rate=$CLS_LEARNING_RATE \
 experiment.num_epochs=$EPOCHS \
 experiment.use_loraxs=True \
 experiment.lora_r=$LORA_R \
 experiment.lora_alpha=$LORA_ALPHA \
 experiment.seed=$SEED \
 experiment.skip_training=False \
 experiment.overwrite=True \
 experiment.lora_dropout=$LORA_DROPOUT \
 experiment.lora_weight_decay=$LORA_WEIGHT_DECAY \
 experiment.classifier_weight_decay=$CLASSIFIER_WEIGHT_DECAY \
 experiment.unfreeze_A=$UNFREEZE_A \
 experiment.unfreeze_B=$UNFREEZE_B \
 experiment.add_lm_head=$ADD_LM_HEAD \
 experiment.extend_target_modules=$EXTEND_TARGET_MODULES


