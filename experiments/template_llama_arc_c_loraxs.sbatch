#!/bin/bash -l
#SBATCH --job-name=$RAWIDENTIFIER
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
#SBATCH --account=plgittossl-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --mem=120G
#SBATCH --output=logs/$RAWIDENTIFIER-%j.out
#SBATCH --error=logs/$RAWIDENTIFIER-%j.err

## Set envs for distributed training
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
# export MASTER_PORT=29637  # Or another free port
export MASTER_PORT=$((12000 + RANDOM % 20000))


# WORKSPACE_DIR is set and not empty, ensure it ends with /
if [ -n "${WORKSPACE_DIR}" ]; then
  [[ "${WORKSPACE_DIR}" != */ ]] && WORKSPACE_DIR="${WORKSPACE_DIR}/"
fi

## Specific for the cluster
ml ML-bundle/24.06a
echo "USING WORKSPACE_DIR=$WORKSPACE_DIR"
source $WORKSPACE_DIR.env/bin/activate


# CONFIGURATION PARAMETERS:
IDENTIFIER="experiment_llama7b_arc_c_loraxs_3407"

LORA_R=64
TASK="arc-c"
SEED=3407

MODEL="llama_7b_chat"

EPOCHS=7
CHECKPOINT_EPOCHS="[4,5,6]"

LEARNING_RATE=1e-3
CLS_LEARNING_RATE=1e-3
LORA_DROPOUT=0.0
LORA_WEIGHT_DECAY=0.1
CLASSIFIER_WEIGHT_DECAY=0.01
DO_LAPLACE=True
LORA_ALPHA=25

UNFREEZE_A=False
UNFREEZE_B=False

ADD_LM_HEAD=False
EXTEND_TARGET_MODULES=False

RECONSTRUCT_CONFIG="reconstruct_config.yaml"

RECONSTRUCT_TYPE="svd"  # overrides the value from the reconstruct_config


## TO RUN ON A SINGLE GPU REPLACE THE FIRST LINE WITH:
# torchrun --standalone --nnodes=1 --nproc_per_node=1 launch_exp_hydra.py \
accelerate launch launch_exp_hydra.py \
 +reconstruct_config=$RECONSTRUCT_CONFIG \
 +reconstruction_type=$RECONSTRUCT_TYPE \
 model=$MODEL \
 experiment.exp_name=$IDENTIFIER \
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
 experiment.extend_target_modules=$EXTEND_TARGET_MODULES \
 experiment.save_checkpoints_at_epochs=$CHECKPOINT_EPOCHS
