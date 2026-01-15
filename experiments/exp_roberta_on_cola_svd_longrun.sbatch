#!/bin/bash -l
#SBATCH --job-name=exp_roberta_on_cola_svd_longrun
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --account=plgbloraxs-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --mem=32G
#SBATCH --output=exp_roberta_on_cola_svd_longrun-%j.out
#SBATCH --error=exp_roberta_on_cola_svd_longrun-%j.err

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
TASK="cola"

MODEL="roberta-large"

EPOCHS=201
CHECKPOINT_EPOCHS="[5,10,15,20,25,30,35,40,45,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]"

DO_LAPLACE=True

UNFREEZE_A=False
UNFREEZE_B=False

ADD_LM_HEAD=False
EXTEND_TARGET_MODULES=False

RECONSTRUCT_CONFIG="reconstruct_config.yaml"


## TO RUN ON A SINGLE GPU REPLACE THE FIRST LINE WITH:
# torchrun --standalone --nnodes=1 --nproc_per_node=1 launch_exp_hydra.py \
accelerate launch launch_exp_hydra.py \
 +reconstruct_config=$RECONSTRUCT_CONFIG \
 +reconstruction_type="svd" \
 model=$MODEL \
 experiment.exp_name="exp_roberta_on_cola_svd_longrun" \
 experiment.task=$TASK \
 method.do_laplace=$DO_LAPLACE \
 experiment.learning_rate=0.001 \
 experiment.cls_learning_rate=0.005 \
 experiment.num_epochs=$EPOCHS \
 experiment.use_loraxs=True \
 experiment.lora_r=8 \
 experiment.lora_alpha=16 \
 experiment.seed=3407 \
 experiment.skip_training=False \
 experiment.overwrite=True \
 experiment.lora_dropout=0.0 \
 experiment.lora_weight_decay=0.01 \
 experiment.classifier_weight_decay=0.01 \
 experiment.unfreeze_A=$UNFREEZE_A \
 experiment.unfreeze_B=$UNFREEZE_B \
 experiment.add_lm_head=$ADD_LM_HEAD \
 experiment.extend_target_modules=$EXTEND_TARGET_MODULES \
 experiment.save_checkpoints_at_epochs=$CHECKPOINT_EPOCHS
 