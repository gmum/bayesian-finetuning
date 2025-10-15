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
#SBATCH --output=prec_logs/job-%j.out
#SBATCH --error=prec_logs/job-%j.err

# Set envs for distributed training
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
# export MASTER_PORT=29637  # Or another free port
export MASTER_PORT=$((12000 + RANDOM % 20000))
# Set environment variables with defaults if not already set
LORA_R="${LORA_R:-32}"
# TASK="${TASK:-cola}"
TASK="${TASK:-obqa}" # obqa, arc-e
SEED="${SEED:-3407}"
# MODEL="${MODEL:-llama_7b_hf}"
MODEL="${MODEL:-llama_7b_chat}"
# optional parameter EPOCHS
EPOCHS="${EPOCHS:-15}" # default - empty
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
CLS_LEARNING_RATE="${CLS_LEARNING_RATE:-1e-3}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_WEIGHT_DECAY="${LORA_WEIGHT_DECAY:-0.01}"
CLASSIFIER_WEIGHT_DECAY="${CLASSIFIER_WEIGHT_DECAY:-0.01}"
DO_LAPLACE="${DO_LAPLACE:-True}"
LORA_ALPHA="${LORA_ALPHA:-16}"
UNFREEZE_A="${UNFREEZE_A:-False}"
UNFREEZE_B="${UNFREEZE_B:-False}"

# # set number of epochs depending on the dataset. If it's cola - 5, obqa or arc-e - 10
# # If epochs wasn't set by the user, set it to the default value
# if [ "$TASK" == "cola" ]; then
#     EPOCHS=10
# elif [ "$TASK" == "obqa" ] || [ "$TASK" == "arc-e" ]; then
#     EPOCHS=15
# fi


echo "MASTER_PORT: $MASTER_PORT"
echo "TASK: $TASK"
echo "EPOCHS: $EPOCHS"
echo "LORA_R: $LORA_R"
echo "SEED: $SEED"
echo "MODEL: $MODEL"
echo "LEARNING_RATE: $LEARNING_RATE"
echo "CLS_LEARNING_RATE: $CLS_LEARNING_RATE"
echo "LORA_DROPOUT: $LORA_DROPOUT"
echo "LORA_WEIGHT_DECAY: $LORA_WEIGHT_DECAY"
echo "CLASSIFIER_WEIGHT_DECAY: $CLASSIFIER_WEIGHT_DECAY"
echo "DO_LAPLACE: $DO_LAPLACE"
echo "LORA_ALPHA: $LORA_ALPHA"
echo "UNFREEZE_A: $UNFREEZE_A"
echo "UNFREEZE_B: $UNFREEZE_B"

ml ML-bundle/24.06a

# source $SCRATCH/bloraxs/bin/activate
source $GRANT_DIR/bloraxs/bin/activate    
# list all the modules installed
# pip list

accelerate launch launch_exp_hydra.py \
  model=$MODEL \
  experiment.task=$TASK \
  experiment.do_laplace=$DO_LAPLACE \
  method.force_save=-1 \
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
  experiment.unfreeze_B=$UNFREEZE_B