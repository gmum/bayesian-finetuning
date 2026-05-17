#!/bin/bash

# Resolve repository root from this script's location so the grid can be
# submitted from any directory. The sbatch job (run_job.sh) is anchored
# in $SCRIPT_DIR and starts with REPO_ROOT as its working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Arrays of values to sweep
tasks=("arc-c" "arc-e" "obqa") # cola
learning_rates=(5e-4) # (1e-3, 8e-4)
cls_learning_rates=(5e-4) # (1e-3, 8e-4)

# tasks=("arc-e") # obqa, arc-e
# learning_rates=(5e-4) # (1e-3, 8e-4)
# cls_learning_rates=(5e-4) # (1e-3, 8e-4)

# tasks=("obqa" "arc-c") # obqa, arc-e
# learning_rates=(1e-3) # (1e-3, 8e-4)
# cls_learning_rates=(1e-3) # (1e-3, 8e-4)

# Freezed version
# lora_rs=(64)
# # lora_rs=(25 32 48 64)
# lora_alphas=(25)
# unfreeze_A=(False)
# unfreeze_B=(False)

# Unfreezed version
lora_rs=(8)
lora_alphas=(25)
unfreeze_A=(True)
unfreeze_B=(True)
EXTEND_TARGET_MODULES=True # Whether to extend target modules to include o_proj and down_proj for Unfreezed version

ADD_LM_HEAD=False

seeds=(111 1337 3407)

lora_dropouts=(0.0) # (0.0, 0.1)
lora_weight_decays=(0.1) # (0.01, 0.001)
classifier_weight_decays=(0.01) # (0.01, 0.001)
epochs=(10) # (10, 15)

DO_LAPLACE=True

for task in "${tasks[@]}"; do
  for lora_r in "${lora_rs[@]}"; do
    for seed in "${seeds[@]}"; do
      for learning_rate in "${learning_rates[@]}"; do
        for cls_learning_rate in "${cls_learning_rates[@]}"; do
        for lora_alpha in "${lora_alphas[@]}"; do
          for lora_dropout in "${lora_dropouts[@]}"; do
            for lora_weight_decay in "${lora_weight_decays[@]}"; do
              for classifier_weight_decay in "${classifier_weight_decays[@]}"; do
                for epoch in "${epochs[@]}"; do
                  for unfreeze_A in "${unfreeze_A[@]}"; do
                    for unfreeze_B in "${unfreeze_B[@]}"; do
                    echo "Submitting: TASK=$task, LORA_R=$lora_r, LORA_ALPHA=$lora_alpha, SEED=$seed, LEARNING_RATE=$learning_rate, CLS_LEARNING_RATE=$cls_learning_rate, LORA_DROPOUT=$lora_dropout, LORA_WEIGHT_DECAY=$lora_weight_decay, CLASSIFIER_WEIGHT_DECAY=$classifier_weight_decay, EPOCHS=$epoch, UNFREEZE_A=$unfreeze_A, UNFREEZE_B=$unfreeze_B, ADD_LM_HEAD=$ADD_LM_HEAD, EXTEND_TARGET_MODULES=$EXTEND_TARGET_MODULES"
                    sbatch --chdir="$REPO_ROOT" --export=ALL,TASK=$task,LORA_R=$lora_r,LORA_ALPHA=$lora_alpha,SEED=$seed,LEARNING_RATE=$learning_rate,CLS_LEARNING_RATE=$cls_learning_rate,LORA_DROPOUT=$lora_dropout,LORA_WEIGHT_DECAY=$lora_weight_decay,CLASSIFIER_WEIGHT_DECAY=$classifier_weight_decay,DO_LAPLACE=$DO_LAPLACE,EPOCHS=$epoch,UNFREEZE_A=$unfreeze_A,UNFREEZE_B=$unfreeze_B,ADD_LM_HEAD=$ADD_LM_HEAD,EXTEND_TARGET_MODULES=$EXTEND_TARGET_MODULES "$SCRIPT_DIR/run_job.sh"
                    done
                  done
                done
              done
            done
          done
          done
        done
      done
    done
  done
done 