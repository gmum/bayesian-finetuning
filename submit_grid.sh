#!/bin/bash

# Arrays of values to sweep
# tasks=("arc-e") # cola
# learning_rates=(5e-4) # (1e-3, 8e-4)
# cls_learning_rates=(5e-4) # (1e-3, 8e-4)

tasks=("obqa") # obqa, arc-e
learning_rates=(1e-3) # (1e-3, 8e-4)
cls_learning_rates=(1e-3) # (1e-3, 8e-4)

lora_rs=(16)
lora_alphas=(16)
# seeds=(42 111 333 777) # 2
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
                echo "Submitting: TASK=$task, LORA_R=$lora_r, LORA_ALPHA=$lora_alpha, SEED=$seed, LEARNING_RATE=$learning_rate, CLS_LEARNING_RATE=$cls_learning_rate, LORA_DROPOUT=$lora_dropout, LORA_WEIGHT_DECAY=$lora_weight_decay, CLASSIFIER_WEIGHT_DECAY=$classifier_weight_decay, EPOCHS=$epoch"
                sbatch --export=TASK=$task,LORA_R=$lora_r,LORA_ALPHA=$lora_alpha,SEED=$seed,LEARNING_RATE=$learning_rate,CLS_LEARNING_RATE=$cls_learning_rate,LORA_DROPOUT=$lora_dropout,LORA_WEIGHT_DECAY=$lora_weight_decay,CLASSIFIER_WEIGHT_DECAY=$classifier_weight_decay,DO_LAPLACE=$DO_LAPLACE,EPOCHS=$epoch run_job.sh
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