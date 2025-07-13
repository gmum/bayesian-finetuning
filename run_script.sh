#!/bin/bash -l

export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=29637  # Or another free port

# Define the different values for experiment.lora_r, method.swag_start, and experiment.task
# lora_r_values=(2 8 16 25)  # Example values for experiment.lora_r
lora_r_values=(8)
# swag_start_values=(25 50 75 100)  # Example values for method.swag_start
# task_values=("cola" "sst2")  # Updated task values
task_values=("cola")
# seed_values=(0 1 2 3 4) # Seeds
seed_values=(0)
# model="roberta-large"
model="llama_7b_hf"


echo "PYTHONPATH: $PYTHONPATH"

# Parse arguments for --slurm
USE_SLURM=true
for arg in "$@"; do
  if [ "$arg" == "--slurm" ]; then
    USE_SLURM=true
    shift # Remove --slurm from arguments
  fi
done

# SLURM parameters (customize as needed)
SBATCH_PARAMS="\
#SBATCH --job-name=laplace\n\
#SBATCH --nodes=1\n\
#SBATCH --ntasks-per-node=1\n\
#SBATCH --cpus-per-task=16\n\
#SBATCH --gres=gpu:1\n\
#SBATCH --time=01:00:00\n\
#SBATCH --account=plgbloraxs-gpu-gh200\n\
#SBATCH --partition=plgrid-gpu-gh200\n\
#SBATCH --output=job-%j.out\n\
#SBATCH --error=job-%j.err\n"

# Loop through each combination of the values
for lora_r in "${lora_r_values[@]}"; do
  # for swag_start in "${swag_start_values[@]}"; do
    for task in "${task_values[@]}"; do
      for seed in "${seed_values[@]}"; do
        # mnli_model_path="./model_checkpoints/RoBERTa-large/MNLI/rank_${lora_r}"

        # Build the command as an array to avoid leading spaces and newlines
        EXP_CMD=(accelerate launch launch_exp_hydra.py
          model=$model
          experiment.task=$task
          method.force_save=-1
          experiment.learning_rate=1e-3
          experiment.cls_learning_rate=5e-3
          experiment.num_epochs=5
          experiment.use_loraxs=True
          experiment.lora_r=$lora_r
          experiment.seed=$seed
          experiment.skip_training=True
          experiment.overwrite=True
        )

        if [ "$USE_SLURM" = true ]; then
          # Create a temporary SLURM script
          SLURM_SCRIPT=$(mktemp /tmp/slurm_script.XXXXXX.sh)
          echo -e "#!/bin/bash\n$SBATCH_PARAMS\nml ML-bundle/24.06a\nsource \$SCRATCH/laplace_loraxs/bin/activate\nexport \n${EXP_CMD[@]}" > "$SLURM_SCRIPT"
          sbatch "$SLURM_SCRIPT"
          echo "Submitted SLURM job for task $task, lora_r $lora_r, seed $seed"
        else
          # Run locally
          "${EXP_CMD[@]}"
        fi
      done
  done
done