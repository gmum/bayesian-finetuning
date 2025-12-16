# Bayesian Fine-tuning in Projected Subspaces

Our repository is based on
[https://github.com/gmum/b-lora-xs](https://github.com/gmum/b-lora-xs)

## Setup
Prepare a work enivornment using [prepare_env.sh](prepare_env.sh) and [requirements.txt](requirements.txt).
Set your workspace directory using `export WORKSPACE_DIR=path`.
Set `save_folder`, `wandb_path` and other WandDB settings in [conf/config.yaml](conf/config.yaml).

We use Accelerate and Hydra to run our experiments. Accelerate expects the following variables exported to the working environment:
```
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=$((12000 + RANDOM % 20000))
```

We use WandDB for monitoring. To log in:
```
pip install wandb weave
wandb login
```
To set it up modify wandb entries in [conf/config.yaml](conf/config.yaml).

To use HuggingFace models one needs to `export HF_TOKEN=...` and request access to the models.
In particular we use `meta-llama/Llama-2-7b-chat-hf`.

The easiest is to set all the exports including `WORKSPACE_DIR`, WandDB API and HF API key in your `.bashrc`.

Test your enviornment on a machine with GPU using [run_test_single_gpu.sh](run_test_single_gpu.sh).

## Implementation

The main logic for replacing LoRA modules is implemented in [loraxs.py](loraxs.py). 

## Experiments

LLAMA experiments were executed using [submit_grid.sh](submit_grid.sh) which relies on [run_job.sh](run_job.sh).


## License
Copyright (C) 2025 Patryk Marszałek, Klaudia Bałazy, Jacek Tabor, Tomasz Kuśmierczyk

This project is distributed under the terms of the [GNU Affero General Public License v3](licenses/LICENSE). 
Portions of the code derived from MIT-licensed sources remain compatible under both the MIT license and AGPL v3. 
Please see the [SWAG LoRA LICENSE file](licenses/SWAG_LORA_LICENSE) for details.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
See the [GNU Affero General Public License v3](licenses/LICENSE) for more details.
