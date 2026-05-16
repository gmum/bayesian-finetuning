# Bayesian Fine-tuning in Projected Subspaces

Our repository is based on
[https://github.com/gmum/b-lora-xs](https://github.com/gmum/b-lora-xs)

## Scope

This repository implements the journal extension's novel contributions: the Laplace
Approximation (L-LoRA-XS and L-LoRA-S variants, with KRON or DIAG covariance) and
the four projection strategies (SVD, Whitened SVD / W-SVD, DCT, RAND) plus their
hybrids, evaluated on `roberta-large` (GLUE) and `meta-llama/Llama-2-7b-chat-hf`
(commonsense MCQA). The previously-published SWAG variant (B-LoRA-XS, EMNLP 2025)
is intentionally **not** implemented here; to reproduce the B-LoRA-XS / LoRA-SWAG
numbers please use the original release at
[gmum/b-lora-xs](https://github.com/gmum/b-lora-xs).

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

Supported flows:

- `roberta-large` + Laplace on GLUE (`cola`, `mrpc`, `rte`, `sst2`) — templates in
  `experiments/template_roberta_*.sbatch`.
- `meta-llama/Llama-2-7b-chat-hf` + Laplace on commonsense MCQA (`arc-c`, `arc-e`,
  `obqa`) — templates in `experiments/template_llama_*.sbatch` and
  `run_job.sh` / `submit_grid.sh`.
- Laplace under alternative projections / reconstruction types — SVD, Whitened SVD
  (W-SVD; implemented in [cca_projections.py](cca_projections.py)), DCT, RAND, plus
  hybrids like `dct-1/2_svd` — via `+reconstruct_config=...` and
  `+reconstruction_type=...` (see [conf/reconstruct_config.yaml](conf/reconstruct_config.yaml),
  [conf/reconstruct_config_halfdct.yaml](conf/reconstruct_config_halfdct.yaml)).
- L-LoRA-S variant (Laplace with `A`, `B` also fine-tuned during MAP) via
  `experiment.unfreeze_A=True experiment.unfreeze_B=True experiment.extend_target_modules=True`.
- For SWAG (B-LoRA-XS / LoRA-SWAG) runs see
  [gmum/b-lora-xs](https://github.com/gmum/b-lora-xs) — the code release for the
  prior EMNLP 2025 publication.


## License
Copyright (C) 2026 Viktar Dubovik, Patryk Marszałek, Jacek Tabor, Tomasz Kuśmierczyk

This project is distributed under the terms of the [GNU Affero General Public License v3](licenses/LICENSE). 
Portions of the code derived from MIT-licensed sources remain compatible under both the MIT license and AGPL v3. 
Please see the [SWAG LoRA LICENSE file](licenses/SWAG_LORA_LICENSE) for details.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
See the [GNU Affero General Public License v3](licenses/LICENSE) for more details.
