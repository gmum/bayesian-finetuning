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

The canonical dependency spec is [requirements.txt](requirements.txt). Either install
the dependencies directly:

```
python3 -m venv .env
source .env/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

or run the convenience wrapper [scripts/prepare_env.sh](scripts/prepare_env.sh) from
the repository root. 

Example environment setup on an HPC cluster is in [scripts/slurm/prepare_env_helios.sbatch](scripts/slurm/prepare_env_helios.sbatch).

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

To use HuggingFace models one needs to `export HF_TOKEN=...`.  
Usage of `meta-llama/Llama-2-7b-chat-hf`require specific request access.

The easiest is to set all the exports including `WORKSPACE_DIR`, WandDB API and HF API key in your `.bashrc`.

Test your enviornment on a machine with GPU using
[scripts/slurm/run_test_single_gpu.sh](scripts/slurm/run_test_single_gpu.sh).
SLURM scripts assume the job working directory is the repository root, so submit
them from the repo root (`sbatch scripts/slurm/run_test_single_gpu.sh`) or pass
`--chdir=/path/to/bay_loraxs`.

## Implementation

The main logic for replacing LoRA modules is implemented in [loraxs.py](loraxs.py). 

The main post-hoc Laplace evaluation is implememented in [laplace_utils.py](laplace_utils.py)

## Experiments

LLAMA experiments were executed using
[scripts/slurm/submit_grid.sh](scripts/slurm/submit_grid.sh) which relies on
[scripts/slurm/run_job.sh](scripts/slurm/run_job.sh).

Supported flows:

- `roberta-large` + Laplace on GLUE (`cola`, `mrpc`, `rte`, `sst2`) — templates in
`experiments/template_roberta_*.sbatch`.
- `meta-llama/Llama-2-7b-chat-hf` + Laplace on commonsense MCQA (`arc-c`, `arc-e`,
`obqa`) — templates in `experiments/template_llama_*.sbatch` and
`scripts/slurm/run_job.sh` / `scripts/slurm/submit_grid.sh`.
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