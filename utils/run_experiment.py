import os

import torch
import wandb
import yaml
from peft import PromptLearningConfig

import numpy
import random
from types import SimpleNamespace

from data import TASK_TYPE_DICT, N_CLASSES_DICT

from loraxs import find_and_initialize
from train import train_laplace
from utils.peft_utils import create_peft_model, get_peft_config
from utils.config_utils import set_save_path
from data import (
    load_glue_data,
    load_mcqa_data,
    MCQA_task_to_context_keys,
    GLUE_task_to_keys,
)
from accelerate import Accelerator
# from accelerate import DeepSpeedPlugin
from utils.peft_utils import WrappedModel



def run_experiment(config):
    # deepspeed and accelerate initialization
    # deepspeed_plugin = DeepSpeedPlugin(
    #     zero_stage=2,
    #     gradient_accumulation_steps=config.experiment.gradient_accumulation_steps,
    # )
    # accelerator = Accelerator(
    #     split_batches=True, log_with="wandb", deepspeed_plugin=deepspeed_plugin
    # )  # mixed_precision='fp16')
    
    accelerator = Accelerator(
        split_batches=True, log_with="wandb"
    )

    wandb_group = config.experiment.wandb_group
    model_name = config.model.model_name
    task = config.experiment.task

    if config.experiment.use_loraxs:
        if config.model.model_name == "roberta-large" or config.model.model_name == "roberta-base":
            config.model.target_modules.extend(
                ["attention.output.dense", "output.dense"])
        elif "meta-llama" in config.model.model_name:
            if config.experiment.add_lm_head:
                config.model.target_modules.extend(['lm_head'])
            if config.experiment.unfreeze_A or config.experiment.unfreeze_B:
                if config.experiment.extend_target_modules:
                    config.model.target_modules.extend(['o_proj', 'down_proj'])
            else:
                # LORA-XS case
                config.model.target_modules.extend(
                    ['k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
                # # test
                # config.model.target_modules.extend(['o_proj', 'down_proj'])
        else:
            raise ValueError(f"Model {config.model.model_name} not supported")

    active_tags = [
        model_name,
        task,
        "loraxs" if config.experiment.use_loraxs else "lora",
    ]

    # Create a descriptive run name
    run_name = f"{model_name}_{task}_{'loraxs' if config.experiment.use_loraxs else 'lora'}_seed{config.experiment.seed}_lr{config.experiment.learning_rate}_cls_lr{config.experiment.cls_learning_rate}_ep{config.experiment.num_epochs}"
    
    accelerator.init_trackers(
        project_name=config.experiment.wandb_project,
        init_kwargs={
            "wandb": {
                "entity": config.experiment.wandb_entity,
                "group": wandb_group,
                "tags": active_tags,
                "name": run_name,
            }
        },
    )

    set_save_path(config, accelerator)

    seed = config.experiment.seed
    print(f"Setting seed to <{seed}>")
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    task = config.experiment.task

    print("offline loading = ", config.experiment.offline)

    ood_dataloader = None

    # Load data
    print("-----------------------Preparing data------------------------")
    if task in GLUE_task_to_keys:
        tokenizer, train_dataloader, eval_dataloader, test_dataloader, num_classes = (
            load_glue_data(
                config,
                task,
                accelerator,
                subtask=config.experiment.subtask,
                offline=config.experiment.offline,
                data_path=config.experiment.data_path,
            )
        )
    elif task in MCQA_task_to_context_keys:
        tokenizer, train_dataloader, eval_dataloader, test_dataloader, num_classes = (
            load_mcqa_data(
                config,
                task,
                accelerator,
                subtask=config.experiment.subtask,
                offline=config.experiment.offline,
                data_path=config.experiment.data_path,
            )
        )
    else:
        raise Exception("Only GLUE tasks and MCQA tasks implemented")
    print("--------------------------------------------------------------")

    if config.experiment.ood_task != "":
        print("-----------------------Preparing OOD data---------------------")
        if task in GLUE_task_to_keys:
            _, _, _, ood_dataloader, _ = load_glue_data(
                config,
                config.experiment.ood_task,
                accelerator,
                batch_size=config.experiment.ood_batch_size,
                subtask=config.experiment.ood_subtask,
                offline=config.experiment.offline,
                data_path=config.experiment.data_path,
            )
        elif task in MCQA_task_to_context_keys:
            _, _, _, ood_dataloader, _ = load_mcqa_data(
                config,
                config.experiment.ood_task,
                accelerator,
                batch_size=config.experiment.ood_batch_size,
                subtask=config.experiment.ood_subtask,
                offline=config.experiment.offline,
                data_path=config.experiment.data_path,
            )
        else:
            raise Exception("Only GLUE tasks and MCQA tasks implemented")
        print("--------------------------------------------------------------")

    peft_config = get_peft_config(config, accelerator=accelerator)

    accelerator.print(f"Method: {config.method.method_name}")

    print("-----------------------Creating PEFT model-----------------------")
    print("RANK ", peft_config.r)
    model = create_peft_model(
        config, peft_config, num_classes=num_classes, tokenizer=tokenizer
    )
    print("-----------------------------------------------------------------")

    if config.experiment.use_loraxs:
        print("-----------------------Creating LORA-XS-----------------------")

        adapter_name = "default"
        peft_config_dict = {}
        if not isinstance(peft_config, PromptLearningConfig):
            peft_config_dict[adapter_name] = peft_config

        with open("conf/reconstruct_config.yaml", "r") as stream:
            reconstr_config = yaml.load(stream, Loader=yaml.FullLoader)
        reconstr_type = reconstr_config["reconstruction_type"]
        reconstr_config[reconstr_type]["rank"] = peft_config_dict[adapter_name].r
        print("XS-RANK ", peft_config_dict[adapter_name].r)

        print("LORA-XS MODE ", config.experiment.loraxs_mode)

        find_and_initialize(
            model,
            peft_config_dict,
            adapter_name=adapter_name,
            reconstr_type=reconstr_type,
            reconstruct_config=reconstr_config,
            writer=None,
            unfreeze_A=config.experiment.unfreeze_A,
            unfreeze_B=config.experiment.unfreeze_B,
            loraxs_sigma=config.experiment.loraxs_sigma,
            loraxs_mode=config.experiment.loraxs_mode,
        )
        model.print_trainable_parameters()

        if config.experiment.task in ["mrpc", "rte", "stsb"]:
            if config.experiment.mnli_model_path is None:
                raise ValueError(
                    f"Path to MNLI model is not provided for {config.experiment.task} task."
                )
            elif peft_config.r in [4, 8, 12, 16, 20, 25]:
                from safetensors import safe_open

                state_dict_mnli = {}
                with safe_open(
                    os.path.join(
                        config.experiment.mnli_model_path, f"rank_{peft_config.r}", "adapter_model.safetensors"
                    ),
                    framework="pt",
                    device="cpu",
                ) as f:
                    for key in f.keys():
                        state_dict_mnli[key] = f.get_tensor(key)
                renamed_state_dict = {
                    k.replace("lora_A", "lora_A.default")
                    .replace("lora_B", "lora_B.default")
                    .replace("_lora_latent", ".default_lora_latent"): v
                    for (k, v) in state_dict_mnli.items()
                    if "classifier.out_proj" not in k
                }

                model.load_state_dict(renamed_state_dict, strict=False)
                print("Loaded pretrained")
        print("-----------------------------------------------------------------")

        model.print_trainable_parameters()

    if "meta-llama" in config.model.model_name and TASK_TYPE_DICT[config.experiment.task] == "MCQA":
        # Use WrappedModel instead, that will output only classification logits in form [batch_size, num_classes]
        model = WrappedModel(model, config.experiment.task, tokenizer)

    for param in model.parameters():
        param.data = param.data.contiguous()

    # Train base (MAP) and optionally run Laplace
    if config.method.method_name == "laplace":

        print("-----------------------TRAIN LAPLACE-----------------------")

        # wandb.run.tags = list(wandb.run.tags) + [f"{len(model.get_peft_model().get_trainable_parameters()) / 1000}K"]
        train_laplace(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            config,
            accelerator,
            tokenizer,
            num_classes=num_classes,
            peft_config=peft_config,
            ood_dataloader=ood_dataloader,
            save_checkpoints_at_epochs=config.experiment.save_checkpoints_at_epochs,
        )
    else:
        raise Exception("Only 'laplace' method is implemented")

    accelerator.end_training()
