import os
from safetensors import safe_open
from peft import PeftModel, PeftMixedModel

from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
)

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
   
)
import yaml

from loraxs import find_and_initialize


def load_run_config(configs_path, task_name, rank, common_pref="", spec_pref=""):
    common_conf_name = os.path.join(configs_path, f"{common_pref}config.yaml")
    with open(common_conf_name, "r") as stream:
        common_conf = yaml.load(stream, Loader=yaml.FullLoader)
    
    # merge common and specific configs, spec overrides common
    config = common_conf
    
    spec_conf_name = os.path.join(configs_path, task_name, f"{spec_pref}{task_name}_r{rank}.yaml")
    if os.path.exists(spec_conf_name):
        with open(spec_conf_name, "r") as stream:
            spec_conf = yaml.load(stream, Loader=yaml.FullLoader)
            config.update(spec_conf)
    
    return config

def get_output_dir_loraxs_args(root_dir,
                          task_name,
                          model_name_or_path,
                          peft_method,
                          lora_rank,
                          learning_rate,
                          cls_learning_rate,
                          seed):
    peft_method = "lora_xs"
    output_dir = os.path.join(
        root_dir,
        task_name,
        f"{model_name_or_path}_{peft_method}_rank_{lora_rank}_lr_{float(learning_rate)}_clslr_{float(cls_learning_rate)}_seed_{seed}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def get_output_dir_loraxs(data_args, model_args, training_args):
    peft_method = "lora_xs"
    output_dir = os.path.join(
        training_args.output_dir,
        data_args.task_name,
        f"rank_{model_args.lora_rank}",
        f"{model_args.model_dir_prefix}{model_args.model_name_or_path}_{peft_method}_lr_{float(training_args.learning_rate)}_clslr_{float(model_args.cls_learning_rate)}_seed_{training_args.seed}_ep_{training_args.num_train_epochs}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def load_loraxs_weights(model: PeftModel | PeftMixedModel, checkpoint_dir:str, load_classifier: bool) -> None:
    """
    Loads adapter weights from a given checkpoint directory into the model.

    Args:
        model (torch.nn.Module): The model to which the adapter weights should be loaded.
        checkpoint_dir (str): The directory containing the adapter weights file `adapter_model.safetensors`.
        load_classifier (bool): Whether to load classifier weights or not. If MNLI checkpoint is loaded,
            don't load classifier weights, just the adapter weights, so set load_classifier=False.

    Raises:
        ValueError: If the `checkpoint_dir` is not provided or the file does not exist.
    """
    if checkpoint_dir is None:
        raise ValueError("Path to the checkpoint directory is not provided.")

    adapter_file_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")

    if not os.path.exists(adapter_file_path):
        raise ValueError(f"Adapter weights file not found at {adapter_file_path}.")
    print(f"Loading adapter weights from {adapter_file_path}.")
    
    if not any("default_lora_latent" in k for k in model.state_dict().keys()):
        print("Given LoRA model, initializing LoRA-XS model.")
        model, peft_config_dict, reconstr_config, adapter_name = create_loraxs_model(model, create_fast=True)
        
    
    state_dict = {}
    with safe_open(adapter_file_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    renamed_state_dict = {
        k.replace("lora_A", "lora_A.default")
         .replace("lora_B", "lora_B.default")
         .replace("_lora_latent", ".default_lora_latent"): v
        for k, v in state_dict.items()
    }
    if not load_classifier:
        print("Not loading classifier weights.")
        renamed_state_dict = {k: v for k, v in renamed_state_dict.items() if "classifier" not in k}
    else:
        # change classifier. name to classifier.modules_to_save.default
        print("Loading classifier weights. Renaming classifier weights to classifier.modules_to_save.default.")
        renamed_state_dict = {
            k.replace("classifier", "classifier.modules_to_save.default"): v
            for k, v in renamed_state_dict.items()
        }
    # print what keys differ between the model and the checkpoint
    model_state_dict = model.state_dict()
    # print(f"Keys in model, but not in checkpoint: {set(model_state_dict.keys()) - set(renamed_state_dict.keys())}")
    print(f"Keys in checkpoint, but not in model: {set(renamed_state_dict.keys()) - set(model_state_dict.keys())}")

    model.load_state_dict(renamed_state_dict, strict=False)

def create_lora_model(model_args, data_args, num_labels):
    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        inference_mode=False,
        r=model_args.lora_rank,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=0.0,
        target_modules=["query", "value", "attention.output.dense", "output.dense"],
    )
    
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    model = get_peft_model(model, peft_config)
    
    return model, tokenizer


def create_loraxs_model(lora_model: PeftModel, create_fast: bool = False, verbose: bool = False):
    """
    If `create_fast` is True, number of svd iterations is set to 1.
    Use if you already have saved weights you want to load in.
    """
    adapter_name = "default"
    peft_config = lora_model.peft_config[adapter_name]
    peft_config_dict = {}
    peft_config_dict[adapter_name] = peft_config

    with open("config/reconstruct_config.yaml", "r") as stream:
        reconstr_config = yaml.load(stream, Loader=yaml.FullLoader)
    reconstr_type = reconstr_config["reconstruction_type"]
    reconstr_config[reconstr_type]["rank"] = peft_config_dict[adapter_name].r
    
    if create_fast:
        reconstr_config["svd"]["n_iter"] = 1
    
    find_and_initialize(
        lora_model,
        peft_config_dict,
        adapter_name=adapter_name,
        reconstr_type=reconstr_type,
        reconstruct_config=reconstr_config,
        writer=None
    )

    for param in lora_model.parameters():
        param.data = param.data.contiguous()
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_model.to(device)
    if verbose:
        print("LoRA-XS model created.")
        print(lora_model)
    
    return lora_model, peft_config_dict, reconstr_config, adapter_name


def load_loraxs_mnli_checkpoint(lora_model, lora_rank, fail_on_missing: bool = True):
    CHECKPOINTS_ROOT = "/shared/results/z1192950/Laplace/mnli/roberta-large/loraxs"
    checkpoint_dir = os.path.join(CHECKPOINTS_ROOT, f"rank_{lora_rank}")
    print(f"Loading MNLI checkpoint from {checkpoint_dir}.")
    if not os.path.exists(checkpoint_dir):
        if fail_on_missing:
            raise ValueError(f"Checkpoint directory not found at {checkpoint_dir}.")
        else:
            print(f"Checkpoint directory not found at {checkpoint_dir}.")
            return
    load_loraxs_weights(lora_model, checkpoint_dir, load_classifier=False)