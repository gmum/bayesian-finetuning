import os
from safetensors import safe_open
from peft import PeftModel

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


def load_loraxs_weights(model: PeftModel, checkpoint_dir:str, load_classifier: bool, verbose: bool = False) -> None:
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
        if verbose:
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
        if verbose:
            print("Not loading classifier weights.")
        renamed_state_dict = {k: v for k, v in renamed_state_dict.items() if "classifier" not in k}
    else:
        # For Roberta-Large, we fine-tune all classifier weights the same way as in Lora Laplace paper
        # change classifier. name to classifier.modules_to_save.default
        if verbose:
            print("Loading classifier weights. Renaming classifier weights to classifier.modules_to_save.default.")
        if any("classifier" in k for k in renamed_state_dict.keys()):
            renamed_state_dict = {
                k.replace("classifier", "classifier.modules_to_save.default"): v
                for k, v in renamed_state_dict.items()
            }
        elif any("lm_head" in k for k in renamed_state_dict.keys()):
            # In Llama-2-7b, we either don't fine-tune lm_head or we apply LoRA-XS adapters to it.
            # Only rename non-LoRA lm_head parameters (i.e., modules_to_save)
            # LoRA parameters (lora_A, lora_B, lora_latent) are already correctly named
            # renamed_state_dict = {
            #     k.replace("lm_head", "lm_head.modules_to_save.default") if not any(x in k for x in ["lora_A", "lora_B", "lora_latent"]) else k: v
            #     for k, v in renamed_state_dict.items()
            # }
            pass
        else:
            print(f"No classifier or lm_head found in the checkpoint. Keys in renamed_state_dict: {renamed_state_dict.keys()}")
    # print what keys differ between the model and the checkpoint
    model_state_dict = model.state_dict()
    if verbose:
        print(f"Keys in checkpoint, but not in model: {set(renamed_state_dict.keys()) - set(model_state_dict.keys())}")
        print(f"Classifier keys in model: {[k for k in model_state_dict.keys() if 'classifier' in k]}")
        print(f"Lm head keys in model: {[k for k in model_state_dict.keys() if 'lm_head' in k]}")

    model.load_state_dict(renamed_state_dict, strict=False)




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

