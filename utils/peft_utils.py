import os
from data import TASK_TYPE_DICT
from peft import PeftModel
from peft import (
    get_peft_model,
)
from transformers import (
    AutoModelForMultipleChoice,
    AutoModelForSequenceClassification,
    LlamaConfig,
    LlamaForSequenceClassification,
    LlamaForCausalLM,
)
from peft.tuners.lora import LoraConfig
import torch
from types import SimpleNamespace


def get_id_list(task_name: str, tokenizer):
    task_name = task_name.lower()
    if task_name == 'boolq':
        return [tokenizer.encode('False')[1], tokenizer.encode('True')[1]]
    elif task_name in ('openbookqa', 'obqa'):
        return [tokenizer.encode('A')[1], tokenizer.encode('B')[1], tokenizer.encode('C')[1], tokenizer.encode('D')[1]]
    elif 'arc' in task_name or task_name in ('mmlu', 'arc-e', 'arc-c'):
        return [tokenizer.encode('A')[1], tokenizer.encode('B')[1], tokenizer.encode('C')[1], tokenizer.encode('D')[1]]
    elif 'winogrande' in task_name:
        return [tokenizer.encode('A')[1], tokenizer.encode('B')[1]]
    elif task_name == 'cqa':
        return [tokenizer.encode('A')[1], tokenizer.encode('B')[1], tokenizer.encode('C')[1], tokenizer.encode('D')[1], tokenizer.encode('E')[1]]
    else:
        # Default to four-way choices if an MCQA task alias is missed, avoiding full-vocab logits
        return [tokenizer.encode('A')[1], tokenizer.encode('B')[1], tokenizer.encode('C')[1], tokenizer.encode('D')[1]]


class WrappedModel(torch.nn.Module):
    """
    WrappedModel class and get_id_list are used to ensure that we can use a generative model 
    in classification MCQA tasks during evaluation, where predefined classes are expected. 
    The same approach is used in Laplace Lora paper and SWAG Lora papers.
    """
    def __init__(self, model, task_name: str, tokenizer, verbose: bool = True):
        super().__init__()
        self.id_list = get_id_list(task_name, tokenizer)
        
        self.model = model

        if verbose:
            print(self.model)

    def get_peft_model(self) -> PeftModel:
        return self.model

    def forward(self, **kwargs):
        kwargs.pop('labels', None)
        output_dict = self.model(**kwargs)
        logits = output_dict['logits']
        selected_logits = logits[:, -1, self.id_list]

        # Return an object with a `.logits` attribute and keep a dummy
        # sequence dimension so existing indexing `[:, -1, :]` works.
        wrapped_output = SimpleNamespace(
            logits=selected_logits.to(torch.float32).unsqueeze(1)
        )
        return wrapped_output

    def __getattr__(self, name):
        # Delegate attribute access to the underlying PEFT model when
        # the attribute is not found on the wrapper.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    # Explicit pass-throughs for common HF/PEFT APIs used elsewhere
    def save_pretrained(self, *args, **kwargs):
        return self.model.save_pretrained(*args, **kwargs)

    def load_adapter(self, *args, **kwargs):
        return self.model.load_adapter(*args, **kwargs)



def create_transformer(config, num_classes=2, cache_dir=None):
    task_type = TASK_TYPE_DICT[config.experiment.task]
    model_name_or_path = (
        os.path.join(config.experiment.model_path, config.model.model_name)
        if config.experiment.offline
        else config.model.model_name
    )
    if task_type == "SEQ_CLS":
        if "meta-llama" in config.model.model_name:
            configuration = LlamaConfig()
            model = LlamaForSequenceClassification(configuration).from_pretrained(
                model_name_or_path,
                return_dict=True,
                num_labels=num_classes,
                cache_dir=cache_dir,
            )
            model.config.pad_token_id = model.config.eos_token_id
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path, return_dict=True, num_labels=num_classes
            )

    elif task_type == "MCQA":
        if "meta-llama" in config.model.model_name:
            configuration = LlamaConfig()
            model = LlamaForCausalLM(configuration).from_pretrained(
                model_name_or_path, return_dict=True, cache_dir=cache_dir
            )
            model.config.pad_token_id = model.config.eos_token_id
        else:
            model = AutoModelForMultipleChoice.from_pretrained(
                model_name_or_path, return_dict=True
            )
    else:
        raise Exception("Only SEQ_CLS and MCQA task types implemented.")
    return model


def create_pretrained_model(
    config, verbose=False, num_classes=2, tokenizer=None
):
    model = create_transformer(config, num_classes)
    return model


def create_peft_model(
    config, peft_config, verbose=False, num_classes=2, tokenizer=None
):
    model = create_transformer(config, num_classes)
    model = get_peft_model(model, peft_config)
    return model


def get_peft_config(config, accelerator=None):
    task_type = TASK_TYPE_DICT[config.experiment.task]

    if "meta-llama" in config.model.model_name and task_type == "MCQA":
        task_type = "CAUSAL_LM"
    else:
        task_type = "SEQ_CLS"

    print_fn = accelerator.print if accelerator else print
    print_fn(
        f"Creating LoRA config. Lora-target modules {config.model.target_modules}")
    peft_config = LoraConfig(
        task_type=task_type,
        inference_mode=False,
        r=config.experiment.lora_r,
        lora_alpha=config.experiment.lora_alpha,
        lora_dropout=config.experiment.lora_dropout,
        bias=config.experiment.train_biases,
        target_modules=list(config.model.target_modules),
        modules_to_save=list(config.model.modules_to_save),
    )
    return peft_config


