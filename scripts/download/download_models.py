"""Download Hugging Face model weights to a local cache directory.

Originally tailored to a specific HPC scratch path; the destination is now
configurable via CLI flags or the ``BAY_LORAXS_MODELS_DIR`` environment
variable so the script is reusable on any machine.

Example::

    python scripts/download/download_models.py \\
        --model meta-llama/Llama-2-7b-chat-hf \\
        --save-dir /path/to/models
"""

import argparse
import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMultipleChoice,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model and tokenizer to a local directory."
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help=(
            "Hugging Face model id to download (e.g. roberta-base, "
            "meta-llama/Llama-2-7b-chat-hf). Repeat to download multiple models."
        ),
    )
    parser.add_argument(
        "--save-dir",
        default=os.environ.get("BAY_LORAXS_MODELS_DIR"),
        help=(
            "Destination directory; each model is saved under <save-dir>/<model_id>. "
            "Defaults to the BAY_LORAXS_MODELS_DIR environment variable when set."
        ),
    )
    args = parser.parse_args()

    if not args.save_dir:
        parser.error(
            "--save-dir is required (or set the BAY_LORAXS_MODELS_DIR environment variable)."
        )

    for model_name in args.model:
        download_model(model_name, args.save_dir)


def download_model(model_name: str, save_directory: str) -> None:
    os.makedirs(save_directory, exist_ok=True)

    cache_dir = os.path.join(save_directory, model_name)
    if model_name in ["roberta-base", "roberta-large"]:
        AutoModelForSequenceClassification.from_pretrained(model_name).save_pretrained(
            cache_dir
        )
        AutoModelForMultipleChoice.from_pretrained(model_name).save_pretrained(
            cache_dir
        )
        AutoTokenizer.from_pretrained(model_name).save_pretrained(cache_dir)

    elif model_name in (
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-7b-chat-hf",
    ):
        dtype = torch.bfloat16
        AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        ).save_pretrained(cache_dir)
        AutoTokenizer.from_pretrained(model_name).save_pretrained(cache_dir)

    else:
        AutoModelForCausalLM.from_pretrained(model_name).save_pretrained(cache_dir)
        AutoTokenizer.from_pretrained(model_name).save_pretrained(cache_dir)

    print(f"Model {model_name} downloaded and saved to {cache_dir}")


if __name__ == "__main__":
    main()
