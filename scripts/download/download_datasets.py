"""Download datasets (GLUE/SuperGLUE and commonsense MCQA) to a local cache.

Originally tailored to a specific HPC scratch path; the destination is now
configurable via the ``--save-dir`` CLI flag or the
``BAY_LORAXS_DATA_DIR`` environment variable so the script is reusable on any
machine.

Example::

    python scripts/download/download_datasets.py \\
        --save-dir /path/to/data \\
        --group glue_superglue mcqa
"""

import argparse
import os

from datasets import load_dataset

GLUE_AND_SUPERGLUE_TASKS = [
    "cola",
    "mnli",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
    "stsb",
    "wnli",
    "boolq",
    "wic",
]

MMLU_SUBTASKS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download datasets used in the bay_loraxs experiments."
    )
    parser.add_argument(
        "--save-dir",
        default=os.environ.get("BAY_LORAXS_DATA_DIR"),
        help=(
            "Destination directory for the downloaded datasets. Defaults to the "
            "BAY_LORAXS_DATA_DIR environment variable when set."
        ),
    )
    parser.add_argument(
        "--group",
        nargs="+",
        choices=["glue_superglue", "mcqa", "all"],
        default=["all"],
        help="Which dataset groups to download (default: all).",
    )
    args = parser.parse_args()

    if not args.save_dir:
        parser.error(
            "--save-dir is required (or set the BAY_LORAXS_DATA_DIR environment variable)."
        )

    groups = set(args.group)
    if "all" in groups:
        groups = {"glue_superglue", "mcqa"}

    if "glue_superglue" in groups:
        download_glue_superglue(args.save_dir)
    if "mcqa" in groups:
        download_mcqa(args.save_dir)


def download_glue_superglue(save_dir: str) -> None:
    """Download GLUE/SuperGLUE tasks and save them under ``save_dir``."""
    os.makedirs(save_dir, exist_ok=True)

    for task in GLUE_AND_SUPERGLUE_TASKS:
        try:
            dataset_name = "super_glue" if task in ["boolq", "wic"] else "glue"
            print(f"Downloading dataset for task '{task}' from '{dataset_name}'...")
            dataset = load_dataset(dataset_name, task)
            task_save_path = os.path.join(save_dir, f"data_{task}")
            dataset.save_to_disk(task_save_path)
        except Exception as exc:
            print(f"Error in downloading task '{task}': {exc}")


def download_mcqa(save_dir: str) -> None:
    """Download commonsense MCQA datasets used in the paper."""
    os.makedirs(save_dir, exist_ok=True)

    _save(load_dataset("swag", "regular"), save_dir, "data_swag")
    _save(load_dataset("commonsense_qa"), save_dir, "data_cqa")
    _save(load_dataset("openbookqa"), save_dir, "data_obqa")
    _save(load_dataset("allenai/ai2_arc", "ARC-Challenge"), save_dir, "data_arc-c")
    _save(load_dataset("allenai/ai2_arc", "ARC-Easy"), save_dir, "data_arc-e")

    for subtask in MMLU_SUBTASKS:
        _save(load_dataset("cais/mmlu", subtask), save_dir, f"data_mmlu_{subtask}")


def _save(dataset, save_dir: str, name: str) -> None:
    dataset.save_to_disk(os.path.join(save_dir, name))


if __name__ == "__main__":
    main()
