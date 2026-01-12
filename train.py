import os
import time
from data import TASK_TYPE_DICT
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import tabulate
import pandas as pd
import wandb
from utils.peft_utils import load_peft_model
import json
import accelerate
import torch.nn.functional as F
from utils.eval_utils import evaluate_task, compute_metrics, ood_metrics_entropy, compute_nll, compute_ece
import shutil

from utils.peft_utils import get_id_list

from laplace_utils import evaluate_laplace_params, get_laplace_params, checkpoints_to_fit, LaplaceWeights, tensor_metrics_to_float
from laplace.utils.enums import Likelihood, SubsetOfWeights, HessianStructure, PredType, LinkApprox, TuningMethod, PriorStructure

from transformers import (
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)


def log_metrics(
    eval_metrics,
    log_prefix,
    *,
    epoch,
    lr_scheduler,
    train_loss,
    train_acc,
    time_diff,
    columns,
    logged_values,
    accelerator,
    counter,
):
    eval_loss = eval_metrics.get("loss", None)
    eval_acc = eval_metrics.get("acc", None)

    nll_value = (
        eval_metrics.get("nll", None).item()
        if isinstance(eval_metrics.get("nll", None), torch.Tensor)
        else eval_metrics.get("nll", None)
    )

    brier_value = (
        eval_metrics.get("brier", None).item()
        if isinstance(eval_metrics.get("brier", None), torch.Tensor)
        else eval_metrics.get("brier", None)
    )

    mcc_value = (
        eval_metrics.get("mcc", None).item()
        if isinstance(eval_metrics.get("mcc", None), torch.Tensor)
        else eval_metrics.get("mcc", None)
    )

    values = [
        epoch,
        lr_scheduler.get_last_lr()[0],
        train_loss,
        train_acc,
        eval_loss,
        eval_acc,
        nll_value,
        eval_metrics.get("ece", None),
        brier_value,
        mcc_value,
        time_diff,
    ]

    table = tabulate.tabulate(
        [values], columns, tablefmt="simple", floatfmt="8.4f")
    print(table)
    logged_values.append(values)

    # Log each metric individually for graphing
    metrics_to_log = {f"{log_prefix}{col}": val for col,
                      val in zip(columns, values)}
    counter += 1
    accelerator.log(metrics_to_log, step=counter)  # here


def get_lr_scheduler(optimizer, num_batches, config):
    num_steps = num_batches * config.experiment.num_epochs
    if config.experiment.scheduler == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.experiment.warmup_length * num_steps,
        )
    elif config.experiment.scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.experiment.warmup_length * num_steps,
            num_training_steps=num_steps,
        )
    elif config.experiment.scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.experiment.warmup_length * num_steps,
            num_training_steps=num_steps,
            num_cycles=config.experiment.num_lr_cycles,
        )
    elif config.experiment.scheduler == "cosine_hard_restarts":
        if not float(config.experiment.num_lr_cycles).is_integer():
            raise Exception(
                "num_lr_cycles must be an integer for hard restarts.")
        lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.experiment.warmup_length * num_steps,
            num_training_steps=num_steps,
            num_cycles=config.experiment.num_lr_cycles,
        )
    else:
        raise Exception("Unimplemented learning rate scheduler given.")

    return lr_scheduler


def save_best_metric_checkpoint(
    model,
    metrics_to_save,
    all_metrics,
    epoch,
    save_path,
    best_checkpoint_prefix,
    wandb_run=None,
):
    """
    Save checkpoint if current metric is the best so far, and remove old checkpoint.
    
    Args:
        model: The model to save
        metrics_to_save: Dict tracking metrics with structure:
            {metric_name: {"best_metric_val": float/None, 
                          "saved_epoch": int/None, 
                          "is_greater_better": bool}}
        all_metrics: Dict of all current metrics (evaluation and training). 
            Should contain keys matching metric_name in metrics_to_save.
        epoch: Current epoch number
        save_path: Base directory to save checkpoints
        best_checkpoint_prefix: Prefix for best checkpoint directory names
    
    Returns:
        Updated metrics_to_save dict
    """
    def get_checkpoint_name(metric_name, metric_step):
        return f'{best_checkpoint_prefix}{metric_name}-{metric_step}'
    
    for metric_name, metric_info in metrics_to_save.items():
        if metric_name not in all_metrics:
            print(f"Warning: {metric_name} not found in metrics dictionary. Skipping checkpoint saving for this metric.")
            continue
            
        cur_metric_val = all_metrics[metric_name]
        
        # Convert tensor to float if needed
        if isinstance(cur_metric_val, torch.Tensor):
            cur_metric_val = cur_metric_val.item()
        
        # Check if this is the best metric value
        if metric_info["best_metric_val"] is None or \
            (metric_info["is_greater_better"] and cur_metric_val > metric_info["best_metric_val"]) or \
            (not metric_info["is_greater_better"] and cur_metric_val < metric_info["best_metric_val"]):
            

            save_prefix = f"{best_checkpoint_prefix}{metric_name}"
            if wandb_run is not None:
                wandb_run.log({f"{save_prefix}/value": cur_metric_val})
                wandb_run.log({f"{save_prefix}/epoch": epoch})
                
            metrics_to_save[metric_name]["best_metric_val"] = cur_metric_val
            previous_epoch = metrics_to_save[metric_name]["saved_epoch"]
            metrics_to_save[metric_name]["saved_epoch"] = epoch
            
            # Save new checkpoint
            save_path_metric = os.path.join(save_path, get_checkpoint_name(metric_name, epoch))
            model.save_pretrained(save_path_metric)
            print(f"Saved best {metric_name} model checkpoint: {cur_metric_val:.4f} at epoch {epoch}")
            print(f"Checkpoint path: {save_path_metric}")
            
            # Remove old checkpoint if it exists
            if previous_epoch is not None:
                old_metric_save_path = os.path.join(save_path, get_checkpoint_name(metric_name, previous_epoch))
                print(f"Removing old checkpoint {old_metric_save_path}")
                if os.path.exists(old_metric_save_path):
                    if os.path.isdir(old_metric_save_path):
                        shutil.rmtree(old_metric_save_path)
                    else:
                        os.remove(old_metric_save_path)
                    print(f"Old checkpoint removed successfully")
    
    return metrics_to_save


def train_laplace(
    model,
    train_dataloader,
    eval_dataloader,
    test_dataloader,
    config,
    accelerator,
    tokenizer,
    num_classes=2,
    peft_config=None,
    ood_dataloader=None,
    save_checkpoints_at_epochs=[3, 4, 5, 6, 7, 8],
):
    n_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {n_gpus}")

    causal_lm = False
    task_type = TASK_TYPE_DICT[config.experiment.task]
    if "meta-llama" in config.model.model_name and task_type == "MCQA":
        causal_lm = True

    num_epochs = config.experiment.num_epochs
    save_path = config.experiment.save_path
    print("save_path in train_laplace is:", save_path)
    print("Base learning rate: ", config.experiment.learning_rate)

    metric_to_optimize = "eval_mcc" if config.experiment.task == "cola" else "eval_acc"

    best_checkpoint_prefix = f"best_"
    # Initialize metric tracking for best checkpoint saving
    # To save checkpoints based on other metrics (e.g. train_loss), add them here.
    # Ensure the metric key exists in all_metrics passed to save_best_metric_checkpoint.
    # To add your own metric:
    # 1. Add it to metrics_to_save dict below
    # 2. Ensure it is available in all_metrics in the training loop (lines ~450)
    # Example:
    # metrics_to_save["train_loss"] = {
    #     "best_metric_val": None,
    #     "saved_epoch": None,
    #     "is_greater_better": False
    # }
    metrics_to_save = {
        metric_to_optimize: {
            "best_metric_val": None,
            "saved_epoch": None,
            "is_greater_better": True
        },
        # "train_loss": {
        #     "best_metric_val": None,
        #     "saved_epoch": None,
        #     "is_greater_better": False
        # }
    }

    print(f"Metric to optimize: {metric_to_optimize}")
    print(f"Metrics to save: {metrics_to_save}")

    cls_lr = (
        config.experiment.learning_rate
        if config.experiment.cls_learning_rate == 1
        else config.experiment.cls_learning_rate
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    i[1] for i in model.named_parameters() if "classifier" in i[0]
                ],
                "lr": cls_lr,
                "weight_decay": config.experiment.classifier_weight_decay,
            },
            {
                "params": [
                    i[1] for i in model.named_parameters() if "classifier" not in i[0]
                ],
                "lr": config.experiment.learning_rate,
                "weight_decay": config.experiment.lora_weight_decay,
            },
        ]
    )
    # print weight decays of optimizer groups for sanity check
    for i, group in enumerate(optimizer.param_groups):
        print(f"Group {i}, lr: {group['lr']}, wd: {group['weight_decay']}")

    loss_fn = CrossEntropyLoss()

    lr_scheduler = get_lr_scheduler(
        optimizer=optimizer, num_batches=len(train_dataloader), config=config
    )

    # prepare model, data, optimizer, scheduler
    (
        model,
        optimizer,
        lr_scheduler,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
    ) = accelerator.prepare(
        model,
        optimizer,
        lr_scheduler,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
    )
    if ood_dataloader is not None:
        ood_dataloader = accelerator.prepare(ood_dataloader)

    print(lr_scheduler.state_dict())

    # save config.experiment parameters
    wandb.config.update({f"experiment/{k}": v for k, v in config.experiment.items()})
    wandb.config.update({f"model/{k}": v for k, v in config.model.items()})
    
    global_step = 0

    columns = [
        "epc",
        "lr_e",
        "tr_loss",
        "tr_acc",
        "loss",
        "acc",
        "nll",
        "ece",
        "brier",
        "mcc",
        "time",
    ]
    print("")
    print("-------------------------START MAP TRAINING-------------------------")

    if min(save_checkpoints_at_epochs) >= num_epochs:
        raise Exception(f"No checkpoints will be saved, all specified epochs exceed num_epochs: "
                        f"save_checkpoints_at_epochs={save_checkpoints_at_epochs}, num_epochs={num_epochs}!")

    # Drop unused references to free up memory
    import gc
    for i in range(5):
        gc.collect()

    learning_rates = []
    logged_values = []

    counter = 1
    for epoch in range(num_epochs):
        if config.experiment.skip_training:
            print("Skipping training")
            break

        start_time = time.time()

        model.train()
        total_loss = 0.0

        all_preds = torch.tensor([], dtype=torch.long,
                                 device=accelerator.device)
        all_labels = torch.tensor(
            [], dtype=torch.long, device=accelerator.device)
        all_probs_list = []

        correct = torch.tensor(0, dtype=torch.long, device=accelerator.device)
        total = torch.tensor(0, dtype=torch.long, device=accelerator.device)

        for step, batch in enumerate(
            tqdm(train_dataloader, disable=not accelerator.is_main_process)
        ):
            with accelerator.accumulate(model):
                output = model(
                    **{
                        "input_ids": batch["input_ids"],
                        "attention_mask": batch["attention_mask"],
                    }
                )
                # if causal lm we use last representation's logits
                logits = output.logits[:, -1,
                                       :] if causal_lm else output.logits

                loss = loss_fn(logits, batch["labels"])
                total_loss += loss.detach()
                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()

                preds = logits.argmax(axis=-1)
                all_preds = torch.cat((all_preds, preds), dim=0)
                all_labels = torch.cat((all_labels, batch["labels"]), dim=0)
                
                probs = F.softmax(logits, dim=-1)
                all_probs_list.append(probs)

                correct += preds.eq(batch["labels"].view_as(preds)).sum()
                total += len(batch["labels"])

                lr_scheduler.step()
                learning_rates.append(optimizer.param_groups[0]["lr"])

            if (
                accelerator.is_main_process
                and step % config.experiment.gradient_accumulation_steps == 0
            ):
                counter += 1
                accelerator.log(
                    {
                        "step": counter,
                        "epoch": epoch,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )

        collected_total_loss = accelerator.gather_for_metrics(
            total_loss).sum().item()
        train_loss = collected_total_loss / (
            len(train_dataloader) / config.experiment.gradient_accumulation_steps
        )

        collected_correct, collected_total = accelerator.gather_for_metrics(
            (correct, total)
        )
        collected_correct = collected_correct.sum().item()
        collected_total = collected_total.sum().item()
        train_acc = 0
        train_nll = 0
        train_ece = 0

        all_probs = torch.cat(all_probs_list, dim=0)
        
        collected_all_probs = accelerator.gather_for_metrics(all_probs)
        collected_all_labels = accelerator.gather_for_metrics(all_labels)
        
        if accelerator.is_main_process:
            train_acc = collected_correct / collected_total
            train_nll = compute_nll(collected_all_probs, collected_all_labels)
            train_ece = compute_ece(collected_all_probs, collected_all_labels)
        accelerator.wait_for_everyone()

        eval_metrics, _ = compute_metrics(
            model,
            eval_dataloader,
            "eval",
            "base",
            accelerator=accelerator,
            causal_lm=causal_lm,
        )
        
        eval_metrics = tensor_metrics_to_float(eval_metrics)

        # Report training metrics for epoch
        counter += 1
        
        global_step += 1

        accelerator.print(
            "eval_metrics",
            {k: v for (k, v) in eval_metrics.items() if "entrop" not in k},
        )

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            
            # Combine metrics for checkpoint saving
            # To add your own metric, add it to this dictionary
            all_metrics = {f"eval_{k}": v for k, v in eval_metrics.items()}
            all_metrics.update({
                "train_acc": train_acc,
                "train_loss": train_loss,
                "train_nll": train_nll,
                "train_ece": train_ece,
                "epoch": epoch,
            })

            wandb.log({f"all_metrics/{k}": v for k, v in all_metrics.items()})
            
            # Save best metric checkpoint (accuracy)
            metrics_to_save = save_best_metric_checkpoint(
                unwrapped_model,
                metrics_to_save,
                all_metrics,
                epoch,
                save_path,
                best_checkpoint_prefix=best_checkpoint_prefix,
                wandb_run=wandb.run
            )
            
            if config.method.do_laplace:
                if epoch in save_checkpoints_at_epochs:
                    checkpoint_path = os.path.join(save_path, f"checkpoint-{epoch}")
                    unwrapped_model.save_pretrained(checkpoint_path)
                    print(f"saved base model to path {checkpoint_path} at epoch {epoch}")

        accelerator.wait_for_everyone()
        time_diff = time.time() - start_time

        # Update values list
        if accelerator.is_main_process:
            log_metrics(
                eval_metrics,
                "val/",
                epoch=epoch,
                lr_scheduler=lr_scheduler,
                train_loss=train_loss,
                train_acc=train_acc,
                time_diff=time_diff,
                columns=columns,
                logged_values=logged_values,
                accelerator=accelerator,
                counter=counter,
            )

        accelerator.wait_for_everyone()

    print("-------------------------END OF MAP TRAINING-------------------------")

    if accelerator.is_main_process:
        counter += 1
        accelerator.log(
            {"Epoch Summary": wandb.Table(
                data=logged_values, columns=columns)},
            step=counter,
        )  # here

        train_log = pd.DataFrame(logged_values, columns=columns)
        train_log = train_log.round(2)

        print("Training complete.")

        csv_file = os.path.join(save_path, "training_log.csv")
        train_log.to_csv(csv_file, index=False)
        print(f"Training metrics saved to {csv_file}")
        lr_file = os.path.join(save_path, "learning_rates.npy")
        np.save(lr_file, learning_rates)
        print(f"Learning rates saved to {lr_file}")
        
    
    # Post-hoc Laplace approximation on chosen checkpoints
    if config.method.do_laplace:
        print("-------------------------POST-HOC LAPLACE HESSIANS-------------------------")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(device=accelerator.device) / 1024**3:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(device=accelerator.device) / 1024**3:.2f} GB")
        print("Post-hoc Laplace approximation on chosen checkpoints")
        # Example: construct laplace_params_list as needed
        # Layerwise prior structure for LORAXS kron
        laplace_params_list = []
        
        # link_approx_list = [LinkApprox.MC, LinkApprox.PROBIT]
        link_approx_list = [LinkApprox.MC]
        

        hessian_structure_list = [HessianStructure.KRON, HessianStructure.DIAG]
        prediction_kwargs_list = [
            (PredType.GLM, {"n_samples": 100000, "joint": True}),
        ]
        
        prior_params_list = [
            (PriorStructure.SCALAR, {"n_steps": 2000, "lr": 0.1}),
        ]
        
        print("Constructing Laplace parameters list...")
        for laplace_weights in [LaplaceWeights.LORAXS]:
            for hessian_structure in hessian_structure_list:
                for link_approx in link_approx_list:
                    for pred_type, pred_kwargs in prediction_kwargs_list:
                        for prior_structure, prior_kwargs in prior_params_list:
                                laplace_params_list.append(get_laplace_params(laplace_weights=laplace_weights, 
                                                                            link_approx=link_approx, 
                                                                            prior_structure=prior_structure,
                                                                            prior_kwargs=prior_kwargs,
                                                                            pred_type=pred_type,
                                                                            prediction_kwargs=pred_kwargs,
                                                                            hessian_structure=hessian_structure))
        
        print(f"Constructed {len(laplace_params_list)} Laplace parameter configurations = {laplace_params_list}")
        print(f"Determining checkpoints to fit Laplace approximation: save_path={save_path}")
        checkpoints_list = checkpoints_to_fit(output_dir=save_path, 
                                              use_best_checkpoints=True,
                                              use_step_checkpoints=True,
                                              best_checkpoint_prefix=best_checkpoint_prefix) # "eval_comb_score", "eval_nll", "eval_acc"

        print("-------------------------LAPLACE EVALUATION-------------------------")
        print(f"Evaluating Laplace parameters on checkpoints: {checkpoints_list} from save_path={save_path}")

        total_laplace_metrics = {}
        for checkpoint in checkpoints_list:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            

            unwrapped_model = accelerator.unwrap_model(model)   
            checkpoint_full_path = os.path.join(save_path, checkpoint)

            prefix = checkpoint
            if best_checkpoint_prefix in checkpoint:
                # Remove step from checkpoint name so that it gets saved to wandb correctly
                # Use rsplit to only remove the last part (epoch number) after the final "-"
                prefix = checkpoint.rsplit("-", 1)[0]
                epoch = int(checkpoint.rsplit("-", 1)[1])

            print(f"Evaluating Laplace parameters on checkpoint: {checkpoint} at epoch {epoch}")

            total_laplace_metrics.update(evaluate_laplace_params(
                unwrapped_model,
                laplace_params_list,
                prefix=prefix,
                checkpoint_full_path=checkpoint_full_path,
                device=accelerator.device,
                num_labels=num_classes,
                config=config,
                train_loader=train_dataloader,
                val_loader=eval_dataloader,
                test_loader=test_dataloader,
                json_metrics_path=save_path,
                accelerator=accelerator,
                causal_lm=causal_lm,
                wandb_run=wandb.run,
                epoch=epoch
            ))
            
        print(f"Total laplace metrics: {total_laplace_metrics}")
        
        laplace_metrics_df = pd.DataFrame.from_dict(total_laplace_metrics, orient="index").reset_index()
        laplace_metrics_df.to_csv(os.path.join(save_path, "new_laplace_metrics.csv"), index=False)
        wandb.log({"laplace/metrics_df": wandb.Table(dataframe=laplace_metrics_df)})
        print("Successfully finished.")
    
    return None
