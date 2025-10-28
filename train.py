import copy
import os
import time
import shutil
from data import TASK_TYPE_DICT
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim.swa_utils import SWALR
from tqdm import tqdm
import tabulate
import pandas as pd
import wandb
from utils.peft_utils import load_peft_model
import json
import accelerate
from utils.eval_utils import evaluate_task, compute_metrics, ood_metrics_entropy

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
    swag_eval_metrics,
    log_prefix,
    *,
    epoch,
    lr_scheduler,
    swag_scheduler,
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

    swag_eval_acc = swag_eval_metrics.get("acc", None)
    swag_eval_loss = swag_eval_metrics.get("loss", None)

    nll_value = (
        eval_metrics.get("nll", None).item()
        if isinstance(eval_metrics.get("nll", None), torch.Tensor)
        else eval_metrics.get("nll", None)
    )
    swag_nll_value = (
        swag_eval_metrics.get("nll", None).item()
        if isinstance(swag_eval_metrics.get("nll", None), torch.Tensor)
        else swag_eval_metrics.get("nll", None)
    )

    brier_value = (
        eval_metrics.get("brier", None).item()
        if isinstance(eval_metrics.get("brier", None), torch.Tensor)
        else eval_metrics.get("brier", None)
    )
    swag_brier_value = (
        swag_eval_metrics.get("brier", None).item()
        if isinstance(swag_eval_metrics.get("brier", None), torch.Tensor)
        else swag_eval_metrics.get("brier", None)
    )

    mcc_value = (
        eval_metrics.get("mcc", None).item()
        if isinstance(eval_metrics.get("mcc", None), torch.Tensor)
        else eval_metrics.get("mcc", None)
    )
    swag_mcc_value = (
        swag_eval_metrics.get("mcc", None).item()
        if isinstance(swag_eval_metrics.get("mcc", None), torch.Tensor)
        else swag_eval_metrics.get("mcc", None)
    )

    values = [
        epoch,
        lr_scheduler.get_last_lr()[0],
        swag_scheduler.get_last_lr()[0],
        train_loss,
        train_acc,
        eval_loss,
        eval_acc,
        swag_eval_loss,
        swag_eval_acc,
        nll_value,
        swag_nll_value,
        eval_metrics.get("ece", None),
        swag_eval_metrics.get("ece", None),
        brier_value,
        swag_brier_value,
        mcc_value,
        swag_mcc_value,
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

    if config.method.method_name == "swag":
        swag_scheduler = SWALR(
            optimizer,
            anneal_strategy=config.method.swag_anneal_strategy,
            anneal_epochs=config.method.swag_anneal_epochs * num_batches,
            swa_lr=config.method.swag_learning_rate,
        )
        return lr_scheduler, swag_scheduler
    return lr_scheduler


def train_swag(
    model,
    swag_model,
    train_dataloader,
    eval_dataloader,
    test_dataloader,
    config,
    accelerator,
    tokenizer,
    num_classes=2,
    peft_config=None,
    ood_dataloader=None,
):
    n_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {n_gpus}")

    causal_lm = False
    task_type = TASK_TYPE_DICT[config.experiment.task]
    if "meta-llama" in config.model.model_name and task_type == "MCQA":
        causal_lm = True

    swag_start = config.method.swag_start
    num_epochs = config.experiment.num_epochs
    save_path = config.experiment.save_path
    print("save_path in train_swag is:", save_path)
    print("Base learning rate: ", config.experiment.learning_rate)

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

    lr_scheduler, swag_scheduler = get_lr_scheduler(
        optimizer=optimizer, num_batches=len(train_dataloader), config=config
    )

    # prepare model, data, optimizer, scheduler
    (
        swag_model,
        optimizer,
        lr_scheduler,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
    ) = accelerator.prepare(
        swag_model,
        optimizer,
        lr_scheduler,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
    )
    if ood_dataloader is not None:
        ood_dataloader = accelerator.prepare(ood_dataloader)

    print(lr_scheduler.state_dict())
    print(swag_scheduler.state_dict())

    best_base_metric = 0
    best_swag_metric = 0
    base_save_info = {}
    swag_save_info = {}
    swag_eval_metrics = {}
    # metric_to_optimize = "mcc" if config.experiment.task == "cola" else "acc"
    metric_to_optimize = "acc"
    
    metrics_to_save = {
        "acc": {
            "is_greater_better": True,
            "best_metric_val": None,
            "saved_epoch": None
        },
        "comb_score": {
            "is_greater_better": False,
            "best_metric_val": None,
            "saved_epoch": None
        },
        "nll": {
            "is_greater_better": False,
            "best_metric_val": None,
            "saved_epoch": None
        }
    }
    
    # save config.experiment parameters
    wandb.config.update({f"experiment/{k}": v for k, v in config.experiment.items()})
    
    global_step = 0

    columns = [
        "epc",
        "lr_e",
        "swag_lr_e",
        "tr_loss",
        "tr_acc",
        "loss",
        "acc",
        "swag_loss",
        "swag_acc",
        "nll",
        "swag_nll",
        "ece",
        "swag_ece",
        "brier",
        "swag_brier",
        "mcc",
        "swag_mcc",
        "time",
    ]
    print("")
    print("-------------------------START TRAINING-------------------------")

    if swag_start < num_epochs:
        print(
            f"SWAG start epoch {swag_start} is <= num_epochs {num_epochs}, SWAG collection will occur!"
        )

    import gc
    for i in range(5):
        gc.collect()

    learning_rates = []
    logged_values = []

    counter = 2
    for epoch in range(num_epochs):
        if config.experiment.skip_training:
            print("Skipping training")
            break
        # load base model with the lowest loss when starting SWA(G)
        if epoch == swag_start:
            print("Beginning SWAG collection")
            if config.method.swag_start_with_lowest_loss:
                print("Loading model from training with lowest validation loss")
                model_load = load_peft_model(
                    os.path.join(save_path, "base_model"),
                    config.experiment.task,
                    is_trainable=True,
                    tokenizer=tokenizer,
                )

                unwrapped_model = accelerator.unwrap_model(swag_model)
                unwrapped_model.base.load_state_dict(
                    model_load.state_dict(), strict=False
                )

        start_time = time.time()
        swag_eval_metric = 0

        model.train()
        swag_model.train()
        total_loss = 0.0

        all_preds = torch.tensor([], dtype=torch.long,
                                 device=accelerator.device)
        all_labels = torch.tensor(
            [], dtype=torch.long, device=accelerator.device)

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

                correct += preds.eq(batch["labels"].view_as(preds)).sum()
                total += len(batch["labels"])

                if epoch < swag_start:
                    lr_scheduler.step()
                else:
                    swag_scheduler.step()
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
        if accelerator.is_main_process:
            train_acc = collected_correct / collected_total
        accelerator.wait_for_everyone()

        if (
            epoch >= swag_start
            and (epoch - swag_start) % config.method.swag_c_epochs == 0
        ):
            # collect model to SWAG
            swag_model.collect_model(model)

            # eval swag on validation
            model_cpu = swag_model.to("cpu")
            swag_state = copy.deepcopy(model_cpu.state_dict())
            swag_model.to(accelerator.device)

            # utils.bn_update (not needed unless we have batch norm)
            swag_eval_metrics, _ = compute_metrics(
                swag_model,
                eval_dataloader,
                "eval",
                "swag",
                swag_samples=config.method.swag_samples,
                swag_sample_scale=config.method.swag_sample_scale,
                swag_cov_mat=config.method.swag_cov_mat,
                accelerator=accelerator,
                causal_lm=causal_lm,
            )

            swag_eval_metric = swag_eval_metrics[metric_to_optimize]
            swag_model.load_state_dict(swag_state, strict=True)
            del swag_state

        eval_metrics, _ = compute_metrics(
            model,
            eval_dataloader,
            "eval",
            "base",
            accelerator=accelerator,
            causal_lm=causal_lm,
        )
        
        wandb_log_metrics = tensor_metrics_to_float(eval_metrics)
        wandb_log_metrics = {f"eval_{k}": v for k, v in wandb_log_metrics.items()}
        wandb.log(wandb_log_metrics)
        

        counter += 1
        
        global_step += 1

        accelerator.print(
            "eval_metrics",
            {k: v for (k, v) in eval_metrics.items() if "entrop" not in k},
        )
        eval_metric = eval_metrics[metric_to_optimize]

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            # save LoRA model if it improves upon the best validation loss thus far
            if config.method.swag_save_base_model or config.experiment.do_laplace:
                if epoch in [3, 4, 5, 6, 7]:
                    checkpoint_path = os.path.join(save_path, f"checkpoint-{epoch}")
                    model.save_pretrained(checkpoint_path)
                    print(f"saved base model to path {checkpoint_path}")
                # def get_checkpoint_name(metric_name, metric_step):
                #     return f'eval_{metric_name}-{metric_step}'
                
                # for metric_name, metric_info in metrics_to_save.items():
                #     cur_metric_val = eval_metrics[metric_name]
                #     if metric_info["best_metric_val"] is None or \
                #         (metric_info["is_greater_better"] and cur_metric_val > metric_info["best_metric_val"]) or \
                #         (not metric_info["is_greater_better"] and cur_metric_val < metric_info["best_metric_val"]):
                #         metrics_to_save[metric_name]["best_metric_val"] = cur_metric_val
                #         previous_epoch = metrics_to_save[metric_name]["saved_epoch"]
                #         metrics_to_save[metric_name]["saved_epoch"] = epoch
                        
                #         save_path_metric = os.path.join(save_path, get_checkpoint_name(metric_name, epoch))
                #         model.save_pretrained(save_path_metric)
                #         print(f"saved base model to path {save_path_metric}")
                #         print(f"Saving (highest metric value) base model checkpoint {metric_name}: {cur_metric_val} at epoch {epoch}")
                #         if previous_epoch is not None:
                #             old_metric_save_path = os.path.join(save_path, get_checkpoint_name(metric_name, previous_epoch))
                #             print(f"Removing old checkpoint {old_metric_save_path}")
                #             if os.path.exists(old_metric_save_path):
                #                 print(f"Old checkpoint {old_metric_save_path} exists, removing")
                #                 if os.path.isdir(old_metric_save_path):
                #                     shutil.rmtree(old_metric_save_path)
                #                 else:
                #                     os.remove(old_metric_save_path)
                        
            # # save last model
            # model.save_pretrained(os.path.join(save_path, "last_model"))
            # print(f"saved last model to path {os.path.join(save_path, 'last_model')}")

            # save swag model
            if swag_eval_metric > best_swag_metric:
                print(
                    "Saving (highest metric value) SWAG model checkpoint ",
                    swag_eval_metric,
                    " ",
                    epoch,
                )
                best_swag_metric = swag_eval_metric
                swag_save_info["saved_epoch"] = epoch
                unwrapped_swag_model = accelerator.unwrap_model(swag_model)
                torch.save(
                    unwrapped_swag_model.state_dict(),
                    os.path.join(save_path, "swag_model.pt"),
                )
                print(
                    "saved swag model to path ",
                    os.path.join(save_path, "swag_model.pt"),
                )

        accelerator.wait_for_everyone()
        time_diff = time.time() - start_time

        # Update values list
        if accelerator.is_main_process:
            log_metrics(
                eval_metrics,
                swag_eval_metrics,
                "val/",
                epoch=epoch,
                lr_scheduler=lr_scheduler,
                swag_scheduler=swag_scheduler,
                train_loss=train_loss,
                train_acc=train_acc,
                time_diff=time_diff,
                columns=columns,
                logged_values=logged_values,
                accelerator=accelerator,
                counter=counter,
            )

            log_metrics(
                eval_metrics,
                swag_eval_metrics,
                "test/",
                epoch=epoch,
                lr_scheduler=lr_scheduler,
                swag_scheduler=swag_scheduler,
                train_loss=train_loss,
                train_acc=train_acc,
                time_diff=time_diff,
                columns=columns,
                logged_values=[],
                accelerator=accelerator,
                counter=counter,
            )
        accelerator.wait_for_everyone()

    print("-------------------------END OF TRAINING-------------------------")

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
    if config.experiment.do_laplace:
        print("-------------------------POST-HOC LAPLACE-------------------------")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(device=accelerator.device) / 1024**3:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(device=accelerator.device) / 1024**3:.2f} GB")
        print("Post-hoc Laplace approximation on chosen checkpoints")
        # Example: construct laplace_params_list as needed
        # Layerwise prior structure for LORAXS kron
        laplace_params_list = []
        
        # prior_params_list = [
        #     (PriorStructure.SCALAR, {"n_steps": 1000, "lr": 0.1}),
        #     (PriorStructure.LAYERWISE, {"n_steps": 2000,"lr": 0.005}),
        #     (PriorStructure.LAYERWISE, {"n_steps": 2000,"lr": 0.01}),
        #     (PriorStructure.LAYERWISE, {"n_steps": 2000,"lr": 0.007}),
        #     (PriorStructure.LAYERWISE, {"n_steps": 2000,"lr": 0.02})
        # ]
        
        # link_approx_list = [LinkApprox.MC, LinkApprox.PROBIT]
        link_approx_list = [LinkApprox.MC]
        
        prediction_kwargs_list = [
            (PredType.GLM, {"n_samples": 100000, "joint": True}),
            # (PredType.NN, {"n_samples": 100, "joint": True}),
        ]
        
        prior_params_list = [
            # (PriorStructure.SCALAR, {"n_steps": 1000, "lr": 0.1}),
            (PriorStructure.SCALAR, {"n_steps": 2000, "lr": 0.1}),
            # (PriorStructure.SCALAR, {"n_steps": 4000, "lr": 0.2}),
            # (PriorStructure.SCALAR, {"n_steps": 8000, "lr": 0.1}),
            # (PriorStructure.LAYERWISE, {"n_steps": 2000, "lr": 0.2}),
            # (PriorStructure.LAYERWISE, {"n_steps": 8000, "lr": 0.05}),
            #  (PriorStructure.LAYERWISE, {"n_steps": 8000, "lr": 0.2}),
        ]
        
        print("Constructing Laplace parameters list...")
        for laplace_weights in [LaplaceWeights.LORAXS]:
            for link_approx in link_approx_list:
                for pred_type, pred_kwargs in prediction_kwargs_list:
                    for prior_structure, prior_kwargs in prior_params_list:
                            laplace_params_list.append(get_laplace_params(laplace_weights=laplace_weights, 
                                                                        link_approx=link_approx, 
                                                                        prior_structure=prior_structure,
                                                                        prior_kwargs=prior_kwargs,
                                                                        pred_type=pred_type,
                                                                        prediction_kwargs=pred_kwargs))
        
        print(f"Constructed {len(laplace_params_list)} Laplace parameter configurations.")
        checkpoints_list = checkpoints_to_fit(output_dir=save_path, use_first_last=False, use_metrics=False) # "eval_comb_score", "eval_nll", "eval_acc"

        
        total_laplace_metrics = {}

    print(f"Evaluating Laplace parameters on checkpoints: {checkpoints_list} from save_path={save_path}")

    for checkpoint in checkpoints_list:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        checkpoint_full_path = os.path.join(save_path, checkpoint)
        total_laplace_metrics.update(evaluate_laplace_params(
            model,
            laplace_params_list,
            prefix=checkpoint,
            checkpoint_full_path=checkpoint_full_path,
            device=accelerator.device,
            num_labels=num_classes,
            config=config,
            train_loader=train_dataloader,
            val_loader=eval_dataloader,
            test_loader=None,
            json_metrics_path=save_path,
            accelerator=accelerator,
            causal_lm=causal_lm,
            wandb_run=wandb.run
        ))
        
    print(f"Total laplace metrics: {total_laplace_metrics}")
    
    laplace_metrics_df = pd.DataFrame(columns=["name", "eval_comb_score", "eval_comb_calib_score", "eval_acc", "eval_nll", "eval_ece", "nl_marglik"])
    new_entries_df = pd.DataFrame.from_dict(total_laplace_metrics, orient="index").reset_index()
    laplace_metrics_df = pd.concat([laplace_metrics_df, new_entries_df], ignore_index=False)
    
    laplace_metrics_df.to_csv(os.path.join(save_path, "new_laplace_metrics.csv"))
    wandb.log({"laplace/metrics_df": wandb.Table(dataframe=laplace_metrics_df)})
    print("Successfully finished.")
    

    if config.method.swag_save_base_model:
        base_path = os.path.join(save_path, "base_model")
        if accelerator.is_main_process:
            print(
                f'Saving LoRA base model saving info to {os.path.join(base_path, "save_info.json")}'
            )
            with open(os.path.join(base_path, "base_save_info.json"), "w") as f:
                json.dump(base_save_info, f)

        if config.experiment.eval_upon_save:
            if accelerator.is_main_process:
                accelerator.print(
                    f'Evaluating best loss LoRA model (using model from epoch {base_save_info["saved_epoch"]})'
                )

            unwrapped_model = accelerator.unwrap_model(swag_model)

            # change buffer shape of base model (we know when it was saved & swag_start)
            if accelerator.is_main_process:
                base_save_tensor = torch.tensor(
                    base_save_info["saved_epoch"], device=accelerator.device
                )
            else:
                base_save_tensor = torch.tensor([1], device=accelerator.device)
            base_save_tensor = accelerate.utils.broadcast(base_save_tensor)
            base_save_epoch = base_save_tensor.item()

            for name, buffer in unwrapped_model.named_buffers():
                if "weight_cov" in name:
                    if base_save_epoch >= swag_start:
                        new_buffer_size = (
                            min(
                                config.method.swag_max_num_models,
                                base_save_epoch - swag_start + 1,
                            ),
                        ) + buffer.size()[1:]
                        new_buffer = torch.empty(*new_buffer_size)

                        buffer.data = new_buffer
                    else:
                        new_buffer_size = (0,) + buffer.size()[1:]
                        new_buffer = torch.empty(*new_buffer_size)

                        buffer.data = new_buffer

            unwrapped_model.base.load_adapter(
                base_path, "default", is_trainable=False)

            evaluate_task(
                model,
                "base",
                train_dataloader,
                eval_dataloader,
                test_dataloader,
                save_path=base_path,
                prefix="base_",
                accelerator=accelerator,
                causal_lm=causal_lm,
            )

            if config.experiment.ood_task != "":
                print("=========== OOD eval ==============")
                if accelerator.is_main_process:
                    accelerator.print(
                        f"Evaluating best loss LoRA model on OOD task "
                        f"{config.experiment.ood_task + config.experiment.ood_subtask} )"
                    )

                ood_metrics, ood_report = compute_metrics(
                    model,
                    ood_dataloader,
                    "OOD",
                    "base",
                    accelerator=accelerator,
                    causal_lm=causal_lm,
                )
                id_metrics, id_report = compute_metrics(
                    model,
                    test_dataloader,
                    "ID",
                    "base",
                    accelerator=accelerator,
                    causal_lm=causal_lm,
                )

                if accelerator.is_main_process:
                    print(id_report)
                    print(ood_report)

                    print("--- Detection (entropies) ---")
                    auroc_score, aupr_in, aupr_ood = ood_metrics_entropy(
                        id_metrics["entropies"], ood_metrics["entropies"]
                    )
                    print(f"AUROC score (entropies) = {auroc_score}")
                    print(f"AUPR_in score (entropies) = {aupr_in}")
                    print(f"AUPR_ood score (entropies) = {aupr_ood}")

                    print("--- Detection (MIs) ---")
                    auroc_score, aupr_in, aupr_ood = ood_metrics_entropy(
                        id_metrics["MIs"], ood_metrics["MIs"]
                    )
                    print(f"AUROC score (MIs) = {auroc_score}")
                    print(f"AUPR_in score (MIs) = {aupr_in}")
                    print(f"AUPR_ood score (MIs) = {aupr_ood}")

                    print("--- Detection (across model entropies) ---")
                    auroc_score, aupr_in, aupr_ood = ood_metrics_entropy(
                        id_metrics["across_model_entropies"],
                        ood_metrics["across_model_entropies"],
                    )
                    print(
                        f"AUROC score (across model entropies) = {auroc_score}")
                    print(
                        f"AUPR_in score (across model entropies) = {aupr_in}")
                    print(
                        f"AUPR_ood score (across model entropies) = {aupr_ood}")

                    print("--- Detection (disagrement ratio) ---")
                    auroc_score, aupr_in, aupr_ood = ood_metrics_entropy(
                        id_metrics["disagreement_ratios"],
                        ood_metrics["disagreement_ratios"],
                    )
                    print(f"AUROC score (disagreement_ratios) = {auroc_score}")
                    print(f"AUPR_in score (disagreement_ratios) = {aupr_in}")
                    print(f"AUPR_ood score (disagreement_ratios) = {aupr_ood}")
                accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(f"Saving SWAG model saving info to {save_path}")
        with open(os.path.join(save_path, "swag_save_info.json"), "w") as f:
            json.dump(swag_save_info, f)
    accelerator.wait_for_everyone()

    if config.experiment.eval_upon_save:
        if accelerator.is_main_process:
            if "saved_epoch" in swag_save_info:
                print(
                    f'Evaluating best loss SWAG model (using model from epoch {swag_save_info["saved_epoch"]})'
                )
            else:
                print(
                    "\n\n[!!!] Warning: Saved epoch information is not available. No loading possible!"
                )
                return None
        accelerator.wait_for_everyone()

        # change buffer shape of base model (we know when it was saved & swag_start)
        if accelerator.is_main_process:
            swag_save_tensor = torch.tensor(
                swag_save_info["saved_epoch"], device=accelerator.device
            )
        else:
            swag_save_tensor = torch.tensor([1], device=accelerator.device)
        swag_save_tensor = accelerate.utils.broadcast(swag_save_tensor)
        swag_save_epoch = swag_save_tensor.item()

        unwrapped_swag_model = accelerator.unwrap_model(swag_model)

        # change buffer shape of base model (we know when it was saved & swag_start)
        for name, buffer in unwrapped_swag_model.named_buffers():
            if "weight_cov" in name:
                new_buffer_size = (
                    min(
                        config.method.swag_max_num_models,
                        swag_save_epoch - swag_start + 1,
                    ),
                ) + buffer.size()[1:]
                new_buffer = torch.empty(*new_buffer_size)
                buffer.data = new_buffer

        state_dict = torch.load(
            os.path.join(save_path, "swag_model.pt"), map_location="cpu"
        )
        if "n_models" not in state_dict:
            print("n_models not in state_dict WTF")
            print(state_dict)
        unwrapped_swag_model.load_state_dict(state_dict, strict=True)
        del state_dict
        swag_model.sample(0.0)

        evaluate_task(
            swag_model,
            "swag",
            train_dataloader,
            eval_dataloader,
            test_dataloader,
            save_path=save_path,
            accelerator=accelerator,
            causal_lm=causal_lm,
        )
        if config.experiment.ood_task != "":
            print("=========== OOD eval ==============")
            if accelerator.is_main_process:
                accelerator.print(
                    f"Evaluating best loss LoRA model on OOD task "
                    f"{config.experiment.ood_task + config.experiment.ood_subtask} )"
                )

            ood_metrics, ood_report = compute_metrics(
                swag_model,
                ood_dataloader,
                "OOD",
                "swag",
                accelerator=accelerator,
                causal_lm=causal_lm,
                swag_samples=config.method.swag_samples,
                swag_sample_scale=config.method.swag_sample_scale,
                swag_cov_mat=config.method.swag_cov_mat,
            )
            id_metrics, id_report = compute_metrics(
                swag_model,
                test_dataloader,
                "ID",
                "swag",
                accelerator=accelerator,
                causal_lm=causal_lm,
                swag_samples=config.method.swag_samples,
                swag_sample_scale=config.method.swag_sample_scale,
                swag_cov_mat=config.method.swag_cov_mat,
            )

            if accelerator.is_main_process:
                print(id_report)
                print(ood_report)
                print("--- Detection (entropies) ---")
                auroc_score, aupr_in, aupr_ood = ood_metrics_entropy(
                    id_metrics["entropies"], ood_metrics["entropies"]
                )
                print(f"AUROC score (entropies) = {auroc_score}")
                print(f"AUPR_in score (entropies) = {aupr_in}")
                print(f"AUPR_ood score (entropies) = {aupr_ood}")

                print("--- Detection (MIs) ---")
                auroc_score, aupr_in, aupr_ood = ood_metrics_entropy(
                    id_metrics["MIs"], ood_metrics["MIs"]
                )
                print(f"AUROC score (MIs) = {auroc_score}")
                print(f"AUPR_in score (MIs) = {aupr_in}")
                print(f"AUPR_ood score (MIs) = {aupr_ood}")

                print("--- Detection (across model entropies) ---")
                auroc_score, aupr_in, aupr_ood = ood_metrics_entropy(
                    id_metrics["across_model_entropies"],
                    ood_metrics["across_model_entropies"],
                )
                print(f"AUROC score (across model entropies) = {auroc_score}")
                print(f"AUPR_in score (across model entropies) = {aupr_in}")
                print(f"AUPR_ood score (across model entropies) = {aupr_ood}")

                print("--- Detection (disagrement ratio) ---")
                auroc_score, aupr_in, aupr_ood = ood_metrics_entropy(
                    id_metrics["disagreement_ratios"],
                    ood_metrics["disagreement_ratios"],
                )
                print(f"AUROC score (disagreement_ratios) = {auroc_score}")
                print(f"AUPR_in score (disagreement_ratios) = {aupr_in}")
                print(f"AUPR_ood score (disagreement_ratios) = {aupr_ood}")
    return None
