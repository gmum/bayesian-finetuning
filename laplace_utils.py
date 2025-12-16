from collections.abc import MutableMapping
from enum import Enum
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Union
import math

from peft.peft_model import PeftModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from laplace import Laplace
from laplace.utils.enums import Likelihood, SubsetOfWeights, HessianStructure, PredType, LinkApprox, TuningMethod, PriorStructure
from laplace.baselaplace import BaseLaplace, KronLaplace, FullLaplace, LowRankLaplace, ParametricLaplace, FunctionalLaplace, DiagLaplace
from laplace.curvature import *
from torchmetrics import Accuracy, CalibrationError

import re
from typing import List

import wandb
from utils.eval_utils import compute_nll, compute_ece, get_classification_metrics, compute_metrics
from loading_utils import load_loraxs_weights
from utils.peft_utils import WrappedModel
# from bayesian_lora_utils import evaluate_bayesian_lora

def tensor_metrics_to_float(metrics: dict) -> dict:
  # iterate over dict (possibly nested) and convert all torch.Tensor to float
  for k, v in metrics.items():
    if isinstance(v, torch.Tensor):
      if v.numel() > 1:
        print(f"Warning: {k} is a tensor with {v.numel()} elements, converting to float")
        print(f"v.shape: {v.shape}")
      metrics[k] = v.item()
    elif isinstance(v, dict):
      metrics[k] = tensor_metrics_to_float(v)
  return metrics


class LaplaceWrapper(nn.Module):
    def __init__(self, peft_model: PeftModel):
        super().__init__()
        self.model = peft_model

    def forward(self, data: MutableMapping) -> torch.Tensor:
        # Remove labels for inference
        inference_data = {k: v for k, v in data.items() if k != 'labels'}
        output = self.model(**inference_data)
        logits = output.logits
        # print(f"LaplaceWrapper logits shape: {logits.shape}")
        # If output is 3D (causal LM: [batch, seq_len, vocab] or [batch, seq_len, num_choices]),
        # take the last token's logits and, if needed, select only the answer tokens
        if logits.dim() == 3:
            # [batch, seq_len, vocab/num_choices] -> [batch, num_choices] (last token)
            logits = logits[:, -1, :]
        # If output is already [batch, num_choices], do nothing
        # print(f"LaplaceWrapper final logits.shape: {logits.shape}")
        return logits.to(torch.float32)

class LaplaceWeights(str, Enum):
    """Valid options for `subset_of_weights`."""

    LORAXS = "loraxs"
    """Calc Laplace approximation for LORAXS adapters"""

    LL = "last_layer"
    """Laplace only for last layer (out_proj) in Roberta-Large case"""
    
    LORAXS_LL = "loraxs_last_layer"
    """Laplace for both loraxs and last layer"""
    
    FULL_CLS = "full_cls"
    """Laplace for all classifier weights"""
    
    FULL = "full"
    """Laplace for all weights(i.e. loraxs and classifier)"""
    
    @staticmethod
    def update_model(model: nn.Module, laplace_weights: 'LaplaceWeights', verbose: bool = False) -> None:
        print(f"Setting LaplaceWeights: {laplace_weights}")
        model.requires_grad_(False)
        if laplace_weights == LaplaceWeights.LORAXS:
            set_loraxs_requires_grad(model, True)
        elif laplace_weights == LaplaceWeights.LL:
            set_classifier_out_proj_requires_grad(model, True)
        elif laplace_weights == LaplaceWeights.LORAXS_LL:
            set_loraxs_requires_grad(model, True)
            set_classifier_out_proj_requires_grad(model, True)
        elif laplace_weights == LaplaceWeights.FULL_CLS:
            set_classifier_requires_grad(model, True)
        elif laplace_weights == LaplaceWeights.FULL:
            set_classifier_requires_grad(model, True)
            set_loraxs_requires_grad(model, True)
        else:
            raise ValueError(f"Invalid LaplaceWeights: {laplace_weights}")
            
        if verbose:
          print_parameters_require_grad_short(model)

def set_classifier_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for name, param in model.named_parameters():
        if "classifier.modules_to_save" in name:
            param.requires_grad = requires_grad

def set_loraxs_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for name, param in model.named_parameters():
        if "lora_latent_mapping" in name:
            param.requires_grad = requires_grad
            
def set_loraxs_output_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for name, param in model.named_parameters():
        if "output" in name and "lora_latent_mapping" in name:
            param.requires_grad = requires_grad

def set_classifier_out_proj_requires_grad(model: nn.Module, requires_grad: bool) -> None:
  for name, param in model.named_parameters():
    if 'modules_to_save.default.out_proj' in name:
      param.requires_grad = True
            
def print_parameters_require_grad_short(model) -> None:
    """
    Print parameter names requiring gradients, combining encoder layer indices.
    
    Args:
        model: PyTorch model
    """
    # Get names of parameters requiring gradients
    trainable_params = [
        name for name, param in model.named_parameters()
        if param.requires_grad
    ]
    # Process and deduplicate names
    processed_names = set()
    for param_name in trainable_params:
        # Match encoder layer pattern
        pattern = r'(.*encoder\.layer\.)\d+(.*)'
        if re.match(pattern, param_name):
            # Replace layer index with generic "layer"
            consolidated_name = re.sub(pattern, r'\1\2', param_name)
            processed_names.add(consolidated_name)
        else:
            processed_names.add(param_name)
    # Print consolidated parameter names
    for name in sorted(processed_names):
        print(name)
        
def get_trainer_checkpoints(model_dirs, use_first_last:bool = True):
  checkpoint_dirs = []
  # eval all checkpoints present in the directory
  checkpoint_pref = "checkpoint-"
  for model_dir in model_dirs:
      if checkpoint_pref in model_dir:
          checkpoint_dirs.append(model_dir)
  # sort checkpoints by steps numbers in the name in ascending order
  checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[-1]))
  if use_first_last and len(checkpoint_dirs) > 1:
      checkpoint_dirs = [checkpoint_dirs[0], checkpoint_dirs[-1]]
  
  return checkpoint_dirs

def get_best_metrics_checkpoints(model_dirs, best_checkpoint_prefix: str):
  checkpoint_dirs = []
  # best checkpoint metrics are saved in directories of the form "{best_checkpoint_prefix}{metric_name}-{step}"
  for model_dir in model_dirs:
    if best_checkpoint_prefix in model_dir:
      checkpoint_dirs.append(model_dir)

  return checkpoint_dirs

def checkpoints_to_fit(output_dir, 
                       *,
                       use_best_checkpoints:bool,
                       use_step_checkpoints:bool,
                       best_checkpoint_prefix: str="best_eval_", 
                       use_first_last:bool = False, 
                       peft_method="lora_xs"):
    """
    Get the checkpoints to fit Laplace on.

    use_best_checkpoints: bool - whether to use best checkpoints based on metrics
    use_step_checkpoints: bool - whether to use step checkpoints
    use_first_last: bool - whether to use only first and last checkpoints by steps, applies to LORAXS. 
    For LORAXS first one is the best and last one is the last one.
    """
    model_dirs = os.listdir(output_dir)
    checkpoint_dirs = []
    if peft_method == "lora":
        steps_to_eval = [3999, 4999, 5999]
        checkpoint_pref = "step_"
        for step in steps_to_eval:
            if f"{checkpoint_pref}{step}" in model_dirs:
                checkpoint_dirs.append(f"{checkpoint_pref}{step}")
    elif peft_method == "lora_xs":
      if use_best_checkpoints:
        checkpoint_dirs.extend(get_best_metrics_checkpoints(model_dirs, best_checkpoint_prefix))
      if use_step_checkpoints:
        checkpoint_dirs.extend(get_trainer_checkpoints(model_dirs, use_first_last))
    
    if len(checkpoint_dirs) == 0:
        raise ValueError(f"No checkpoints to fit, use_best_checkpoints: {use_best_checkpoints}, use_step_checkpoints: {use_step_checkpoints}")

    print(f"Fitting Laplace on the following checkpoints: {checkpoint_dirs}")
    return checkpoint_dirs

  
def evaluate_linearized_prediction(
    data_loader,
    prefix: str,
    la : BaseLaplace,
    device,
    pred_type,
    link_approx,
    num_labels,
    prediction_kwargs
):
    """
    Evaluate model using linearized prediction.
    
    Args:
        data_loader: DataLoader for validation set
        la: Laplace approximator
        device: torch device
        pred_type: prediction type
        link_approx: link approximation method
        num_labels: number of classes
        
    Returns:
        dict: Metrics including NLL, accuracy and calibration error
    """
    print(f"Doing linearized prediction {prefix}")
    if la.likelihood != Likelihood.CLASSIFICATION:
        raise ValueError("Only classification likelihood is supported for this function (evaluate_linearized_prediction)")
    
    # print gpu memory usage
    verbose = False
    if verbose:
      print(f"GPU memory allocated: {torch.cuda.memory_allocated(device=device) / 1024**3:.2f} GB")
      print(f"GPU memory reserved: {torch.cuda.memory_reserved(device=device) / 1024**3:.2f} GB")
    # total_loss = 0
    total_samples = 0
    
    # metric_kwargs = {"task": "multiclass", "num_classes": num_labels}
    # acc_metric = Accuracy(**metric_kwargs).to(device)
    # ece_metric = CalibrationError(n_bins=20, **metric_kwargs).to(device)
    # nll_metric = torch.nn.NLLLoss(reduction="sum")
    all_probas = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            labels = batch["labels"].to(device)
            preds = la(batch, pred_type=pred_type, link_approx=link_approx, **prediction_kwargs)
            
            probas = preds # For now, we consider only Classification
            all_probas.append(probas)
            all_labels.append(labels)
            # acc_metric.update(probas, labels)
            # total_loss += nll_metric(torch.log(probas), labels).item()
            # ece_metric.update(probas, labels)
            total_samples += len(labels)

    # acc = acc_metric.compute().item()
    # total_loss /= total_samples
    # ece = ece_metric.compute().item()
    
    all_probas = torch.cat(all_probas)
    all_labels = torch.cat(all_labels)
    
    metrics = get_classification_metrics(all_probas, all_labels, prefix=prefix)
    
    print(f"Metrics for {prefix}: {metrics}, total samples: {total_samples}")
    
    return metrics


def print_hessian_info(la):
  if isinstance(la, KronLaplace):
    print("Kronecker factors shapes:")
    string = ""
    for layer, kfac_list in enumerate(la.H.kfacs):
      string += f"Layer {layer}: "
      for kfac in kfac_list:
        string += f"{kfac.shape}, "
      string += '\n'
    print(string)

  
  if isinstance(la, FullLaplace):
    print(f"Full Hessian shape: {la.H.shape}")


@dataclass
class LaplaceParams:
  name: str
  laplace_weights: LaplaceWeights
  hessian_structure: HessianStructure
  likelihood: Likelihood
  pred_type: PredType
  link_approx: LinkApprox
  prior_optim_method: TuningMethod
  prior_structure: PriorStructure
  backend: Any # Defaults to CurvlinopsGGN if None. if FULL choose AsdlGGN - default doesn't work for attention module with FULL
  prior_kwargs: dict
  prediction_kwargs: dict
  
  def update_model(self, model):
    # set requires_grad for the model parameters used in Laplace approximation
    self.laplace_weights.update_model(model, self.laplace_weights)

  def get_short_name(self):
    return f"{self.laplace_weights.name}_{self.hessian_structure.name}"


def evaluate_laplace(model,
                     *,
                     device,
                     train_loader,
                     val_loader,
                     test_loader=None,
                     num_labels,
                     laplace_save_path=None,
                     laplace_params: LaplaceParams,
                     test_precision_importance: bool = False,
                     **kwargs):
    
  laplace_params.update_model(model)
  # backend for access to curvature/Hessian approximations. Defaults to CurvlinopsGGN if None.
  la = Laplace(
      LaplaceWrapper(model),
      likelihood=laplace_params.likelihood,
      subset_of_weights="all", # even though "all", only weights with gradients turned on will be used
      hessian_structure=laplace_params.hessian_structure,
      backend=laplace_params.backend
  )
#   print_hessian_info(la)
  
  la.fit(train_loader, progress_bar=True)
  print("Prior precision before optimization: ", la.prior_precision)
  # marginal likelihood optimization using Adam
  la.optimize_prior_precision(pred_type=laplace_params.pred_type,
                              method=laplace_params.prior_optim_method,
                              prior_structure=laplace_params.prior_structure,
                              **laplace_params.prior_kwargs,
                              progress_bar=True)
  print("Prior precision after optimization: ", la.prior_precision)

  # Drop unused references to free up memory
  import gc
  for i in range(5):
      gc.collect()
  
  total_metrics = {
      "nl_marglik" : -la.log_marginal_likelihood().item()
  }
  print(f"Negative log marginal likelihood: {total_metrics['nl_marglik']}")
  if val_loader:
    val_metrics = evaluate_linearized_prediction(val_loader, "eval_", la, device, laplace_params.pred_type, laplace_params.link_approx, num_labels, prediction_kwargs=laplace_params.prediction_kwargs)
    total_metrics.update(val_metrics)
    
  if test_precision_importance:
    # Run evaluation for precision equal 100 * prior precision and 1 / 100 * prior precision
    real_prior_precision = la.prior_precision.clone()
    with torch.no_grad():
      # 100 * prior precision
      la.prior_precision.mul_(10.0)
      print("Prior precision after multiplying by 10: ", la.prior_precision)
      prior_mult_10_metrics = evaluate_linearized_prediction(val_loader, "mult_10_", la, device, laplace_params.pred_type, laplace_params.link_approx, num_labels, prediction_kwargs=laplace_params.prediction_kwargs)
    
      la.prior_precision.copy_(real_prior_precision)
    
  if test_loader:
    test_metrics = evaluate_linearized_prediction(test_loader, "test_", la, device, laplace_params.pred_type, laplace_params.link_approx, num_labels, prediction_kwargs=laplace_params.prediction_kwargs)
    total_metrics.update(test_metrics)
  
  if laplace_save_path is not None:
    # save state_dict of laplace
    torch.save(la.state_dict(), laplace_save_path)
    
  # delete laplace object to free up memory
  del la
  return total_metrics


def get_laplace_params(*, 
                       name: Optional[str]=None, 
                       laplace_weights: LaplaceWeights, 
                       hessian_structure: Union[HessianStructure, str] = HessianStructure.KRON,
                       likelihood: Likelihood = Likelihood.CLASSIFICATION,
                       link_approx: LinkApprox = LinkApprox.MC,
                       prior_structure: PriorStructure = PriorStructure.SCALAR,
                       prior_kwargs: dict = {"n_steps" : 1000, "lr" : 0.1},
                       prediction_kwargs: dict = {"n_samples" : 100000, "joint": True},
                       pred_type: PredType = PredType.GLM,
                       backend=CurvlinopsGGN) -> LaplaceParams:
    if isinstance(hessian_structure, str):
      hessian_structure = HessianStructure(hessian_structure)
    # Use AsdlGGN for FULL and DIAG - CurvlinopsGGN doesn't have efficient diag implementation
    # For DIAG, CurvlinopsGGN falls back to computing full Jacobians which causes OOM
    use_asdl = hessian_structure in [HessianStructure.FULL, HessianStructure.DIAG]
    
    laplace_params = LaplaceParams(
      name="",
      laplace_weights=laplace_weights,
      hessian_structure=hessian_structure,
      likelihood=likelihood,
      pred_type=pred_type,  
      link_approx=link_approx,
      prior_optim_method=TuningMethod.MARGLIK,
      prior_structure=prior_structure,
      backend=AsdlGGN if use_asdl else backend,
      prior_kwargs=prior_kwargs,
      prediction_kwargs=prediction_kwargs
    )

    if name is None:
      base_name = f"{laplace_weights.name}_{hessian_structure.name}_{link_approx.name}"
      if likelihood != Likelihood.CLASSIFICATION:
        base_name += f"_{likelihood.name}"
        
      if prior_structure != PriorStructure.SCALAR:
        base_name += f"_pr_{prior_structure.name}"
      
      if prior_kwargs != {"n_steps" : 1000, "lr" : 0.1}:
        base_name += f"_st_{prior_kwargs['n_steps']}_lr_{prior_kwargs['lr']}"
      
      if prediction_kwargs != {"n_samples" : 100000, "joint": True}:
        base_name += f"_ns_{prediction_kwargs['n_samples']}_j_{prediction_kwargs['joint']}"
        
      if laplace_params.backend != AsdlGGN:
        base_name += f"_backend_{backend.__name__}"

      laplace_params.name = base_name
    else:
      laplace_params.name = name
    
    return laplace_params


def evaluate_laplace_params(
    model : WrappedModel | PeftModel,
    laplace_params_list,
    *,
    prefix,
    checkpoint_full_path=None,
    device,
    num_labels,
    config,
    train_loader,
    val_loader,
    test_loader=None,
    json_metrics_path=None,
    json_metric_file="laplace_metrics.json",
    test_run=False,
    accelerator=None,
    causal_lm=False,
    wandb_run=None,
):
    if checkpoint_full_path is not None:
        if isinstance(model, WrappedModel):
            load_loraxs_weights(model.get_peft_model(), checkpoint_full_path, load_classifier=True)
        else:
            load_loraxs_weights(model, checkpoint_full_path, load_classifier=True)
    else:
        print("No checkpoint provided. Using model as is.")

    total_laplace_metrics = {}
    json_metrics_full_path = None
    if json_metrics_path is not None:
        json_metrics_full_path = os.path.join(json_metrics_path, json_metric_file)

    if json_metrics_full_path is not None and os.path.exists(json_metrics_full_path):
        with open(json_metrics_full_path, "r") as f:
            total_laplace_metrics = json.load(f)

    base_name = f"{prefix}_base"
    if base_name in total_laplace_metrics:
        print(f"{base_name} already evaluated")
    else:
        base_metrics, _ = compute_metrics(
            model,
            val_loader,
            split="eval",
            method="base",
            accelerator=accelerator,
            causal_lm=causal_lm,
        )
        base_metrics = tensor_metrics_to_float(base_metrics)
        # TODO: If there're multiple Laplace methods evaluated, they will be logged under the same metric name. 
        # Add differentiation for diag and kronecker
        base_metrics = {f"eval_{k}": v for k, v in base_metrics.items()}
        
        # Evaluate on test set if available
        if test_loader is not None and len(test_loader) > 0:
            test_metrics, _ = compute_metrics(
                model,
                test_loader,
                split="test",
                method="base",
                accelerator=accelerator,
                causal_lm=causal_lm,
            )
            test_metrics = tensor_metrics_to_float(test_metrics)
            test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}
            base_metrics.update(test_metrics)
        
        if wandb_run is not None:
            wandb_run.log({f"{prefix}/base_{key}": value for key, value in base_metrics.items()})
            
            wandb_run.log({f"laplace/base_{key}": value for key, value in base_metrics.items()})
        
        total_laplace_metrics[base_name] = base_metrics
        total_laplace_metrics[base_name]["nl_marglik"] = 0
        total_laplace_metrics[base_name]["name"] = f"{prefix}_base"

    for laplace_params in laplace_params_list:
        print(f"laplace_params: {laplace_params}")
        method_name = laplace_params.name
        full_name = f"{prefix}_{method_name}"
        if full_name in total_laplace_metrics:
            print(f"{full_name} already evaluated")
        elif test_run:
            total_laplace_metrics[full_name] = {"name": f"{prefix}_{method_name}"}
            for key in total_laplace_metrics[base_name].keys():
                if key != "name":
                    total_laplace_metrics[full_name][key] = 0.0
        else:
            total_laplace_metrics[full_name] = evaluate_laplace(
                model,
                device=device,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                num_labels=num_labels,
                laplace_save_path=None,
                laplace_params=laplace_params,
            )
            total_laplace_metrics[full_name] = tensor_metrics_to_float(total_laplace_metrics[full_name])
            total_laplace_metrics[full_name]["name"] = f"{prefix}_{method_name}"
        if json_metrics_full_path is not None:
            with open(json_metrics_full_path, "w") as f:
                json.dump(total_laplace_metrics, f)
        
        if wandb_run is not None:
            wandb_run.log({f"{prefix}/{laplace_params.get_short_name()}_{key}": value for key, value in total_laplace_metrics[full_name].items()})
            
            wandb_run.log({f"laplace_{laplace_params.get_short_name()}/{key}": value for key, value in total_laplace_metrics[full_name].items()})

    return total_laplace_metrics