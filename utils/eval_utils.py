from sklearn.metrics import matthews_corrcoef
import torch.nn.functional as F
import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


def set_dropout_training(module):
    if type(module) == torch.nn.modules.dropout.Dropout:
        module.train()


def mean_logits(logits):
    return torch.stack(logits).mean(dim=0)


def compute_nll(probabilities, labels, normalize=True):
    labels = labels.long()

    n_classes = probabilities.shape[-1]
    y_true_one_hot = F.one_hot(labels.squeeze(), num_classes=n_classes)

    nll = -torch.sum(y_true_one_hot * torch.log(probabilities + 1e-9))
    if normalize:
        nll /= len(labels)
    return nll


def compute_brier(probs, labels, normalize=False):
    labels = labels.long()

    # Ensure labels are on the same device as probs
    if labels.device != probs.device:
        labels = labels.to(probs.device)

    n_classes = probs.size(-1)

    y_true_one_hot = F.one_hot(labels.squeeze(), num_classes=n_classes)

    if normalize:
        brier_score = torch.mean(
            torch.sum((probs - y_true_one_hot) ** 2, dim=-1))
    else:
        brier_score = torch.sum(
            torch.sum((probs - y_true_one_hot) ** 2, dim=-1))
    return brier_score


def compute_ece(probabilities, labels, num_bins=20, div_factor=None):

    labels = torch.as_tensor(labels)
    confidences = torch.max(probabilities, dim=1)[0]

    if labels.device != probabilities.device:
        labels = labels.to(probabilities.device)

    denom = confidences.shape[0]
    if div_factor is not None:
        denom = div_factor

    predictions = torch.argmax(probabilities, dim=1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        # samples in current bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = torch.sum(in_bin).item()
        if bin_size > 0:
            accuracy_in_bin = torch.mean(accuracies[in_bin].float()).item()
            confidence_in_bin = torch.mean(confidences[in_bin]).item()
            ece += abs(accuracy_in_bin - confidence_in_bin) * \
                (bin_size / denom)
    return ece


def compute_mcc(all_preds, all_labels):
    if isinstance(all_preds, torch.Tensor):
        all_preds = all_preds.cpu()
    if isinstance(all_preds, torch.Tensor):
        all_labels = all_labels.cpu()

    mcc = matthews_corrcoef(all_labels, all_preds)
    return mcc


def get_classification_metrics(all_probas: torch.Tensor, all_labels: torch.Tensor, prefix: str = ""):
    if not isinstance(all_probas, torch.Tensor):
        all_probas = torch.tensor(all_probas)
    if not isinstance(all_labels, torch.Tensor):
        all_labels = torch.tensor(all_labels)
    
    assert(len(all_labels.shape) == 1), f"Labels should be 1D tensor, got {all_labels.shape}"
    
    if all_probas.abs().max() > 1:
        print(f"Probas are not in [0, 1] range, max value: {all_probas.abs().max()}, applying softmax")
        all_probas = F.softmax(all_probas, dim=-1)
    
    metrics = {}

    print(f"all_probas shape: {all_probas.shape}, all_labels shape: {all_labels.shape}")

    metrics["acc"] = torch.mean(all_labels.eq(all_probas.argmax(dim=-1)).float()).item()
    metrics["nll"] = compute_nll(all_probas, all_labels, normalize=True)
    metrics["ece"] = compute_ece(all_probas, all_labels, num_bins=20)
    
    metrics["comb_score"] = -metrics["acc"] + metrics["nll"] + metrics["ece"]
    metrics["comb_calib_score"] = metrics["nll"] + metrics["ece"]
    
    metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
    
    return metrics


def compute_metrics(
    model,
    dataloader,
    split,
    method,
    dropout_samples=None,
    accelerator=None,
    causal_lm=False,
):

    model.eval()
    loss_fn = CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0
    nll = 0
    brier = 0
    all_preds = []
    all_labels = []
    all_probs_list = []

    for batch in tqdm(dataloader, disable=not accelerator.is_main_process):
        with torch.no_grad():
            if method == "base":
                losses = []
                dropout_logits = []

                model.eval()
                if dropout_samples is None or dropout_samples == 1:
                    dropout_samples = 1
                else:
                    model.apply(set_dropout_training)
                for i in range(dropout_samples):
                    output = model(
                        **{
                            "input_ids": batch["input_ids"],
                            "attention_mask": batch["attention_mask"],
                        }
                    )
                    logits = (
                        output.logits[:, -1, :] if causal_lm else output.logits
                    )  # if causal lm we use last representation's logits
                    dropout_logits.append(logits)
                    loss = loss_fn(logits, batch["labels"])
                    losses.append(loss)
                total_loss += sum(tensor for tensor in losses) / len(losses)

                probs = torch.stack(
                    [F.softmax(logs, dim=-1) for logs in dropout_logits]
                )  # (num_models, bsz, classes)

                preds = mean_logits(dropout_logits).argmax(
                    dim=-1
                )  # ensemble predictions (bsz)

                probs = probs.mean(
                    dim=0
                )  # get mc dropout 'ensemble' probs (bsz, classes)

            else:
                raise Exception("No such method implemented: ", method)

            (
                collected_preds,
                collected_labels,
                collected_probs,
            ) = accelerator.gather_for_metrics(
                (preds, batch["labels"], probs)
            )
            if accelerator.is_main_process:
                all_preds += collected_preds.tolist()
                all_labels += collected_labels.tolist()
                all_probs_list.append(collected_probs)  # [(bsz, classes)...]
            accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            nll += compute_nll(collected_probs,
                               collected_labels, normalize=False)

            brier += compute_brier(collected_probs, collected_labels)
            correct += (
                collected_preds.eq(collected_labels.view_as(collected_preds))
                .sum()
                .item()
            )
            total += len(collected_labels)
        accelerator.wait_for_everyone()

    collected_loss = accelerator.gather_for_metrics(total_loss).sum().item()

    metrics = {
        "loss": torch.tensor(0.0, device=accelerator.device),
        "acc": torch.tensor(0.0, device=accelerator.device),
        "ece": torch.tensor(0.0, device=accelerator.device),
        "brier": torch.tensor(0.0, device=accelerator.device),
        "nll": torch.tensor(0.0, device=accelerator.device),
        "mcc": torch.tensor(0.0, device=accelerator.device),
    }

    if accelerator.is_main_process:
        all_probs = torch.cat(all_probs_list, dim=0)  # (all_samples, classes)
        ece = compute_ece(all_probs, all_labels)
        mcc = compute_mcc(all_preds, all_labels)
        nll /= len(dataloader.dataset)
        brier /= len(dataloader.dataset)

        metrics = {
            "loss": collected_loss / len(dataloader),
            "acc": correct / total,
            "ece": ece,
            "brier": brier,
            "nll": nll,
            "mcc": mcc,
        }

        metrics["comb_score"] = -5 * metrics["acc"] + metrics["nll"]
        metrics["comb_calib_score"] = metrics["nll"] + metrics["ece"]

    accelerator.wait_for_everyone()

    report_str = split + ":\n" + str(metrics)

    return metrics, report_str
