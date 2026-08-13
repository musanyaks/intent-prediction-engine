"""Offline model evaluation with business-relevant metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    mean_absolute_percentage_error,
    roc_auc_score,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_intent_classifier(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Evaluate intent classification with macro-F1 and per-class recall."""
    from sklearn.metrics import classification_report, confusion_matrix

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, output_dict=True)

    metrics = {
        "macro_f1": macro_f1,
        "accuracy": report["accuracy"],
    }
    for cls in ["0", "1", "2", "3"]:
        if cls in report:
            metrics[f"recall_class_{cls}"] = report[cls]["recall"]

    logger.info(f"Intent classifier - Macro F1: {macro_f1:.4f}")
    return metrics


def evaluate_purchase_probability(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Evaluate purchase probability with AUC-PR and log-loss."""
    auc_pr = average_precision_score(y_true, y_prob)
    ll = log_loss(y_true, y_prob, eps=1e-15)
    auc_roc = roc_auc_score(y_true, y_prob)

    metrics = {
        "auc_pr": auc_pr,
        "log_loss": ll,
        "auc_roc": auc_roc,
    }
    logger.info(f"Purchase probability - AUC-PR: {auc_pr:.4f}, LogLoss: {ll:.4f}")
    return metrics


def evaluate_ltv(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Evaluate LTV prediction with MAPE and under-prediction rate."""
    mape = mean_absolute_percentage_error(y_true, y_pred)
    under_pred_rate = (y_pred < y_true).mean()

    metrics = {
        "mape": mape,
        "under_prediction_rate": under_pred_rate,
        "mean_absolute_error": np.mean(np.abs(y_true - y_pred)),
    }
    logger.info(f"LTV - MAPE: {mape:.4f}, Under-prediction rate: {under_pred_rate:.4f}")
    return metrics


def evaluate_churn(y_true: np.ndarray, y_prob: np.ndarray, fpr_threshold: float = 0.05) -> dict:
    """Evaluate churn prediction with AUC-ROC and FPR constraint."""
    from sklearn.metrics import roc_curve

    auc_roc = roc_auc_score(y_true, y_prob)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    # Find threshold where FPR <= 5%
    valid_idx = np.where(fpr <= fpr_threshold)[0]
    if len(valid_idx) > 0:
        best_threshold = thresholds[valid_idx[-1]]
        tpr_at_constraint = tpr[valid_idx[-1]]
    else:
        best_threshold = 0.5
        tpr_at_constraint = 0.0

    metrics = {
        "auc_roc": auc_roc,
        "tpr_at_fpr_5pct": tpr_at_constraint,
        "optimal_threshold": best_threshold,
    }
    logger.info(f"Churn - AUC-ROC: {auc_roc:.4f}, TPR@FPR<=5%: {tpr_at_constraint:.4f}")
    return metrics


def generate_evaluation_report(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> pd.DataFrame:
    """Generate a comprehensive evaluation report."""
    model.eval()
    all_outputs = {"intent": [], "purchase": [], "ltv": [], "churn": []}
    all_targets = {"intent": [], "purchase": [], "ltv": [], "churn": []}

    with torch.no_grad():
        for batch in dataloader:
            event_ids = batch["event_ids"].to(device)
            user_features = batch["user_features"].to(device)
            product_features = batch["product_features"].to(device)
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(device)

            outputs = model(event_ids, user_features, product_features, mask)

            all_outputs["intent"].extend(outputs["intent_logits"].argmax(dim=1).cpu().numpy())
            all_outputs["purchase"].extend(torch.sigmoid(outputs["purchase_logit"]).cpu().numpy())
            all_outputs["ltv"].extend(outputs["ltv"].cpu().numpy())
            all_outputs["churn"].extend(torch.sigmoid(outputs["churn_logit"]).cpu().numpy())

            all_targets["intent"].extend(batch["intent"].numpy())
            all_targets["purchase"].extend(batch["purchase"].numpy())
            all_targets["ltv"].extend(batch["ltv"].numpy())
            all_targets["churn"].extend(batch["churn"].numpy())

    report = {
        **evaluate_intent_classifier(
            np.array(all_targets["intent"]),
            np.array(all_outputs["intent"]),
        ),
        **evaluate_purchase_probability(
            np.array(all_targets["purchase"]),
            np.array(all_outputs["purchase"]),
        ),
        **evaluate_ltv(
            np.array(all_targets["ltv"]),
            np.array(all_outputs["ltv"]),
        ),
        **evaluate_churn(
            np.array(all_targets["churn"]),
            np.array(all_outputs["churn"]),
        ),
    }

    return pd.DataFrame([report])