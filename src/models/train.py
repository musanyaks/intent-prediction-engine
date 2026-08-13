"""Training loop with MLflow tracking."""

from __future__ import annotations

import os
from typing import Any

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models.architecture import MultiTaskIntentModel
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    def __init__(
        self,
        model: MultiTaskIntentModel,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.global_step = 0

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Training"):
            self.optimizer.zero_grad()

            event_ids = batch["event_ids"].to(self.device)
            user_features = batch["user_features"].to(self.device)
            product_features = batch["product_features"].to(self.device)
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(self.device)

            targets = {
                "intent": batch["intent"].to(self.device),
                "purchase": batch["purchase"].to(self.device),
                "ltv": batch["ltv"].to(self.device),
                "churn": batch["churn"].to(self.device),
            }

            outputs = self.model(event_ids, user_features, product_features, mask)
            loss = self.model.compute_loss(outputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

        return {"train_loss": total_loss / max(num_batches, 1)}

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_preds = {"intent": [], "purchase": [], "ltv": [], "churn": []}
        all_targets = {"intent": [], "purchase": [], "ltv": [], "churn": []}

        with torch.no_grad():
            for batch in dataloader:
                event_ids = batch["event_ids"].to(self.device)
                user_features = batch["user_features"].to(self.device)
                product_features = batch["product_features"].to(self.device)
                mask = batch.get("mask")
                if mask is not None:
                    mask = mask.to(self.device)

                targets = {
                    "intent": batch["intent"].to(self.device),
                    "purchase": batch["purchase"].to(self.device),
                    "ltv": batch["ltv"].to(self.device),
                    "churn": batch["churn"].to(self.device),
                }

                outputs = self.model(event_ids, user_features, product_features, mask)
                loss = self.model.compute_loss(outputs, targets)

                total_loss += loss.item()
                num_batches += 1

                all_preds["intent"].extend(outputs["intent_logits"].argmax(dim=1).cpu().numpy())
                all_preds["purchase"].extend((torch.sigmoid(outputs["purchase_logit"]) > 0.5).cpu().numpy())
                all_preds["ltv"].extend(outputs["ltv"].cpu().numpy())
                all_preds["churn"].extend((torch.sigmoid(outputs["churn_logit"]) > 0.5).cpu().numpy())

                all_targets["intent"].extend(targets["intent"].cpu().numpy())
                all_targets["purchase"].extend(targets["purchase"].cpu().numpy())
                all_targets["ltv"].extend(targets["ltv"].cpu().numpy())
                all_targets["churn"].extend(targets["churn"].cpu().numpy())

        metrics = {
            "val_loss": total_loss / max(num_batches, 1),
            "val_intent_accuracy": sum(
                p == t for p, t in zip(all_preds["intent"], all_targets["intent"])
            ) / len(all_targets["intent"]),
        }
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        early_stopping_patience: int = 3,
    ) -> dict[str, Any]:
        mlflow.set_experiment("intent-prediction-engine")
        best_val_loss = float("inf")
        patience_counter = 0

        with mlflow.start_run():
            mlflow.log_params({
                "epochs": epochs,
                "lr": self.optimizer.defaults["lr"],
                "device": self.device,
            })

            for epoch in range(epochs):
                train_metrics = self.train_epoch(train_loader)
                val_metrics = self.evaluate(val_loader)

                mlflow.log_metrics({**train_metrics, **val_metrics}, step=epoch)
                logger.info(f"Epoch {epoch+1}/{epochs}: {val_metrics}")

                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    patience_counter = 0
                    self.save_checkpoint("best_model.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info("Early stopping triggered")
                        break

        return {"best_val_loss": best_val_loss}

    def save_checkpoint(self, path: str) -> None:
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }, path)
        logger.info(f"Checkpoint saved to {path}")