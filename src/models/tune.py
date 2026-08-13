"""Hyperparameter optimization with Optuna."""

from __future__ import annotations

import optuna
import torch
from optuna.integration import PyTorchLightningPruningCallback
from torch.utils.data import DataLoader
from src.models.architecture import MultiTaskIntentModel
from src.models.train import Trainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_optuna_study(
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_trials: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> optuna.Study:
    """
    Run Optuna hyperparameter search for the multi-task model.
    """

    def objective(trial: optuna.Trial) -> float:
        # Hyperparameter search space
        embed_dim = trial.suggest_categorical("embed_dim", [32, 64, 128])
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        num_layers = trial.suggest_int("num_layers", 1, 4)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        model = MultiTaskIntentModel(
            event_vocab_size=1000,  # Will be overridden by actual vocab
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_intent_classes=4,
            dropout=dropout,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        trainer = Trainer(model, optimizer, device=device)
        metrics = trainer.fit(
            train_loader,
            val_loader,
            epochs=5,  # Short epochs for tuning
            early_stopping_patience=2,
        )

        return metrics["best_val_loss"]

    study = optuna.create_study(
        study_name="intent_model_optimization",
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
    )

    logger.info(f"Starting Optuna study with {n_trials} trials")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best hyperparameters: {study.best_params}")
    logger.info(f"Best validation loss: {study.best_value:.4f}")

    return study