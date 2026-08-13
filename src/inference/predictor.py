"""Main prediction class for production serving."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import torch
from src.inference.feature_retrieval import FeatureRetriever
from src.inference.postprocess import PostProcessor
from src.models.architecture import MultiTaskIntentModel
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class IntentPredictor:
    """
    Production predictor supporting both PyTorch and ONNX backends.
    """

    def __init__(
        self,
        model_path: str | None = None,
        use_onnx: bool = True,
        device: str = "cpu",
    ) -> None:
        self.config = load_config().inference
        self.use_onnx = use_onnx
        self.device = device
        self.feature_retriever = FeatureRetriever()
        self.post_processor = PostProcessor()

        if use_onnx and model_path and model_path.endswith(".onnx"):
            self.session = ort.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            logger.info(f"ONNX predictor loaded from {model_path}")
        elif model_path:
            self.model = MultiTaskIntentModel(
                event_vocab_size=self.config.event_vocab_size,
                num_intent_classes=4,
            )
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.model.to(device)
            logger.info(f"PyTorch predictor loaded from {model_path}")
        else:
            raise ValueError("Model path required")

    def predict(
        self,
        user_id: str,
        session_id: str,
        current_event: dict | None = None,
    ) -> dict:
        """
        Predict intent and scores for a user session.
        Returns structured output for the frontend/business logic layer.
        """
        # Retrieve features
        features = self.feature_retriever.get_features(user_id, session_id)

        # Prepare model inputs
        event_ids = features["event_ids"].unsqueeze(0)  # (1, seq_len)
        user_feats = features["user_features"].unsqueeze(0)  # (1, num_user_features)
        product_feats = features["product_features"].unsqueeze(0)  # (1, num_product_features)
        mask = features.get("mask")
        if mask is not None:
            mask = mask.unsqueeze(0)

        # Run inference
        if self.use_onnx:
            outputs = self.session.run(
                None,
                {
                    "event_ids": event_ids.numpy(),
                    "user_features": user_feats.numpy(),
                    "product_features": product_feats.numpy(),
                    "mask": mask.numpy() if mask is not None else np.zeros_like(event_ids.numpy(), dtype=bool),
                },
            )
            result = {
                "intent_logits": torch.tensor(outputs[0]),
                "purchase_logit": torch.tensor(outputs[1]),
                "ltv": torch.tensor(outputs[2]),
                "churn_logit": torch.tensor(outputs[3]),
            }
        else:
            with torch.no_grad():
                result = self.model(
                    event_ids.to(self.device),
                    user_feats.to(self.device),
                    product_feats.to(self.device),
                    mask.to(self.device) if mask is not None else None,
                )

        # Post-process
        return self.post_processor.process(result, user_id, session_id)