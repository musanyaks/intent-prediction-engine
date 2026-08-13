"""Model export to ONNX and TensorRT for production serving."""

from __future__ import annotations

import os

import torch
import torch.onnx
from src.models.architecture import MultiTaskIntentModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


def export_to_onnx(
    model: MultiTaskIntentModel,
    save_path: str,
    batch_size: int = 1,
    seq_len: int = 128,
    num_user_features: int = 32,
    num_product_features: int = 16,
) -> str:
    """
    Export PyTorch model to ONNX format for CPU inference.
    """
    model.eval()

    dummy_event_ids = torch.randint(0, 1000, (batch_size, seq_len))
    dummy_user_features = torch.randn(batch_size, num_user_features)
    dummy_product_features = torch.randn(batch_size, num_product_features)
    dummy_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    torch.onnx.export(
        model,
        (dummy_event_ids, dummy_user_features, dummy_product_features, dummy_mask),
        save_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["event_ids", "user_features", "product_features", "mask"],
        output_names=["intent_logits", "purchase_logit", "ltv", "churn_logit"],
        dynamic_axes={
            "event_ids": {0: "batch_size", 1: "seq_len"},
            "user_features": {0: "batch_size"},
            "product_features": {0: "batch_size"},
            "mask": {0: "batch_size", 1: "seq_len"},
            "intent_logits": {0: "batch_size"},
            "purchase_logit": {0: "batch_size"},
            "ltv": {0: "batch_size"},
            "churn_logit": {0: "batch_size"},
        },
    )

    logger.info(f"Model exported to ONNX: {save_path}")
    return save_path


def optimize_onnx_with_tensorrt(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
) -> str:
    """
    Build TensorRT engine from ONNX for GPU inference.
    Requires TensorRT and GPU environment.
    """
    try:
        import tensorrt as trt
    except ImportError:
        logger.warning("TensorRT not available, skipping optimization")
        return onnx_path

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                logger.error(parser.get_error(error))
            raise RuntimeError("ONNX parsing failed")

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    engine = builder.build_engine(network, config)

    with open(engine_path, "wb") as f:
        f.write(engine.serialize())

    logger.info(f"TensorRT engine saved to {engine_path}")
    return engine_path