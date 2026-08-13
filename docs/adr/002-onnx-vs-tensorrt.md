# ADR-002: Model Serving Format

## Status
Accepted

## Context
The trained PyTorch model must be exported to a production-serving format. Requirements:
- <50ms inference latency on CPU (p99)
- Batch size 1 (real-time) and 32 (batch scoring)
- Easy deployment to Kubernetes without GPU dependency

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| **ONNX Runtime** | CPU-optimized, cross-platform, easy deployment, supports both PyTorch and TensorFlow | Slightly slower than TensorRT on GPU |
| **TensorRT** | Fastest GPU inference, NVIDIA optimized | Requires GPU nodes, harder to debug, NVIDIA-only |
| **TorchScript** | Native PyTorch, no conversion needed | Slower than ONNX, limited dynamic shapes |
| **TensorFlow Serving** | Mature ecosystem, batching | Requires TensorFlow model (not PyTorch) |

## Decision
Use **ONNX Runtime** for CPU inference. Use **TensorRT** only for batch scoring jobs on GPU nodes.

## Rationale
- ONNX Runtime achieves <35ms on CPU for our model size
- CPU-only pods are cheaper and easier to scale
- Fallback to TorchScript if ONNX conversion fails for a specific model architecture

## Consequences
- **Positive:** Cost-effective scaling, fast CPU inference, portable
- **Negative:** ONNX conversion adds build step, some PyTorch ops not supported

## Mitigations
- Automated ONNX conversion test in CI pipeline
- Fallback to TorchScript if ONNX export fails
- Monitor ONNX Runtime version compatibility
