# System Architecture — MERLIN Intent Prediction Engine

## Overview

MERLIN is a real-time multi-task prediction system that infers user intent during live e-commerce sessions. It powers dynamic UX adaptation, personalized pricing, and proactive support intervention.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Web App    │  │  Mobile App │  │  Mobile Web │  │  Partner API        │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
└─────────┼────────────────┼────────────────┼────────────────────┼────────────┘
          │                │                │                    │
          └────────────────┴────────────────┴────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INGESTION LAYER                                  │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────────┐   │
│  │  Kafka (Events) │     │  Segment        │     │  Webhooks (3rd Party)│   │
│  │  120K evt/sec   │     │  80M/day        │     │  IP Reputation, etc  │   │
│  └────────┬────────┘     └────────┬────────┘     └──────────┬──────────┘   │
└───────────┼─────────────────────┼─────────────────────────┼────────────────┘
            │                     │                         │
            └─────────────────────┴─────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FEATURE ENGINEERING LAYER                          │
│  ┌─────────────────────────┐     ┌─────────────────────────────────────┐    │
│  │  Real-Time (Flink)      │     │  Batch (Spark / dbt)              │    │
│  │  • Session aggregations │     │  • Historical user features       │    │
│  │  • Event velocity       │     │  • Product embeddings             │    │
│  │  • Mouse patterns       │     │  • Cohort-level statistics        │    │
│  └───────────┬─────────────┘     └──────────────┬────────────────────┘    │
│              │                                  │                         │
│              └────────────────┬─────────────────┘                         │
│                               ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │  FEATURE STORE (Feast / Redis)                                     │ │
│  │  • Online store: <10ms read latency                                 │ │
│  │  • Offline store: point-in-time correct training data               │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE LAYER                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  API Gateway (Kong / AWS API Gateway)                               │   │
│  │  • Rate limiting: 10K req/sec per API key                           │   │
│  │  • Authentication: JWT validation                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Prediction Service (Kubernetes / Triton)                           │   │
│  │  • ONNX Runtime for CPU inference                                   │   │
│  │  • TensorRT for GPU batch inference                                 │   │
│  │  • p99 latency: <200ms end-to-end                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Business Logic Layer                                               │   │
│  │  • Discount policy engine                                             │   │
│  │  • Inventory / margin guardrails                                      │   │
│  │  • A/B test variant assignment                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FEEDBACK & LEARNING LAYER                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│  │  Event Logging  │  │  MLflow         │  │  Causal Bandit              │ │
│  │  (Kafka → S3)   │  │  (Model Registry)│  │  (Thompson Sampling)        │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Monitoring & Alerting (WhyLabs / Grafana / PagerDuty)              │   │
│  │  • Data drift detection                                             │   │
│  │  • Prediction distribution tracking                                 │   │
│  │  • Business KPI dashboards                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

| Component | Technology | Responsibility | SLA |
|-----------|-----------|----------------|-----|
| Event Ingestion | Kafka + Flink | Real-time event streaming | 99.99% uptime |
| Feature Store | Feast + Redis | Low-latency feature serving | p99 < 10ms read |
| Model Serving | Triton + K8s | ONNX model inference | p99 < 50ms |
| API Gateway | Kong | Auth, rate limiting, routing | 99.95% uptime |
| Training | SageMaker + Spark | Model training at scale | Daily batch |
| Monitoring | WhyLabs + Grafana | Drift, performance, alerts | Real-time |

## Data Flow

1. **User Action** → Web/Mobile app captures clickstream event
2. **Event Stream** → Kafka receives event within 50ms
3. **Feature Update** → Flink aggregates and updates Redis feature store
4. **Prediction Request** → Frontend calls prediction API with user/session ID
5. **Feature Retrieval** → Service fetches real-time + batch features
6. **Model Inference** → ONNX model returns 4-task predictions
7. **Business Logic** → Policy engine applies guardrails and selects action
8. **Response** → Frontend receives intent + recommended action
9. **Feedback Loop** → Outcome logged to Kafka for model retraining

## Scaling Considerations

- **Horizontal scaling:** Triton pods auto-scale on GPU utilization (target 70%)
- **Feature store:** Redis Cluster with 6 shards, replication factor 2
- **Kafka:** 24 partitions per topic, 3x replication
- **Cold start:** Model weights cached in memory; no disk reads on pod startup

## Security

- All data encrypted at rest (AES-256) and in transit (TLS 1.3)
- PII hashed with HMAC-SHA256 before feature storage
- Model artifacts signed and verified on deployment
- API keys scoped per environment (dev/staging/prod)
