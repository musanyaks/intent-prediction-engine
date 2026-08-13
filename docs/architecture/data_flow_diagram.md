# Data Flow Diagram

## Event Ingestion Flow

```
User Browser/App
       │
       │ HTTPS POST /event
       ▼
┌──────────────┐
│  CDN / WAF   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  API Gateway │
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌──────────────┐
│  Kafka       │────▶│  Flink       │
│  (raw-events)│     │  (aggregate) │
└──────────────┘     └──────┬───────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌─────────┐   ┌─────────┐   ┌─────────┐
        │  Redis  │   │S3 (raw) │   │Snowflake│
        │(online) │   │(archive)│   │(warehouse)
        └─────────┘   └─────────┘   └─────────┘
```

## Prediction Request Flow

```
Frontend
   │
   │ GET /predict?session_id=abc123
   ▼
┌──────────────┐
│ API Gateway  │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────┐
│  Prediction Service                      │
│  ┌─────────────┐  ┌───────────────────┐ │
│  │Feature Fetch│  │  Batch Feature    │ │
│  │  (Redis)    │  │  (Feature Store)  │ │
│  │  <10ms      │  │  <30ms            │ │
│  └──────┬──────┘  └─────────┬─────────┘ │
│         │                   │           │
│         └─────────┬─────────┘           │
│                   ▼                     │
│         ┌───────────────┐              │
│         │  Feature Vector │              │
│         │  (concatenate)  │              │
│         └───────┬───────┘              │
│                 ▼                       │
│         ┌───────────────┐              │
│         │  ONNX Runtime   │              │
│         │  (inference)    │              │
│         │  <50ms          │              │
│         └───────┬───────┘              │
│                 ▼                       │
│         ┌───────────────┐              │
│         │  Post-process │              │
│         │  (guardrails) │              │
│         └───────┬───────┘              │
└─────────────────┼──────────────────────┘
                  │
                  ▼
            Response JSON
```

## Training Data Flow

```
Snowflake Warehouse
       │
       │ SQL query (point-in-time join)
       ▼
┌──────────────┐
│  Spark       │
│  (feature    │
│   engineering)│
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  S3          │
│  (parquet)   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  SageMaker   │
│  Training    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  MLflow      │
│  (registry)  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Model       │
│  Promotion   │
│  (staging    │
│   → prod)    │
└──────────────┘
```

## Feedback Loop

```
User Session
    │
    ├─► Purchase? ──► Kafka (outcome-events)
    │
    ├─► Churn? ─────► Kafka (outcome-events)
    │
    └─► Support? ───► Kafka (outcome-events)
                        │
                        ▼
                ┌──────────────┐
                │  S3 (daily   │
                │   batch)     │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │  Label       │
                │  Generation  │
                │  (Spark)     │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │  Training    │
                │  Dataset     │
                │  (S3)        │
                └──────────────┘
```
