# Intent Prediction Engine (Project MERLIN)

Real-time customer intent prediction and dynamic experience optimization for e-commerce.

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B)](https://intent-demo.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)](https://pytorch.org/)

## What It Does

Predicts a shopper's intent in real-time during their browsing session and dynamically adapts the experience, pricing, and support intervention to maximize both conversion and margin.

|Output|Type|Business Use|
|-|-|-|
|`intent\_class`|browse / compare / ready-to-buy / at-risk|Route to different UX flows|
|`purchase\_prob`|Probability \[0,1]|Trigger urgency mechanisms|
|`ltv\_30d`|Continuous ($)|Decide discount depth|
|`churn\_7d`|Probability \[0,1]|Trigger retention offers|

**Latency constraint:** < 200ms p99 per prediction.

\---

## Live Demo

**Try it now:** [Hugging Face Spaces](https://huggingface.co/spaces/yourname/intent-prediction-demo)

The demo simulates a real shopping session with sliders for page views, cart adds, searches, and user history. It returns predicted intent, purchase probability, LTV estimate, and recommended actions in real-time.

\---

## Architecture

```
User Action
    |
    v
+------------------+
|  FastAPI         |  <- Pydantic validation, structured logging
|  /predict        |
+------------------+
    |
    v
+------------------+     +------------------+
|  Feature         |     |  Redis Cache     |
|  Retrieval       | <-> |  (hot features)  |
|  (< 30ms)        |     +------------------+
+------------------+
    |
    v
+------------------+
|  ONNX Runtime    |  <- Multi-task Transformer
|  Inference       |     Intent | Purchase | LTV | Churn
|  (< 50ms)        |
+------------------+
    |
    v
+------------------+
|  Business Logic  |  <- Dynamic discounting, guardrails
|  (< 100ms)       |
+------------------+
    |
    v
  Streamlit Dashboard
```

**Production stack:** Kafka, Snowflake, Feast, Redis, Kubernetes (EKS), MLflow.

\---

## Quick Start

### Local (Docker)

```bash
git clone https://github.com/yourname/intent-prediction-engine.git
cd intent-prediction-engine
docker compose up --build
```

* Dashboard: http://localhost:8501
* API docs: http://localhost:8000/docs

### Python Environment

```bash
poetry install --with dev
make test
make train
make serve
```

\---

## API Endpoints

|Endpoint|Method|Description|
|-|-|-|
|`/health`|GET|Service health \& model status|
|`/predict`|POST|Intent prediction for a user session|

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "user\_id": "usr\_abc123",
    "session\_id": "sess\_xyz789",
    "events": \[
      {"event\_type": "page\_view", "timestamp": "2026-08-01T10:00:00Z", "product\_id": null},
      {"event\_type": "search", "timestamp": "2026-08-01T10:00:15Z", "product\_id": null},
      {"event\_type": "cart\_add", "timestamp": "2026-08-01T10:02:00Z", "product\_id": "prod\_laptop\_001"}
    ],
    "user\_history\_orders": 3,
    "user\_avg\_order\_value": 450.0,
    "device\_type": "mobile"
  }'
```

**Example response:**

```json
{
  "user\_id": "usr\_abc123",
  "session\_id": "sess\_xyz789",
  "intent": {
    "class": "ready\_to\_buy",
    "confidence": 0.78,
    "probabilities": {
      "browse": 0.05,
      "compare": 0.12,
      "ready\_to\_buy": 0.78,
      "at\_risk": 0.05
    }
  },
  "purchase\_probability": 0.78,
  "ltv\_30d\_estimate": 350.0,
  "churn\_7d\_probability": 0.02,
  "recommended\_action": "show\_urgency\_messaging",
  "discount\_depth\_pct": 0.0,
  "explanation": "User shows strong purchase signals.",
  "inference\_time\_ms": 45.2
}
```

\---

## Project Structure

```
intent-prediction-engine/
├── config/              # Environment-specific configs (Hydra/OmegaConf)
├── src/
│   ├── data/            # ETL: Kafka, Snowflake, validation
│   ├── features/        # Real-time \& batch feature engineering
│   ├── models/          # PyTorch architecture, training, ONNX export
│   ├── inference/       # FastAPI predictor, feature retrieval, post-processing
│   ├── causal/          # Thompson Sampling bandit, ATE estimation
│   ├── monitoring/      # Drift detection, performance tracking, alerting
│   └── utils/           # Logging, config loader, constants
├── tests/               # Unit \& integration tests
├── notebooks/           # EDA \& experiment tracking
├── deployment/          # Docker, K8s, Terraform, Helm
├── docs/                # Architecture docs, ADRs, runbooks
└── scripts/             # Operational scripts
```

\---

## Model Architecture

Multi-task Transformer with uncertainty weighting (Kendall et al.):

```
Event Sequence -> Transformer Encoder -> Shared Representation
                                              |
          +------------------+------------------+------------------+
          |                  |                  |                  |
    Intent Classifier   Purchase Head      LTV Regressor     Churn Head
    (4-class)           (Sigmoid)          (ReLU)            (Sigmoid)
```

* **Sequence encoder:** Transformer over clickstream events
* **Context features:** User history + product embeddings
* **Uncertainty weighting:** Automatically balances task losses
* **Export:** ONNX for CPU-optimized inference

\---

## Business Impact (Simulated)

|Metric|Before|After|Annual Value|
|-|-|-|-|
|Conversion Rate|12.0%|14.5%|+$60M GMV|
|Discount Burn|18% revenue|12% revenue|+$28M margin|
|Customer Support Cost|$0.45/session|$0.32/session|-$6.5M cost|

\---

## Tech Stack

|Layer|Tools|
|-|-|
|**Modeling**|PyTorch, ONNX, scikit-learn, Optuna|
|**API**|FastAPI, Pydantic, Uvicorn|
|**Frontend**|Streamlit|
|**Data**|Kafka, Snowflake, pandas, Pandera|
|**Features**|Feast, Redis|
|**Training**|MLflow, SageMaker|
|**Deployment**|Docker, Kubernetes, Terraform, Helm|
|**Monitoring**|Prometheus, Grafana, PagerDuty|
|**Causal**|Custom Thompson Sampling, doubly robust ATE|

\---

## Testing

```bash
make test           # Unit tests
make test-integration  # Integration tests (needs Docker services)
make lint           # Black, isort, flake8, mypy
make format         # Auto-format code
```

\---

## Deployment

### Hugging Face Spaces (Free)

```bash
git clone https://huggingface.co/spaces/yourname/intent-prediction-demo
cd intent-prediction-demo
# Copy Dockerfile, api.py, app.py, supervisord.conf
git add . \&\& git commit -m "Deploy" \&\& git push
```

### Render (Free Tier)

1. Push repo to GitHub
2. Connect to Render (Docker environment)
3. Auto-deploys on every push

\---

## Documentation

* [System Architecture](docs/architecture/system_architecture.md)
* [Inference Pipeline](docs/architecture/inference_pipeline.md)
* [API Specification](docs/api/prediction_api_spec.md)
* [Incident Response](docs/runbooks/incident_response.md)
* [Retraining Procedure](docs/runbooks/retraining_procedure.md)

\---

## License

MIT

\---

Built by [Your Name](https://linkedin.com/in/yourprofile) | [Email](mailto:you@email.com)

