# \# Intent Prediction Engine (Project MERLIN)

# 

# Real-time customer intent prediction and dynamic experience optimization for e-commerce.

# 

# \[!\[Live Demo](https://img.shields.io/badge/Live\_Demo-Streamlit-FF4B4B)](https://intent-demo.streamlit.app/)

# \[!\[Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)

# \[!\[FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)](https://fastapi.tiangolo.com/)

# \[!\[PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)](https://pytorch.org/)

# 

# \---

# 

# \## What It Does

# 

# Predicts a shopper's intent in real-time during their browsing session and dynamically adapts the experience, pricing, and support intervention to maximize both conversion and margin.

# 

# | Output | Type | Business Use |

# |--------|------|-------------|

# | `intent\_class` | browse / compare / ready-to-buy / at-risk | Route to different UX flows |

# | `purchase\_prob` | Probability \[0,1] | Trigger urgency mechanisms |

# | `ltv\_30d` | Continuous ($) | Decide discount depth |

# | `churn\_7d` | Probability \[0,1] | Trigger retention offers |

# 

# \*\*Latency constraint:\*\* < 200ms p99 per prediction.

# 

# \---

# 

# \## Live Demo

# 

# \*\*Try it now:\*\* \[Streamlit Cloud](https://intent-demo.streamlit.app/)

# 

# The demo simulates a real shopping session with sliders for page views, cart adds, searches, and user history. It returns predicted intent, purchase probability, LTV estimate, and recommended actions in real-time.

# 

# \---

# 

# \## Architecture

# 

# ```

# User Action

# &#x20;   |

# &#x20;   v

# +------------------+

# |  FastAPI         |  <- Pydantic validation, structured logging

# |  /predict        |

# +------------------+

# &#x20;   |

# &#x20;   v

# +------------------+     +------------------+

# |  Feature         |     |  Redis Cache     |

# |  Retrieval       | <-> |  (hot features)  |

# |  (< 30ms)        |     +------------------+

# +------------------+

# &#x20;   |

# &#x20;   v

# +------------------+

# |  ONNX Runtime    |  <- Multi-task Transformer

# |  Inference       |     Intent | Purchase | LTV | Churn

# |  (< 50ms)        |

# +------------------+

# &#x20;   |

# &#x20;   v

# +------------------+

# |  Business Logic  |  <- Dynamic discounting, guardrails

# |  (< 100ms)       |

# +------------------+

# &#x20;   |

# &#x20;   v

# &#x20; Streamlit Dashboard

# ```

# 

# \*\*Production stack:\*\* Kafka, Snowflake, Feast, Redis, Kubernetes (EKS), MLflow.

# 

# \---

# 

# \## Quick Start

# 

# \### Local (Docker)

# 

# ```bash

# git clone https://github.com/yourname/intent-prediction-engine.git

# cd intent-prediction-engine

# docker compose up --build

# ```

# 

# \- Dashboard: http://localhost:8501

# \- API docs: http://localhost:8000/docs

# 

# \### Python Environment

# 

# ```bash

# poetry install --with dev

# make test

# make train

# make serve

# ```

# 

# \---

# 

# \## API Endpoints

# 

# | Endpoint | Method | Description |

# |----------|--------|-------------|

# | `/health` | GET | Service health \& model status |

# | `/predict` | POST | Intent prediction for a user session |

# 

# \*\*Example request:\*\*

# 

# ```bash

# curl -X POST http://localhost:8000/predict \\

# &#x20; -H "Content-Type: application/json" \\

# &#x20; -d '{

# &#x20;   "user\_id": "usr\_abc123",

# &#x20;   "session\_id": "sess\_xyz789",

# &#x20;   "events": \[

# &#x20;     {"event\_type": "page\_view", "timestamp": "2026-08-01T10:00:00Z", "product\_id": null},

# &#x20;     {"event\_type": "search", "timestamp": "2026-08-01T10:00:15Z", "product\_id": null},

# &#x20;     {"event\_type": "cart\_add", "timestamp": "2026-08-01T10:02:00Z", "product\_id": "prod\_laptop\_001"}

# &#x20;   ],

# &#x20;   "user\_history\_orders": 3,

# &#x20;   "user\_avg\_order\_value": 450.0,

# &#x20;   "device\_type": "mobile"

# &#x20; }'

# ```

# 

# \*\*Example response:\*\*

# 

# ```json

# {

# &#x20; "user\_id": "usr\_abc123",

# &#x20; "session\_id": "sess\_xyz789",

# &#x20; "intent": {

# &#x20;   "class": "ready\_to\_buy",

# &#x20;   "confidence": 0.78,

# &#x20;   "probabilities": {

# &#x20;     "browse": 0.05,

# &#x20;     "compare": 0.12,

# &#x20;     "ready\_to\_buy": 0.78,

# &#x20;     "at\_risk": 0.05

# &#x20;   }

# &#x20; },

# &#x20; "purchase\_probability": 0.78,

# &#x20; "ltv\_30d\_estimate": 350.0,

# &#x20; "churn\_7d\_probability": 0.02,

# &#x20; "recommended\_action": "show\_urgency\_messaging",

# &#x20; "discount\_depth\_pct": 0.0,

# &#x20; "explanation": "User shows strong purchase signals.",

# &#x20; "inference\_time\_ms": 45.2

# }

# ```

# 

# \---

# 

# \## Project Structure

# 

# ```

# intent-prediction-engine/

# ├── config/              # Environment-specific configs (Hydra/OmegaConf)

# ├── src/

# │   ├── data/            # ETL: Kafka, Snowflake, validation

# │   ├── features/        # Real-time \& batch feature engineering

# │   ├── models/          # PyTorch architecture, training, ONNX export

# │   ├── inference/       # FastAPI predictor, feature retrieval, post-processing

# │   ├── causal/          # Thompson Sampling bandit, ATE estimation

# │   ├── monitoring/      # Drift detection, performance tracking, alerting

# │   └── utils/           # Logging, config loader, constants

# ├── tests/               # Unit \& integration tests

# ├── notebooks/           # EDA \& experiment tracking

# ├── deployment/          # Docker, K8s, Terraform, Helm

# ├── docs/                # Architecture docs, ADRs, runbooks

# └── scripts/             # Operational scripts

# ```

# 

# \---

# 

# \## Model Architecture

# 

# Multi-task Transformer with uncertainty weighting (Kendall et al.):

# 

# ```

# Event Sequence -> Transformer Encoder -> Shared Representation

# &#x20;                                             |

# &#x20;         +------------------+------------------+------------------+

# &#x20;         |                  |                  |                  |

# &#x20;   Intent Classifier   Purchase Head      LTV Regressor     Churn Head

# &#x20;   (4-class)           (Sigmoid)          (ReLU)            (Sigmoid)

# ```

# 

# \- \*\*Sequence encoder:\*\* Transformer over clickstream events

# \- \*\*Context features:\*\* User history + product embeddings

# \- \*\*Uncertainty weighting:\*\* Automatically balances task losses

# \- \*\*Export:\*\* ONNX for CPU-optimized inference

# 

# \---

# 

# \## Business Impact (Simulated)

# 

# | Metric | Before | After | Annual Value |

# |--------|--------|-------|-------------|

# | Conversion Rate | 12.0% | 14.5% | +$60M GMV |

# | Discount Burn | 18% revenue | 12% revenue | +$28M margin |

# | Customer Support Cost | $0.45/session | $0.32/session | -$6.5M cost |

# 

# \---

# 

# \## Tech Stack

# 

# | Layer | Tools |

# |-------|-------|

# | \*\*Modeling\*\* | PyTorch, ONNX, scikit-learn, Optuna |

# | \*\*API\*\* | FastAPI, Pydantic, Uvicorn |

# | \*\*Frontend\*\* | Streamlit |

# | \*\*Data\*\* | Kafka, Snowflake, pandas, Pandera |

# | \*\*Features\*\* | Feast, Redis |

# | \*\*Training\*\* | MLflow, SageMaker |

# | \*\*Deployment\*\* | Docker, Kubernetes, Terraform, Helm |

# | \*\*Monitoring\*\* | Prometheus, Grafana, PagerDuty |

# | \*\*Causal\*\* | Custom Thompson Sampling, doubly robust ATE |

# 

# \---

# 

# \## Testing

# 

# ```bash

# make test           # Unit tests

# make test-integration  # Integration tests (needs Docker services)

# make lint           # Black, isort, flake8, mypy

# make format         # Auto-format code

# ```

# 

# \---

# 

# \## Deployment

# 

# \### Streamlit Community Cloud (Free)

# 

# 1\. Push `streamlit\_app.py` + `requirements.txt` to a public GitHub repo

# 2\. Go to \[streamlit.io/cloud](https://streamlit.io/cloud) and connect your repo

# 3\. Deploy — auto-updates on every push

# 

# \*\*Live demo:\*\* https://intent-demo.streamlit.app/

# 

# \### Render (Free Tier)

# 

# 1\. Push repo to GitHub

# 2\. Connect to Render (Python environment)

# 3\. Auto-deploys on every push

# 

# \---

# 

# \## Documentation

# 

# \- \[System Architecture](docs/architecture/system\_architecture.md)

# \- \[Inference Pipeline](docs/architecture/inference\_pipeline.md)

# \- \[API Specification](docs/api/prediction\_api\_spec.md)

# \- \[Incident Response](docs/runbooks/incident\_response.md)

# \- \[Retraining Procedure](docs/runbooks/retraining\_procedure.md)

# 

# \---

# 

# \## License

# 

# MIT

# 

# \---

# 

# Built by \[Your Name](https://linkedin.com/in/yourprofile) | \[Email](mailto:you@email.com)



