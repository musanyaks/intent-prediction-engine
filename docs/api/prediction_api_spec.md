# Prediction API Specification

## Base URL

| Environment | URL |
|-------------|-----|
| Development | `http://localhost:8000` |
| Staging | `https://api-staging.intent-engine.io` |
| Production | `https://api.intent-engine.io` |

## Authentication

All requests require an API key in the header:
```
X-API-Key: your_api_key_here
```

API keys are scoped per client and rate-limited.

## Endpoints

### POST /v1/predict

Predict customer intent and recommended actions for a session event.

#### Request

```json
{
  "session_id": "string (required)",
  "user_id": "string (required)",
  "event": {
    "type": "string (enum: page_view, product_view, search, cart_add, checkout_start, checkout_complete, scroll, click)",
    "timestamp": "string (ISO 8601)",
    "product_id": "string (optional)",
    "category": "string (optional)",
    "price": "number (optional)",
    "page_url": "string",
    "device": {
      "type": "string (enum: desktop, mobile, tablet)",
      "os": "string",
      "browser": "string"
    }
  },
  "context": {
    "traffic_source": "string",
    "campaign_id": "string (optional)",
    "geo_country": "string (ISO 3166-1 alpha-2)",
    "geo_region": "string (optional)"
  }
}
```

#### Response (200 OK)

```json
{
  "prediction_id": "string",
  "timestamp": "string (ISO 8601)",
  "model_version": "string",
  "features_used": "integer",
  "latency_ms": "integer",
  "predictions": {
    "intent": {
      "class": "string (enum: browse, compare, ready_to_buy, at_risk)",
      "confidence": "number (0-1)",
      "probabilities": {
        "browse": "number",
        "compare": "number",
        "ready_to_buy": "number",
        "at_risk": "number"
      }
    },
    "purchase_probability": "number (0-1)",
    "ltv_30d_estimate": "number",
    "churn_7d_probability": "number (0-1)"
  },
  "recommended_action": {
    "type": "string (enum: none, show_urgency_banner, offer_discount, show_compare_tool, trigger_chat, human_review)",
    "discount_eligible": "boolean",
    "discount_depth": "integer (0, 5, 10, 15)",
    "human_review_required": "boolean"
  },
  "explanation": {
    "top_features": [
      {
        "feature": "string",
        "importance": "number"
      }
    ]
  },
  "fallback_used": "boolean"
}
```

#### Error Responses

| Status | Code | Description |
|--------|------|-------------|
| 400 | INVALID_REQUEST | Missing required fields |
| 401 | UNAUTHORIZED | Invalid API key |
| 429 | RATE_LIMITED | Too many requests |
| 500 | INTERNAL_ERROR | Model inference failed |
| 503 | SERVICE_UNAVAILABLE | Feature store unreachable |

### GET /v1/health

Health check endpoint for load balancers.

#### Response (200 OK)
```json
{
  "status": "healthy",
  "model_version": "v1.1.0",
  "uptime_seconds": 3600,
  "last_prediction": "2026-08-12T14:30:00Z"
}
```

### GET /v1/metrics

Prometheus-compatible metrics endpoint.

#### Response (200 OK)
```
# HELP inference_latency_ms Prediction latency
# TYPE inference_latency_ms histogram
inference_latency_ms_bucket{le="50"} 1024
inference_latency_ms_bucket{le="100"} 8934
inference_latency_ms_bucket{le="200"} 9876

# HELP predictions_total Total predictions
# TYPE predictions_total counter
predictions_total{intent="ready_to_buy"} 5432

# HELP fallback_total Fallback predictions
# TYPE fallback_total counter
fallback_total{reason="timeout"} 23
```

## Rate Limits

| Tier | Requests/min | Burst |
|------|-------------|-------|
| Free | 100 | 10 |
| Standard | 10,000 | 100 |
| Enterprise | 100,000 | 1,000 |

Rate limit headers included in all responses:
```
X-RateLimit-Limit: 10000
X-RateLimit-Remaining: 9999
X-RateLimit-Reset: 1691846400
```

## SDK Examples

### Python
```python
import requests

response = requests.post(
    "https://api.intent-engine.io/v1/predict",
    headers={"X-API-Key": "your_key"},
    json={
        "session_id": "sess_123",
        "user_id": "user_456",
        "event": {
            "type": "product_view",
            "timestamp": "2026-08-12T14:32:01Z",
            "product_id": "prod_789"
        },
        "context": {"traffic_source": "organic_search"}
    }
)
result = response.json()
print(result["predictions"]["intent"]["class"])
```

### JavaScript
```javascript
const response = await fetch('https://api.intent-engine.io/v1/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': 'your_key'
  },
  body: JSON.stringify({
    session_id: 'sess_123',
    user_id: 'user_456',
    event: { type: 'product_view', product_id: 'prod_789' }
  })
});
const result = await response.json();
```
