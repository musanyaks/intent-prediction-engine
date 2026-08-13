# Inference Pipeline Specification

## API Endpoint

```
POST /v1/predict
Content-Type: application/json
Authorization: Bearer <jwt_token>
```

### Request Body

```json
{
  "session_id": "sess_abc123def456",
  "user_id": "user_789xyz",
  "timestamp": "2026-08-12T14:32:00Z",
  "context": {
    "device_type": "mobile_ios",
    "app_version": "4.2.1",
    "traffic_source": "organic_search",
    "geo_country": "US",
    "geo_region": "CA"
  }
}
```

### Response Body

```json
{
  "session_id": "sess_abc123def456",
  "predictions": {
    "intent_class": {
      "value": "ready_to_buy",
      "confidence": 0.87,
      "probabilities": {
        "browse": 0.05,
        "compare": 0.08,
        "ready_to_buy": 0.87,
        "at_risk": 0.00
      }
    },
    "purchase_probability": {
      "value": 0.87,
      "calibrated": true
    },
    "ltv_30d": {
      "value": 245.50,
      "currency": "USD",
      "confidence_interval": [198.00, 293.00]
    },
    "churn_7d": {
      "value": 0.12
    }
  },
  "recommended_action": {
    "type": "show_urgency_banner",
    "parameters": {
      "message": "Only 3 left in stock",
      "discount_eligible": false
    }
  },
  "model_version": "intent_classifier/v1.1.0",
  "inference_time_ms": 142,
  "feature_freshness_ms": 23
}
```

## Latency Budget Breakdown

| Stage | Budget | Typical | p99 |
|-------|--------|---------|-----|
| Network (client → gateway) | — | 15ms | 50ms |
| Gateway processing | — | 5ms | 15ms |
| Feature retrieval (Redis) | <30ms | 8ms | 20ms |
| Feature retrieval (batch cache) | <50ms | 25ms | 45ms |
| Feature vectorization | <10ms | 3ms | 8ms |
| Model inference (ONNX) | <50ms | 35ms | 48ms |
| Post-processing / guardrails | <30ms | 10ms | 25ms |
| Network (response) | — | 15ms | 40ms |
| **Total** | **<200ms** | **116ms** | **196ms** |

## Fallback Strategy

If any stage exceeds its latency budget:

1. **Feature timeout:** Use cached batch features (may be stale up to 1 hour)
2. **Model timeout:** Return rule-based heuristic (last known intent + simple thresholds)
3. **Total timeout:** Return `"recommended_action": {"type": "default_experience"}`

All fallback events are logged with `fallback_reason` for monitoring.

## Caching Strategy

| Layer | TTL | Invalidation |
|-------|-----|--------------|
| Batch features (local) | 1 hour | Feature store update |
| User embedding | 6 hours | Daily recompute |
| Product embedding | 24 hours | Catalog update |
| Prediction (read-only) | None | Never cached |

## Error Handling

| HTTP Status | Scenario | Response |
|-------------|----------|----------|
| 200 | Success | Full prediction + action |
| 202 | Partial (fallback) | Prediction with `fallback_used: true` |
| 400 | Invalid request | Error details |
| 401 | Unauthorized | Auth error |
| 429 | Rate limited | Retry-After header |
| 500 | Internal error | Incident auto-created |

## Batch Prediction

For offline scoring (e.g., nightly cohort analysis):

```
POST /v1/predict/batch
Content-Type: application/json

{
  "sessions": [
    {"session_id": "sess_001", "user_id": "user_001"},
    {"session_id": "sess_002", "user_id": "user_002"}
  ],
  "synchronous": false
}
```

Returns `job_id` for async processing. Results written to S3.
