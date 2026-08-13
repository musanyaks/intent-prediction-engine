# Rollback Checklist

## When to Rollback

Rollback IMMEDIATELY if:
- [ ] Error rate > 1% for >5 minutes
- [ ] p99 latency > 500ms for >10 minutes
- [ ] Conversion rate drops >3% vs. baseline (confirmed by A/B test)
- [ ] Support tickets spike >200% (indicating false positives)
- [ ] PII or fairness violation detected
- [ ] Regulatory audit flags model decision

## Rollback Steps

### 1. Stop Traffic to New Model
```bash
# Revert Kubernetes deployment
kubectl rollout undo deployment/inference

# Or set specific previous version
kubectl set image deployment/inference   inference=intent-engine:v1.1.0

# Verify rollback
kubectl get pods -n inference
kubectl logs -l app=intent-inference --tail=20
```

### 2. Verify Fallback Model
```bash
# Test health endpoint
curl -f http://inference-service/health

# Test prediction endpoint
curl -X POST http://inference-service/v1/predict   -H "Content-Type: application/json"   -d @tests/fixtures/sample_session.json
```

### 3. Preserve Evidence
```bash
# Export logs for post-mortem
kubectl logs -l app=intent-inference --since=1h > /tmp/rollback_logs_$(date +%Y%m%d_%H%M).log

# Export metrics
aws cloudwatch get-metric-data   --metric-data-queries file://queries.json   --start-time $(date -d '1 hour ago' -u +%Y-%m-%dT%H:%M:%SZ)   --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) > /tmp/rollback_metrics.json
```

### 4. Notify Stakeholders
- Post in #ml-alerts: "Rollback to v1.1.0 completed at [timestamp]"
- Update PagerDuty incident
- If P1: Call VP Engineering within 15 minutes

### 5. Post-Rollback Monitoring
- [ ] Monitor error rate for 30 minutes
- [ ] Verify conversion rate returns to baseline
- [ ] Check feature store health
- [ ] Confirm no data loss in prediction logs

### 6. Post-Mortem (within 24 hours)
- [ ] Schedule post-mortem meeting
- [ ] Document root cause in `docs/incidents/YYYY-MM-DD_rollback.md`
- [ ] Create JIRA tickets for fixes
- [ ] Update runbooks if new failure mode discovered

## Rollback Verification

After rollback, confirm:

```python
# Quick validation script
from src.inference.predictor import IntentPredictor

predictor = IntentPredictor(model_version="v1.1.0")
result = predictor.predict(session_id="test_rollback")

assert result["model_version"] == "v1.1.0"
assert result["fallback_used"] == False
assert result["predictions"]["intent"]["confidence"] > 0.5
print("Rollback verified successfully")
```

## Communication Template

```
Subject: [RESOLVED] Intent Prediction Engine Rollback

Impact: Dynamic personalization was degraded from [time] to [time].
Root Cause: [Brief description]
Action Taken: Rolled back to model v1.1.0 at [time].
Current Status: All systems nominal. Monitoring closely.
Next Steps: Post-mortem scheduled for [time].
```
