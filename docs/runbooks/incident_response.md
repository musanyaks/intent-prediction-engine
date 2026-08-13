# Incident Response Runbook

## Model Returning Nulls / Default Predictions

**Symptoms:** All predictions return "browse" with 0.25 confidence, or API returns 500 errors.

**Impact:** Personalization disabled, revenue impact ~15% conversion drop.

**Detection:** PagerDuty alert "inference_fallback_rate > 10%"

### Response Steps

1. **Check model health**
   ```bash
   kubectl get pods -n inference
   kubectl logs -l app=intent-inference --tail=100
   ```
   Look for: OOMKilled, model load failures, Redis connection errors.

2. **Verify feature store connectivity**
   ```bash
   python -c "from src.features.feature_store import check_health; check_health()"
   ```
   If Redis down: Feature store falls back to Snowflake (higher latency but functional).

3. **Check model artifact integrity**
   ```bash
   md5sum models/intent_classifier/v1.1.0/model.onnx
   # Compare with registered hash in MLflow
   ```

4. **Emergency rollback**
   ```bash
   # Rollback to previous model version
   kubectl set image deployment/inference      inference=intent-engine:v1.0.0
   ```

5. **Notify stakeholders**
   - Post in #ml-alerts Slack channel
   - Update incident status in PagerDuty
   - If >30 min outage, notify VP Engineering

## Feature Drift Detected

**Symptoms:** Drift monitor reports KL-divergence > 0.3 on key features.

**Impact:** Model accuracy degrading, but not immediately catastrophic.

### Response Steps

1. **Identify drifting features**
   ```bash
   python src/monitoring/drift_detector.py --report
   ```

2. **Check upstream data pipelines**
   - Did a new app version change event schema?
   - Did a marketing campaign shift user demographics?
   - Is there a data quality issue (nulls, outliers)?

3. **Short-term mitigation**
   - If drift is on non-critical features: Monitor for 24h
   - If drift is on critical features (e.g., cart_add events): Trigger emergency retraining

4. **Root cause analysis**
   - Review data lineage in MLflow
   - Check Kafka topic lag and event volume
   - Interview product team about recent changes

## Inference Latency Spike

**Symptoms:** p99 latency > 500ms, HPA scaling up rapidly.

### Response Steps

1. **Check infrastructure**
   ```bash
   kubectl top pods -n inference
   aws cloudwatch get-metric-statistics --metric-name CPUUtilization
   ```

2. **Identify bottleneck**
   - If CPU high: Model inference overloaded → Scale pods
   - If memory high: Feature cache evicted → Increase pod memory
   - If network high: Redis latency spike → Check Redis cluster health

3. **Emergency scaling**
   ```bash
   kubectl scale deployment inference --replicas=50
   ```

4. **Enable circuit breaker**
   ```bash
   # Temporarily reject non-critical traffic
   kubectl apply -f deployment/kubernetes/circuit-breaker.yaml
   ```

## False Positive Surge (Discount Abuse)

**Symptoms:** Support tickets spike: "I got a discount I didn't need." Marketing reports margin compression.

### Response Steps

1. **Disable dynamic discounting immediately**
   ```bash
   # Set discount eligibility to false for all users
   kubectl set env deployment/inference DISCOUNT_ENABLED=false
   ```

2. **Analyze affected sessions**
   ```sql
   SELECT * FROM prediction_logs
   WHERE recommended_action = 'discount_10pct'
   AND purchase_probability > 0.8  -- These users would have bought anyway
   AND timestamp > NOW() - INTERVAL '24' HOUR;
   ```

3. **Adjust business logic**
   - Raise discount threshold from purchase_prob < 0.6 to < 0.4
   - Add guardrail: Never discount if LTV tier = "vip"

4. **Retrain with corrected labels**
   - Identify sessions where discount was wasted
   - Add "discount_wasted" as a negative label in training data

## Contact Escalation

| Severity | Response Time | Escalation Path |
|----------|--------------|-----------------|
| P1 (Revenue down) | 5 min | On-call → Staff Engineer → VP Eng → CTO |
| P2 (Degraded) | 15 min | On-call → Team Lead |
| P3 (Warning) | 1 hour | On-call (async) |
