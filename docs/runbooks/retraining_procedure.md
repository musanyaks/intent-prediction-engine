# Model Retraining Procedure

## Trigger Conditions

Retraining is triggered by ANY of the following:

1. **Scheduled:** Every Sunday at 02:00 UTC (weekly retraining)
2. **Drift detected:** KL-divergence > 0.3 on >3 features for >6 hours
3. **Performance degradation:** AUC-PR drops >5% on holdout set for 3 consecutive days
4. **Business request:** New product launch, major UX change, or regulatory requirement
5. **Manual trigger:** Data scientist initiates experiment

## Pre-Training Checklist

- [ ] Verify data freshness: Last transaction data is within 24h
- [ ] Check label completeness: >95% of sessions from 30 days ago have outcomes
- [ ] Validate feature store: All batch features computed successfully
- [ ] Review experiment tracking: MLflow server accessible
- [ ] Confirm GPU availability: SageMaker quota not exceeded
- [ ] Notify stakeholders: Post in #ml-training channel

## Training Pipeline

### Step 1: Data Preparation
```bash
# Extract and validate training data
make extract-training-data   START_DATE=2026-01-01   END_DATE=2026-07-31

# Run data quality checks
python src/data/validation.py   --input data/processed/training_v1.2.0.parquet   --expectations config/data_expectations.yaml
```

### Step 2: Feature Engineering
```bash
# Compute embeddings and sequences
python src/features/embeddings.py   --input data/processed/training_v1.2.0.parquet   --output data/processed/training_with_features_v1.2.0.parquet

# Register features to feature store
feast apply
```

### Step 3: Model Training
```bash
# Launch SageMaker training job
python src/models/train.py   --config config/models/intent_classifier.yaml   --data data/processed/training_with_features_v1.2.0.parquet   --experiment-name "intent_classifier_v1.2.0"   --tracking-uri "${MLFLOW_TRACKING_URI}"
```

### Step 4: Evaluation
```bash
# Run offline evaluation
python src/models/evaluate.py   --model-uri "${MLFLOW_MODEL_URI}"   --test-data data/processed/test_v1.2.0.parquet   --output reports/evaluation_v1.2.0.html
```

**Evaluation criteria:**
- Macro-F1 >= 0.72 (intent classifier)
- AUC-PR >= 0.42 (purchase probability)
- MAPE <= 0.35 (LTV regressor)
- Calibration slope in [0.85, 1.15]
- No per-class recall < 0.65
- Bias audit: disparate impact ratio > 0.8 for all protected groups

### Step 5: Model Registration
```bash
# Register to MLflow Model Registry
mlflow models register   --model-uri "runs:/${RUN_ID}/model"   --name "intent_classifier"

# Transition to Staging
mlflow models transition   --model-name "intent_classifier"   --version "${VERSION}"   --stage "Staging"
```

### Step 6: Shadow Deployment
```bash
# Deploy to staging, shadow mode (0% traffic, logging only)
make deploy-staging VERSION=v1.2.0

# Monitor for 48 hours
# Check: latency, error rate, prediction distribution vs. production
```

### Step 7: Canary A/B Test
```bash
# Route 5% traffic to new model
make deploy-canary VERSION=v1.2.0 TRAFFIC=5

# Monitor for 1 week
# Primary metric: conversion rate (lift >= 2%)
# Guardrail metrics: false positive rate, latency, error rate
```

### Step 8: Full Rollout
```bash
# Ramp to 25% → 50% → 100% over 3 days
make deploy-production VERSION=v1.2.0

# Update active model version in config
# Archive old model artifacts (keep last 3 versions)
```

## Post-Training

- [ ] Update model card in `models/intent_classifier/v1.2.0/model_card.md`
- [ ] Document training parameters and data lineage in MLflow
- [ ] Notify stakeholders of performance changes
- [ ] Schedule model review meeting (within 1 week)
- [ ] Update runbook with any new failure modes observed

## Rollback Plan

If issues detected during canary:
```bash
# Immediate rollback
make rollback-production VERSION=v1.1.0

# Preserve canary model for debugging
mlflow models transition   --model-name "intent_classifier"   --version "${CANARY_VERSION}"   --stage "Archived"
```
