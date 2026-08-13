# ADR-003: Multi-Task vs. Single-Task Learning

## Status
Accepted

## Context
We need to predict four outputs: intent class, purchase probability, LTV, and churn probability. Should we train separate models or a single shared model?

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| **Multi-Task (Shared Backbone)** | Shared representations, faster inference (one forward pass), regularization via auxiliary tasks | Harder to tune, risk of negative transfer |
| **Single-Task (4 Separate Models)** | Independent optimization, easier debugging, no task interference | 4x inference cost, 4x training cost, no shared learning |
| **Ensemble of Specialists** | Best of both worlds, can swap individual models | Complex serving, higher latency |

## Decision
Use **Multi-Task Learning with shared Transformer backbone** and task-specific heads.

## Rationale
- Shared backbone reduces inference latency (one forward pass vs. four)
- Auxiliary tasks (intent, churn) improve main task (purchase) via regularization
- Empirical evidence: MTL improved AUC-PR by 3% vs. single-task baseline
- Uncertainty weighting (Kendall et al.) handles different task scales

## Architecture
```
Input → Transformer Encoder (shared) → [Intent Head, Purchase Head, LTV Head, Churn Head]
```

## Consequences
- **Positive:** Lower latency, better generalization, simpler deployment
- **Negative:** Harder hyperparameter tuning, if one task degrades all are affected

## Mitigations
- Gradient clipping and uncertainty weighting
- Task-specific learning rate multipliers
- Can extract single-task heads if negative transfer detected
