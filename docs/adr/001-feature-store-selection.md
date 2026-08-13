# ADR-001: Feature Store Selection

## Status
Accepted

## Context
We need a centralized feature store to serve real-time and batch features to the inference service. Requirements:
- Low-latency online serving (<30ms p99)
- Point-in-time correctness for training data
- Feature versioning and lineage
- Integration with existing stack (Snowflake, Redis, Kubernetes)

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| **Feast** | Open source, Redis integration, Snowflake offline store, active community | Requires self-hosted infrastructure |
| **Tecton** | Managed service, advanced monitoring, enterprise support | Expensive ($$$), vendor lock-in |
| **Custom (Redis + Snowflake)** | Full control, no external dependency | Rebuild feature store primitives (versioning, lineage, point-in-time joins) |
| **AWS SageMaker Feature Store** | Native AWS integration, managed | Limited to AWS, expensive at scale |

## Decision
Use **Feast** with Redis online store and Snowflake offline store.

## Rationale
- Feast meets all latency and correctness requirements
- Open source avoids vendor lock-in and reduces cost
- Team has prior experience with Feast at previous company
- Can migrate to Tecton later if Feast becomes a bottleneck

## Consequences
- **Positive:** Cost-effective, flexible, community support
- **Negative:** Operational overhead (self-hosted), need to manage Redis cluster

## Alternatives Revisited
Re-evaluate Tecton if team grows beyond 10 ML engineers or feature count exceeds 10,000.
