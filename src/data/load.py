"""Data loading: write to feature store, warehouse, and serving caches."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import redis
from feast import FeatureStore
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureStoreLoader:
    """Load features into Feast feature store for real-time serving."""

    def __init__(self) -> None:
        cfg = load_config().feature_store
        self.store = FeatureStore(repo_path=cfg.repo_path)
        logger.info(f"FeatureStore initialized at {cfg.repo_path}")

    def push_realtime_features(
        self, df: pd.DataFrame, feature_view: str = "user_session_features"
    ) -> None:
        """Push real-time computed features to online store."""
        self.store.push(feature_view, df)
        logger.info(f"Pushed {len(df)} rows to {feature_view}")

    def materialize_batch_features(self, start_date: str, end_date: str) -> None:
        """Materialize batch features from offline to online store."""
        self.store.materialize(start_date, end_date)
        logger.info(f"Materialized batch features from {start_date} to {end_date}")


class RedisLoader:
    """Load hot features into Redis for sub-millisecond serving."""

    def __init__(self) -> None:
        cfg = load_config().redis
        self.client = redis.Redis(
            host=cfg.host,
            port=cfg.port,
            db=cfg.db,
            decode_responses=True,
        )
        logger.info(f"Redis client connected to {cfg.host}:{cfg.port}")

    def cache_user_features(self, user_id: str, features: dict[str, Any], ttl: int = 3600) -> None:
        """Cache user features with TTL."""
        key = f"user_features:{user_id}"
        self.client.setex(key, ttl, json.dumps(features))
        logger.debug(f"Cached features for user {user_id}")

    def get_cached_features(self, user_id: str) -> dict[str, Any] | None:
        """Retrieve cached user features."""
        key = f"user_features:{user_id}"
        raw = self.client.get(key)
        return json.loads(raw) if raw else None


class SnowflakeLoader:
    """Load processed data back to Snowflake for analytics."""

    def __init__(self) -> None:
        from src.data.extract import SnowflakeExtractor

        self.extractor = SnowflakeExtractor()

    def write_predictions(
        self, df: pd.DataFrame, table: str = "model_predictions"
    ) -> None:
        """Write model predictions to warehouse for BI and audit."""
        from snowflake.connector.pandas_tools import write_pandas

        success, num_chunks, num_rows, _ = write_pandas(
            self.extractor.conn, df, table, auto_create_table=True, overwrite=False
        )
        logger.info(f"Wrote {num_rows} predictions to {table}")