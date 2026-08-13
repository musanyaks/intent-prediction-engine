"""Feast feature store client wrapper."""

from __future__ import annotations

from typing import Any

import pandas as pd
from feast import FeatureStore
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureStoreClient:
    """Unified interface for online and offline feature retrieval."""

    def __init__(self) -> None:
        cfg = load_config().feature_store
        self.store = FeatureStore(repo_path=cfg.repo_path)
        logger.info(f"FeatureStoreClient initialized at {cfg.repo_path}")

    def get_online_features(
        self,
        entity_rows: list[dict[str, Any]],
        feature_refs: list[str],
    ) -> pd.DataFrame:
        """
        Fetch real-time features from online store.
        entity_rows: [{"user_id": "abc", "session_id": "xyz"}]
        feature_refs: ["user_session_features:session_duration_sec", ...]
        """
        features = self.store.get_online_features(
            features=feature_refs,
            entity_rows=entity_rows,
        ).to_df()
        logger.info(f"Retrieved online features: {len(features)} rows, {len(feature_refs)} features")
        return features

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        feature_refs: list[str],
    ) -> pd.DataFrame:
        """
        Fetch point-in-time correct features for training.
        entity_df must have 'event_timestamp' column.
        """
        features = self.store.get_historical_features(
            entity_df=entity_df,
            features=feature_refs,
        ).to_df()
        logger.info(f"Retrieved historical features: {len(features)} rows")
        return features

    def push_realtime(self, df: pd.DataFrame, feature_view: str) -> None:
        """Push streaming feature computations to online store."""
        self.store.push(feature_view, df)
        logger.info(f"Pushed {len(df)} rows to {feature_view}")