"""Real-time feature engineering from streaming clickstream events."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_session_features(events: pd.DataFrame) -> pd.DataFrame:
    """
    Compute real-time session-level features from a window of events.
    Expected to run in Flink/Spark Streaming with 60-second windows.
    """
    features = (
        events.groupby("session_id")
        .agg(
            session_duration_sec=("event_timestamp", lambda x: (x.max() - x.min()).total_seconds()),
            event_count=("event_id", "count"),
            page_view_count=("event_type", lambda x: (x == "page_view").sum()),
            cart_add_count=("event_type", lambda x: (x == "cart_add").sum()),
            search_count=("event_type", lambda x: (x == "search").sum()),
            unique_products=("product_id", "nunique"),
            unique_categories=("product_category", "nunique"),
        )
        .reset_index()
    )

    # Velocity features
    features["events_per_minute"] = features["event_count"] / (features["session_duration_sec"] / 60 + 1)
    features["product_view_velocity"] = features["unique_products"] / (features["session_duration_sec"] / 60 + 1)

    # Behavioral patterns
    features["search_to_view_ratio"] = features["search_count"] / (features["page_view_count"] + 1)
    features["cart_add_intensity"] = features["cart_add_count"] / (features["event_count"] + 1)

    logger.info(f"Computed real-time features for {len(features)} sessions")
    return features


def compute_mouse_features(events: pd.DataFrame) -> pd.DataFrame:
    """
    Compute behavioral biometrics from mouse/touch events.
    Requires raw x, y, timestamp columns in events.
    """
    if not {"mouse_x", "mouse_y", "event_timestamp"}.issubset(events.columns):
        logger.warning("Mouse coordinates not available in events")
        return pd.DataFrame()

    mouse_events = events[events["event_type"] == "mouse_move"].copy()
    mouse_events = mouse_events.sort_values(["session_id", "event_timestamp"])

    mouse_events["dx"] = mouse_events.groupby("session_id")["mouse_x"].diff()
    mouse_events["dy"] = mouse_events.groupby("session_id")["mouse_y"].diff()
    mouse_events["dt"] = mouse_events.groupby("session_id")["event_timestamp"].diff().dt.total_seconds()

    mouse_events["velocity"] = np.sqrt(mouse_events["dx"]**2 + mouse_events["dy"]**2) / (mouse_events["dt"] + 0.001)
    mouse_events["acceleration"] = mouse_events.groupby("session_id")["velocity"].diff() / (mouse_events["dt"] + 0.001)

    features = (
        mouse_events.groupby("session_id")
        .agg(
            mouse_mean_velocity=("velocity", "mean"),
            mouse_max_velocity=("velocity", "max"),
            mouse_velocity_std=("velocity", "std"),
            mouse_mean_acceleration=("acceleration", "mean"),
            hesitation_score=("velocity", lambda x: (x < 50).mean()),  # % time moving slowly
        )
        .reset_index()
    )

    features["hesitation_score"] = features["hesitation_score"].fillna(0)
    logger.info(f"Computed mouse features for {len(features)} sessions")
    return features