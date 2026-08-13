"""Data transformation: cleaning, type casting, deduplication."""

from __future__ import annotations

import hashlib

import pandas as pd
from pandera import Check, Column, DataFrameSchema
from src.utils.logger import get_logger

logger = get_logger(__name__)

CLICKSTREAM_SCHEMA = DataFrameSchema(
    {
        "event_id": Column(str, Check.str_length(min_value=1), nullable=False),
        "user_id": Column(str, Check.str_length(min_value=1), nullable=False),
        "session_id": Column(str, Check.str_length(min_value=1), nullable=False),
        "event_type": Column(
            str,
            Check.isin(["page_view", "click", "scroll", "search", "cart_add", "checkout", "purchase"]),
            nullable=False,
        ),
        "event_timestamp": Column("datetime64[ns, UTC]", nullable=False),
        "product_id": Column(str, nullable=True),
        "page_url": Column(str, nullable=True),
        "device_type": Column(str, Check.isin(["mobile", "desktop", "tablet"]), nullable=True),
        "ip_address": Column(str, nullable=True),
    }
)


def clean_clickstream(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean raw clickstream events."""
    logger.info(f"Cleaning clickstream: {len(df)} raw records")
    before = len(df)
    df = df.drop_duplicates(subset=["event_id"])
    logger.info(f"Dropped {before - len(df)} duplicate events")
    df = df.sort_values(["user_id", "session_id", "event_timestamp"])

    try:
        df = CLICKSTREAM_SCHEMA.validate(df, lazy=True)
    except Exception as e:
        logger.error(f"Schema validation failed: {e}")
        raise

    logger.info(f"Cleaned clickstream: {len(df)} valid records")
    return df


def deduplicate_sessions(df: pd.DataFrame, max_gap_minutes: int = 30) -> pd.DataFrame:
    """Split sessions if gap between events exceeds threshold."""
    df = df.sort_values(["user_id", "event_timestamp"])
    df["time_diff"] = df.groupby("user_id")["event_timestamp"].diff().dt.total_seconds() / 60
    df["new_session"] = (df["time_diff"] > max_gap_minutes).astype(int)
    df["session_suffix"] = df.groupby("user_id")["new_session"].cumsum()
    df["session_id"] = df["user_id"] + "_" + df["session_suffix"].astype(str)
    df = df.drop(columns=["time_diff", "new_session", "session_suffix"])
    logger.info("Session deduplication complete")
    return df


def anonymize_pii(df: pd.DataFrame) -> pd.DataFrame:
    """Hash PII fields to ensure privacy compliance."""
    pii_cols = ["ip_address", "email", "phone"]
    for col in pii_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: hashlib.sha256(str(x).encode()).hexdigest() if pd.notna(x) else None
            )
    logger.info("PII anonymization complete")
    return df