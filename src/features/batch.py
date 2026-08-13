"""Batch feature engineering from data warehouse aggregations."""

from __future__ import annotations

import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_user_history_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate user features from transaction history."""
    features = (
        transactions.groupby("user_id")
        .agg(
            total_orders=("order_id", "nunique"),
            total_spend=("total_amount", "sum"),
            avg_order_value=("total_amount", "mean"),
            max_order_value=("total_amount", "max"),
            min_order_value=("total_amount", "min"),
            std_order_value=("total_amount", "std"),
            avg_discount_used=("discount_applied", "mean"),
            favorite_category=("product_category", lambda x: x.mode()[0] if len(x.mode()) > 0 else None),
            unique_categories=("product_category", "nunique"),
            mobile_purchase_ratio=("device_type", lambda x: (x == "mobile").mean()),
        )
        .reset_index()
    )

    # Recency features
    transactions["order_timestamp"] = pd.to_datetime(transactions["order_timestamp"])
    last_purchase = transactions.groupby("user_id")["order_timestamp"].max().reset_index()
    last_purchase["days_since_last_purchase"] = (
        pd.Timestamp.now(tz="UTC") - last_purchase["order_timestamp"]
    ).dt.days
    features = features.merge(
        last_purchase[["user_id", "days_since_last_purchase"]],
        on="user_id",
        how="left",
    )

    # Frequency segments
    features["purchase_frequency_segment"] = pd.cut(
        features["total_orders"],
        bins=[0, 1, 3, 10, float("inf")],
        labels=["one_time", "occasional", "regular", "vip"],
    )

    features["std_order_value"] = features["std_order_value"].fillna(0)
    logger.info(f"Computed batch features for {len(features)} users")
    return features


def compute_product_affinity_features(
    transactions: pd.DataFrame, product_catalog: pd.DataFrame
) -> pd.DataFrame:
    """Compute user-product category affinity vectors."""
    user_category_counts = (
        transactions.groupby(["user_id", "product_category"])
        .size()
        .unstack(fill_value=0)
    )

    # Normalize to probability distribution
    user_category_counts = user_category_counts.div(user_category_counts.sum(axis=1), axis=0)

    # Entropy of category distribution (high = diverse interests, low = focused)
    user_category_counts["category_entropy"] = -(
        user_category_counts * np.log(user_category_counts + 1e-10)
    ).sum(axis=1)

    logger.info(f"Computed product affinity for {len(user_category_counts)} users")
    return user_category_counts.reset_index()