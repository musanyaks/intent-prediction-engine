"""Data extraction layer: read from Kafka, Snowflake, APIs, and external sources."""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import snowflake.connector
from kafka import KafkaConsumer
from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class KafkaExtractor:
    """Extract real-time clickstream events from Kafka."""

    def __init__(self, topic: str = "user-events", group_id: str = "intent-engine") -> None:
        cfg = load_config().kafka
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=cfg.bootstrap_servers,
            group_id=group_id,
            auto_offset_reset="latest",
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )
        logger.info(f"Kafka consumer connected to {cfg.bootstrap_servers}")

    def poll(self, timeout_ms: int = 1000, max_records: int = 1000) -> list[dict[str, Any]]:
        """Poll Kafka for new events."""
        raw = self.consumer.poll(timeout_ms=timeout_ms, max_records=max_records)
        records: list[dict[str, Any]] = []
        for tp, messages in raw.items():
            for msg in messages:
                records.append(msg.value)
        logger.info(f"Polled {len(records)} events from Kafka")
        return records


class SnowflakeExtractor:
    """Extract batch data from Snowflake data warehouse."""

    def __init__(self) -> None:
        cfg = load_config().snowflake
        self.conn = snowflake.connector.connect(
            user=cfg.user,
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            account=cfg.account,
            warehouse=cfg.warehouse,
            database=cfg.database,
            schema=cfg.schema,
        )
        logger.info("Snowflake connection established")

    def query(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query and return a DataFrame."""
        logger.info(f"Executing query: {sql[:80]}...")
        return self.conn.cursor().execute(sql).fetch_pandas_all()

    def get_user_transactions(
        self, start_date: str, end_date: str, user_ids: list[str] | None = None
    ) -> pd.DataFrame:
        """Fetch transaction history for feature engineering."""
        sql = f"""
        SELECT
            user_id,
            order_id,
            order_timestamp,
            total_amount,
            product_category,
            payment_method,
            discount_applied,
            device_type
        FROM transactions
        WHERE order_timestamp BETWEEN '{start_date}' AND '{end_date}'
        """
        if user_ids:
            ids = ",".join(f"'{u}'" for u in user_ids)
            sql += f" AND user_id IN ({ids})"
        return self.query(sql)

    def get_product_catalog(self) -> pd.DataFrame:
        """Fetch full product catalog with embeddings."""
        sql = """
        SELECT
            product_id,
            product_name,
            category,
            subcategory,
            price,
            brand,
            embedding_vector
        FROM product_catalog
        WHERE is_active = TRUE
        """
        return self.query(sql)


class ExternalDataExtractor:
    """Fetch third-party enrichment data (IP reputation, device fingerprinting)."""

    def __init__(self) -> None:
        cfg = load_config().external_apis
        self.ip_reputation_endpoint = cfg.ip_reputation_url
        self.device_fingerprint_endpoint = cfg.device_fingerprint_url
        self.api_key = os.getenv("EXTERNAL_API_KEY")

    def get_ip_reputation(self, ip_address: str) -> dict[str, Any]:
        """Check IP reputation score from third-party provider."""
        import requests

        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = requests.get(
            f"{self.ip_reputation_endpoint}/{ip_address}",
            headers=headers,
            timeout=5,
        )
        resp.raise_for_status()
        return resp.json()

    def get_device_fingerprint(self, device_id: str) -> dict[str, Any]:
        """Fetch device fingerprint enrichment data."""
        import requests

        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = requests.get(
            f"{self.device_fingerprint_endpoint}/{device_id}",
            headers=headers,
            timeout=5,
        )
        resp.raise_for_status()
        return resp.json()