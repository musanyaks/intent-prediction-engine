"""Data validation using Great Expectations and Pandera."""

from __future__ import annotations

import great_expectations as gx
import pandas as pd
from great_expectations.core.expectation_suite import ExpectationSuite
from great_expectations.expectations import (
    ExpectColumnValuesToBeBetween,
    ExpectColumnValuesToNotBeNull,
    ExpectTableRowCountToBeBetween,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_expectation_suite(suite_name: str = "intent_engine_suite") -> ExpectationSuite:
    """Build a Great Expectations suite for clickstream data."""
    suite = ExpectationSuite(name=suite_name)
    suite.add_expectation(ExpectTableRowCountToBeBetween(min_value=1, max_value=100_000_000))
    suite.add_expectation(ExpectColumnValuesToNotBeNull(column="user_id"))
    suite.add_expectation(ExpectColumnValuesToNotBeNull(column="event_timestamp"))
    suite.add_expectation(ExpectColumnValuesToBeBetween(column="total_amount", min_value=0))
    return suite


def validate_batch(df: pd.DataFrame, suite: ExpectationSuite | None = None) -> dict:
    """Validate a DataFrame batch against expectations."""
    if suite is None:
        suite = build_expectation_suite()

    context = gx.get_context()
    batch = context.pandas_default_reader.read_dataframe(df)
    validator = context.get_validator(batch_request=batch.batch_request, expectation_suite=suite)
    results = validator.validate()

    success = results.success
    if not success:
        failed = [r for r in results.results if not r.success]
        logger.warning(f"Validation failed for {len(failed)} expectations")
        for f in failed:
            logger.warning(f"  - {f.expectation_config.expectation_type}: {f.result}")
    else:
        logger.info("Batch validation passed")

    return {"success": success, "results": results}


def check_data_drift(
    current: pd.DataFrame, reference: pd.DataFrame, threshold: float = 0.05
) -> dict:
    """Simple drift check: compare column distributions via KS test."""
    from scipy import stats

    drift_report = {}
    numeric_cols = current.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        if col in reference.columns:
            statistic, pvalue = stats.ks_2samp(current[col].dropna(), reference[col].dropna())
            drift_report[col] = {
                "ks_statistic": float(statistic),
                "p_value": float(pvalue),
                "drift_detected": pvalue < threshold,
            }

    drifted = [c for c, r in drift_report.items() if r["drift_detected"]]
    if drifted:
        logger.warning(f"Data drift detected in columns: {drifted}")
    else:
        logger.info("No data drift detected")

    return drift_report