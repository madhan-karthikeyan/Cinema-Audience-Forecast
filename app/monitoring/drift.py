from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from app.monitoring.logging import get_logger
from app.monitoring.metrics import FEATURE_DRIFT, PREDICTION_DRIFT

logger = get_logger(__name__)


@dataclass
class DriftReport:
    statistic: float = 0.0
    p_value: float = 1.0
    drifted: bool = False
    sample_count: int = 0


class DriftMonitor:
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    def compare_distributions(
        self, current: np.ndarray, reference: np.ndarray
    ) -> DriftReport:
        if len(current) < 5 or len(reference) < 5:
            return DriftReport(sample_count=min(len(current), len(reference)))

        statistic, p_value = ks_2samp(current, reference)
        drifted = statistic > self.threshold

        report = DriftReport(
            statistic=statistic,
            p_value=p_value,
            drifted=drifted,
            sample_count=min(len(current), len(reference)),
        )

        PREDICTION_DRIFT.set(statistic)

        if drifted:
            logger.warning(
                "prediction_drift_detected",
                ks_statistic=round(statistic, 4),
                p_value=round(p_value, 6),
                threshold=self.threshold,
            )

        return report

    def check_feature_drift(
        self, current_features: pd.DataFrame, reference_features: pd.DataFrame
    ) -> dict[str, DriftReport]:
        reports = {}
        for col in current_features.columns:
            if col in reference_features.columns:
                report = self.compare_distributions(
                    current_features[col].values,
                    reference_features[col].values,
                )
                reports[col] = report
                FEATURE_DRIFT.labels(feature_name=col).set(report.statistic)
        return reports

    def record_ground_truth(
        self, theater_id: str, date: str, predicted: float, actual: float
    ) -> None:
        residual = abs(predicted - actual)
        logger.info(
            "ground_truth_recorded",
            theater_id=theater_id,
            date=date,
            predicted=round(predicted, 2),
            actual=round(actual, 2),
            residual=round(residual, 2),
        )
