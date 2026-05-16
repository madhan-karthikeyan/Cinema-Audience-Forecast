from __future__ import annotations

import re
from datetime import date

import pandas as pd

from app.features.state import RollingWindowState
from app.monitoring.logging import get_logger

logger = get_logger(__name__)


LAG_PATTERN = re.compile(r"^lag_(\d+)$")


class LagFeatureComputer:
    def __init__(self, state: RollingWindowState):
        self.state = state

    @staticmethod
    def is_lag_feature(name: str) -> bool:
        return bool(LAG_PATTERN.match(name))

    @staticmethod
    def parse_lag_days(name: str) -> int | None:
        match = LAG_PATTERN.match(name)
        if match:
            return int(match.group(1))
        return None

    def compute_single(
        self,
        theater_id: str,
        target_date: date,
        lag_days: int,
        cold_start_default: float = 25.0,
    ) -> float:
        return self.state.resolve_lag(
            theater_id, target_date, lag_days, cold_start_default
        )

    def compute_batch(
        self,
        theater_ids: list[str],
        prediction_dates: list[date],
        lag_days_list: list[int],
        cold_start_defaults: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        rows = []
        for tid in theater_ids:
            for dt in prediction_dates:
                row: dict = {
                    "theater_id": tid,
                    "date": dt,
                }
                for lag in lag_days_list:
                    default = 25.0
                    if cold_start_defaults and f"lag_{lag}" in cold_start_defaults:
                        default = cold_start_defaults[f"lag_{lag}"]
                    val = self.compute_single(tid, dt, lag, default)
                    row[f"lag_{lag}"] = val
                rows.append(row)
        df = pd.DataFrame(rows)
        logger.info(
            "lag_features_computed",
            theater_count=len(theater_ids),
            date_count=len(prediction_dates),
            lags=lag_days_list,
        )
        return df
