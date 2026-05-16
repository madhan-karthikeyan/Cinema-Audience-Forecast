from __future__ import annotations

import re
from datetime import date

import numpy as np
import pandas as pd

from app.features.state import RollingWindowState
from app.monitoring.logging import get_logger

logger = get_logger(__name__)


ROLLING_MEAN_PATTERN = re.compile(r"^rolling_mean_(\d+)$")
ROLLING_STD_PATTERN = re.compile(r"^rolling_std_(\d+)$")


class RollingFeatureComputer:
    def __init__(self, state: RollingWindowState):
        self.state = state

    @staticmethod
    def is_rolling_mean_feature(name: str) -> bool:
        return bool(ROLLING_MEAN_PATTERN.match(name))

    @staticmethod
    def is_rolling_std_feature(name: str) -> bool:
        return bool(ROLLING_STD_PATTERN.match(name))

    @staticmethod
    def parse_window(name: str) -> int | None:
        for pat in (ROLLING_MEAN_PATTERN, ROLLING_STD_PATTERN):
            match = pat.match(name)
            if match:
                return int(match.group(1))
        return None

    @staticmethod
    def _compute_rolling_mean(
        window_values: list[float],
        min_periods: int = 1,
    ) -> float:
        arr = np.array(window_values, dtype=np.float64)
        valid = arr[~np.isnan(arr)]
        if len(valid) < min_periods:
            return float(np.nan)
        return float(valid.mean())

    @staticmethod
    def _compute_rolling_std(
        window_values: list[float],
        min_periods: int = 2,
    ) -> float:
        arr = np.array(window_values, dtype=np.float64)
        valid = arr[~np.isnan(arr)]
        if len(valid) < min_periods:
            return float(np.nan)
        return float(valid.std(ddof=1))

    def compute_rolling_mean_single(
        self,
        theater_id: str,
        target_date: date,
        window: int = 7,
        shift: int = 1,
        min_periods: int = 1,
        cold_start_default: float = 25.0,
    ) -> float:
        vals = self.state.get_window_values(
            theater_id, target_date, window_days=window, shift=shift
        )
        vals_filled = [cold_start_default if v is None else v for v in vals]
        mean_val = self._compute_rolling_mean(vals_filled, min_periods)
        if np.isnan(mean_val):
            return cold_start_default
        return max(mean_val, 0.0)

    def compute_rolling_std_single(
        self,
        theater_id: str,
        target_date: date,
        window: int = 7,
        shift: int = 1,
        min_periods: int = 2,
        cold_start_default: float = 15.0,
    ) -> float:
        vals = self.state.get_window_values(
            theater_id, target_date, window_days=window, shift=shift
        )
        vals_filled = [cold_start_default if v is None else v for v in vals]
        std_val = self._compute_rolling_std(vals_filled, min_periods)
        if np.isnan(std_val):
            return cold_start_default
        return max(std_val, 0.0)

    def compute_batch(
        self,
        theater_ids: list[str],
        prediction_dates: list[date],
        windows_mean: list[int] | None = None,
        windows_std: list[int] | None = None,
        cold_start_defaults: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        windows_mean = windows_mean or [7, 14, 28]
        windows_std = windows_std or [7]
        rows = []
        for tid in theater_ids:
            for dt in prediction_dates:
                row: dict = {"theater_id": tid, "date": dt}
                for w in windows_mean:
                    default = 25.0
                    if cold_start_defaults and f"rolling_mean_{w}" in cold_start_defaults:
                        default = cold_start_defaults[f"rolling_mean_{w}"]
                    row[f"rolling_mean_{w}"] = self.compute_rolling_mean_single(
                        tid, dt, window=w, cold_start_default=default
                    )
                for w in windows_std:
                    default = 15.0
                    if cold_start_defaults and f"rolling_std_{w}" in cold_start_defaults:
                        default = cold_start_defaults[f"rolling_std_{w}"]
                    row[f"rolling_std_{w}"] = self.compute_rolling_std_single(
                        tid, dt, window=w, cold_start_default=default
                    )
                rows.append(row)
        df = pd.DataFrame(rows)
        logger.info(
            "rolling_features_computed",
            theater_count=len(theater_ids),
            date_count=len(prediction_dates),
            mean_windows=windows_mean,
            std_windows=windows_std,
        )
        return df
