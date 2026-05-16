from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from app.monitoring.logging import get_logger

logger = get_logger(__name__)


class CalendarFeatureComputer:
    def compute(self, target_dates: list[date]) -> pd.DataFrame:
        dt_index = pd.DatetimeIndex(target_dates)
        df = pd.DataFrame({"date": target_dates, "_dt": dt_index})

        df["day_of_week"] = df["_dt"].dt.weekday.astype("int32")
        df["is_weekend"] = (df["_dt"].dt.weekday >= 5).astype("int32")
        df["month"] = df["_dt"].dt.month.astype("int32")
        df["day"] = df["_dt"].dt.day.astype("int32")

        df["dow_sin"] = np.sin(2 * np.pi * df["_dt"].dt.weekday / 7).astype("float64")
        df["dow_cos"] = np.cos(2 * np.pi * df["_dt"].dt.weekday / 7).astype("float64")
        df["month_sin"] = np.sin(2 * np.pi * (df["_dt"].dt.month - 1) / 12).astype("float64")
        df["month_cos"] = np.cos(2 * np.pi * (df["_dt"].dt.month - 1) / 12).astype("float64")

        df["week_of_year"] = df["_dt"].dt.isocalendar()["week"].astype("int32")

        df.drop(columns=["_dt"], inplace=True)

        logger.info(
            "calendar_features_computed",
            date_count=len(target_dates),
        )
        return df

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "day_of_week",
            "is_weekend",
            "month",
            "day",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
            "week_of_year",
        ]
