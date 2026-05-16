from __future__ import annotations

from datetime import date

import pandas as pd

from app.features.calendar import CalendarFeatureComputer
from app.features.lag import LagFeatureComputer
from app.features.rolling import RollingFeatureComputer
from app.features.schema import FeatureSchema
from app.features.state import RollingWindowState
from app.monitoring.logging import get_logger
from app.storage.history import HistoryStore

logger = get_logger(__name__)


class FeaturePipeline:
    def __init__(
        self,
        history_store: HistoryStore,
        schema: FeatureSchema | None = None,
    ):
        self.history = history_store
        self.schema = schema or FeatureSchema.load()
        self.state = RollingWindowState(history_store)
        self.lag_computer = LagFeatureComputer(self.state)
        self.rolling_computer = RollingFeatureComputer(self.state)
        self.calendar_computer = CalendarFeatureComputer()

        self._identify_features()

    def _identify_features(self) -> None:
        lag_days = set()
        rolling_means = set()
        rolling_stds = set()
        for name in self.schema.feature_names:
            days = self.lag_computer.parse_lag_days(name)
            if days is not None:
                lag_days.add(days)
            if self.rolling_computer.is_rolling_mean_feature(name):
                w = self.rolling_computer.parse_window(name)
                if w is not None:
                    rolling_means.add(w)
            if self.rolling_computer.is_rolling_std_feature(name):
                w = self.rolling_computer.parse_window(name)
                if w is not None:
                    rolling_stds.add(w)

        self._lag_days = sorted(lag_days)
        self._rolling_mean_windows = sorted(rolling_means)
        self._rolling_std_windows = sorted(rolling_stds)

        cal_features = set(CalendarFeatureComputer.feature_names())
        self._calendar_features = [
            c for c in self.schema.feature_names if c in cal_features
        ]
        self._static_features = [
            c
            for c in self.schema.feature_names
            if c not in cal_features
            and not self.lag_computer.is_lag_feature(c)
            and not self.rolling_computer.is_rolling_mean_feature(c)
            and not self.rolling_computer.is_rolling_std_feature(c)
        ]

        logger.info(
            "feature_pipeline_configured",
            lag_days=self._lag_days,
            rolling_mean_windows=self._rolling_mean_windows,
            rolling_std_windows=self._rolling_std_windows,
            calendar_features=len(self._calendar_features),
            static_features=len(self._static_features),
        )

    def build_batch_features(
        self,
        prediction_dates: list[date],
        theater_ids: list[str],
    ) -> pd.DataFrame:
        all_features: list[pd.DataFrame] = []

        self.state._prediction_buffer.clear()

        for tid in theater_ids:
            theater_dates = sorted(d for d in prediction_dates)
            for target_date in theater_dates:
                row = self._build_single_row(tid, target_date)
                all_features.append(row)

        if all_features:
            df = pd.DataFrame(all_features)
        else:
            df = pd.DataFrame(columns=self.schema.feature_names)

        df = self.schema.apply_cold_start_defaults(df)
        df = self.schema.ensure_column_order(df)

        logger.info(
            "batch_features_built",
            theater_count=len(theater_ids),
            date_count=len(prediction_dates),
            shape=list(df.shape),
        )
        return df

    def build_single_features(
        self,
        theater_id: str,
        target_date: date,
    ) -> pd.Series:
        row = self._build_single_row(theater_id, target_date)
        df = pd.DataFrame([row])
        df = self.schema.apply_cold_start_defaults(df)
        df = self.schema.ensure_column_order(df)
        return df.iloc[0]

    def _build_single_row(
        self, theater_id: str, target_date: date
    ) -> dict:
        row: dict = {
            "theater_id": theater_id,
            "date": target_date,
        }

        for lag in self._lag_days:
            default = self.schema.cold_start_defaults.get(f"lag_{lag}", 25.0)
            val = self.lag_computer.compute_single(
                theater_id, target_date, lag, default
            )
            row[f"lag_{lag}"] = val

        for w in self._rolling_mean_windows:
            default = self.schema.cold_start_defaults.get(
                f"rolling_mean_{w}", 25.0
            )
            val = self.rolling_computer.compute_rolling_mean_single(
                theater_id, target_date, window=w, cold_start_default=default
            )
            row[f"rolling_mean_{w}"] = val

        for w in self._rolling_std_windows:
            default = self.schema.cold_start_defaults.get(
                f"rolling_std_{w}", 15.0
            )
            val = self.rolling_computer.compute_rolling_std_single(
                theater_id, target_date, window=w, cold_start_default=default
            )
            row[f"rolling_std_{w}"] = val

        cal_df = self.calendar_computer.compute([target_date])
        for col in self._calendar_features:
            if col in cal_df.columns:
                row[col] = cal_df.iloc[0][col]

        for col in self._static_features:
            if col in row:
                continue
            if col == "tickets_sold_daily" or col == "tickets_booked_daily":
                row[col] = 0.0
            elif col in (
                "theater_type_standard",
                "theater_type_premium",
            ):
                row[col] = 1 if col == "theater_type_standard" else 0
            elif col == "theater_area":
                row[col] = 1.0
            else:
                row[col] = 0.0

        return row

    def compute_batch_chronological(
        self,
        prediction_dates: list[date],
        theater_ids: list[str],
        chunk_size: int = 10,
    ) -> pd.DataFrame:
        self.state._prediction_buffer.clear()
        sorted_dates = sorted(prediction_dates)
        all_rows: list[dict] = []

        for i in range(0, len(theater_ids), chunk_size):
            chunk = theater_ids[i : i + chunk_size]
            for target_date in sorted_dates:
                for tid in chunk:
                    row = self._build_single_row(tid, target_date)
                    row["_predicted"] = None
                    all_rows.append(row)

        df = pd.DataFrame(all_rows)
        df = self.schema.apply_cold_start_defaults(df)
        df = self.schema.ensure_column_order(df)

        logger.info(
            "chronological_features_built",
            theater_count=len(theater_ids),
            date_count=len(prediction_dates),
            shape=list(df.shape),
        )
        return df
