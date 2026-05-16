from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

import numpy as np

from app.monitoring.logging import get_logger
from app.storage.history import HistoryStore

logger = get_logger(__name__)


class RollingWindowState:
    def __init__(self, history_store: HistoryStore):
        self.history = history_store
        self._prediction_buffer: dict[str, dict[date, float]] = defaultdict(dict)

    def get_mixed_history(
        self,
        theater_id: str,
        target_date: date,
        lookback: int = 28,
    ) -> list[float | None]:
        results: list[float | None] = []
        for offset in range(lookback, 0, -1):
            d = target_date - timedelta(days=offset)
            val = self._resolve_value(theater_id, d)
            results.append(val)
        return results

    def get_mixed_series(
        self,
        theater_id: str,
        target_date: date,
        lookback: int = 28,
    ) -> np.ndarray:
        vals = self.get_mixed_history(theater_id, target_date, lookback)
        arr = np.array(vals, dtype=np.float64)
        arr[np.isnan(arr)] = 25.0
        return arr

    def resolve_lag(
        self,
        theater_id: str,
        target_date: date,
        lag_days: int,
        cold_start_default: float = 25.0,
    ) -> float:
        lookup = target_date - timedelta(days=lag_days)
        val = self._resolve_value(theater_id, lookup)
        return val if val is not None else cold_start_default

    def append_prediction(
        self, theater_id: str, d: date, value: float
    ) -> None:
        self._prediction_buffer[theater_id][d] = value

    def flush_predictions(self) -> None:
        for theater_id, preds in self._prediction_buffer.items():
            if not preds:
                continue
            import pandas as pd

            records = [
                {
                    "theater_id": theater_id,
                    "date": d,
                    "predicted": v,
                }
                for d, v in sorted(preds.items())
            ]
            df = pd.DataFrame(records)
            self.history.store_predictions(df)

        total = sum(len(v) for v in self._prediction_buffer.values())
        self._prediction_buffer.clear()
        if total:
            logger.info("predictions_flushed", count=total)

    def get_window_values(
        self,
        theater_id: str,
        target_date: date,
        window_days: int = 7,
        shift: int = 1,
    ) -> list[float | None]:
        values: list[float | None] = []
        for offset in range(shift + window_days - 1, shift - 1, -1):
            d = target_date - timedelta(days=offset)
            val = self._resolve_value(theater_id, d)
            values.append(val)
        return values

    def _resolve_value(
        self, theater_id: str, d: date
    ) -> float | None:
        buf = self._prediction_buffer.get(theater_id, {})
        if d in buf:
            return buf[d]
        truth = self.history.get_ground_truth(theater_id, d)
        if truth is not None:
            return truth
        pred = self.history.get_prediction(theater_id, d)
        if pred is not None:
            return pred
        return None

    def get_first_unpredicted_date(
        self, theater_id: str, start: date, end: date
    ) -> date | None:
        for d in self._date_range(start, end):
            if d not in self._prediction_buffer.get(theater_id, {}):
                return d
        return None

    @staticmethod
    def _date_range(start: date, end: date):
        current = start
        while current <= end:
            yield current
            current += timedelta(days=1)
