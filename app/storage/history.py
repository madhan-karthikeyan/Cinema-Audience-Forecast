from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from app.config import settings
from app.monitoring.logging import get_logger

logger = get_logger(__name__)


class HistoryStore:
    def __init__(self, parquet_path: Path | None = None):
        self.parquet_path = parquet_path or settings.history_store_path
        self.parquet_path.mkdir(parents=True, exist_ok=True)
        self._cold_start_cache: dict[str, float] = {}
        logger.info("history_store_initialized", path=str(self.parquet_path))

    def get_ground_truth(self, theater_id: str, target_date: date) -> float | None:
        df = self._read_partition(theater_id, target_date)
        if df is not None and not df.empty and "ground_truth" in df.columns:
            val = df["ground_truth"].iloc[0]
            if pd.notna(val):
                return float(val)
        return None

    def get_prediction(self, theater_id: str, target_date: date) -> float | None:
        df = self._read_partition(theater_id, target_date)
        if df is not None and not df.empty and "predicted" in df.columns:
            val = df["predicted"].iloc[0]
            if pd.notna(val):
                return float(val)
        return None

    def has_any_record(self, theater_id: str, target_date: date) -> bool:
        return self._read_partition(theater_id, target_date) is not None

    def get_lag_value(
        self,
        theater_id: str,
        target_date: date,
        lag_days: int,
        cold_start_default: float = 25.0,
    ) -> float | None:
        lookup = target_date - timedelta(days=lag_days)
        truth = self.get_ground_truth(theater_id, lookup)
        if truth is not None:
            return truth
        pred = self.get_prediction(theater_id, lookup)
        if pred is not None:
            return pred
        return cold_start_default

    def get_history_window(
        self, theater_id: str, end_date: date, window_size: int = 28
    ) -> list[float]:
        values: list[float] = []
        for offset in range(window_size, 0, -1):
            d = end_date - timedelta(days=offset)
            val = self.get_lag_value(theater_id, d, 0)
            values.append(val if val is not None else 25.0)
        return values

    def get_mixed_history(
        self, theater_id: str, target_date: date, lookback: int = 28
    ) -> list[float | None]:
        results: list[float | None] = []
        for offset in range(lookback, 0, -1):
            d = target_date - timedelta(days=offset)
            truth = self.get_ground_truth(theater_id, d)
            if truth is not None:
                results.append(truth)
            else:
                pred = self.get_prediction(theater_id, d)
                results.append(pred)
        return results

    def get_value_sequence(
        self,
        theater_id: str,
        start_date: date,
        end_date: date,
        prefer: str = "truth",
    ) -> pd.Series:
        dates = []
        values: list[float] = []
        current = start_date
        while current <= end_date:
            val: float | None = None
            if prefer == "truth":
                val = self.get_ground_truth(theater_id, current)
                if val is None:
                    val = self.get_prediction(theater_id, current)
            else:
                val = self.get_prediction(theater_id, current)
                if val is None:
                    val = self.get_ground_truth(theater_id, current)
            if val is not None:
                dates.append(current)
                values.append(val)
            current += timedelta(days=1)
        return pd.Series(values, index=pd.DatetimeIndex(dates))

    def store_predictions(self, predictions: pd.DataFrame) -> None:
        required = {"theater_id", "date", "predicted"}
        missing = required - set(predictions.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        records = predictions.to_dict("records")
        for rec in records:
            d = (
                rec["date"]
                if isinstance(rec["date"], date)
                else pd.Timestamp(rec["date"]).date()
            )
            self._write_record(
                theater_id=str(rec["theater_id"]),
                d=d,
                predicted=float(rec["predicted"]),
            )

        logger.info(
            "predictions_stored",
            count=len(records),
            path=str(self.parquet_path),
        )

    def store_ground_truth_bulk(self, df: pd.DataFrame) -> None:
        required = {"theater_id", "date", "audience_count"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        records = df.to_dict("records")
        for rec in records:
            d = (
                rec["date"]
                if isinstance(rec["date"], date)
                else pd.Timestamp(rec["date"]).date()
            )
            self._write_record(
                theater_id=str(rec["theater_id"]),
                d=d,
                ground_truth=float(rec["audience_count"]),
            )

        logger.info(
            "ground_truth_stored",
            count=len(records),
            path=str(self.parquet_path),
        )

    def cold_start_fallback(self, theater_id: str) -> float:
        if theater_id in self._cold_start_cache:
            return self._cold_start_cache[theater_id]
        try:
            if self.parquet_path.exists():
                files = sorted(self.parquet_path.glob("**/*.parquet"))
                if files:
                    dfs = []
                    for f in files[:10]:
                        try:
                            dfs.append(pd.read_parquet(f))
                        except Exception:
                            continue
                    if dfs:
                        combined = pd.concat(dfs, ignore_index=True)
                        if "ground_truth" in combined.columns:
                            mean_val = float(combined["ground_truth"].dropna().mean())
                            self._cold_start_cache[theater_id] = mean_val
                            return mean_val
        except Exception as e:
            logger.warning("cold_start_fallback_error", error=str(e))
        return 25.0

    def theater_exists(self, theater_id: str) -> bool:
        pattern = f"theater_id={theater_id}"
        return any(p.is_dir() for p in self.parquet_path.glob(pattern))

    def get_all_theater_ids(self) -> list[str]:
        ids: set[str] = set()
        for p in self.parquet_path.glob("theater_id=*"):
            ids.add(p.name.split("=", 1)[1])
        return sorted(ids)

    def _partition_path(self, theater_id: str, d: date) -> Path:
        return (
            self.parquet_path
            / f"theater_id={theater_id}"
            / f"date={d.isoformat()}"
            / "data.parquet"
        )

    def _read_partition(self, theater_id: str, d: date) -> pd.DataFrame | None:
        path = self._partition_path(theater_id, d)
        if path.exists():
            return pd.read_parquet(path)
        alt = (
            self.parquet_path
            / f"theater_id={theater_id}"
            / f"date={d.isoformat()}.parquet"
        )
        if alt.exists():
            return pd.read_parquet(alt)
        return None

    def _write_record(
        self,
        theater_id: str,
        d: date,
        ground_truth: float | None = None,
        predicted: float | None = None,
    ) -> None:
        existing = self._read_partition(theater_id, d)
        if existing is not None:
            if ground_truth is not None:
                existing["ground_truth"] = ground_truth
            if predicted is not None:
                existing["predicted"] = predicted
            self._write_partition(theater_id, d, existing)
        else:
            row_data: dict = {"theater_id": theater_id, "date": d.isoformat()}
            if ground_truth is not None:
                row_data["ground_truth"] = ground_truth
            if predicted is not None:
                row_data["predicted"] = predicted
            row = pd.DataFrame([row_data])
            self._write_partition(theater_id, d, row)

    def _write_partition(self, theater_id: str, d: date, df: pd.DataFrame) -> None:
        path = (
            self.parquet_path
            / f"theater_id={theater_id}"
            / f"date={d.isoformat()}"
        )
        path.mkdir(parents=True, exist_ok=True)
        filepath = path / "data.parquet"
        df.to_parquet(filepath, index=False)
