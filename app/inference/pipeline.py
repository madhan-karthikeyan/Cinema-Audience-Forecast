from __future__ import annotations

import time
from datetime import date

import numpy as np
import pandas as pd

from app.features.builder import FeaturePipeline
from app.inference.blender import BlendConfig, Blender
from app.inference.orchestrator import EnsembleOrchestrator
from app.monitoring.logging import get_logger
from app.monitoring.metrics import (
    BATCH_SIZE,
    LAST_BATCH_TIMESTAMP,
    PREDICTION_REQUESTS,
)
from app.storage.history import HistoryStore

logger = get_logger(__name__)


class InferencePipeline:
    def __init__(
        self,
        feature_pipeline: FeaturePipeline,
        orchestrator: EnsembleOrchestrator,
        history_store: HistoryStore,
        blender: Blender | None = None,
        blend_config: BlendConfig | None = None,
    ):
        self.features = feature_pipeline
        self.orchestrator = orchestrator
        self.history = history_store
        self.blender = blender or Blender(blend_config)
        self.blend_config = blend_config or BlendConfig()

    async def run_batch(
        self,
        prediction_dates: list[date],
        theater_ids: list[str],
        chunk_size: int = 10,
    ) -> pd.DataFrame:
        start = time.monotonic()
        sorted_dates = sorted(prediction_dates)

        all_predictions: list[dict] = []

        for i in range(0, len(theater_ids), chunk_size):
            chunk = theater_ids[i : i + chunk_size]
            chunk_start = time.monotonic()

            for target_date in sorted_dates:
                feature_df = self.features.build_batch_features(
                    [target_date], chunk
                )

                feature_cols = [
                    c
                    for c in feature_df.columns
                    if c not in ("theater_id", "date", "lag_7")
                ]
                feature_array = feature_df[feature_cols].values

                lag7 = feature_df["lag_7"].values
                lag7_available = ~np.isnan(lag7)

                result = await self.orchestrator.predict(
                    features=feature_array,
                    lag7_values=lag7,
                    lag7_available=lag7_available,
                    theater_ids=chunk,
                )

                if not result.success or result.predictions is None:
                    logger.error(
                        "batch_chunk_prediction_failed",
                        chunk_index=i // chunk_size,
                        error=result.error,
                    )
                    continue

                predictions_list = result.predictions.tolist()
                for idx, tid in enumerate(chunk):
                    all_predictions.append({
                        "theater_id": tid,
                        "date": target_date,
                        "predicted": float(predictions_list[idx]),
                        "lag_7": float(lag7[idx]) if not np.isnan(lag7[idx]) else None,
                        "models_used": ",".join(result.models_used),
                        "fallback_used": result.fallback_used,
                    })

                self.features.state.append_prediction(
                    chunk[0] if chunk else "",
                    target_date,
                    float(predictions_list[0]) if predictions_list else 0.0,
                )

            chunk_elapsed = time.monotonic() - chunk_start
            logger.info(
                "chunk_complete",
                chunk_index=i // chunk_size,
                theaters=len(chunk),
                dates=len(sorted_dates),
                latency_ms=round(chunk_elapsed * 1000, 2),
            )

        self.features.state.flush_predictions()

        result_df = pd.DataFrame(all_predictions)
        total_predictions = len(result_df)
        elapsed = time.monotonic() - start

        BATCH_SIZE.observe(total_predictions)
        LAST_BATCH_TIMESTAMP.set(time.time())

        PREDICTION_REQUESTS.labels(
            endpoint="/v1/predict/batch", status="success"
        ).inc()

        logger.info(
            "batch_predictions_complete",
            theater_count=len(theater_ids),
            date_count=len(sorted_dates),
            prediction_count=total_predictions,
            total_latency_ms=round(elapsed * 1000, 2),
        )

        return result_df

    async def run_single(
        self,
        theater_id: str,
        target_date: date,
    ) -> dict:
        start = time.monotonic()

        feature_series = self.features.build_single_features(
            theater_id, target_date
        )

        feature_array = feature_series.values.reshape(1, -1)
        lag7 = feature_series.get("lag_7", np.nan)
        lag7_val = np.array([lag7 if not pd.isna(lag7) else np.nan])
        lag7_avail = np.array([not pd.isna(lag7)])

        result = await self.orchestrator.predict(
            features=feature_array,
            lag7_values=lag7_val,
            lag7_available=lag7_avail,
            theater_ids=[theater_id],
        )

        elapsed = time.monotonic() - start
        prediction = None
        if result.success and result.predictions is not None:
            prediction = float(result.predictions[0])

        if prediction is not None:
            self.features.state.append_prediction(
                theater_id, target_date, prediction
            )
            self.features.state.flush_predictions()

        status_label = "success" if prediction is not None else "failure"
        PREDICTION_REQUESTS.labels(
            endpoint="/v1/predict/theater", status=status_label
        ).inc()

        response = {
            "theater_id": theater_id,
            "target_date": target_date,
            "prediction": prediction,
            "model_version": ",".join(result.models_used) if result.models_used else "none",
            "latency_ms": round(elapsed * 1000, 2),
            "models_used": result.models_used,
            "fallback_used": result.fallback_used,
            "success": result.success,
            "error": result.error,
        }

        logger.info(
            "single_prediction_complete",
            theater_id=theater_id,
            target_date=target_date.isoformat(),
            prediction=prediction,
            latency_ms=round(elapsed * 1000, 2),
        )

        return response
