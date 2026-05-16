from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from app.inference.blender import BlendConfig, Blender
from app.models.registry import ModelRegistry
from app.monitoring.logging import get_logger
from app.monitoring.metrics import (
    ENSEMBLE_FALLBACK,
    MODEL_INFERENCE_TIME,
    PREDICTION_LATENCY,
)

logger = get_logger(__name__)


@dataclass
class EnsembleResult:
    models_used: list[str] = field(default_factory=list)
    predictions: Optional[np.ndarray] = None
    blend_weight: float = 0.0
    model_times: dict[str, float] = field(default_factory=dict)
    fallback_used: bool = False
    success: bool = True
    error: Optional[str] = None


class EnsembleOrchestrator:
    def __init__(
        self,
        registry: ModelRegistry,
        blender: Optional[Blender] = None,
        blend_config: Optional[BlendConfig] = None,
    ):
        self.registry = registry
        self.blender = blender or Blender(blend_config)
        self.blend_config = blend_config or BlendConfig()
        self._executor = ThreadPoolExecutor(max_workers=3)

    async def predict(
        self,
        features: np.ndarray,
        lag7_values: Optional[np.ndarray] = None,
        lag7_available: Optional[np.ndarray] = None,
        theater_ids: Optional[list[str]] = None,
    ) -> EnsembleResult:
        model_names = ["lightgbm", "xgboost", "catboost"]
        predictions: dict[str, np.ndarray] = {}
        model_times: dict[str, float] = {}
        models_used: list[str] = []

        futures = {}
        for name in model_names:
            model = self.registry.get_loaded_model(name)
            if model is None:
                logger.warning("model_not_loaded", model=name)
                continue
            future = self._executor.submit(
                self._predict_single, name, model, features
            )
            futures[future] = name

        if not futures:
            logger.error("no_models_available_for_prediction")
            return self._build_fallback(
                lag7_values if lag7_values is not None else np.zeros(features.shape[0])
            )

        for future in as_completed(futures):
            name = futures[future]
            try:
                pred, elapsed = future.result()
                predictions[name] = pred
                model_times[name] = elapsed
                models_used.append(name)
                MODEL_INFERENCE_TIME.labels(
                    model_name=name, model_version="active"
                ).observe(elapsed)
                PREDICTION_LATENCY.labels(
                    endpoint="ensemble", model=name
                ).observe(elapsed)
            except Exception as e:
                logger.error(
                    "model_prediction_failed",
                    model=name,
                    error=str(e),
                    exc_info=True,
                )
                ENSEMBLE_FALLBACK.labels(
                    fallback_reason="single_model_failure"
                ).inc()

        if not predictions:
            logger.critical("all_models_failed_using_fallback")
            ENSEMBLE_FALLBACK.labels(
                fallback_reason="all_models_failure"
            ).inc()
            fallback_lag = (
                lag7_values
                if lag7_values is not None
                else np.zeros(features.shape[0])
            )
            return self._build_fallback(fallback_lag)

        pred_array = np.mean(list(predictions.values()), axis=0)

        if lag7_values is not None and lag7_available is not None:
            blended = self.blender.blend(
                model_pred=pred_array,
                lag7=lag7_values,
                lag7_available=lag7_available,
                config=self.blend_config,
                theater_ids=theater_ids,
            )
        else:
            blended = np.clip(
                pred_array,
                self.blend_config.clip_min,
                self.blend_config.clip_max or np.inf,
            )

        return EnsembleResult(
            models_used=models_used,
            predictions=blended,
            blend_weight=self.blend_config.alpha,
            model_times=model_times,
            fallback_used=False,
            success=True,
        )

    def _predict_single(
        self, name: str, model: Any, features: np.ndarray
    ) -> tuple[np.ndarray, float]:
        start = time.monotonic()
        if name == "lightgbm":
            pred = model.predict(features)
        elif name == "xgboost":
            import xgboost as xgb

            pred = model.predict(xgb.DMatrix(features))
        elif name == "catboost":
            pred = model.predict(features)
        else:
            pred = model.predict(features)
        elapsed = time.monotonic() - start
        pred = np.asarray(pred, dtype=np.float64).ravel()
        return pred, elapsed

    def _build_fallback(
        self, lag7_values: np.ndarray
    ) -> EnsembleResult:
        fallback_pred = np.clip(
            lag7_values,
            self.blend_config.clip_min,
            self.blend_config.clip_max or np.inf,
        )
        return EnsembleResult(
            models_used=[],
            predictions=fallback_pred,
            blend_weight=0.0,
            fallback_used=True,
            success=True,
            error="All models failed, using lag_7 fallback",
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
        logger.info("ensemble_executor_shutdown")
