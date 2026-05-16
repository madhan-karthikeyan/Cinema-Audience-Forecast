from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.monitoring.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BlendConfig:
    alpha: float = 0.2
    per_theater_alphas: dict[str, float] | None = None
    clip_min: float = 0.0
    clip_max: float | None = None


class Blender:
    def __init__(self, config: BlendConfig | None = None):
        self.config = config or BlendConfig()

    def blend(
        self,
        model_pred: np.ndarray,
        lag7: np.ndarray,
        lag7_available: np.ndarray,
        config: BlendConfig | None = None,
        theater_ids: list[str] | None = None,
    ) -> np.ndarray:
        cfg = config or self.config
        result = np.copy(model_pred)

        if lag7_available.any():
            alpha = cfg.alpha
            result[lag7_available] = (
                1 - alpha
            ) * model_pred[lag7_available] + alpha * lag7[lag7_available]

        result = np.clip(result, cfg.clip_min, cfg.clip_max or np.inf)

        logger.debug(
            "blend_computed",
            alpha=cfg.alpha,
            lag7_available_count=int(lag7_available.sum()),
            model_pred_mean=float(np.mean(model_pred)),
            blended_mean=float(np.mean(result)),
        )
        return result
