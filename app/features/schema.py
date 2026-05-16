from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.config import settings
from app.monitoring.logging import get_logger

logger = get_logger(__name__)


class FeatureSchema:
    def __init__(
        self,
        version: str = "",
        feature_names: list[str] | None = None,
        feature_dtypes: dict[str, str] | None = None,
        cold_start_defaults: dict[str, float] | None = None,
        target_column: str = "audience_count",
        training_metrics: dict | None = None,
        categorical_features: list[str] | None = None,
    ):
        self.version = version
        self.feature_names = feature_names or []
        self.feature_dtypes = feature_dtypes or {}
        self.cold_start_defaults = cold_start_defaults or {}
        self.target_column = target_column
        self.training_metrics = training_metrics or {}
        self.categorical_features = categorical_features or []

    @classmethod
    def load(cls, path: Path | None = None) -> FeatureSchema:
        path = path or settings.feature_schema_path
        if not path.exists():
            logger.warning("feature_schema_not_found", path=str(path))
            return cls()
        with open(path) as f:
            data = json.load(f)
        return cls(
            version=data.get("version", ""),
            feature_names=data.get("feature_names", []),
            feature_dtypes=data.get("feature_dtypes", {}),
            cold_start_defaults=data.get("cold_start_defaults", {}),
            target_column=data.get("target_column", "audience_count"),
            training_metrics=data.get("training_metrics", {}),
            categorical_features=data.get("categorical_features", []),
        )

    def validate_inference_features(self, df: pd.DataFrame) -> bool:
        missing = set(self.feature_names) - set(df.columns)
        if missing:
            logger.error(
                "feature_validation_failed",
                missing_columns=list(missing),
            )
            return False

        expected_order = self.feature_names
        actual = list(df.columns[: len(expected_order)])
        if actual != expected_order:
            logger.warning(
                "feature_column_order_mismatch",
                expected=expected_order,
                actual=actual,
            )

        for col in self.feature_names:
            if col in self.feature_dtypes:
                expected_dtype = self.feature_dtypes[col]
                actual_dtype = str(df[col].dtype)
                if not actual_dtype.startswith(expected_dtype.split(".")[0]):
                    logger.warning(
                        "feature_dtype_mismatch",
                        column=col,
                        expected=expected_dtype,
                        actual=actual_dtype,
                    )

        logger.info(
            "feature_validation_passed",
            feature_count=len(self.feature_names),
            row_count=len(df),
        )
        return True

    def apply_cold_start_defaults(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for col, default in self.cold_start_defaults.items():
            if col in result.columns:
                result[col] = result[col].fillna(default)
        return result

    def ensure_column_order(self, df: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in self.feature_names if c in df.columns]
        extra = [c for c in df.columns if c not in self.feature_names]
        return df[available + extra]

    def export(self, path: Path | None = None) -> None:
        path = path or settings.feature_schema_path
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": self.version or "1.0.0",
            "feature_names": self.feature_names,
            "feature_dtypes": self.feature_dtypes,
            "cold_start_defaults": self.cold_start_defaults,
            "target_column": self.target_column,
            "training_metrics": self.training_metrics,
            "categorical_features": self.categorical_features,
            "created_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(
            "feature_schema_exported",
            path=str(path),
            feature_count=len(self.feature_names),
        )
