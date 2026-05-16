import json
from pathlib import Path

import pandas as pd
import pytest

from app.features.schema import FeatureSchema


class TestFeatureSchema:
    @pytest.fixture
    def schema(self):
        return FeatureSchema(
            version="1.0.0",
            feature_names=["lag_7", "lag_14", "rolling_mean_7", "day_of_week"],
            feature_dtypes={
                "lag_7": "float64",
                "lag_14": "float64",
                "rolling_mean_7": "float64",
                "day_of_week": "int32",
            },
            cold_start_defaults={"lag_7": 25.0, "lag_14": 25.0, "rolling_mean_7": 20.0},
            target_column="audience_count",
            training_metrics={"rmse": 21.6},
        )

    def test_load_from_file(self, schema, tmp_path):
        path = tmp_path / "feature_schema.json"
        schema.export(path)
        loaded = FeatureSchema.load(path)
        assert loaded.version == "1.0.0"
        assert loaded.feature_names == schema.feature_names
        assert loaded.cold_start_defaults == schema.cold_start_defaults
        assert loaded.training_metrics == schema.training_metrics

    def test_validate_inference_features_passes(self, schema):
        df = pd.DataFrame(
            {
                "lag_7": [1.0],
                "lag_14": [2.0],
                "rolling_mean_7": [3.0],
                "day_of_week": [0],
            }
        )
        assert schema.validate_inference_features(df) is True

    def test_validate_inference_features_missing_columns(self, schema):
        df = pd.DataFrame({"lag_7": [1.0]})
        assert schema.validate_inference_features(df) is False

    def test_apply_cold_start_defaults(self, schema):
        df = pd.DataFrame(
            {
                "lag_7": [None],
                "lag_14": [None],
                "rolling_mean_7": [30.0],
                "day_of_week": [0],
            }
        )
        result = schema.apply_cold_start_defaults(df)
        assert result["lag_7"].iloc[0] == 25.0
        assert result["lag_14"].iloc[0] == 25.0
        assert result["rolling_mean_7"].iloc[0] == 30.0

    def test_ensure_column_order(self, schema):
        df = pd.DataFrame(
            {
                "day_of_week": [0],
                "lag_14": [2.0],
                "extra_col": [99],
                "lag_7": [1.0],
                "rolling_mean_7": [3.0],
            }
        )
        result = schema.ensure_column_order(df)
        expected = ["lag_7", "lag_14", "rolling_mean_7", "day_of_week", "extra_col"]
        assert list(result.columns) == expected

    def test_export_creates_valid_json(self, schema, tmp_path):
        path = tmp_path / "exported.json"
        schema.export(path)
        with open(path) as f:
            data = json.load(f)
        assert data["version"] == "1.0.0"
        assert len(data["feature_names"]) == 4
        assert "created_at" in data

    def test_load_nonexistent_returns_default(self):
        schema = FeatureSchema.load(Path("/nonexistent/path.json"))
        assert schema.version == ""
        assert schema.feature_names == []
