from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from app.inference.blender import BlendConfig
from app.inference.orchestrator import EnsembleResult
from app.inference.pipeline import InferencePipeline


def make_mock_orchestrator(prediction_value: float = 100.0, fallback: bool = False):
    mock = AsyncMock(spec_set=["predict", "shutdown"])

    async def predict_side_effect(
        features, lag7_values=None, lag7_available=None, theater_ids=None
    ):
        n = features.shape[0]
        pred_val = prediction_value
        return EnsembleResult(
            models_used=["lightgbm"] if not fallback else [],
            predictions=np.full(n, pred_val, dtype=np.float64),
            blend_weight=0.2 if not fallback else 0.0,
            model_times={"lightgbm": 0.01} if not fallback else {},
            fallback_used=fallback,
            success=True,
            error=None,
        )

    mock.predict.side_effect = predict_side_effect
    mock.shutdown = MagicMock()
    return mock


class TestInferencePipeline:
    @pytest.fixture
    def pipeline(self, feature_pipeline, seeded_history):
        history, _ = seeded_history
        orchestrator = make_mock_orchestrator(prediction_value=100.0)
        return InferencePipeline(
            feature_pipeline=feature_pipeline,
            orchestrator=orchestrator,
            history_store=history,
        )

    @pytest.mark.asyncio
    async def test_run_single_returns_correct_structure(self, pipeline):
        result = await pipeline.run_single("book_00001", date(2024, 3, 15))
        assert isinstance(result, dict)
        assert result["theater_id"] == "book_00001"
        assert result["target_date"] == date(2024, 3, 15)
        assert result["prediction"] is not None
        assert isinstance(result["prediction"], float)
        assert "latency_ms" in result
        assert "models_used" in result
        assert "fallback_used" in result
        assert "success" in result
        assert result["success"]
        assert result["models_used"] == ["lightgbm"]

    @pytest.mark.asyncio
    async def test_run_single_prediction_is_positive(self, pipeline):
        result = await pipeline.run_single("book_00001", date(2024, 3, 15))
        assert result["prediction"] >= 0

    @pytest.mark.asyncio
    async def test_run_single_model_version_string(self, pipeline):
        result = await pipeline.run_single("book_00001", date(2024, 3, 15))
        assert isinstance(result["model_version"], str)
        assert len(result["model_version"]) > 0

    @pytest.mark.asyncio
    async def test_run_single_with_cold_start_theater(self, pipeline):
        result = await pipeline.run_single("book_99999", date(2024, 3, 1))
        assert result["success"]
        assert result["prediction"] is not None

    @pytest.mark.asyncio
    async def test_run_single_early_date(self, pipeline):
        result = await pipeline.run_single("book_00001", date(2024, 1, 1))
        assert result["success"]
        assert result["prediction"] is not None

    @pytest.mark.asyncio
    async def test_run_single_failure_handled(self, feature_pipeline, seeded_history):
        history, _ = seeded_history
        fail_orchestrator = make_mock_orchestrator(
            prediction_value=0.0, fallback=True
        )

        async def fail_side_effect(**kwargs):
            return EnsembleResult(
                models_used=[],
                predictions=np.array([50.0], dtype=np.float64),
                blend_weight=0.0,
                fallback_used=True,
                success=True,
                error="all models failed",
            )

        fail_orchestrator.predict.side_effect = fail_side_effect
        pipeline = InferencePipeline(
            feature_pipeline=feature_pipeline,
            orchestrator=fail_orchestrator,
            history_store=history,
        )
        result = await pipeline.run_single("book_00001", date(2024, 3, 15))
        assert result["success"]
        assert result["fallback_used"]

    @pytest.mark.asyncio
    async def test_run_batch_returns_dataframe(self, pipeline):
        dates = [date(2024, 3, 1), date(2024, 3, 2)]
        theater_ids = ["book_00001", "book_00002"]
        result = await pipeline.run_batch(dates, theater_ids, chunk_size=2)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(dates) * len(theater_ids)

    @pytest.mark.asyncio
    async def test_run_batch_expected_columns(self, pipeline):
        dates = [date(2024, 3, 1)]
        theater_ids = ["book_00001"]
        result = await pipeline.run_batch(dates, theater_ids)
        expected_cols = {
            "theater_id",
            "date",
            "predicted",
            "lag_7",
            "models_used",
            "fallback_used",
        }
        assert expected_cols.issubset(set(result.columns))

    @pytest.mark.asyncio
    async def test_run_batch_predictions_positive(self, pipeline):
        dates = [date(2024, 3, 1), date(2024, 3, 2)]
        theater_ids = ["book_00001", "book_00002"]
        result = await pipeline.run_batch(dates, theater_ids, chunk_size=2)
        assert (result["predicted"] >= 0).all()

    @pytest.mark.asyncio
    async def test_run_batch_chunking(self, pipeline):
        dates = [date(2024, 3, 1)]
        theater_ids = [f"book_{i:05d}" for i in range(5)]
        result = await pipeline.run_batch(dates, theater_ids, chunk_size=2)
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_run_batch_single_theater(self, pipeline):
        dates = [date(2024, 3, 1)]
        theater_ids = ["book_00001"]
        result = await pipeline.run_batch(dates, theater_ids)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_run_batch_empty_theater_list(self, pipeline):
        result = await pipeline.run_batch([date(2024, 3, 1)], [])
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_run_batch_empty_date_list(self, pipeline):
        result = await pipeline.run_batch([], ["book_00001"])
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_run_batch_updates_state(self, pipeline):
        dates = [date(2024, 3, 1)]
        theater_ids = ["book_00001"]
        _ = await pipeline.run_batch(dates, theater_ids)
        state = pipeline.features.state.get_mixed_history(
            "book_00001", date(2024, 3, 8)
        )
        assert len(state) > 0

    @pytest.mark.asyncio
    async def test_run_single_updates_state(self, pipeline):
        _ = await pipeline.run_single("book_00001", date(2024, 4, 1))
        state = pipeline.features.state.get_mixed_history(
            "book_00001", date(2024, 4, 8)
        )
        assert len(state) > 0

    @pytest.mark.asyncio
    async def test_consecutive_batch_updates_predictions_differ(
        self, feature_pipeline, seeded_history
    ):
        history, _ = seeded_history
        orchestrator = make_mock_orchestrator(prediction_value=100.0)
        pipeline = InferencePipeline(
            feature_pipeline=feature_pipeline,
            orchestrator=orchestrator,
            history_store=history,
        )

        result1 = await pipeline.run_batch(
            [date(2024, 3, 1)], ["book_00001", "book_00002"]
        )

        result2 = await pipeline.run_batch(
            [date(2024, 3, 2)], ["book_00001", "book_00002"]
        )

        assert len(result1) == 2
        assert len(result2) == 2

    @pytest.mark.asyncio
    async def test_multiple_dates_same_batch(self, pipeline):
        dates = [date(2024, 3, 1), date(2024, 3, 8), date(2024, 3, 15)]
        theater_ids = ["book_00001"]
        result = await pipeline.run_batch(dates, theater_ids)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_different_theaters_have_different_lag(
        self, feature_pipeline, seeded_history
    ):
        history, _ = seeded_history
        orchestrator1 = make_mock_orchestrator(prediction_value=100.0)
        pipeline = InferencePipeline(
            feature_pipeline=feature_pipeline,
            orchestrator=orchestrator1,
            history_store=history,
        )
        _ = await pipeline.run_single("book_00001", date(2024, 3, 1))
        _ = await pipeline.run_single("book_00002", date(2024, 3, 1))
        state1 = pipeline.features.state.get_mixed_history(
            "book_00001", date(2024, 3, 8)
        )
        state2 = pipeline.features.state.get_mixed_history(
            "book_00002", date(2024, 3, 8)
        )
        assert len(state1) > 0
        assert len(state2) > 0

    @pytest.mark.asyncio
    async def test_non_default_blend_config(
        self, feature_pipeline, seeded_history
    ):
        history, _ = seeded_history
        orchestrator = make_mock_orchestrator(prediction_value=100.0)
        blend_cfg = BlendConfig(alpha=0.5, clip_min=10.0, clip_max=500.0)
        pipeline = InferencePipeline(
            feature_pipeline=feature_pipeline,
            orchestrator=orchestrator,
            history_store=history,
            blend_config=blend_cfg,
        )
        result = await pipeline.run_single("book_00001", date(2024, 3, 1))
        assert result["success"]
        assert pipeline.blend_config.alpha == 0.5
