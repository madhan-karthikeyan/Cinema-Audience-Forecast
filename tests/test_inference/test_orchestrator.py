from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.inference.blender import BlendConfig
from app.inference.orchestrator import EnsembleOrchestrator
from app.models.registry import ModelRegistry


@pytest.fixture
def temp_models_dir():
    tmp = Path(tempfile.mkdtemp())
    yield tmp
    shutil.rmtree(str(tmp))


@pytest.fixture
def empty_registry(temp_models_dir):
    registry = ModelRegistry(base_path=temp_models_dir)
    return registry


@pytest.fixture
def registry_with_manifest(temp_models_dir):
    manifest = {
        "models": [
            {
                "name": "lightgbm",
                "version": "1.0.0",
                "path": "/fake/lightgbm.txt",
                "metrics": {"rmse": 21.0},
                "params": {"num_leaves": 31},
                "created_at": "2024-01-01T00:00:00Z",
                "checksum": "abc",
                "active": True,
            },
            {
                "name": "xgboost",
                "version": "1.0.0",
                "path": "/fake/xgboost.json",
                "metrics": {"rmse": 22.0},
                "params": {"max_depth": 6},
                "created_at": "2024-01-01T00:00:00Z",
                "checksum": "def",
                "active": True,
            },
            {
                "name": "catboost",
                "version": "1.0.0",
                "path": "/fake/catboost.cbm",
                "metrics": {"rmse": 20.5},
                "params": {"iterations": 1000},
                "created_at": "2024-01-01T00:00:00Z",
                "checksum": "ghi",
                "active": True,
            },
        ]
    }
    (temp_models_dir / "registry.json").write_text(json.dumps(manifest))
    registry = ModelRegistry(base_path=temp_models_dir)
    return registry


class FakeModel:
    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], 100.0, dtype=np.float64)


class TestEnsembleOrchestrator:
    @pytest.fixture
    def orchestrator(self, empty_registry):
        return EnsembleOrchestrator(registry=empty_registry)

    @pytest.fixture
    def sample_features(self):
        return np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float64)

    @pytest.fixture
    def multi_row_features(self):
        return np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64
        )

    @pytest.mark.asyncio
    async def test_no_models_fallback_to_lag7(self, orchestrator, sample_features):
        lag7 = np.array([80.0])
        available = np.array([True])
        result = await orchestrator.predict(
            features=sample_features,
            lag7_values=lag7,
            lag7_available=available,
        )
        assert result.success
        assert result.fallback_used
        assert result.predictions is not None
        assert result.predictions[0] == pytest.approx(80.0)
        assert result.models_used == []
        assert result.blend_weight == 0.0

    @pytest.mark.asyncio
    async def test_no_models_no_lag7_returns_zero(self, orchestrator, sample_features):
        result = await orchestrator.predict(features=sample_features)
        assert result.success
        assert result.fallback_used
        assert result.predictions is not None
        assert result.predictions[0] == 0.0

    @pytest.mark.asyncio
    async def test_no_models_lag7_unavailable_returns_nan(
        self, orchestrator, sample_features
    ):
        lag7 = np.array([np.nan])
        available = np.array([False])
        result = await orchestrator.predict(
            features=sample_features,
            lag7_values=lag7,
            lag7_available=available,
        )
        assert result.success
        assert result.fallback_used
        assert np.isnan(result.predictions[0])

    @pytest.mark.asyncio
    async def test_single_model_loaded(
        self, orchestrator, sample_features, empty_registry
    ):
        empty_registry._loaded_models["lightgbm"] = FakeModel()
        lag7 = np.array([80.0])
        available = np.array([True])
        result = await orchestrator.predict(
            features=sample_features,
            lag7_values=lag7,
            lag7_available=available,
        )
        assert result.success
        assert not result.fallback_used
        assert result.predictions is not None
        assert result.models_used == ["lightgbm"]
        assert len(result.model_times) == 1
        raw_model_pred = 100.0
        expected = 0.8 * raw_model_pred + 0.2 * 80.0
        assert result.predictions[0] == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_all_three_models_loaded_graceful_degradation(
        self, empty_registry, sample_features
    ):
        empty_registry._loaded_models["lightgbm"] = FakeModel()
        empty_registry._loaded_models["xgboost"] = FakeModel()
        empty_registry._loaded_models["catboost"] = FakeModel()
        orchestrator = EnsembleOrchestrator(registry=empty_registry)
        lag7 = np.array([80.0])
        available = np.array([True])
        result = await orchestrator.predict(
            features=sample_features,
            lag7_values=lag7,
            lag7_available=available,
        )
        assert result.success
        assert not result.fallback_used
        assert len(result.models_used) >= 2
        assert result.predictions is not None
        raw_avg = 100.0
        expected = 0.8 * raw_avg + 0.2 * 80.0
        assert result.predictions[0] == pytest.approx(expected, abs=1.0)

    @pytest.mark.asyncio
    async def test_some_models_fail_graceful_degradation(
        self, empty_registry, sample_features
    ):
        good_model = FakeModel()

        class FailingModel:
            def predict(self, features):
                raise RuntimeError("model crashed")

        empty_registry._loaded_models["lightgbm"] = good_model
        empty_registry._loaded_models["xgboost"] = FailingModel()
        empty_registry._loaded_models["catboost"] = FailingModel()
        orchestrator = EnsembleOrchestrator(registry=empty_registry)
        lag7 = np.array([80.0])
        available = np.array([True])
        result = await orchestrator.predict(
            features=sample_features,
            lag7_values=lag7,
            lag7_available=available,
        )
        assert result.success
        assert not result.fallback_used
        assert result.models_used == ["lightgbm"]
        assert result.predictions is not None
        expected = 0.8 * 100.0 + 0.2 * 80.0
        assert result.predictions[0] == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_all_models_fail_triggers_fallback(
        self, empty_registry, sample_features
    ):
        class FailingModel:
            def predict(self, features):
                raise RuntimeError("crashed")

        empty_registry._loaded_models["lightgbm"] = FailingModel()
        empty_registry._loaded_models["xgboost"] = FailingModel()
        empty_registry._loaded_models["catboost"] = FailingModel()
        orchestrator = EnsembleOrchestrator(registry=empty_registry)
        lag7 = np.array([80.0])
        available = np.array([True])
        result = await orchestrator.predict(
            features=sample_features,
            lag7_values=lag7,
            lag7_available=available,
        )
        assert result.success
        assert result.fallback_used
        assert result.predictions is not None
        assert result.predictions[0] == pytest.approx(80.0)

    @pytest.mark.asyncio
    async def test_multi_row_prediction(self, empty_registry, multi_row_features):
        empty_registry._loaded_models["lightgbm"] = FakeModel()
        empty_registry._loaded_models["xgboost"] = FakeModel()
        orchestrator = EnsembleOrchestrator(registry=empty_registry)
        lag7 = np.array([80.0, 90.0, 100.0])
        available = np.array([True, True, True])
        result = await orchestrator.predict(
            features=multi_row_features,
            lag7_values=lag7,
            lag7_available=available,
        )
        assert result.success
        assert result.predictions is not None
        assert len(result.predictions) == 3
        raw_avg = 100.0
        expected = 0.8 * raw_avg + 0.2 * lag7
        np.testing.assert_array_almost_equal(result.predictions, expected)

    @pytest.mark.asyncio
    async def test_shutdown(self, orchestrator):
        orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_orchestrator_with_manifest_registry(self, registry_with_manifest):
        assert registry_with_manifest.list_models() == ["lightgbm", "xgboost", "catboost"]
        orchestrator = EnsembleOrchestrator(registry=registry_with_manifest)
        features = np.array([[1.0, 2.0]], dtype=np.float64)
        result = await orchestrator.predict(features=features)
        assert result.fallback_used
        assert result.success

    @pytest.mark.asyncio
    async def test_custom_blend_config(self, empty_registry, sample_features):
        empty_registry._loaded_models["lightgbm"] = FakeModel()
        config = BlendConfig(alpha=0.5, clip_min=0.0, clip_max=200.0)
        orchestrator = EnsembleOrchestrator(
            registry=empty_registry, blend_config=config
        )
        lag7 = np.array([80.0])
        available = np.array([True])
        result = await orchestrator.predict(
            features=sample_features,
            lag7_values=lag7,
            lag7_available=available,
        )
        assert result.blend_weight == 0.5
        assert result.predictions is not None
        expected = 0.5 * 100.0 + 0.5 * 80.0
        assert result.predictions[0] == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_no_lag7_available_with_models(
        self, empty_registry, sample_features
    ):
        empty_registry._loaded_models["lightgbm"] = FakeModel()
        orchestrator = EnsembleOrchestrator(registry=empty_registry)
        result = await orchestrator.predict(features=sample_features)
        assert result.success
        assert not result.fallback_used
        assert result.predictions is not None
        assert result.predictions[0] == pytest.approx(100.0)

    @pytest.mark.asyncio
    async def test_ensemble_result_dataclass(self):
        from app.inference.orchestrator import EnsembleResult

        result = EnsembleResult(
            models_used=["lgb"],
            predictions=np.array([1.0, 2.0]),
            blend_weight=0.2,
            model_times={"lgb": 0.01},
            fallback_used=False,
            success=True,
        )
        assert result.models_used == ["lgb"]
        assert result.blend_weight == 0.2
        assert result.success
        assert not result.fallback_used
