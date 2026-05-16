import numpy as np
import pytest

from app.inference.blender import BlendConfig, Blender


class TestBlender:
    @pytest.fixture
    def blender(self):
        return Blender()

    @pytest.fixture
    def model_pred(self):
        return np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float64)

    def test_default_alpha_blend(self, blender, model_pred):
        lag7 = np.array([50.0, 180.0, 350.0, 420.0], dtype=np.float64)
        available = np.array([True, True, True, True])
        result = blender.blend(model_pred, lag7, available)
        expected = 0.8 * model_pred + 0.2 * lag7
        np.testing.assert_array_almost_equal(result, expected)

    def test_no_lag7_available_returns_pure_model(self, blender, model_pred):
        lag7 = np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
        available = np.array([False, False, False, False])
        result = blender.blend(model_pred, lag7, available)
        np.testing.assert_array_almost_equal(result, model_pred)

    def test_mixed_availability_blends_partially(self, blender, model_pred):
        lag7 = np.array([50.0, np.nan, 350.0, np.nan], dtype=np.float64)
        available = np.array([True, False, True, False])
        result = blender.blend(model_pred, lag7, available)
        expected = model_pred.copy()
        expected[0] = 0.8 * model_pred[0] + 0.2 * lag7[0]
        expected[2] = 0.8 * model_pred[2] + 0.2 * lag7[2]
        np.testing.assert_array_almost_equal(result, expected)

    def test_all_available_with_custom_alpha(self, blender, model_pred):
        lag7 = np.array([50.0, 180.0, 350.0, 420.0], dtype=np.float64)
        available = np.array([True, True, True, True])
        config = BlendConfig(alpha=0.5)
        result = blender.blend(model_pred, lag7, available, config=config)
        expected = 0.5 * model_pred + 0.5 * lag7
        np.testing.assert_array_almost_equal(result, expected)

    def test_alpha_zero_returns_pure_model(self, blender, model_pred):
        lag7 = np.array([50.0, 180.0, 350.0, 420.0], dtype=np.float64)
        available = np.array([True, True, True, True])
        config = BlendConfig(alpha=0.0)
        result = blender.blend(model_pred, lag7, available, config=config)
        np.testing.assert_array_almost_equal(result, model_pred)

    def test_alpha_one_returns_pure_lag7(self, blender, model_pred):
        lag7 = np.array([50.0, 180.0, 350.0, 420.0], dtype=np.float64)
        available = np.array([True, True, True, True])
        config = BlendConfig(alpha=1.0)
        result = blender.blend(model_pred, lag7, available, config=config)
        np.testing.assert_array_almost_equal(result, lag7)

    def test_clip_min_only(self, blender, model_pred):
        lag7 = np.array([50.0, 180.0, 350.0, 420.0], dtype=np.float64)
        available = np.array([True, True, True, True])
        config = BlendConfig(alpha=0.2, clip_min=150.0, clip_max=None)
        result = blender.blend(model_pred, lag7, available, config=config)
        assert result.min() >= 150.0
        blended_expected = 0.8 * model_pred + 0.2 * lag7
        assert result[1] == pytest.approx(blended_expected[1])

    def test_clip_min_and_max(self):
        blender = Blender()
        model_pred = np.array([-10.0, 500.0, 100.0], dtype=np.float64)
        lag7 = np.array([0.0, 600.0, 90.0], dtype=np.float64)
        available = np.array([True, True, True])
        config = BlendConfig(alpha=0.2, clip_min=0.0, clip_max=450.0)
        result = blender.blend(model_pred, lag7, available, config=config)
        assert result[0] >= 0.0
        assert result[1] <= 450.0

    def test_single_element_arrays(self, blender):
        model_pred = np.array([100.0])
        lag7 = np.array([90.0])
        available = np.array([True])
        result = blender.blend(model_pred, lag7, available)
        expected = 0.8 * 100.0 + 0.2 * 90.0
        assert result[0] == pytest.approx(expected)

    def test_no_lag7_clips_only(self, blender):
        model_pred = np.array([-5.0, 300.0], dtype=np.float64)
        lag7 = np.array([np.nan, np.nan], dtype=np.float64)
        available = np.array([False, False])
        config = BlendConfig(clip_min=0.0, clip_max=250.0)
        result = blender.blend(model_pred, lag7, available, config=config)
        assert result[0] == 0.0
        assert result[1] == 250.0

    def test_lag7_available_any_false_does_not_modify(self, blender, model_pred):
        lag7 = np.array([np.inf, np.inf, np.inf, np.inf])
        available = np.array([False, False, False, False])
        result = blender.blend(model_pred, lag7, available)
        np.testing.assert_array_almost_equal(result, model_pred)

    def test_output_is_float64(self, blender, model_pred):
        lag7 = np.array([50.0, 180.0, 350.0, 420.0], dtype=np.float64)
        available = np.array([True, True, True, True])
        result = blender.blend(model_pred, lag7, available)
        assert result.dtype == np.float64

    def test_blender_uses_instance_config_when_no_config_passed(self):
        config = BlendConfig(alpha=0.3, clip_min=10.0)
        blender = Blender(config=config)
        model_pred = np.array([100.0])
        lag7 = np.array([80.0])
        available = np.array([True])
        result = blender.blend(model_pred, lag7, available)
        expected = 0.7 * 100.0 + 0.3 * 80.0
        assert result[0] == pytest.approx(expected)
        assert blender.config.alpha == 0.3

    def test_blend_config_defaults(self):
        config = BlendConfig()
        assert config.alpha == 0.2
        assert config.clip_min == 0.0
        assert config.clip_max is None
        assert config.per_theater_alphas is None

    def test_large_array_no_memory_error(self, blender):
        n = 10000
        model_pred = np.random.randn(n).astype(np.float64) * 100 + 200
        lag7 = np.random.randn(n).astype(np.float64) * 50 + 180
        available = np.ones(n, dtype=bool)
        result = blender.blend(model_pred, lag7, available)
        assert len(result) == n
        assert result.dtype == np.float64

    def test_preserves_input_unmodified(self, blender):
        model_pred = np.array([100.0, 200.0])
        lag7 = np.array([50.0, 180.0])
        available = np.array([True, True])
        original_model = model_pred.copy()
        original_lag7 = lag7.copy()
        _ = blender.blend(model_pred, lag7, available)
        np.testing.assert_array_equal(model_pred, original_model)
        np.testing.assert_array_equal(lag7, original_lag7)

    def test_blend_with_no_config_uses_default_clip(self):
        blender = Blender()
        model_pred = np.array([-1.0, 100.0])
        lag7 = np.array([np.nan, np.nan])
        available = np.array([False, False])
        result = blender.blend(model_pred, lag7, available)
        assert result[0] == 0.0
        assert result[1] == 100.0
