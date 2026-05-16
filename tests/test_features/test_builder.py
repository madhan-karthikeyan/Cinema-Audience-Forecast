from datetime import date

import pandas as pd
import pytest


class TestFeaturePipeline:
    def test_build_single_features_returns_series(self, feature_pipeline):
        features = feature_pipeline.build_single_features(
            "book_00001", date(2024, 3, 15)
        )
        assert isinstance(features, pd.Series)
        assert len(features) > 0

    def test_build_single_features_all_present(self, feature_pipeline, default_schema):
        features = feature_pipeline.build_single_features(
            "book_00001", date(2024, 3, 15)
        )
        for col in default_schema.feature_names:
            assert col in features.index, f"Missing feature: {col}"

    def test_build_single_has_no_nans(self, feature_pipeline):
        features = feature_pipeline.build_single_features(
            "book_00001", date(2024, 3, 15)
        )
        assert not features.isnull().any(), "Features should not contain NaN"

    def test_build_batch_features_shape(
        self, feature_pipeline, two_week_window
    ):
        theater_ids = ["book_00001", "book_00002"]
        df = feature_pipeline.build_batch_features(
            two_week_window, theater_ids
        )
        expected_rows = len(theater_ids) * len(two_week_window)
        assert len(df) == expected_rows
        assert all(c in df.columns for c in ["lag_7", "lag_14", "day_of_week"])

    def test_build_batch_chronological_preserves_order(
        self, feature_pipeline, two_week_window
    ):
        theater_ids = ["book_00001", "book_00002"]
        df = feature_pipeline.compute_batch_chronological(
            two_week_window, theater_ids, chunk_size=2
        )
        assert len(df) == len(theater_ids) * len(two_week_window)

    def test_different_theaters_get_different_features(
        self, feature_pipeline
    ):
        f1 = feature_pipeline.build_single_features(
            "book_00001", date(2024, 1, 15)
        )
        f2 = feature_pipeline.build_single_features(
            "book_00002", date(2024, 1, 15)
        )
        assert f1["rolling_mean_7"] != pytest.approx(f2["rolling_mean_7"], abs=1e-6)

    def test_calendar_features_deterministic(
        self, feature_pipeline
    ):
        f1 = feature_pipeline.build_single_features(
            "book_00001", date(2024, 12, 25)
        )
        f2 = feature_pipeline.build_single_features(
            "book_00005", date(2024, 12, 25)
        )
        assert f1["day_of_week"] == f2["day_of_week"]
        assert f1["month"] == f2["month"]
        assert f1["day"] == f2["day"]

    def test_empty_theater_list_returns_empty_df(
        self, feature_pipeline, two_week_window
    ):
        df = feature_pipeline.build_batch_features(
            two_week_window, []
        )
        assert len(df) == 0

    def test_empty_date_list_returns_empty_df(self, feature_pipeline):
        df = feature_pipeline.build_batch_features([], ["book_00001"])
        assert len(df) == 0

    def test_lag_features_vary_by_date(
        self, feature_pipeline
    ):
        d1 = feature_pipeline.build_single_features("book_00001", date(2024, 3, 1))
        d2 = feature_pipeline.build_single_features("book_00001", date(2024, 3, 8))
        assert d1["lag_7"] != d2["lag_7"], "Lag-7 should differ for dates 7 days apart"

    def test_lag_28_for_target_early_in_month(self, feature_pipeline):
        features = feature_pipeline.build_single_features(
            "book_00001", date(2024, 1, 7)
        )
        assert features["lag_28"] == 25.0
