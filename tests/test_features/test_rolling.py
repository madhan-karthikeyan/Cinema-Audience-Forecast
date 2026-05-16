from datetime import date, timedelta

import numpy as np
import pandas as pd

from app.features.rolling import RollingFeatureComputer
from app.features.state import RollingWindowState


class TestRollingFeatureComputer:
    def test_is_rolling_mean_feature_detection(self):
        assert RollingFeatureComputer.is_rolling_mean_feature("rolling_mean_7") is True
        assert RollingFeatureComputer.is_rolling_mean_feature("rolling_mean_14") is True
        assert RollingFeatureComputer.is_rolling_mean_feature("lag_7") is False
        assert RollingFeatureComputer.is_rolling_mean_feature("rolling_std_7") is False

    def test_is_rolling_std_feature_detection(self):
        assert RollingFeatureComputer.is_rolling_std_feature("rolling_std_7") is True
        assert RollingFeatureComputer.is_rolling_std_feature("rolling_std_14") is True
        assert RollingFeatureComputer.is_rolling_std_feature("lag_7") is False
        assert RollingFeatureComputer.is_rolling_std_feature("rolling_mean_7") is False

    def test_parse_window(self):
        assert RollingFeatureComputer.parse_window("rolling_mean_7") == 7
        assert RollingFeatureComputer.parse_window("rolling_std_7") == 7
        assert RollingFeatureComputer.parse_window("rolling_mean_14") == 14
        assert RollingFeatureComputer.parse_window("lag_7") is None

    def test_compute_rolling_mean_constant_values(self, seeded_history):
        history, tid = seeded_history
        state = RollingWindowState(history)
        computer = RollingFeatureComputer(state)
        for i in range(1, 30):
            d = date(2024, 3, 1) - timedelta(days=i)
            history.store_ground_truth_bulk(
                pd.DataFrame(
                    [
                        {
                            "theater_id": tid,
                            "date": d,
                            "audience_count": 50.0,
                        }
                    ]
                )
            )
        target = date(2024, 3, 1)
        val = computer.compute_rolling_mean_single(tid, target, window=7)
        assert val == 50.0, "Rolling mean of constant values should be the constant"

    def test_compute_rolling_std_known_values(self, seeded_history):
        history, tid = seeded_history
        state = RollingWindowState(history)
        computer = RollingFeatureComputer(state)
        vals = [10, 20, 10, 20, 10, 20, 10]
        for i, v in enumerate(vals, 1):
            d = date(2024, 3, 1) - timedelta(days=i)
            history.store_ground_truth_bulk(
                pd.DataFrame(
                    [
                        {
                            "theater_id": tid,
                            "date": d,
                            "audience_count": float(v),
                        }
                    ]
                )
            )
        target = date(2024, 3, 1)
        val = computer.compute_rolling_std_single(tid, target, window=7)
        expected = np.std(vals, ddof=1)
        assert abs(val - expected) < 0.01, f"Expected std ~{expected:.2f}, got {val:.2f}"

    def test_compute_rolling_mean_respects_shift(self, seeded_history):
        history, tid = seeded_history
        state = RollingWindowState(history)
        computer = RollingFeatureComputer(state)
        for i in range(1, 15):
            d = date(2024, 3, 1) - timedelta(days=i)
            history.store_ground_truth_bulk(
                pd.DataFrame(
                    [
                        {
                            "theater_id": tid,
                            "date": d,
                            "audience_count": float(i * 10),
                        }
                    ]
                )
            )
        target = date(2024, 3, 1)
        val_shift1 = computer.compute_rolling_mean_single(tid, target, window=7, shift=1)
        val_shift2 = computer.compute_rolling_mean_single(tid, target, window=7, shift=2)
        assert val_shift1 != val_shift2, "Different shifts should give different results"

    def test_cold_start_returns_default(self, seeded_history):
        history, tid = seeded_history
        state = RollingWindowState(history)
        computer = RollingFeatureComputer(state)
        target = date(2024, 5, 1)
        val = computer.compute_rolling_mean_single(
            tid, target, window=7, cold_start_default=30.0
        )
        assert val == 30.0

    def test_compute_batch_shape(self, seeded_history):
        history, tid = seeded_history
        state = RollingWindowState(history)
        computer = RollingFeatureComputer(state)
        df = computer.compute_batch(
            [tid, "book_00002"],
            [date(2024, 3, 1), date(2024, 3, 2)],
            windows_mean=[7, 14],
            windows_std=[7],
        )
        assert len(df) == 4
        expected_cols = [
            "theater_id",
            "date",
            "rolling_mean_7",
            "rolling_mean_14",
            "rolling_std_7",
        ]
        for c in expected_cols:
            assert c in df.columns
