from datetime import date, timedelta

import pandas as pd

from app.features.state import RollingWindowState


class TestRollingWindowState:
    def test_mixed_history_prefers_ground_truth(self, seeded_history):
        history, tid = seeded_history
        state = RollingWindowState(history)
        target = date(2024, 2, 15)
        history.store_ground_truth_bulk(
            pd.DataFrame(
                [
                    {
                        "theater_id": tid,
                        "date": target - timedelta(days=7),
                        "audience_count": 42.0,
                    }
                ]
            )
        )
        history.store_predictions(
            pd.DataFrame(
                [
                    {
                        "theater_id": tid,
                        "date": target - timedelta(days=7),
                        "predicted": 10.0,
                    }
                ]
            )
        )
        val = state.resolve_lag(tid, target, 7)
        assert val == 42.0, "Should prefer ground truth over prediction"

    def test_lag_falls_back_to_prediction(self, seeded_history):
        history, tid = seeded_history
        state = RollingWindowState(history)
        target = date(2024, 3, 10)
        history.store_predictions(
            pd.DataFrame(
                [
                    {
                        "theater_id": tid,
                        "date": target - timedelta(days=7),
                        "predicted": 35.0,
                    }
                ]
            )
        )
        val = state.resolve_lag(tid, target, 7)
        assert val == 35.0, "Should fall back to prediction when truth unavailable"

    def test_cold_start_returns_default(self, seeded_history):
        history, tid = seeded_history
        state = RollingWindowState(history)
        target = date(2024, 5, 1)
        val = state.resolve_lag(tid, target, 999, cold_start_default=50.0)
        assert val == 50.0, "Should return configured default for missing data"

    def test_prediction_buffer_takes_priority(self, seeded_history):
        history, tid = seeded_history
        state = RollingWindowState(history)
        target = date(2024, 3, 10)
        state.append_prediction(tid, target - timedelta(days=7), 99.0)
        val = state.resolve_lag(tid, target, 7, cold_start_default=0.0)
        assert val == 99.0, "In-memory buffer should take priority"

    def test_append_and_flush_predictions(self, seeded_history):
        history, tid = seeded_history
        state = RollingWindowState(history)
        state.append_prediction(tid, date(2024, 3, 1), 30.0)
        state.append_prediction(tid, date(2024, 3, 2), 35.0)
        state.flush_predictions()
        stored = history.get_prediction(tid, date(2024, 3, 1))
        assert stored == 30.0
        stored2 = history.get_prediction(tid, date(2024, 3, 2))
        assert stored2 == 35.0

    def test_get_window_values_respects_shift(self, seeded_history):
        history, tid = seeded_history
        state = RollingWindowState(history)
        target = date(2024, 3, 10)
        vals_7 = state.get_window_values(tid, target, window_days=7, shift=1)
        assert len(vals_7) == 7
        vals_14 = state.get_window_values(tid, target, window_days=14, shift=1)
        assert len(vals_14) == 14

    def test_get_mixed_series_no_nans(self, seeded_history):
        history, tid = seeded_history
        state = RollingWindowState(history)
        target = date(2024, 3, 1)
        series = state.get_mixed_series(tid, target, lookback=28)
        import numpy as np

        assert not np.any(np.isnan(series)), "Mixed series should have no NaN values"

    def test_prediction_propagation(self, seeded_history):
        history, tid = seeded_history
        state = RollingWindowState(history)
        for day_offset in range(1, 15):
            d = date(2024, 3, 1) + timedelta(days=day_offset)
            history.store_predictions(
                pd.DataFrame(
                    [
                        {
                            "theater_id": tid,
                            "date": d,
                            "predicted": float(day_offset * 10),
                        }
                    ]
                )
            )
        target = date(2024, 3, 15)
        lag_7 = state.resolve_lag(tid, target, 7)
        offset_7 = 7
        expected = offset_7 * 10
        assert lag_7 == expected, f"Lag-7 should use prediction: expected {expected}, got {lag_7}"
        lag_14 = state.resolve_lag(tid, target, 14)
        assert lag_14 == 25.0, "Lag-14 has no prediction stored, should return cold-start default"

    def test_flush_clears_buffer(self, seeded_history):
        history, tid = seeded_history
        state = RollingWindowState(history)
        state.append_prediction(tid, date(2024, 3, 1), 30.0)
        state.flush_predictions()
        assert tid not in state._prediction_buffer or not state._prediction_buffer[tid]
