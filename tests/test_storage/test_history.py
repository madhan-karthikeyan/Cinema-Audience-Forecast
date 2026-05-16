from datetime import date, timedelta

import pandas as pd
import pytest


class TestHistoryStore:
    def test_store_and_retrieve_ground_truth(self, history_store):
        tid = "book_00001"
        d = date(2024, 1, 15)
        df = pd.DataFrame(
            [{"theater_id": tid, "date": d, "audience_count": 42.0}]
        )
        history_store.store_ground_truth_bulk(df)
        val = history_store.get_ground_truth(tid, d)
        assert val == 42.0

    def test_store_and_retrieve_prediction(self, history_store):
        tid = "book_00001"
        d = date(2024, 3, 1)
        df = pd.DataFrame(
            [{"theater_id": tid, "date": d, "predicted": 35.0}]
        )
        history_store.store_predictions(df)
        val = history_store.get_prediction(tid, d)
        assert val == 35.0

    def test_get_missing_ground_truth_returns_none(self, history_store):
        val = history_store.get_ground_truth("nonexistent", date(2024, 5, 1))
        assert val is None

    def test_get_lag_value_prefers_truth(self, history_store):
        tid = "book_00001"
        d = date(2024, 3, 8)
        lag_date = date(2024, 3, 1)
        history_store.store_ground_truth_bulk(
            pd.DataFrame(
                [{"theater_id": tid, "date": lag_date, "audience_count": 50.0}]
            )
        )
        history_store.store_predictions(
            pd.DataFrame(
                [{"theater_id": tid, "date": lag_date, "predicted": 10.0}]
            )
        )
        val = history_store.get_lag_value(tid, d, 7)
        assert val == 50.0

    def test_get_lag_value_falls_back_to_prediction(self, history_store):
        tid = "book_00001"
        d = date(2024, 3, 8)
        lag_date = date(2024, 3, 1)
        history_store.store_predictions(
            pd.DataFrame(
                [{"theater_id": tid, "date": lag_date, "predicted": 33.0}]
            )
        )
        val = history_store.get_lag_value(tid, d, 7)
        assert val == 33.0

    def test_get_lag_value_cold_start(self, history_store):
        val = history_store.get_lag_value(
            "nonexistent", date(2024, 5, 1), 7, cold_start_default=99.0
        )
        assert val == 99.0

    def test_get_history_window_length(self, history_store):
        tid = "book_00001"
        for i in range(35):
            d = date(2024, 2, 1) + timedelta(days=i)
            history_store.store_ground_truth_bulk(
                pd.DataFrame(
                    [{"theater_id": tid, "date": d, "audience_count": float(20 + i)}]
                )
            )
        vals = history_store.get_history_window(
            tid, date(2024, 3, 1), window_size=28
        )
        assert len(vals) == 28

    def test_get_mixed_history_length(self, history_store):
        tid = "book_00001"
        for i in range(28):
            d = date(2024, 2, 2) + timedelta(days=i)
            history_store.store_ground_truth_bulk(
                pd.DataFrame(
                    [{"theater_id": tid, "date": d, "audience_count": float(30)}]
                )
            )
        history_store.store_predictions(
            pd.DataFrame(
                [{"theater_id": tid, "date": date(2024, 3, 1), "predicted": 40.0}]
            )
        )
        mixed = history_store.get_mixed_history(tid, date(2024, 3, 1), lookback=28)
        assert len(mixed) == 28

    def test_store_bulk_predictions(self, history_store):
        n = 5
        df = pd.DataFrame(
            {
                "theater_id": ["book_00001"] * n,
                "date": [date(2024, 3, i) for i in range(1, n + 1)],
                "predicted": [float(i * 10) for i in range(1, n + 1)],
            }
        )
        history_store.store_predictions(df)
        for i in range(1, n + 1):
            val = history_store.get_prediction("book_00001", date(2024, 3, i))
            assert val == float(i * 10)

    def test_store_predictions_missing_columns(self, history_store):
        with pytest.raises(ValueError, match="Missing"):
            history_store.store_predictions(
                pd.DataFrame({"theater_id": ["x"], "date": [date(2024, 1, 1)]})
            )

    def test_overwrite_existing_record(self, history_store):
        tid = "book_00001"
        d = date(2024, 1, 1)
        history_store.store_ground_truth_bulk(
            pd.DataFrame(
                [{"theater_id": tid, "date": d, "audience_count": 10.0}]
            )
        )
        history_store.store_predictions(
            pd.DataFrame(
                [{"theater_id": tid, "date": d, "predicted": 20.0}]
            )
        )
        assert history_store.get_ground_truth(tid, d) == 10.0
        assert history_store.get_prediction(tid, d) == 20.0

    def test_cold_start_fallback_returns_float(self, history_store):
        val = history_store.cold_start_fallback("book_new")
        assert isinstance(val, float)
        assert val > 0

    def test_get_all_theater_ids(self, history_store):
        for tid in ["book_a", "book_b", "book_c"]:
            history_store.store_ground_truth_bulk(
                pd.DataFrame(
                    [
                        {
                            "theater_id": tid,
                            "date": date(2024, 1, 1),
                            "audience_count": 30.0,
                        }
                    ]
                )
            )
        ids = history_store.get_all_theater_ids()
        assert "book_a" in ids
        assert "book_b" in ids
        assert "book_c" in ids
