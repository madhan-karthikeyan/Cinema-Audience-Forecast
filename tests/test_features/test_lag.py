from datetime import date, timedelta

from app.features.lag import LagFeatureComputer
from app.features.state import RollingWindowState


class TestLagFeatureComputer:
    def test_is_lag_feature_detection(self):
        assert LagFeatureComputer.is_lag_feature("lag_7") is True
        assert LagFeatureComputer.is_lag_feature("lag_14") is True
        assert LagFeatureComputer.is_lag_feature("lag_100") is True
        assert LagFeatureComputer.is_lag_feature("roll_mean_7") is False
        assert LagFeatureComputer.is_lag_feature("day_of_week") is False

    def test_parse_lag_days(self):
        assert LagFeatureComputer.parse_lag_days("lag_7") == 7
        assert LagFeatureComputer.parse_lag_days("lag_14") == 14
        assert LagFeatureComputer.parse_lag_days("lag_28") == 28
        assert LagFeatureComputer.parse_lag_days("lag_100") == 100
        assert LagFeatureComputer.parse_lag_days("rolling_mean_7") is None
        assert LagFeatureComputer.parse_lag_days("random") is None

    def test_compute_single_with_truth(self, seeded_history):
        history, tid = seeded_history
        state = RollingWindowState(history)
        computer = LagFeatureComputer(state)
        target = date(2024, 2, 1)
        val = computer.compute_single(tid, target, 7)
        assert isinstance(val, float)
        assert val >= 0

    def test_compute_single_cold_start(self, seeded_history):
        history, tid = seeded_history
        state = RollingWindowState(history)
        computer = LagFeatureComputer(state)
        target = date(2024, 5, 1)
        val = computer.compute_single(tid, target, 7, cold_start_default=42.0)
        assert val == 42.0

    def test_compute_single_with_prediction_buffer(self, seeded_history):
        history, tid = seeded_history
        state = RollingWindowState(history)
        computer = LagFeatureComputer(state)
        target = date(2024, 3, 15)
        state.append_prediction(tid, target - timedelta(days=7), 55.0)
        val = computer.compute_single(tid, target, 7, cold_start_default=0.0)
        assert val == 55.0

    def test_compute_batch_shape(self, seeded_history):
        pass

    def test_compute_batch_all_lags(self, seeded_history):
        history, tid = seeded_history
        state = RollingWindowState(history)
        computer = LagFeatureComputer(state)
        theater_ids = [tid, "book_00002"]
        dates = [date(2024, 3, 1), date(2024, 3, 2)]
        df = computer.compute_batch(theater_ids, dates, [7, 14, 28])
        assert len(df) == 4
        assert list(df.columns) == ["theater_id", "date", "lag_7", "lag_14", "lag_28"]
        assert all(df["lag_7"] > 0)

    def test_compute_batch_with_defaults(self, seeded_history):
        history, tid = seeded_history
        state = RollingWindowState(history)
        computer = LagFeatureComputer(state)
        df = computer.compute_batch(
            [tid],
            [date(2024, 5, 1)],
            [7],
            cold_start_defaults={"lag_7": 99.0},
        )
        assert df.iloc[0]["lag_7"] == 99.0
