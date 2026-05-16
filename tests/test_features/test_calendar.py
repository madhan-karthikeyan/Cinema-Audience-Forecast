from datetime import date

from app.features.calendar import CalendarFeatureComputer


class TestCalendarFeatureComputer:
    def test_single_date(self):
        computer = CalendarFeatureComputer()
        df = computer.compute([date(2024, 3, 15)])
        assert len(df) == 1
        assert df.iloc[0]["day"] == 15
        assert df.iloc[0]["month"] == 3
        assert df.iloc[0]["week_of_year"] == 11

    def test_weekend_detection(self):
        computer = CalendarFeatureComputer()
        saturday = date(2024, 3, 16)
        sunday = date(2024, 3, 17)
        monday = date(2024, 3, 18)
        df = computer.compute([saturday, sunday, monday])
        assert df.iloc[0]["is_weekend"] == 1
        assert df.iloc[1]["is_weekend"] == 1
        assert df.iloc[2]["is_weekend"] == 0

    def test_cyclic_encoding_bounds(self):
        computer = CalendarFeatureComputer()
        df = computer.compute([date(2024, 1, 1), date(2024, 7, 1)])
        for col in ["dow_sin", "dow_cos", "month_sin", "month_cos"]:
            assert all(-1.0 <= v <= 1.0 for v in df[col])

    def test_sin_squared_plus_cos_squared(self):
        computer = CalendarFeatureComputer()
        df = computer.compute([date(2024, 3, 15)])
        dow_sum = df.iloc[0]["dow_sin"] ** 2 + df.iloc[0]["dow_cos"] ** 2
        month_sum = df.iloc[0]["month_sin"] ** 2 + df.iloc[0]["month_cos"] ** 2
        assert abs(dow_sum - 1.0) < 0.001
        assert abs(month_sum - 1.0) < 0.001

    def test_feature_names_complete(self):
        names = CalendarFeatureComputer.feature_names()
        expected = [
            "day_of_week",
            "is_weekend",
            "month",
            "day",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
            "week_of_year",
        ]
        assert names == expected

    def test_multiple_dates(self):
        computer = CalendarFeatureComputer()
        dates = [date(2024, m, 1) for m in range(1, 13)]
        df = computer.compute(dates)
        assert len(df) == 12
        assert list(df["month"]) == list(range(1, 13))

    def test_output_columns(self):
        computer = CalendarFeatureComputer()
        df = computer.compute([date(2024, 6, 15)])
        expected_cols = {
            "date",
            "day_of_week",
            "is_weekend",
            "month",
            "day",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
            "week_of_year",
        }
        assert set(df.columns) == expected_cols
