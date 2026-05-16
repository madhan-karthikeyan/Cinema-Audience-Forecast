import shutil
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.features.builder import FeaturePipeline
from app.features.schema import FeatureSchema
from app.features.state import RollingWindowState
from app.main import create_app
from app.storage.history import HistoryStore


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_theater_id():
    return "book_00001"


@pytest.fixture
def temp_dir():
    tmp = Path(tempfile.mkdtemp())
    yield tmp
    shutil.rmtree(str(tmp))


@pytest.fixture
def history_store(temp_dir):
    store_path = temp_dir / "history"
    return HistoryStore(parquet_path=store_path)


@pytest.fixture
def seeded_history(history_store):
    theater_id = "book_00001"
    base = date(2024, 1, 1)
    rows = []
    for i in range(60):
        d = base + timedelta(days=i)
        rows.append(
            {
                "theater_id": theater_id,
                "date": d,
                "audience_count": float(np.random.poisson(30)),
            }
        )
    df = pd.DataFrame(rows)
    history_store.store_ground_truth_bulk(df)
    return history_store, theater_id


@pytest.fixture
def rolling_state(seeded_history):
    history, tid = seeded_history
    return RollingWindowState(history), tid


@pytest.fixture
def default_schema():
    return FeatureSchema(
        version="1.0.0",
        feature_names=[
            "lag_7",
            "lag_14",
            "lag_21",
            "lag_28",
            "rolling_mean_7",
            "rolling_mean_14",
            "rolling_mean_28",
            "rolling_std_7",
            "day_of_week",
            "is_weekend",
            "month",
            "day",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
            "tickets_sold_daily",
            "tickets_booked_daily",
            "theater_type_standard",
            "theater_type_premium",
            "theater_area",
            "week_of_year",
        ],
        feature_dtypes={
            "lag_7": "float64",
            "lag_14": "float64",
            "day_of_week": "int32",
            "rolling_mean_7": "float64",
        },
        cold_start_defaults={
            "lag_7": 25.0,
            "lag_14": 25.0,
            "lag_21": 25.0,
            "lag_28": 25.0,
            "rolling_mean_7": 25.0,
            "rolling_mean_14": 25.0,
            "rolling_mean_28": 25.0,
            "rolling_std_7": 15.0,
        },
        target_column="audience_count",
        training_metrics={"rmse": 21.6, "r2": 0.54},
    )


@pytest.fixture
def feature_pipeline(seeded_history, default_schema):
    history, _ = seeded_history
    return FeaturePipeline(history_store=history, schema=default_schema)


@pytest.fixture
def prediction_dates():
    return [date(2024, 3, 1) + timedelta(days=i) for i in range(61)]


@pytest.fixture
def sample_theater_ids():
    return [f"book_{i:05d}" for i in range(1, 4)]


@pytest.fixture
def two_week_window():
    base = date(2024, 3, 1)
    return [base + timedelta(days=i) for i in range(14)]
