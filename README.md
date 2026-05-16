# Cinema Audience Forecast

Production-grade ML inference serving for cinema audience forecasting. Transforms a notebook-based forecasting model into a containerized, observable, ensemble inference platform with stateful feature reconstruction.

**Three-model ensemble** (LightGBM, XGBoost, CatBoost) with lag-blending, deployed as a FastAPI service with Prometheus metrics, structured logging, and Grafana dashboards.

## Architecture

```
Client → FastAPI API → EnsembleOrchestrator → FeaturePipeline → HistoryStore
                          │                        │
                     ┌────┴────┐              ┌────┴────┐
                     │ LightGBM │              │  Lag    │
                     │ XGBoost  │              │ Rolling │
                     │ CatBoost │              │Calendar │
                     └────┬────┘              └─────────┘
                          │
                     ┌────┴────┐
                     │ Blender │  (0.8×ensemble + 0.2×lag_7)
                     └─────────┘
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Parquet-partitioned HistoryStore** | O(1) lookup per (theater, date); atomic overwrite |
| **RollingWindowState in-memory buffer** | Dict-based prediction cache avoids repeated Parquet I/O across sequential batch dates |
| **ThreadPoolExecutor(3) for ensemble** | LightGBM/XGBoost/CatBoost release GIL during predict(); threading avoids multiprocess serialization |
| **feature_schema.json contract** | Single source of truth for feature names, dtypes, cold-start defaults; shared between train and serve |
| **Truth>prediction>default resolution** | Lag features use ground truth when available, prior predictions when not, configured defaults for cold-start |
| **Shift-1 rolling semantics** | `shift(1).rolling(N).mean()` matches notebook — no data leakage from current day |
| **Graceful degradation** | 3-model avg → 2-model → single → lag_7 fallback → clip to defaults — each level logged and metrified |
| **α=0.2 blend** | Matches notebook exactly: `(1-α) * ensemble_pred + α * lag_7` |

## Directory Structure

```
app/
├── main.py                     # FastAPI app factory, lifespan, middleware
├── config.py                   # pydantic-settings (FORECAST_ prefix)
├── api/
│   ├── routes.py               # 9 HTTP endpoints
│   ├── deps.py                 # Dependency injection
│   └── errors.py               # Exception handlers
├── models/
│   ├── schemas.py              # Pydantic request/response models
│   └── registry.py             # Model registry with versioning
├── features/
│   ├── builder.py              # FeaturePipeline orchestrator
│   ├── lag.py                  # LagFeatureComputer
│   ├── rolling.py              # RollingFeatureComputer
│   ├── calendar.py             # CalendarFeatureComputer
│   ├── schema.py               # FeatureSchema (train-serve contract)
│   └── state.py                # RollingWindowState (truth/prediction mix)
├── inference/
│   ├── pipeline.py             # InferencePipeline (feature→predict→blend→store)
│   ├── orchestrator.py         # EnsembleOrchestrator (parallel predict)
│   └── blender.py              # Blender (lag blend + clip)
├── storage/
│   ├── history.py              # HistoryStore (Parquet)
│   └── cache.py                # AsyncRedisCache
├── monitoring/
│   ├── metrics.py              # 15 Prometheus metrics
│   └── logging.py              # loguru (stdout + rotation)
├── middleware/
│   └── tracing.py              # RequestIDMiddleware
└── scheduler/
    └── tasks.py                # APScheduler daily batch

deployment/
├── Dockerfile                  # Multi-stage (builder + runtime)
├── docker-compose.yml          # API + Redis + Prometheus + Grafana
├── prometheus/
│   ├── prometheus.yml
│   └── alerts.yml              # 5 alert rules
└── grafana/
    ├── dashboards/
    └── datasources/

tests/
├── conftest.py                 # Shared fixtures
├── test_api/                   # Routes + health + errors
├── test_features/              # Lag, rolling, calendar, schema, state, builder
├── test_inference/             # Orchestrator, blender, pipeline
├── test_monitoring/            # Metrics, logging
└── test_storage/               # HistoryStore, cache

models/
└── registry.json               # Model manifest

scripts/
├── seed_history.py             # Initialize HistoryStore from training data
└── export_models.py            # Export trained models to registry
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & docker-compose (optional, for full stack)

### Install & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Install dev dependencies (for testing/lint)
pip install -r requirements-dev.txt

# Run API server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
# Full stack (API + Redis + Prometheus + Grafana)
docker compose -f deployment/docker-compose.yml up --build

# Or just the API
docker build -f deployment/Dockerfile -t cinema-forecast-api .
docker run -p 8000:8000 cinema-forecast-api
```

### Makefile

```bash
make install        # Install production deps
make install-dev    # Install dev deps
make test           # Run tests with coverage
make lint           # Run ruff linter
make run            # docker-compose up
make api            # uvicorn directly
```

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/health` | Deep health check (models loaded, schema valid) |
| GET | `/v1/ready` | Lightweight liveness probe |
| GET | `/v1/models` | List registered models |
| GET | `/v1/models/{name}` | Model details + version history |
| POST | `/v1/predict/batch` | Multi-theater batch forecast |
| GET | `/v1/predict/theater/{id}` | Single theater forecast |
| POST | `/v1/predict/theater/{id}` | Single theater with custom config |
| GET | `/v1/metrics` | Prometheus scrape endpoint |
| GET | `/v1/docs` | OpenAPI documentation |

### Batch Prediction

```bash
curl -X POST http://localhost:8000/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_dates": ["2024-03-01", "2024-03-02"],
    "theater_ids": ["book_00001", "book_00002"],
    "blend_alpha": 0.2
  }'
```

### Single Prediction

```bash
curl "http://localhost:8000/v1/predict/theater/book_00001?target_date=2024-04-15"
```

## Feature Pipeline

The feature reconstruction pipeline is the core engineering challenge. During inference, lag features must use predictions (not ground truth) for future dates:

```
Training:  feature[t] = target[t-7]       (known ground truth)
Inference: feature[t] = prediction[t-7]   (model output from 7 days ago)
```

### Feature Types

| Type | Features | Deterministic |
|------|----------|--------------|
| **Lag** | `lag_7`, `lag_14`, `lag_21`, `lag_28` | No (depends on predictions) |
| **Rolling** | `rolling_mean_7/14/28`, `rolling_std_7` | No (depends on lag buffer) |
| **Calendar** | `day_of_week`, `is_weekend`, `month`, `day`, `dow_sin/cos`, `month_sin/cos`, `week_of_year` | Yes |
| **Static** | `theater_type_standard`, `theater_type_premium`, `theater_area` | Yes (metadata) |

### Cold-Start Strategy

| History Available | Lag Resolution |
|---|---|
| 28+ days | Truth → Prediction (mixed) |
| 7-27 days | Partial truth + defaults |
| < 7 days | `cold_start_defaults` from schema |
| New theater | Global mean fallback |

## Configuration

All config via environment variables with `FORECAST_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `FORECAST_LOG_LEVEL` | `INFO` | Log level |
| `FORECAST_REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `FORECAST_MODEL_PATH` | `models` | Model artifact directory |
| `FORECAST_FEATURE_SCHEMA_PATH` | `feature_schema.json` | Feature schema path |
| `FORECAST_HISTORY_STORE_PATH` | `data/processed` | History parquet store |
| `FORECAST_PREDICTION_STORE_PATH` | `data/predictions` | Prediction output path |
| `FORECAST_DEFAULT_BLEND_ALPHA` | `0.2` | Lag-7 blend weight |
| `FORECAST_PREDICTION_CLIP_MIN` | `0.0` | Lower prediction bound |
| `FORECAST_MAX_BATCH_THEATERS` | `500` | Max batch size |
| `FORECAST_RATE_LIMIT_PER_SECOND` | `100` | Request rate limit |

## Testing

```bash
# Run all tests
make test

# Run with coverage
python -m pytest tests/ -v --cov=app --cov-report=term-missing

# Run specific test file
python -m pytest tests/test_features/test_lag.py -v
```

## Observability

### Metrics (15 Prometheus metrics)

- `forecast_requests_total` — Request count by endpoint + status
- `forecast_prediction_latency_seconds` — Latency histogram by endpoint + model
- `forecast_model_inference_seconds` — Per-model inference time
- `forecast_feature_computation_seconds` — Feature computation by type
- `forecast_model_loaded` — Binary gauge per model
- `forecast_ensemble_fallback_total` — Fallback event counter
- `forecast_batch_size` — Predictions per batch
- `forecast_last_batch_timestamp_seconds` — Last successful batch time

### Alerts (5 rules)

- `HighPredictionLatency` — p95 > 30s
- `EnsembleDegraded` — Fallback rate > 10%
- `FeatureDriftDetected` — KS statistic > 0.15
- `CacheHitRatioLow` — Hit ratio < 50%
- `BatchNotRunning` — No batch in 25 hours

## Deployment Stack

```
┌──────────┐    ┌──────────┐    ┌───────────┐    ┌────────┐
│  FastAPI  │───▶│  Redis   │    │ Prometheus │    │ Grafana│
│  :8000    │    │  :6379   │    │  :9090     │    │ :3000  │
└──────────┘    └──────────┘    └───────────┘    └────────┘
```

## Feature Schema Contract

`feature_schema.json` is the canonical contract between training and inference — cold-start defaults, dtypes, and column order are all defined here:

```json
{
  "version": "1.0.0",
  "feature_names": ["lag_7", "lag_14", ..., "week_of_year"],
  "cold_start_defaults": {"lag_7": 25.0, ...},
  "training_metrics": {"rmse": 21.6, "r2": 0.54}
}
```

## License

MIT
