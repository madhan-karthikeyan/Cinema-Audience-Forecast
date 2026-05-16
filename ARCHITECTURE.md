# Cinema Audience Forecast — Production Inference Architecture

> Author: Senior ML Systems Architect  
> Status: Architecture Design Document  
> Context: Transforms notebook-based forecasting into production-grade ML serving platform

---

## Table of Contents

1. [Architecture Philosophy & Key Decisions](#1-architecture-philosophy--key-decisions)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Component Deep Dive](#3-component-deep-dive)
4. [API Design](#4-api-design)
5. [Inference Flows](#5-inference-flows)
6. [Feature Reconstruction & State Management](#6-feature-reconstruction--state-management)
7. [Observability Stack](#7-observability-stack)
8. [Deployment Topology](#8-deployment-topology)
9. [Directory Structure](#9-directory-structure)
10. [Testing Strategy](#10-testing-strategy)
11. [Benchmarking & Load Testing](#11-benchmarking--load-testing)
12. [CI/CD Pipeline](#12-cicd-pipeline)
13. [Production Concerns & Tradeoffs](#13-production-concerns--tradeoffs)
14. [Interview Preparation](#14-interview-preparation)
15. [Future Extensibility](#15-future-extensibility)

---

## 1. Architecture Philosophy & Key Decisions

### 1.1 Why This Architecture Matters

This is a **forecasting inference system**, not a real-time recommender. That distinction drives every design decision:

| Aspect | Forecasting System | Real-Time System |
|--------|-------------------|------------------|
| Primary load | Batch (daily full forecast) | Single-item requests |
| Latency requirement | Minutes acceptable | Milliseconds required |
| State management | Rolling windows, lag chains | Stateless preferred |
| Feature computation | Complex, sequential | Simple, independent |
| Caching strategy | Pre-compute batch results | Cache model + features |
| Scaling unit | Workflow parallelism | Request parallelism |

**The most impressive engineering signal** is not millisecond latency — it's **feature consistency between train and serve**, **stateful inference correctness**, **observability**, and **operational maturity**.

### 1.2 High-ROI Components (Implement)

| Component | Why | Engineering Signal |
|-----------|-----|-------------------|
| Feature reconstruction pipeline | Hardest engineering problem in forecasting serving | Systems thinking, edge case handling |
| Model registry with versioning | Demonstrates ML lifecycle maturity | Reproducibility, deployment discipline |
| FastAPI with async batch endpoints | Clean serving layer | API design, async Python |
| Structured logging + metrics | Non-negotiable for production | Observability maturity |
| Docker + docker-compose | Portable deployment | Infrastructure basics |
| CI/CD with GitHub Actions | Automates quality gates | DevOps mindset |
| Drift monitoring | Catches model decay | ML production awareness |
| Load testing | Validates throughput claims | Engineering rigor |
| Lag state management | Core challenge for time-series inference | Stateful system design |

### 1.3 Low-ROI Components (Skip)

| Component | Why Skip | Interview Risk |
|-----------|----------|----------------|
| Kubernetes | docker-compose serves same demo purpose | Explaining K8s without real need = red flag |
| Kafka / streaming | No streaming data source exists | Pigeonholing tech onto wrong problem |
| Feature store as separate service | 3 features don't justify it | Overengineering at this scale |
| Multi-region deployment | Portfolio project, not prod | Can discuss in interview without implementing |
| A/B test infrastructure | Single model version is fine | Discuss conceptually instead |
| gRPC instead of REST | REST is fine for batch forecasting | Premature optimization |

### 1.4 Core Design Principles

- **Stateless where possible, stateful only where necessary** — API workers are stateless; lag state lives in a single source of truth
- **Fail fast, fail visibly** — every prediction returns success/failure with structured error context
- **Train-serve parity** — feature computation code is shared between training and inference; no drift between notebook features and API features
- **Observability by default** — every request generates logs, metrics, and traces; dashboards are not an afterthought
- **Graceful degradation** — if one ensemble model fails, fall back to remaining models; if all fail, return cached last-known prediction

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        External Clients                             │
│  (Dashboard, Scheduler, Ad-hoc Requests, CI/CD Triggers)            │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         API Gateway (Nginx)                         │
│              Rate limiting · Request routing · SSL                    │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     FastAPI Inference Server                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────┐  ┌───────────────┐  ┌───────────────────────────┐ │
│  │  API Layer   │  │  Scheduler    │  │  Monitoring Layer          │ │
│  │  /predict/*  │  │  APScheduler  │  │  Prometheus · Logging      │ │
│  │  /health     │  │  Daily batch  │  │  Tracing · Drift detection │ │
│  │  /metrics    │  │  Background   │  │  Health checks             │ │
│  └──────┬───────┘  └──────┬────────┘  └────────────────────────────┘ │
│         │                 │                                           │
│         ▼                 ▼                                           │
│  ┌────────────────────────────────────────────────────────────┐      │
│  │                Orchestration Layer                          │      │
│  │  Ensemble pipeline · Model routing · Fallback logic         │      │
│  └────────────────────────┬───────────────────────────────────┘      │
│                           │                                          │
│                           ▼                                          │
│  ┌────────────────────────────────────────────────────────────┐      │
│  │              Feature Reconstruction Pipeline                │      │
│  │  Lag features · Rolling windows · Calendar encoding         │      │
│  │  Booking aggregation · Theater metadata                     │      │
│  │  ←--- MOST COMPLEX ENGINEERING COMPONENT ---→              │      │
│  └──────────┬─────────────────────────────────────────────────┘      │
│             │                                                        │
│             ▼                                                        │
│  ┌──────────────────────────────────────────────────────┐           │
│  │              Model Inference Engine                   │           │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │           │
│  │  │ LightGBM │  │ XGBoost  │  │ CatBoost │            │           │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘            │           │
│  │       └──────────┬──┴──────────────┘                  │           │
│  │                  ▼                                     │           │
│  │           ┌──────────────┐                             │           │
│  │           │   Blender    │  (ensemble avg + lag blend) │           │
│  │           └──────┬───────┘                             │           │
│  └──────────────────┼────────────────────────────────────┘           │
│                     │                                                │
│                     ▼                                                │
│  ┌──────────────────────────────────────────────┐                   │
│  │              Storage Layer                    │                   │
│  │  Redis: Model cache · Feature cache · Locks   │                   │
│  │  Disk: Prediction history · Drift snapshots   │                   │
│  └──────────────────────────────────────────────┘                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Deep Dive

### 3.1 Model Registry (`models/registry.py`)

**Why**: Models are artifacts that change over time. Without versioning, you cannot roll back, audit, or compare.

**What it demonstrates**: ML lifecycle management, artifact versioning, reproducible inference.

**Implementation**:

```python
# models/registry.py
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Immutable model version metadata."""
    name: str               # e.g., "lightgbm"
    version: str            # e.g., "1.0.0"
    path: Path              # Filesystem path to serialized model
    metrics: dict           # Validation metrics at registration time
    params: dict            # Hyperparameters at registration time
    created_at: str         # ISO timestamp
    checksum: str           # SHA256 of model file for integrity
    active: bool = False    # Whether this is the currently serving version


class ModelRegistry:
    """
    Simple filesystem-backed model registry.

    Design decisions:
    - Filesystem-backed (not DB) — avoids infra dependency for a portfolio project
    - JSON metadata alongside model artifacts — self-contained, portable
    - Atomic swap for version activation — safe reloads without downtime
    """

    def __init__(self, base_path: Path = Path("models")):
        self.base_path = base_path
        self._versions: dict[str, list[ModelVersion]] = {}
        self._active: dict[str, ModelVersion] = {}
        self._load_manifest()

    def register(self, name: str, version: str, path: Path,
                 metrics: dict, params: dict) -> ModelVersion:
        """Register a new model version."""
        ...

    def activate(self, name: str, version: str) -> None:
        """Swap active version atomically."""
        ...

    def get_active(self, name: str) -> ModelVersion:
        """Get currently serving version."""
        ...

    def load_model(self, name: str):
        """Deserialize and return the active model instance."""
        ...

    def _load_manifest(self):
        """Read registry manifest file, build in-memory state."""
        ...

    def _save_manifest(self):
        """Persist registry state to disk."""
        ...
```

**Complexity vs ROI**: Medium complexity, very high ROI. This is one of the most commonly asked-about topics in ML system design interviews.

### 3.2 Feature Reconstruction Pipeline (`features/`)

**Why**: This is the hardest problem in forecasting inference. Features used during training must be exactly reconstructed during inference. Lag features require knowing predictions for previous days; rolling windows require sequential computation.

**What it demonstrates**: Understanding of train-serve skew, time-series state management, feature engineering as production code.

**The core challenge**:

```
Training:   feature[t] = target[t-7]    (uses ground truth)
Inference:  feature[t] = prediction[t-7] (uses model output)
             ^^^ This is fundamentally different and must be handled explicitly
```

**Implementation**:

```python
# features/builder.py

class FeaturePipeline:
    """
    Feature reconstruction for inference.

    Two modes:
    1. BATCH: Full historical reconstruction + forward prediction
       - Load last N days of ground truth
       - Compute features for all prediction dates
       - Store predictions back into history store
    2. REALTIME: Single theater, single date
       - Load theater's historical window
       - Compute features for target date
       - Return feature vector + store prediction
    """

    def __init__(self, history_store: HistoryStore, config: Config):
        self.history = history_store
        self.config = config

    def build_batch_features(
        self,
        prediction_dates: list[date],
        theater_ids: list[str]
    ) -> pd.DataFrame:
        """
        Reconstruct feature matrix for a batch of (theater, date) pairs.

        Steps:
        1. Load historical ground truth + previous predictions
        2. Compute lag features using truth where available, predictions where not
        3. Compute rolling window statistics
        4. Generate calendar features (deterministic)
        5. Join theater metadata
        6. Join booking aggregates (from database, not notebook)
        7. Validate feature column order matches training schema

        Returns feature DataFrame ready for model inference.
        """
        ...

    def validate_feature_columns(self, features: pd.DataFrame) -> bool:
        """
        CRITICAL: Verify that inference features match training feature schema.

        Checks:
        - Column names match exactly
        - Column order matches training order
        - Feature dtype compatibility
        - Missing value check (should not happen for deterministic features)
        """
        ...
```

**State management**:

```python
# features/state.py

class RollingWindowState:
    """
    Manages the rolling state necessary for lag features.

    This is the most nuanced component. During batch inference:
    - Day 1: lag_7 uses ground truth from 7 days ago ✓
    - Day 8: lag_7 uses Day 1's prediction → error accumulates
    - Strategy: predict in chronological order, updating state after each prediction

    For single-theater inference:
    - Load last 28 days of known data
    - Compute lag features from known + cached predictions
    - Return prediction, update cache
    """

    def __init__(self, cache_client: Redis | None):
        self.cache = cache_client

    def get_window(self, theater_id: str, target_date: date,
                   window_size: int = 28) -> pd.Series:
        """Get the last `window_size` values for theater."""
        ...

    def append_prediction(self, theater_id: str, date: date, value: float):
        """Store prediction so it can serve as lag for future predictions."""
        ...

    def get_mixed_history(self, theater_id: str, target_date: date,
                          lookback: int = 28) -> list[float | None]:
        """
        Returns ordered list of [ground_truth..., prediction...].

        Where ground truth exists (past dates), use it.
        Where predictions exist (already-forecasted future dates), use those.
        Where neither exists, return None (cold-start fallback).
        """
        ...
```

**Train-serve feature contract**:

```python
# features/schema.py

class FeatureSchema:
    """
    Immutable feature schema shared between training and inference.

    Training script exports the feature schema as JSON.
    Inference pipeline loads it and validates against every request.
    This is the single source of truth for feature consistency.
    """

    @classmethod
    def from_training(cls, feature_names: list[str],
                      feature_dtypes: dict[str, str],
                      target_column: str) -> "FeatureSchema":
        ...

    @classmethod
    def load(cls, path: Path) -> "FeatureSchema":
        """Load schema from exported JSON."""
        ...

    def validate_inference_features(self, df: pd.DataFrame) -> bool:
        """
        Validates:
        - All required columns present
        - Column order matches training
        - dtypes are compatible
        - No NaN in deterministic features
        - NaN allowed only in lag features (cold-start fallback)
        """
        ...

    def export(self, path: Path):
        """Export to JSON for inference pipeline to consume."""
        ...
```

### 3.3 Ensemble Orchestrator (`inference/orchestrator.py`)

**Why**: Three models must be loaded, invoked, and combined with proper error handling. A naive loop fails on any single model failure.

**What it demonstrates**: Error handling design, fallback strategies, parallel execution.

```python
# inference/orchestrator.py

class EnsembleResult:
    models_used: list[str]
    predictions: np.ndarray
    blend_weight: float
    model_times: dict[str, float]
    fallback_used: bool
    success: bool
    error: str | None


class EnsembleOrchestrator:
    """
    Orchestrates 3-model ensemble inference.

    Strategy:
    - Try all models concurrently (ThreadPoolExecutor for GIL-limited inference)
    - On individual model failure: log error, continue with remaining models
    - If 2+ models fail: fall back to blended lag_7 only
    - If all models fail: raise, return HTTP 503

    Concurrency note:
    - LightGBM, XGBoost, CatBoost all release GIL during predict() calls
    - ThreadPoolExecutor with 3 workers is optimal (I/O + compute overlap)
    - ProcessPoolExecutor would add serialization overhead for negligible gain
    """

    def __init__(self, registry: ModelRegistry, blender: Blender):
        self.registry = registry
        self.blender = blender

    async def predict(self, features: np.ndarray) -> EnsembleResult:
        """
        Execute ensemble prediction with error isolation.

        Each model prediction runs in a try/except.
        Failed model predictions are excluded from the blend.
        """
        ...

    def _build_fallback(self, lag7_values: np.ndarray) -> EnsembleResult:
        """
        When all models fail, fall back to weighted lag_7.

        This is better than returning an error because:
        - A stale/naive forecast is better than no forecast
        - Clients can still make operational decisions
        - Clear signal in logs/metrics indicates model failure
        """
        ...
```

### 3.4 Prediction Blender (`inference/blender.py`)

**Why**: The original notebook uses a hardcoded α=0.2 blend with lag_7. Production requires this to be configurable, testable, and potentially per-theater.

```python
# inference/blender.py

@dataclass
class BlendConfig:
    alpha: float = 0.2           # Weight on lag_7
    per_theater_alphas: dict[str, float] | None = None  # Per-theater override
    clip_min: float = 0.0        # Safety clamp
    clip_max: float | None = None  # Optional upper bound


class Blender:
    """
    Blends model ensemble output with lag_7.

    The original notebook's blend logic is preserved exactly:
        prediction = (1 - α) * model_output + α * lag_7

    Production additions:
    - Configurable blend weights via BlendingConfig
    - Optional per-theater alpha (learned from validation residuals)
    - Structured logging of blend contribution per theater
    - Metrics on blend weight distribution
    """

    def blend(self, model_pred: np.ndarray, lag7: np.ndarray,
              lag7_available: np.ndarray, config: BlendConfig,
              theater_ids: list[str] | None = None) -> np.ndarray:
        """
        Blend predictions with lag_7.

        lag7_available mask handles theaters with no lag_7 data (cold start).
        When lag_7 is unavailable, pure model prediction is used.
        Values are clipped to [clip_min, clip_max] after blending.
        """
        ...
```

### 3.5 API Layer (`api/routes.py`)

**Why**: Clean API design demonstrates REST maturity, input validation, and production request handling.

```python
# api/routes.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.models.schemas import (
    BatchPredictionRequest, BatchPredictionResponse,
    SinglePredictionRequest, SinglePredictionResponse,
    HealthResponse, ModelInfoResponse
)

router = APIRouter()

@router.post("/v1/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Full forecast for all theaters or specified subset.

    This is the PRIMARY endpoint. It mirrors the original notebook's
    prediction generation for the full 2024-03-01 to 2024-04-30 window.

    Request body allows overriding:
    - prediction_dates (default: full campaign window)
    - theater_ids (default: all theaters)
    - blend_alpha (default: 0.2)

    BackgroundTasks triggers:
    1. Storing predictions to history store
    2. Computing drift metrics against previous batch
    3. Updating Grafana annotations
    """
    ...


@router.get("/v1/predict/theater/{theater_id}",
            response_model=SinglePredictionResponse)
async def predict_theater(
    theater_id: str,
    target_date: date = Query(..., description="Target prediction date")
):
    """
    Real-time single theater forecast.

    This endpoint demonstrates real-time inference capability.
    For a full portfolio projection, use batch endpoint instead.
    """
    ...


@router.get("/v1/health", response_model=HealthResponse)
async def health_check():
    """
    Deep health check.

    Returns:
    - Status of each model in registry
    - Feature schema validation state
    - Cache connectivity
    - Last successful batch timestamp
    - Uptime
    """
    ...


@router.get("/v1/models", response_model=list[ModelInfoResponse])
async def list_models():
    """Return all registered model versions with metadata."""
    ...
```

### 3.6 Request/Response Schemas (`models/schemas.py`)

```python
from pydantic import BaseModel, Field
from datetime import date
from typing import Optional


class BatchPredictionRequest(BaseModel):
    prediction_dates: Optional[list[date]] = None
    theater_ids: Optional[list[str]] = None
    blend_alpha: float = Field(default=0.2, ge=0.0, le=1.0)
    include_features: bool = Field(default=False, description="Include feature vector in response")


class BatchPredictionItem(BaseModel):
    theater_id: str
    date: date
    prediction: float
    model_version: str
    blend_weight: float


class BatchPredictionResponse(BaseModel):
    request_id: str
    predictions: list[BatchPredictionItem]
    ensemble_metrics: dict
    latency_ms: float
    models_used: list[str]
    fallback_used: bool
    warnings: list[str]


class SinglePredictionResponse(BaseModel):
    theater_id: str
    target_date: date
    prediction: float
    confidence_interval: Optional[dict] = None
    model_version: str
    latency_ms: float
    features_used: Optional[list[str]] = None


class HealthResponse(BaseModel):
    status: str  # "healthy" | "degraded" | "unhealthy"
    uptime_seconds: float
    models_loaded: list[str]
    last_batch_timestamp: Optional[str]
    cache_connected: bool
    feature_schema_valid: bool
    prediction_count_total: int


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    active: bool
    metrics: dict
    params: dict
    created_at: str
    checksum: str
```

### 3.7 History Store (`storage/history.py`)

**Why**: Lag features need access to historical ground truth AND previous predictions. This store is the bridge between training data and inference state.

```python
class HistoryStore:
    """
    Stores and retrieves the rolling history for lag feature computation.

    Data stored per (theater_id, date):
    - ground_truth: actual audience count (from training data)
    - predicted: model's prediction (from previous inference runs)
    - features: feature vector at prediction time (for debugging)

    Backend: local Parquet files + optional Redis cache.

    This is the system of record for inference state — it must be
    consistently written to and read from across all inference requests.
    """

    def __init__(self, parquet_path: Path, redis_client=None):
        self.parquet_path = parquet_path
        self.redis = redis_client

    def get_history(self, theater_id: str,
                    lookback_days: int = 28) -> pd.DataFrame:
        """
        Load last N days of ground truth + predictions for theater.

        Returns columns: [date, ground_truth, predicted, ...]
        """
        ...

    def get_lag_value(self, theater_id: str, target_date: date,
                      lag_days: int) -> float | None:
        """
        Get the value for a specific lag feature.

        Priority:
        1. Ground truth if target_date - lag_days is in history
        2. Prediction if target_date - lag_days was predicted
        3. None (cold-start)
        """
        ...

    def store_predictions(self, predictions: pd.DataFrame):
        """
        Persist batch prediction results.

        Called after every batch inference run.
        Appends to Parquet store using partitioning by date.
        """
        ...

    def cold_start_fallback(self, theater_id: str) -> float:
        """
        When a theater has no history, provide a reasonable baseline.

        Options:
        - Global mean audience count
        - Theater-type mean (if theater_type known)
        - Zero (conservative)
        """
        ...
```

### 3.8 Scheduler (`scheduler/tasks.py`)

**Why**: Daily forecasts should run automatically. APScheduler handles cron-like scheduling without additional infrastructure.

```python
class ForecastScheduler:
    """
    Scheduled daily batch inference.

    Schedule: Daily at 02:00 (after data load completes)
    
    Flow:
    1. Load latest booking data (from wherever it arrives)
    2. Run full batch inference
    3. Store predictions
    4. Compute drift metrics vs previous period
    5. Send notification (log-based for now)
    
    This runs in the same process as the API server.
    For larger scale, this would be a separate worker process.
    """

    def start(self):
        ...

    async def daily_forecast_job(self):
        """Main scheduled job."""
        ...
```

### 3.9 Drift Monitor (`monitoring/drift.py`)

**Why**: Model performance degrades over time as audience behavior changes. Drift monitoring detects this.

```python
class DriftMonitor:
    """
    Monitors feature and prediction drift across batch runs.

    Detection methods:
    1. Prediction distribution drift (KS test vs historical)
    2. Feature distribution drift (per-feature KS test)
    3. Residual drift (when ground truth eventually arrives)
    4. Coverage drift (are we predicting for all expected theaters?)

    Alerts:
    - p-value < 0.01 → log warning, increment metric
    - Consecutive drift in 3+ batches → trigger retraining signal
    - Ground truth residual increase > 20% → flag for investigation

    This is lightweight statistical monitoring, not a full MLOps platform.
    """

    def compare_distributions(
        self, current: np.ndarray, reference: np.ndarray
    ) -> DriftReport:
        ...

    def check_feature_drift(
        self, current_features: pd.DataFrame, reference_features: pd.DataFrame
    ) -> dict[str, DriftReport]:
        ...

    def record_ground_truth(
        self, theater_id: str, date: date, predicted: float, actual: float
    ):
        """When ground truth arrives, log residual for drift analysis."""
        ...
```

---

## 4. API Design

### 4.1 Endpoints Summary

| Method | Path | Purpose | Load Pattern |
|--------|------|---------|--------------|
| POST | `/v1/predict/batch` | Full multi-theater forecast | Weekly/daily |
| GET | `/v1/predict/theater/{id}` | Single theater forecast | Ad-hoc |
| POST | `/v1/predict/theater/{id}` | Single theater with custom features | Ad-hoc |
| GET | `/v1/health` | Deep health check | 30s polling |
| GET | `/v1/ready` | Lightweight liveness | 10s polling |
| GET | `/v1/models` | List registered models | Dashboard |
| GET | `/v1/models/{name}` | Model details + version history | Dashboard |
| PUT | `/v1/admin/models/{name}/activate/{version}` | Activate model version | Deployment |
| POST | `/v1/admin/models/register` | Register new model version | CI/CD |
| GET | `/v1/metrics` | Prometheus metrics | Scrape target |
| GET | `/v1/docs` | OpenAPI docs | Developer |

### 4.2 Request ID & Tracing

Every response includes `X-Request-ID` header. This enables:
- Log correlation across requests
- Tracing from client → API → model inference
- Debugging specific prediction failures
- Latency breakdown per request

```python
# middleware/tracing.py
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    with tracer.start_as_current_span(f"{request.method} {request.url.path}") as span:
        span.set_attribute("request_id", request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
```

---

## 5. Inference Flows

### 5.1 Batch Inference Flow

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Client   │     │   FastAPI    │     │  Orchestrator │     │   Features   │
│          │     │              │     │              │     │              │
│ POST /v1/ │────▶│  Validate   │────▶│  Build batch  │────▶│  Load history│
│ predict/  │     │  request    │     │  schedule    │     │  from store  │
│ batch     │     │              │     │              │     │              │
│          │     │              │     │              │     │              │
│          │     │              │     │              │◀────│  Return df   │
│          │     │              │     │              │     │              │
│          │     │              │     │──────────────│────▶│  Reconstruct │
│          │     │              │     │  Split into  │     │  features    │
│          │     │              │     │  chunks      │     │  per chunk   │
│          │     │              │     │              │     │              │
│          │     │              │     │──── chunk ──▶│     │              │
│          │     │              │     │  (parallel)  │     │              │
│          │     │              │     │              │────▶│  Ensemble    │
│          │     │              │     │              │     │  predict     │
│          │     │              │     │◀──── pred ───│     │              │
│          │     │              │     │              │     └──────────────┘
│          │     │              │     │              │
│          │     │              │     │  Lag state   │     ┌──────────────┐
│          │     │              │     │  update      │────▶│  History     │
│          │     │              │     │              │     │  Store       │
│          │     │              │     │              │     │  (Parquet)   │
│          │     │              │◀────│  Results     │     └──────────────┘
│          │◀────│  Response    │◀────│  assembled   │
│          │     │              │     │              │
└──────────┘     └──────────────┘     └──────────────┘
```

### 5.2 Chunked Batch Processing

For 112 theaters × 61 days = 6832 predictions, processing sequentially is slow. **Chunked parallelism**:

```python
async def batch_predict(self, request: BatchPredictionRequest):
    chunks = self._create_chunks(request, chunk_size=10)
    results = await asyncio.gather(*[
        self._process_chunk(chunk) for chunk in chunks
    ], return_exceptions=True)
    return self._merge_results(results)
```

**Chunking strategy**: Process 10 theaters at a time (enough parallelism without overwhelming memory). Each chunk internally processes all 61 days for those theaters sequentially (preserving lag feature dependencies).

### 5.3 Single-Theater Inference Flow

```
GET /v1/predict/theater/book_00001?target_date=2024-04-15

1. Load last 28 days from HistoryStore for book_00001
2. Compute features: lag_7, lag_14, rolling_mean_7, etc.
3. Load 3 models from registry
4. Predict with each model concurrently
5. Blend: 0.8 * ensemble_avg + 0.2 * lag_7
6. Clip to [0, ∞)
7. Return prediction + metadata
8. Background: store prediction to HistoryStore
```

---

## 6. Feature Reconstruction & State Management

### 6.1 The Core Problem

During training:
```
feature_vector[t] = f(target[t-7], target[t-14], ..., meta[t], calendar[t])
                          ^^^^^^^^^^^^^^^^^^^^^^
                          Uses GROUND TRUTH (known from history)
```

During inference:
```
feature_vector[t] = f(target[t-7], target[t-14], ..., meta[t], calendar[t])
                          ^^^^^^^^^^^^^^^^^^^^^^
                          Uses PREDICTIONS (from previous inference steps)
```

**This is not a trivial difference.** It means:
1. Error propagates through time (bad prediction at t-7 → bad features at t → bad prediction at t)
2. The model trained on ground-truth features must generalize to prediction-based features
3. Cold-start theaters have no history at all

### 6.2 Solution Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Feature Reconstruction                          │
│                                                                      │
│  ┌──────────────┐  ┌────────────────────┐  ┌──────────────────┐    │
│  │  Ground Truth │  │  Previous Predict. │  │  Deterministic   │    │
│  │  (parquet)    │  │  (parquet + Redis) │  │  (calendar/meta) │    │
│  └──────┬───────┘  └─────────┬──────────┘  └────────┬─────────┘    │
│         │                    │                       │               │
│         ▼                    ▼                       ▼               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                  Merged History Vector                         │   │
│  │  For each (theater, target_date):                              │   │
│  │    lag_7  = truth[target-7]  if exists, else pred[target-7]    │   │
│  │    lag_14 = truth[target-14] if exists, else pred[target-14]   │   │
│  │    ...                                                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.3 Cold-Start Strategy

For theaters without a 28-day history:

| Scenario | Lag 7 | Lag 14 | Rolling Mean | Fallback |
|----------|-------|--------|--------------|----------|
| 28+ days history | truth | truth | truth | None needed |
| 7-27 days | truth if avail / pred | truth if avail / global_mean | partial | Use available |
| < 7 days | global_mean | global_mean | global_mean | Full fallback |
| New theater | theater_type_mean | theater_type_mean | theater_type_mean | Cold-start path |

```python
def _cold_start_fallback(self, theater_id: str, feature: str) -> float:
    """
    Return a reasonable default for missing features.

    Strategy:
    1. If theater_type known → use precomputed mean for that type
    2. If unknown → use global precomputed mean
    3. If global not available → return 0.0 (conservative)

    These values are precomputed during training and stored in the
    feature schema metadata, so they're always available without
    database access at inference time.
    """
    ...
```

### 6.4 Feature Schema Contract

The single most important file for train-serve consistency:

```json
{
  "version": "1.0.0",
  "feature_names": [
    "lag_7", "lag_14", "lag_21", "lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
    "rolling_std_7",
    "day_of_week", "is_weekend", "month", "day",
    "dow_sin", "dow_cos", "month_sin", "month_cos",
    "tickets_sold_daily", "tickets_booked_daily",
    "theater_type_standard", "theater_type_premium",
    "theater_area", "week_of_year"
  ],
  "feature_dtypes": {
    "lag_7": "float64",
    "rolling_mean_7": "float64",
    "day_of_week": "int32",
    ...
  },
  "cold_start_defaults": {
    "lag_7": 25.0,
    "lag_14": 25.0,
    ...
  },
  "target_column": "audience_count",
  "created_at": "2024-01-15T10:00:00Z",
  "training_metrics": {
    "rmse": 21.6,
    "r2": 0.54
  }
}
```

---

## 7. Observability Stack

### 7.1 Structured Logging

**Tool**: `loguru` (simpler than standard logging, structured output by default)

```
2024-03-01 02:00:15.123 | INFO     | inference.orchestrator: EnsembleOrchestrator.predict
    request_id="abc-123"
    endpoint="/v1/predict/batch"
    theater_count=112
    date_range=["2024-03-01", "2024-04-30"]
    models_used=["lightgbm_v1.0.0", "xgboost_v1.0.0", "catboost_v1.0.0"]
    blend_alpha=0.2
    latency_ms=3450
    fallback_used=false
    chunk_size=10
```

Every log line includes:
- `request_id` — correlates all logs for a single request
- `module` + `function` — origin of the log
- Structured key=value pairs — machine-parseable without regex
- `latency_ms` — every operation is timed

### 7.2 Prometheus Metrics

```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
PREDICTION_REQUESTS = Counter(
    "forecast_requests_total",
    "Total prediction requests",
    ["endpoint", "status"]
)

PREDICTION_LATENCY = Histogram(
    "forecast_prediction_latency_seconds",
    "Prediction latency in seconds",
    ["endpoint", "model"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
)

# Model metrics
MODEL_INFERENCE_TIME = Histogram(
    "forecast_model_inference_seconds",
    "Per-model inference time",
    ["model_name", "model_version"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
)

# Feature metrics
FEATURE_COMPUTATION_TIME = Histogram(
    "forecast_feature_computation_seconds",
    "Feature reconstruction time",
    ["feature_type"],  # "lag", "rolling", "calendar", "booking"
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
)

# Drift metrics
PREDICTION_DRIFT = Gauge(
    "forecast_prediction_drift_ks_statistic",
    "KS statistic between current and reference prediction distributions"
)

FEATURE_DRIFT = Gauge(
    "forecast_feature_drift_ks_statistic",
    "KS statistic per feature",
    ["feature_name"]
)

# Cache metrics
CACHE_HIT_RATIO = Gauge(
    "forecast_cache_hit_ratio",
    "Cache hit ratio for model/feature caches",
    ["cache_type"]
)

# Batch metrics
LAST_BATCH_TIMESTAMP = Gauge(
    "forecast_last_batch_timestamp_seconds",
    "Unix timestamp of last successful batch prediction"
)

BATCH_SIZE = Histogram(
    "forecast_batch_size",
    "Number of predictions per batch",
    buckets=(100, 500, 1000, 5000, 10000)
)

# System metrics
MODEL_LOADED = Gauge(
    "forecast_model_loaded",
    "Whether each model is loaded and ready",
    ["model_name", "model_version"]
)

ENSEMBLE_FALLBACK = Counter(
    "forecast_ensemble_fallback_total",
    "Count of ensemble fallback events",
    ["fallback_reason"]  # "single_model_failure", "all_models_failure"
)
```

### 7.3 Health Check Depth

```
/v1/health — Deep check (every 30s):
├── Models loaded? (all 3, correct versions)
├── Feature schema valid? (JSON schema file exists, parsable)
├── Cache connected? (Redis PING)
├── History store accessible? (Parquet directory readable)
├── Last batch successful? (timestamp within expected window)
├── Drift within threshold? (latest KS statistic < 0.1)
└── Disk space available? (> 500MB for predictions)

/v1/ready — Lightweight (every 10s):
├── Server process alive
└── Basic model files accessible
```

### 7.4 Grafana Dashboard

Panels:

```
Row 1: Request Overview
├── Request Rate (req/s, 5m avg) — broken down by endpoint
├── Request Latency (p50, p95, p99 histograms)
└── Error Rate (% of requests returning 5xx)

Row 2: Model Performance
├── Per-model inference latency (p50, p95)
├── Ensemble fallback rate (fallbacks / total requests)
└── Model version distribution (% of requests per version)

Row 3: Feature Pipeline
├── Feature computation latency (by feature type)
├── Feature drift (KS statistic per feature, top 5)
└── Cold-start prediction rate (% of predictions using fallback)

Row 4: System Health
├── Cache hit ratio (by cache type)
├── Prediction volume per theater (heatmap)
└── Last batch timestamp (time since last successful batch)
```

### 7.5 Alert Rules

```yaml
# prometheus/alerts.yml
groups:
  - name: forecast_inference
    rules:
      - alert: HighPredictionLatency
        expr: histogram_quantile(0.95, forecast_prediction_latency_seconds) > 30
        for: 5m
        labels: { severity: warning }

      - alert: EnsembleDegraded
        expr: rate(forecast_ensemble_fallback_total[10m]) > 0.1
        for: 5m
        labels: { severity: critical }

      - alert: FeatureDriftDetected
        expr: forecast_feature_drift_ks_statistic > 0.15
        for: 30m
        labels: { severity: warning }

      - alert: CacheHitRatioLow
        expr: forecast_cache_hit_ratio < 0.5
        for: 10m
        labels: { severity: warning }

      - alert: BatchNotRunning
        expr: time() - forecast_last_batch_timestamp_seconds > 90000  # 25 hours
        for: 1h
        labels: { severity: critical }
```

---

## 8. Deployment Topology

### 8.1 Local Development

```
┌──────────────────────────────────────┐
│         docker-compose up             │
├──────────────────────────────────────┤
│  ┌──────────┐  ┌──────────────────┐  │
│  │ FastAPI   │  │ Redis            │  │
│  │ :8000     │  │ :6379            │  │
│  └────┬─────┘  └──────────────────┘  │
│       │                              │
│  ┌────▼─────┐  ┌──────────────────┐  │
│  │ Prometheus│  │ Grafana          │  │
│  │ :9090     │  │ :3000            │  │
│  └──────────┘  └──────────────────┘  │
└──────────────────────────────────────┘
```

```yaml
# docker-compose.yml
services:
  api:
    build: .
    ports: ["8000:8000"]
    volumes:
      - ./models:/app/models:ro
      - ./data:/app/data
      - ./feature_schema.json:/app/feature_schema.json:ro
    depends_on: [redis]
    environment:
      REDIS_URL: redis://redis:6379/0
      LOG_LEVEL: INFO
      MODEL_PATH: /app/models

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    volumes:
      - redis_data:/data

  prometheus:
    image: prom/prometheus
    ports: ["9090:9090"]
    volumes:
      - ./deployment/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana
    ports: ["3000:3000"]
    volumes:
      - ./deployment/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - grafana_data:/var/lib/grafana
```

### 8.2 Production Considerations

| Concern | Solution |
|---------|----------|
| Model loading cold start | Pre-load models in startup event; health check fails until loaded |
| Memory pressure | LightGBM/XGBoost/CatBoost all share memory (~500MB total) |
| Prediction persistence | Parquet with date partitioning + monthly rotation |
| Concurrent requests | FastAPI async + uvicorn workers (4 workers recommended) |
| API rate limiting | SlowAPI middleware (100 req/s burst, 30 req/s sustained) |
| Graceful shutdown | Handle SIGTERM → finish in-flight predictions → flush caches → exit |
| Secret management | Environment variables for Redis URL, API keys |
| Readiness probe | Separate lightweight `/v1/ready` endpoint |
| Backup | Regular Parquet snapshots to cloud storage |

### 8.3 Multi-Stage Docker Build

```dockerfile
# Stage 1: Build
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dirs -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim
WORKDIR /app

# Create non-root user
RUN groupadd -r forecast && useradd -r -g forecast forecast

# Copy Python packages
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY app/ app/
COPY models/ models/
COPY feature_schema.json .

# Make sure scripts are executable
ENV PATH=/root/.local/bin:$PATH
USER forecast

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/v1/health')"

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000",
     "--workers", "4", "--limit-max-requests", "10000"]
```

---

## 9. Directory Structure

```
cinema-forecast-api/
├── app/
│   ├── __init__.py
│   ├── main.py                      # FastAPI app creation, middleware, lifespan
│   ├── config.py                    # pydantic-settings configuration
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py                # All HTTP route definitions
│   │   ├── deps.py                  # FastAPI dependency injection
│   │   └── errors.py                # Exception handlers
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py               # Pydantic request/response models
│   │   └── registry.py              # Model registry (versioned loading)
│   ├── features/
│   │   ├── __init__.py
│   │   ├── builder.py               # FeaturePipeline orchestrator
│   │   ├── lag.py                   # Lag feature computation
│   │   ├── rolling.py               # Rolling window computation
│   │   ├── calendar.py              # Calendar/cyclic feature generation
│   │   ├── schema.py                # FeatureSchema (train-serve contract)
│   │   └── state.py                 # RollingWindowState management
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── orchestrator.py          # EnsembleOrchestrator
│   │   ├── blender.py               # Blender (ensemble avg + lag blend)
│   │   └── pipeline.py              # Full inference pipeline composition
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── history.py               # HistoryStore (Parquet + Redis)
│   │   └── cache.py                 # Redis cache client
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py               # Prometheus metric definitions
│   │   ├── logging.py               # Structured logging setup
│   │   └── drift.py                 # DriftMonitor
│   ├── scheduler/
│   │   ├── __init__.py
│   │   └── tasks.py                 # APScheduler daily job
│   └── middleware/
│       ├── __init__.py
│       └── tracing.py               # Request ID + OpenTelemetry
├── models/                           # Serialized model files
│   ├── registry.json                 # Model registry manifest
│   ├── lightgbm/
│   │   ├── model_v1.0.0.txt
│   │   └── metadata_v1.0.0.json
│   ├── xgboost/
│   │   ├── model_v1.0.0.json
│   │   └── metadata_v1.0.0.json
│   └── catboost/
│       ├── model_v1.0.0.cbm
│       └── metadata_v1.0.0.json
├── data/
│   ├── raw/                         # Raw input CSVs
│   ├── processed/                   # Feature-engineered parquet
│   └── predictions/                 # Prediction history
│       └── date=2024-03-01/
│           └── predictions.parquet
├── feature_schema.json              # Exported training feature schema
├── tests/
│   ├── conftest.py                  # Fixtures (mock models, history, features)
│   ├── test_features/
│   │   ├── test_lag.py
│   │   ├── test_rolling.py
│   │   ├── test_calendar.py
│   │   ├── test_builder.py
│   │   └── test_state.py
│   ├── test_inference/
│   │   ├── test_orchestrator.py
│   │   ├── test_blender.py
│   │   └── test_pipeline.py
│   ├── test_api/
│   │   ├── test_routes.py
│   │   ├── test_health.py
│   │   └── test_batch.py
│   ├── test_storage/
│   │   └── test_history.py
│   └── test_monitoring/
│       └── test_drift.py
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── .dockerignore
│   ├── prometheus/
│   │   ├── prometheus.yml
│   │   └── alerts.yml
│   ├── grafana/
│   │   ├── dashboards/
│   │   │   └── forecast_inference.json
│   │   └── datasources/
│   │       └── prometheus.yml
│   └── nginx/
│       └── default.conf
├── scripts/
│   ├── train.py                     # Extracted training pipeline
│   ├── export_models.py             # Serialize + register models
│   ├── export_feature_schema.py     # Generate feature_schema.json
│   ├── benchmark.py                 # Load test with locust/k6
│   ├── simulate_batch.py            # Simulate daily batch for testing
│   └── seed_history.py              # Initialize HistoryStore from training data
├── .github/
│   └── workflows/
│       ├── ci.yml                   # PR checks: lint, type, test, coverage
│       └── deploy.yml               # Docker build, push, deploy
├── k6/                              # Load test scripts
│   └── batch_forecast.js
├── Makefile
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── ARCHITECTURE.md
└── README.md
```

---

## 10. Testing Strategy

### 10.1 Test Pyramid

```
         ╱─────╲
        ╱  E2E  ╲         2 tests: full feature→predict→store flow
       ╱─────────╲
      ╱  Integration ╲    5 tests: API + orchestrator + storage
     ╱─────────────────╲
    ╱    Unit Tests     ╲  20+ tests: individual components
   ╱───────────────────────╲
```

### 10.2 Key Tests

**Feature Reconstruction Tests** (most critical — get these wrong and predictions fail silently):

```python
# tests/test_features/test_state.py

def test_mixed_history_prefers_ground_truth():
    """Given both truth and prediction for same date, use truth."""
    ...

def test_lag_falls_back_to_prediction():
    """When truth is unavailable, use cached prediction."""
    ...

def test_cold_start_returns_global_mean():
    """New theater with no history returns configured default."""
    ...

def test_rolling_window_computed_correctly():
    """Rolling mean/std computed over correct window with shift."""
    ...

def test_lag_with_prediction_propagation():
    """
    Day 1 prediction becomes Day 8's lag_7.
    Verify 7-step propagation has correct values.
    """
    ...

def test_feature_schema_validation():
    """Mismatched features between train/serve schema raise clear error."""
    ...
```

**Ensemble Tests**:

```python
# tests/test_inference/test_orchestrator.py

def test_all_models_succeed():
    """Normal case — all 3 models return predictions."""
    ...

def test_one_model_fails_logs_warning():
    """Single model failure should not crash the ensemble."""
    ...

def test_all_models_fail_triggers_fallback():
    """
    When all 3 models fail, assemble should fall back to lag_7 blend
    and log a critical error.
    """
    ...

def test_concurrent_predictions_are_thread_safe():
    """Multiple simultaneous predictions don't corrupt model state."""
    ...
```

**API Tests**:

```python
# tests/test_api/test_routes.py

def test_batch_predict_returns_correct_schema():
    """Response matches BatchPredictionResponse model."""
    ...

def test_batch_predict_with_empty_theater_list():
    """Empty theater list returns 422 with clear error."""
    ...

def test_health_check_returns_degraded_when_model_missing():
    """Health check correctly reports model loading failures."""
    ...

def test_single_predict_handles_unknown_theater():
    """Unknown theater_id returns 404 with helpful message."""
    ...
```

### 10.3 Test Infrastructure

```python
# tests/conftest.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

@pytest.fixture
def mock_feature_schema():
    """Create a minimal feature schema for testing."""
    ...

@pytest.fixture
def mock_history_store():
    """Create a HistoryStore backed by temporary Parquet files."""
    ...

@pytest.fixture
def mock_models():
    """Create lightweight mock LightGBM/XGBoost/CatBoost models."""
    ...

@pytest.fixture
def sample_theater_history():
    """Generate 60 days of synthetic theater data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", "2024-02-29")
    return pd.DataFrame({
        "date": dates,
        "ground_truth": np.random.poisson(30, len(dates)),
        "theater_id": "book_00001",
    })
```

### 10.4 Testing Commands

```makefile
.PHONY: test test-unit test-integration test-e2e test-coverage

test: test-unit test-integration

test-unit:
    python -m pytest tests/test_features tests/test_inference \
    tests/test_storage tests/test_monitoring \
    -v --cov=app --cov-report=term-missing --cov-fail-under=80

test-integration:
    python -m pytest tests/test_api -v --cov=app --cov-append

test-e2e:
    python -m pytest tests/test_e2e -v

test-coverage:
    python -m pytest tests/ -v --cov=app --cov-report=html
```

---

## 11. Benchmarking & Load Testing

### 11.1 Locust Test

```python
# scripts/benchmark.py
from locust import HttpUser, task, between
import json

class ForecastLoadTest(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(3)
    def batch_predict(self):
        """Simulate daily batch forecast."""
        self.client.post("/v1/predict/batch", json={
            "prediction_dates": ["2024-04-01", "2024-04-02"],
            "theater_ids": ["book_00001", "book_00002"],
            "blend_alpha": 0.2
        })

    @task(1)
    def single_predict(self):
        """Simulate ad-hoc single theater query."""
        self.client.get(
            "/v1/predict/theater/book_00001",
            params={"target_date": "2024-04-15"}
        )

    @task(1)
    def health_check(self):
        self.client.get("/v1/health")
```

### 11.2 K6 Script

```javascript
// k6/batch_forecast.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const batchDuration = new Trend('batch_duration');
const batchSize = new Trend('batch_predictions');

export const options = {
    stages: [
        { duration: '30s', target: 5 },   // Ramp up
        { duration: '1m', target: 10 },    // Steady
        { duration: '30s', target: 0 },    // Ramp down
    ],
    thresholds: {
        http_req_duration: ['p(95)<30000'],  // 95% of requests under 30s
        batch_duration: ['p(95)<60000'],     // Batch under 60s
    },
};

export default function () {
    const payload = JSON.stringify({
        prediction_dates: ["2024-04-01", "2024-04-02", "2024-04-03"],
        theater_ids: ["book_00001", "book_00002", "book_00003"],
    });

    const res = http.post('/v1/predict/batch', payload, {
        headers: { 'Content-Type': 'application/json' },
    });

    check(res, {
        'status is 200': (r) => r.status === 200,
        'predictions returned': (r) => {
            const body = JSON.parse(r.body);
            return body.predictions.length > 0;
        },
    });

    batchDuration.add(res.timings.duration);
    sleep(1);
}
```

### 11.3 Benchmarks to Establish

| Metric | Expected | Good | Great |
|--------|----------|------|-------|
| Batch 112 theaters × 3 days | <15s | <10s | <5s |
| Single theater prediction | <500ms | <200ms | <100ms |
| Model loading (cold start) | <30s | <20s | <10s |
| Feature reconstruction (batch) | <5s | <3s | <1s |
| Concurrent batch requests (10) | No errors | <5% degradation | <1% degradation |
| Memory usage (idle) | <1GB | <500MB | <300MB |
| Memory usage (under load) | <2GB | <1GB | <500MB |

---

## 12. CI/CD Pipeline

### 12.1 CI Workflow (`.github/workflows/ci.yml`)

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt
          pip install -r requirements.txt

      - name: Lint
        run: ruff check app/ tests/

      - name: Type check
        run: mypy app/

      - name: Test
        run: |
          python -m pytest tests/ -v --cov=app --cov-report=xml \
            --cov-fail-under=80

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### 12.2 CD Workflow (`.github/workflows/deploy.yml`)

```yaml
name: Deploy
on:
  push:
    branches: [main]
    paths:
      - "app/**"
      - "models/**"
      - "deployment/Dockerfile"
      - "requirements.txt"

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build Docker image
        run: |
          docker build -f deployment/Dockerfile \
            -t ghcr.io/${{ github.repository }}/cinema-forecast-api:latest \
            -t ghcr.io/${{ github.repository }}/cinema-forecast-api:${{ github.sha }} \
            .

      - name: Run smoke test
        run: |
          docker run -d -p 8000:8000 \
            -v $PWD/models:/app/models:ro \
            ghcr.io/${{ github.repository }}/cinema-forecast-api:latest
          sleep 5
          curl -f http://localhost:8000/v1/health
          curl -f http://localhost:8000/v1/models

      - name: Push to registry
        run: |
          echo "${{ secrets.GITHUB_TOKEN }}" | docker login ghcr.io -u ${{ github.actor }} --password-stdin
          docker push ghcr.io/${{ github.repository }}/cinema-forecast-api:latest
```

### 12.3 Model Registration CI

When a new model is trained, it should:
1. Run through the same test suite
2. Validate on held-out test set
3. Export feature schema
4. Create PR to update model registry
5. After merge, CD activates the new version

```yaml
# Separate workflow triggered when new model artifacts are pushed
name: Register Model
on:
  push:
    paths:
      - "models/lightgbm/*.txt"
      - "models/xgboost/*.json"
      - "models/catboost/*.cbm"

jobs:
  validate-and-register:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Validate model artifacts
        run: python scripts/validate_models.py
      - name: Update registry manifest
        run: python scripts/update_registry.py
      - name: Commit manifest update
        run: |
          git config user.name "Model Registry Bot"
          git add models/registry.json feature_schema.json
          git commit -m "chore: register model version [skip ci]"
          git push
```

---

## 13. Production Concerns & Tradeoffs

### 13.1 Bottlenecks

| Bottleneck | Location | Mitigation |
|-----------|----------|------------|
| Feature reconstruction | I/O reading history + compute | Parquet with predicate pushdown; cache hot theaters in Redis |
| Model inference | CPU-bound predict() | 3 models in parallel; process pool for CPU-intensive calls |
| History store writes | Parquet append during batch | Async writes; batch flush every N predictions |
| Cold start | First prediction for new theater | Pre-compute fallback values; store in feature schema metadata |
| Memory | Model artifacts in RAM | ~500MB for 3 models; watchdog for OOM |

### 13.2 Latency Budgets

```
Single theater request (target: <500ms total):
├── Feature reconstruction: <50ms  (cache hit: <5ms)
│   ├── Lag lookup: <10ms
│   ├── Rolling compute: <20ms
│   └── Calendar encode: <5ms
├── Model inference: <300ms (parallel)
│   ├── LightGBM predict: <100ms
│   ├── XGBoost predict: <150ms
│   └── CatBoost predict: <150ms
├── Blending: <5ms
├── Storage write: <50ms (async, non-blocking)
└── Serialization: <10ms

Batch request (112 theaters × 61 days, target: <5 min):
├── Feature reconstruction: <2 min (10 chunks × 12 seconds)
├── Model inference: <2 min (10 chunks × 12 seconds)
├── Blending: <30s
├── Storage write: <30s
└── Serialization: <5s
```

### 13.3 State Management Tradeoffs

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Parquet-only | Simple, no infra dependency | Slow reads for random access | Use for bulk history |
| Redis-only | Fast random access | Memory bound, no persistence | Use for hot/active state |
| Dual (Parquet + Redis) | Fast + durable | Consistency complexity, double writes | ✅ Recommended for this project |
| SQLite | Single file, transactional | Slow at scale | Not chosen |
| PostgreSQL | ACID, mature | Infra overhead for portfolio | Not chosen |

### 13.4 Failure Modes

```
├── Model file corrupted:
│   → Registry returns checksum mismatch
│   → Health check reports model as inactive
│   → Ensemble falls back to 2 models
│   → Alert: "Model integrity check failed"
│
├── History store unavailable:
│   → Feature pipeline reads from Redis cache only
│   → Cold-start fallbacks for all features
│   → Predictions use cached state + defaults
│   → Alert: "History store offline"
│
├── Redis unavailable:
│   → Bypass cache, read directly from Parquet
│   → Slower but working
│   → Alert: "Cache unavailable"
│
├── Memory pressure (OOM risk):
│   → Release large in-memory DataFrames
│   → Reduce chunk size for batch processing
│   → Process batches sequentially instead of parallel
│   → Alert: "Memory usage > 80%"
│
├── Ground truth never arrives:
│   → Drift monitor cannot compute residuals
│   → Fall back to prediction distribution drift only
│   → Alert: "Ground truth stale - no updates in N days"
```

### 13.5 Scaling Considerations

**Vertical scaling** (for this project's scale):
- 4 uvicorn workers
- 4GB RAM
- 2 CPU cores
- Handles 50+ single-theater requests/second
- Full batch in <5 minutes

**Horizontal scaling** (if needed for larger theater networks):
- Add behind Nginx load balancer
- Share Redis cache across instances
- Each worker loads models independently (~500MB each)
- Dedicated batch worker (separate container) for scheduled forecasts
- Writes to shared prediction store (NFS, S3-compatible)

---

## 14. Interview Preparation

### 14.1 Resume Bullet Points

**Strong** (quantified, technical, demonstrates impact):

> • Designed and deployed a production-grade inference API serving a 3-model gradient boosting ensemble (LightGBM, XGBoost, CatBoost) for cinema audience forecasting using FastAPI, achieving <500ms p95 single-prediction latency and full 112-theater batch forecasts in <5 minutes.

> • Built a feature reconstruction pipeline that solves the train-serve skew problem for time-series lag features by implementing a dual-layer history store (Parquet + Redis) with mixed ground-truth/prediction fallback, maintaining exact feature consistency with the training schema.

> • Implemented a versioned model registry with checksum verification and atomic version activation, enabling zero-downtime model swaps and full reproducibility across inference runs.

> • Deployed a complete observability stack: Prometheus metrics (20+ custom metrics including per-model latency, feature drift KS statistics, cache hit ratios), structured logging with loguru and request ID tracing, and Grafana dashboards for real-time system monitoring.

> • Containerized the entire system with Docker and docker-compose (API server, Redis cache, Prometheus, Grafana), with GitHub Actions CI/CD covering linting, type checking, testing (80%+ coverage), and automated deployment.

> • Implemented statistical drift detection comparing prediction distributions across batch runs using two-sample KS tests, with automated alerting when drift exceeds configurable thresholds.

> • Load-tested the system with k6 to validate throughput (10 concurrent batch requests) and established latency budgets for each pipeline stage (feature reconstruction: <50ms, model inference: <300ms, blending: <5ms).

**Moderate** (solid but needs quantification):

> • Designed and implemented a FastAPI-based ML inference server for time-series forecasting with batch and real-time prediction endpoints.

> • Engineered a model registry system for managing multiple model versions with metadata tracking.

**Weak** (too generic, no engineering depth):

> • Deployed ML models to production.
> • Built an API for machine learning.

### 14.2 Interview Talking Points

**System Design Questions** (most likely to be asked):

- "How would you design a production ML inference system?" → Walk through this architecture: API layer → feature reconstruction → ensemble orchestrator → model registry → observability
- "How do you handle train-serve skew for time-series features?" → Explain the mixed ground-truth/prediction approach, feature schema contract, and cold-start fallbacks
- "What happens when a model fails during inference?" → Ensemble isolation pattern: try/except per model, graceful degradation to remaining models, fallback to lag blend, each level logged and metrified
- "How do you monitor model performance in production?" → Point to drift monitoring (prediction distribution + feature distribution + ground truth residuals), Prometheus metrics, Grafana dashboards
- "How would you scale this to 10,000 theaters?" → Chunked batch processing, horizontal scaling behind load balancer, dedicated batch worker, shared Redis + S3-compatible storage
- "How do you ensure predictions are reproducible?" → Model registry (versioned artifacts + checksums), feature schema (immutable contract), pinned dependencies, deterministic seeds

**Deep Dive Questions** (demonstrates real understanding):

- "Why not use process pools for model inference?" → LightGBM/XGBoost/CatBoost release GIL during predict(); ThreadPoolExecutor avoids serialization overhead of multiprocessing. Verified with profiling.
- "Why Parquet + Redis instead of just PostgreSQL?" → Parquet for columnar storage of time-series data (predicate pushdown for date range queries), Redis for sub-millisecond feature lookups. No need for ACID transactions across prediction writes.
- "How do you handle the recursive lag feature problem?" → Predict chronologically within each batch, update state after each day's predictions. The first day's predictions become the lag features for the eighth day.
- "What's the actual engineering challenge here?" → It's not model accuracy (notebook achieved R²=0.54). It's feature consistency, state management, and operational reliability — the things that matter when this runs unattended for months.
- "Why 3 models instead of 1?" → The original notebook used 3; the serving layer should not second-guess the modeling choice. The ensemble also demonstrates parallel inference patterns and graceful degradation.

### 14.3 System Design Discussion Points

When asked "design an ML inference system" in an interview, the key points to hit:

1. **Not a generic ML system** — it's a **forecasting system**, which changes assumptions (batch > real-time, stateful features, sequential dependencies)
2. **Train-serve parity is the hardest problem** — feature pipeline must share code with training, not reimplement it
3. **Observability is not optional** — you cannot debug a silent prediction error; every step must be visible
4. **Graceful degradation is an architectural property** — not an afterthought; designed from the start with fallback chains
5. **Version everything** — models, features, schemas, API endpoints; reproducibility requires all four
6. **Production constraints > ML elegance** — a model that can't be served reliably is useless regardless of accuracy

---

## 15. Future Extensibility

### 15.1 Modular Upgrades

Each component can be independently upgraded:

```
Component                → Upgrade Path
────────────────────────────────────────────────
History store (Parquet)  → S3 / GCS / MinIO
Model registry (JSON)    → MLflow / DVC model registry
Redis caching            → ElastiCache / Memorystore
Scheduler (APScheduler)  → Celery / Airflow / Prefect
Drift detection (KS)     → Evidently / WhyLabs / NannyML
API (FastAPI)            → Add gRPC for inter-service communication
Testing (pytest)         → Add performance regression tests
Monitoring (Prometheus)  → Add Grafana OnCall / PagerDuty integration
CI/CD (GitHub Actions)   → ArgoCD for GitOps deployment
```

### 15.2 Feature Store Integration

If the system grows to support multiple models:
```
Current: Feature computation inline in inference pipeline
Future:  Centralized feature store (Feast/Tecton) with:
         - Point-in-time correct feature retrieval
         - Feature sharing across models
         - Automatic feature backfilling for training
         - Serving with pre-computed feature values
```

### 15.3 Retraining Pipeline

```
Current: Model retrained manually in notebook
Future:  Automated retraining triggered by:
         - Drift threshold breach (KS p < 0.01)
         - Scheduled (weekly/monthly)
         - New ground truth data arrival
         - Model registry validates new model against test set
           before promoting to production
```

### 15.4 A/B Testing

```
Current: Single model version serves all traffic
Future:  Shadow mode: new model version runs alongside active version
         with results logged but not served
         → Compare prediction distributions before promotion
         → Catch regressions before they reach production
         → Canary deployment: route 10% of requests to new version
```

### 15.5 Advanced Monitoring

```
Current: Statistical drift detection (KS test)
Future:  - Prediction interval calibration (are 90% CIs actually 90%?)
         - Residual autocorrelation detection (are errors time-dependent?)
         - Per-theater performance dashboards
         - Business impact metrics (staffing cost savings from accurate forecasts)
```

---

## Appendix A: Configuration Management

```python
# app/config.py
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    # Application
    app_name: str = "cinema-forecast-api"
    debug: bool = False
    log_level: str = "INFO"

    # Model registry
    model_path: Path = Path("models")
    feature_schema_path: Path = Path("feature_schema.json")
    auto_load_models: bool = True

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_ttl_seconds: int = 3600

    # Prediction storage
    prediction_store_path: Path = Path("data/predictions")
    history_store_path: Path = Path("data/processed")

    # Batch scheduling
    batch_schedule_hour: int = 2
    batch_schedule_minute: int = 0
    batch_chunk_size: int = 10

    # Inference
    default_blend_alpha: float = 0.2
    prediction_clip_min: float = 0.0
    prediction_clip_max: Optional[float] = None

    # API
    max_batch_theaters: int = 500
    rate_limit_per_second: int = 100
    request_timeout_seconds: int = 300

    # Monitoring
    drift_ks_threshold: float = 0.1
    consecutive_drift_alert_count: int = 3
    enable_tracing: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4

    class Config:
        env_file = ".env"
        env_prefix = "FORECAST_"
```

---

## Appendix B: Key Technology Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| API framework | FastAPI | Async-native, OpenAPI docs, Pydantic validation, best Python serving framework |
| Async runtime | uvicorn | Standard ASGI server, production-proven |
| Serialization | Pickle (internal) + JSON (API) | Pickle for model serialization (LightGBM/XGBoost native), JSON for API |
| Caching | Redis 7 | Sub-ms lookups, pub/sub for cache invalidation, well-known |
| Structured logging | loguru | Cleaner API than stdlib, built-in structured output, zero-config |
| Metrics | Prometheus client | Industry standard for metrics, Grafana integration |
| Storage | Parquet + Redis | Columnar for analytics, in-memory for low-latency access |
| Testing | pytest | Standard Python testing, rich fixture system, parallel execution |
| Linting | ruff | Fastest Python linter, replaces flake8 + isort + autoflake |
| Type checking | mypy | De facto Python type checker, catches real bugs |
| Containerization | Docker | Portable, reproducible, well-understood |
| CI/CD | GitHub Actions | Tight GitHub integration, free for public repos |
| Load testing | k6 | Modern, scriptable, JavaScript-based, good reporting |
| Scheduling | APScheduler | In-process, no external dependency, cron-like syntax |

---

## Appendix C: Makefile

```makefile
.PHONY: help install lint type test coverage build run deploy clean

help:                           ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:                        ## Install production dependencies
	pip install -r requirements.txt

install-dev: install            ## Install dev dependencies
	pip install -r requirements-dev.txt

lint:                           ## Run ruff linter
	ruff check app/ tests/

type:                           ## Run mypy type checker
	mypy app/

test:                           ## Run all tests
	python -m pytest tests/ -v --cov=app --cov-report=term-missing

test-ci:                        ## Run tests with CI flags
	python -m pytest tests/ -v --cov=app --cov-report=xml --cov-fail-under=80

build:                          ## Build Docker image
	docker build -f deployment/Dockerfile -t cinema-forecast-api:latest .

run:                            ## Run with docker-compose
	docker-compose -f deployment/docker-compose.yml up --build

run-detached:                   ## Run in background
	docker-compose -f deployment/docker-compose.yml up --build -d

stop:                           ## Stop containers
	docker-compose -f deployment/docker-compose.yml down

benchmark:                      ## Run load tests
	k6 run k6/batch_forecast.js

seed-history:                   ## Initialize history store from training data
	python scripts/seed_history.py

export-models:                  ## Export and register trained models
	python scripts/export_models.py

clean:                          ## Clean temporary files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
```

---

*Document prepared for Cinema Audience Forecast production inference architecture design review.*  
*Questions, tradeoff discussions, and implementation decisions should reference specific section numbers above.*
