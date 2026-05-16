from datetime import date, datetime

from pydantic import BaseModel, Field


class BatchPredictionRequest(BaseModel):
    prediction_dates: list[date] | None = None
    theater_ids: list[str] | None = None
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


class SinglePredictionRequest(BaseModel):
    target_date: date
    blend_alpha: float = Field(default=0.2, ge=0.0, le=1.0)
    include_features: bool = False


class SinglePredictionResponse(BaseModel):
    theater_id: str
    target_date: date
    prediction: float
    confidence_interval: dict | None = None
    model_version: str
    latency_ms: float
    features_used: list[str] | None = None


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    models_loaded: list[str]
    last_batch_timestamp: str | None = None
    cache_connected: bool = False
    feature_schema_valid: bool = False
    prediction_count_total: int = 0


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    active: bool
    metrics: dict
    params: dict
    created_at: str
    checksum: str


class ErrorResponse(BaseModel):
    request_id: str
    error: str
    detail: str | None = None
    status_code: int


class ModelVersionMetadata(BaseModel):
    name: str
    version: str
    path: str
    metrics: dict
    params: dict
    created_at: datetime
    checksum: str
    active: bool = False
