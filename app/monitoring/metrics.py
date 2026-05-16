from prometheus_client import Counter, Gauge, Histogram

PREDICTION_REQUESTS = Counter(
    "forecast_requests_total",
    "Total prediction requests",
    ["endpoint", "status"],
)

PREDICTION_LATENCY = Histogram(
    "forecast_prediction_latency_seconds",
    "Prediction latency in seconds",
    ["endpoint", "model"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)

MODEL_INFERENCE_TIME = Histogram(
    "forecast_model_inference_seconds",
    "Per-model inference time",
    ["model_name", "model_version"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

FEATURE_COMPUTATION_TIME = Histogram(
    "forecast_feature_computation_seconds",
    "Feature reconstruction time",
    ["feature_type"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
)

PREDICTION_DRIFT = Gauge(
    "forecast_prediction_drift_ks_statistic",
    "KS statistic between current and reference prediction distributions",
)

FEATURE_DRIFT = Gauge(
    "forecast_feature_drift_ks_statistic",
    "KS statistic per feature",
    ["feature_name"],
)

CACHE_HIT_RATIO = Gauge(
    "forecast_cache_hit_ratio",
    "Cache hit ratio for model/feature caches",
    ["cache_type"],
)

LAST_BATCH_TIMESTAMP = Gauge(
    "forecast_last_batch_timestamp_seconds",
    "Unix timestamp of last successful batch prediction",
)

BATCH_SIZE = Histogram(
    "forecast_batch_size",
    "Number of predictions per batch",
    buckets=(100, 500, 1000, 5000, 10000),
)

MODEL_LOADED = Gauge(
    "forecast_model_loaded",
    "Whether each model is loaded and ready",
    ["model_name", "model_version"],
)

ENSEMBLE_FALLBACK = Counter(
    "forecast_ensemble_fallback_total",
    "Count of ensemble fallback events",
    ["fallback_reason"],
)

HEALTH_CHECK_DURATION = Histogram(
    "forecast_health_check_duration_seconds",
    "Health check endpoint duration",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
)

REQUEST_SIZE = Histogram(
    "forecast_request_size_bytes",
    "Request payload size in bytes",
    ["endpoint"],
    buckets=(100, 1000, 10000, 100000, 1000000),
)

RESPONSE_SIZE = Histogram(
    "forecast_response_size_bytes",
    "Response payload size in bytes",
    ["endpoint"],
    buckets=(100, 1000, 10000, 100000, 1000000, 10000000),
)
