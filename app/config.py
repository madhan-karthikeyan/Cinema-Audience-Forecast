from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "cinema-forecast-api"
    debug: bool = False
    log_level: str = "INFO"

    model_path: Path = Path("models")
    feature_schema_path: Path = Path("feature_schema.json")
    auto_load_models: bool = True

    redis_url: str = "redis://localhost:6379/0"
    redis_ttl_seconds: int = 3600

    prediction_store_path: Path = Path("data/predictions")
    history_store_path: Path = Path("data/processed")

    batch_schedule_hour: int = 2
    batch_schedule_minute: int = 0
    batch_chunk_size: int = 10

    default_blend_alpha: float = 0.2
    prediction_clip_min: float = 0.0
    prediction_clip_max: float | None = None

    max_batch_theaters: int = 500
    rate_limit_per_second: int = 100
    request_timeout_seconds: int = 300

    drift_ks_threshold: float = 0.1
    consecutive_drift_alert_count: int = 3
    enable_tracing: bool = False

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4

    model_config = {"env_file": ".env", "env_prefix": "FORECAST_"}


settings = Settings()
