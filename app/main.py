import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.errors import (
    http_exception_handler,
    not_implemented_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from app.api.routes import router as api_router
from app.config import settings
from app.middleware.tracing import RequestIDMiddleware
from app.monitoring.logging import get_logger, setup_logging
from app.monitoring.metrics import PREDICTION_LATENCY

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()

    app.state.settings = settings
    app.state.start_time = time.monotonic()

    for path in [settings.model_path, settings.feature_schema_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    settings.prediction_store_path.mkdir(parents=True, exist_ok=True)
    settings.history_store_path.mkdir(parents=True, exist_ok=True)

    app.state.registry = None
    app.state.history_store = None
    app.state.feature_pipeline = None
    app.state.orchestrator = None
    app.state.inference_pipeline = None
    app.state.cache = None
    app.state.scheduler = None
    app.state.drift_monitor = None

    from app.features.builder import FeaturePipeline
    from app.features.schema import FeatureSchema
    from app.inference.blender import BlendConfig, Blender
    from app.inference.orchestrator import EnsembleOrchestrator
    from app.inference.pipeline import InferencePipeline
    from app.models.registry import ModelRegistry
    from app.storage.history import HistoryStore

    try:
        history_store = HistoryStore(parquet_path=settings.history_store_path)
        app.state.history_store = history_store
    except Exception as e:
        logger.error("history_store_init_failed", error=str(e))
        history_store = None

    try:
        schema = FeatureSchema.load(settings.feature_schema_path)
        if schema.feature_names:
            feature_pipeline = FeaturePipeline(
                history_store=history_store or HistoryStore(
                    parquet_path=settings.history_store_path
                ),
                schema=schema,
            )
            app.state.feature_pipeline = feature_pipeline
        else:
            logger.warning("feature_schema_empty")
            feature_pipeline = None
            app.state.feature_pipeline = None
    except Exception as e:
        logger.error("feature_pipeline_init_failed", error=str(e))
        feature_pipeline = None
        app.state.feature_pipeline = None

    try:
        registry = ModelRegistry(base_path=settings.model_path)
        app.state.registry = registry
        load_results = registry.load_all_models()
        loaded = sum(1 for v in load_results.values() if v)
        logger.info("model_loading_complete", loaded=loaded, total=len(load_results))
    except Exception as e:
        logger.error("registry_init_failed", error=str(e))
        registry = None
        app.state.registry = None

    try:
        blend_cfg = BlendConfig(
            alpha=settings.default_blend_alpha,
            clip_min=settings.prediction_clip_min,
            clip_max=settings.prediction_clip_max,
        )
        blender = Blender(config=blend_cfg)

        if registry:
            orchestrator = EnsembleOrchestrator(
                registry=registry,
                blender=blender,
                blend_config=blend_cfg,
            )
            app.state.orchestrator = orchestrator
        else:
            orchestrator = None
            app.state.orchestrator = None
    except Exception as e:
        logger.error("orchestrator_init_failed", error=str(e))
        orchestrator = None
        app.state.orchestrator = None

    try:
        if feature_pipeline and orchestrator and history_store:
            inference_pipeline = InferencePipeline(
                feature_pipeline=feature_pipeline,
                orchestrator=orchestrator,
                history_store=history_store,
                blend_config=blend_cfg,
            )
            app.state.inference_pipeline = inference_pipeline
        else:
            inference_pipeline = None
            app.state.inference_pipeline = None
            logger.warning(
                "inference_pipeline_not_initialized",
                has_features=feature_pipeline is not None,
                has_orchestrator=orchestrator is not None,
                has_history=history_store is not None,
            )
    except Exception as e:
        logger.error("inference_pipeline_init_failed", error=str(e))
        app.state.inference_pipeline = None

    logger.info(
        "application_starting",
        app_name=settings.app_name,
        log_level=settings.log_level,
        model_path=str(settings.model_path),
        registry_loaded=registry is not None,
        pipeline_ready=app.state.inference_pipeline is not None,
    )

    yield

    if app.state.scheduler:
        app.state.scheduler.stop()
    if app.state.orchestrator:
        app.state.orchestrator.shutdown()
    if app.state.cache:
        await app.state.cache.disconnect()
    logger.info("application_shutdown_complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Cinema Audience Forecast API",
        description="Production-grade inference serving for cinema audience forecasting. "
        "Provides ensemble predictions from LightGBM, XGBoost, and CatBoost models "
        "with stateful feature reconstruction, drift monitoring, and structured observability.",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/v1/docs",
        redoc_url="/v1/redoc",
        openapi_url="/v1/openapi.json",
    )

    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestIDMiddleware)

    app.add_exception_handler(NotImplementedError, not_implemented_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    app.include_router(api_router)

    metrics_app = make_asgi_app()
    app.mount("/v1/metrics", metrics_app)

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start = time.monotonic()
        response = await call_next(request)
        elapsed = time.monotonic() - start
        PREDICTION_LATENCY.labels(
            endpoint=request.url.path, model="all"
        ).observe(elapsed)
        return response

    logger.info("application_created", docs_url="/v1/docs")
    return app


app = create_app()
