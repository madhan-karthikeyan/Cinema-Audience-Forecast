import datetime
import time
from datetime import date

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import (
    get_pipeline,
    get_registry,
    get_request_id,
)
from app.models.schemas import (
    BatchPredictionItem,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    SinglePredictionRequest,
    SinglePredictionResponse,
)
from app.monitoring.logging import get_logger
from app.monitoring.metrics import HEALTH_CHECK_DURATION

logger = get_logger(__name__)
router = APIRouter()


@router.get("/v1/health", response_model=HealthResponse)
async def health_check(
    request_id: str = Depends(get_request_id),
    registry=Depends(get_registry),
    pipeline=Depends(get_pipeline),
):
    with HEALTH_CHECK_DURATION.time():
        models_loaded = []
        all_models_loaded = True
        if registry:
            for name in ["lightgbm", "xgboost", "catboost"]:
                if registry.get_loaded_model(name) is not None:
                    mv = registry.get_active(name)
                    v_str = f"{name}_v{mv.version}" if mv else name
                    models_loaded.append(v_str)
                else:
                    all_models_loaded = False

        feature_schema_valid = (
            pipeline is not None
            and pipeline.features.schema is not None
            and len(pipeline.features.schema.feature_names) > 0
        )

        if all_models_loaded and len(models_loaded) == 3:
            status = "healthy"
        elif len(models_loaded) > 0:
            status = "degraded"
        else:
            status = "degraded"

        logger.info(
            "health_check",
            request_id=request_id,
            status=status,
            models_loaded=len(models_loaded),
        )

        return HealthResponse(
            status=status,
            uptime_seconds=time.monotonic(),
            models_loaded=models_loaded,
            feature_schema_valid=feature_schema_valid,
        )


@router.get("/v1/ready")
async def readiness():
    return {"status": "ready"}


@router.get("/v1/models", response_model=list[ModelInfoResponse])
async def list_models(registry=Depends(get_registry)):
    if registry is None:
        raise HTTPException(status_code=501, detail="Model registry not initialized")

    result = []
    for name in registry.list_models():
        for mv in registry.list_versions(name):
            result.append(
                ModelInfoResponse(
                    name=mv.name,
                    version=mv.version,
                    active=mv.active,
                    metrics=mv.metrics,
                    params=mv.params,
                    created_at=mv.created_at,
                    checksum=mv.checksum,
                )
            )
    return result


@router.get("/v1/models/{name}", response_model=ModelInfoResponse)
async def get_model(name: str, registry=Depends(get_registry)):
    if registry is None:
        raise HTTPException(status_code=501, detail="Model registry not initialized")

    mv = registry.get_active(name)
    if mv is None:
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")

    return ModelInfoResponse(
        name=mv.name,
        version=mv.version,
        active=mv.active,
        metrics=mv.metrics,
        params=mv.params,
        created_at=mv.created_at,
        checksum=mv.checksum,
    )


@router.put("/v1/admin/models/{name}/activate/{version}")
async def activate_model(
    name: str,
    version: str,
    registry=Depends(get_registry),
):
    if registry is None:
        raise HTTPException(status_code=501, detail="Model registry not initialized")

    mv = registry.get_version(name, version)
    if mv is None:
        raise HTTPException(
            status_code=404,
            detail=f"Version {version} for model '{name}' not found",
        )

    registry.activate(name, version)
    registry.load_model(name)
    return {"status": "ok", "model": name, "version": version, "active": True}


@router.post("/v1/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(
    request: BatchPredictionRequest,
    request_id: str = Depends(get_request_id),
    pipeline=Depends(get_pipeline),
):
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Inference pipeline not initialized",
        )

    start = time.monotonic()
    logger.info(
        "batch_predict_requested",
        request_id=request_id,
        theater_count=len(request.theater_ids) if request.theater_ids else "all",
        date_range=request.prediction_dates,
    )

    prediction_dates = request.prediction_dates
    if not prediction_dates:
        prediction_dates = [
            datetime.date(2024, 3, 1) + datetime.timedelta(days=i)
            for i in range(61)
        ]

    theater_ids = request.theater_ids
    if not theater_ids:
        theater_ids = pipeline.history.get_all_theater_ids()
        if not theater_ids:
            theater_ids = [f"book_{i:05d}" for i in range(1, 113)]

    result_df = await pipeline.run_batch(
        prediction_dates=prediction_dates,
        theater_ids=theater_ids,
        chunk_size=10,
    )

    elapsed = time.monotonic() - start
    predictions_list = []
    for _, row in result_df.iterrows():
        predictions_list.append(
            BatchPredictionItem(
                theater_id=str(row["theater_id"]),
                date=(
                    row["date"]
                    if isinstance(row["date"], date)
                    else pd.Timestamp(row["date"]).date()
                ),
                prediction=float(row["predicted"]),
                model_version="ensemble",
                blend_weight=0.2,
            )
        )

    models_used = set()
    for m in result_df.get("models_used", []):
        for name in str(m).split(","):
            if name:
                models_used.add(name)

    return BatchPredictionResponse(
        request_id=request_id,
        predictions=predictions_list,
        ensemble_metrics={
            "model_count": len(models_used),
            "fallback_count": (
                int(result_df["fallback_used"].sum())
                if "fallback_used" in result_df.columns
                else 0
            ),
        },
        latency_ms=round(elapsed * 1000, 2),
        models_used=sorted(models_used),
        fallback_used=(
            bool(result_df["fallback_used"].any())
            if "fallback_used" in result_df.columns
            else False
        ),
        warnings=[],
    )


@router.get("/v1/predict/theater/{theater_id}", response_model=SinglePredictionResponse)
async def predict_theater(
    theater_id: str,
    target_date: date = Query(..., description="Target prediction date"),
    request_id: str = Depends(get_request_id),
    pipeline=Depends(get_pipeline),
):
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Inference pipeline not initialized",
        )

    logger.info(
        "single_predict_requested",
        request_id=request_id,
        theater_id=theater_id,
        target_date=target_date.isoformat(),
    )

    result = await pipeline.run_single(theater_id, target_date)

    if not result["success"]:
        raise HTTPException(
            status_code=500,
            detail=result.get("error", "Prediction failed"),
        )

    return SinglePredictionResponse(
        theater_id=result["theater_id"],
        target_date=result["target_date"],
        prediction=result["prediction"],
        model_version=result["model_version"],
        latency_ms=result["latency_ms"],
    )


@router.post("/v1/predict/theater/{theater_id}", response_model=SinglePredictionResponse)
async def predict_theater_post(
    theater_id: str,
    request: SinglePredictionRequest,
    request_id: str = Depends(get_request_id),
    pipeline=Depends(get_pipeline),
):
    return await predict_theater(theater_id, request.target_date, request_id, pipeline)
