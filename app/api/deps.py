from fastapi import Request

from app.config import Settings, settings as app_settings


def get_settings(request: Request) -> Settings:
    return getattr(request.app.state, "settings", app_settings)


def get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


async def get_registry(request: Request):
    return getattr(request.app.state, "registry", None)


async def get_history_store(request: Request):
    return getattr(request.app.state, "history_store", None)


async def get_orchestrator(request: Request):
    return getattr(request.app.state, "orchestrator", None)


async def get_feature_pipeline(request: Request):
    return getattr(request.app.state, "feature_pipeline", None)


async def get_pipeline(request: Request):
    return getattr(request.app.state, "inference_pipeline", None)
