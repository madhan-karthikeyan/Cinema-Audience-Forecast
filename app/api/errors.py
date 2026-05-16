import uuid

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.monitoring.logging import get_logger

logger = get_logger(__name__)


def get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", str(uuid.uuid4()))


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = get_request_id(request)
    logger.warning(
        "request_validation_error",
        request_id=request_id,
        errors=exc.errors(),
    )
    return JSONResponse(
        status_code=422,
        content={
            "request_id": request_id,
            "error": "Validation Error",
            "detail": exc.errors(),
            "status_code": 422,
        },
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    request_id = get_request_id(request)
    logger.warning(
        "http_exception",
        request_id=request_id,
        status_code=exc.status_code,
        detail=exc.detail,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "request_id": request_id,
            "error": exc.detail,
            "status_code": exc.status_code,
        },
    )


async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = get_request_id(request)
    logger.error(
        "unhandled_exception",
        request_id=request_id,
        error=str(exc),
        exc_info=True,
    )
    settings = getattr(request.app.state, "settings", None)
    debug = settings.debug if settings else False
    return JSONResponse(
        status_code=500,
        content={
            "request_id": request_id,
            "error": "Internal Server Error",
            "detail": str(exc) if debug else None,
            "status_code": 500,
        },
    )


async def not_implemented_handler(request: Request, exc: NotImplementedError):
    request_id = get_request_id(request)
    logger.warning(
        "not_implemented",
        request_id=request_id,
        path=str(request.url.path),
    )
    return JSONResponse(
        status_code=501,
        content={
            "request_id": request_id,
            "error": "Not Implemented",
            "detail": str(exc),
            "status_code": 501,
        },
    )
