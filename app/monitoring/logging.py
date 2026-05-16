import sys

from loguru import logger

from app.config import settings


def setup_logging() -> None:
    logger.remove()

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stdout,
        format=log_format,
        level=settings.log_level.upper(),
        colorize=True,
    )

    logger.add(
        "data/logs/{time:YYYY-MM-DD}.log",
        format="{time} | {level: <8} | {name}:{function} | {message}",
        level="INFO",
        rotation="1 day",
        retention="30 days",
        compression="gz",
    )


def get_logger(name: str):
    return logger.bind(module=name)
