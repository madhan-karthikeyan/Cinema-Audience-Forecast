from __future__ import annotations

from app.config import settings
from app.monitoring.logging import get_logger

logger = get_logger(__name__)


class ForecastScheduler:
    def __init__(self):
        self._scheduler = None

    def start(self) -> None:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        self._scheduler = AsyncIOScheduler()
        self._scheduler.add_job(
            self.daily_forecast_job,
            "cron",
            hour=settings.batch_schedule_hour,
            minute=settings.batch_schedule_minute,
            id="daily_forecast",
            name="Daily batch forecast",
            replace_existing=True,
        )
        self._scheduler.start()
        logger.info(
            "scheduler_started",
            schedule=(
                f"daily at {settings.batch_schedule_hour:02d}"
                f":{settings.batch_schedule_minute:02d}"
            ),
        )

    def stop(self) -> None:
        if self._scheduler:
            self._scheduler.shutdown(wait=False)
            logger.info("scheduler_stopped")

    async def daily_forecast_job(self) -> None:
        logger.info("daily_forecast_job_started")
        raise NotImplementedError("Scheduled batch prediction will be implemented in Step 3")
