import asyncio
from uuid import UUID

from celery import Celery

from .config import get_settings
from .services.workflow import discover_project, plan_selected_gap

settings = get_settings()
celery_app = Celery("researchflow", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)


@celery_app.task(name="researchflow.discover_project")
def discover_project_task(project_id: str) -> None:
    asyncio.run(discover_project(UUID(project_id)))


@celery_app.task(name="researchflow.plan_selected_gap")
def plan_selected_gap_task(project_id: str, gap_id: str) -> None:
    asyncio.run(plan_selected_gap(UUID(project_id), UUID(gap_id)))
