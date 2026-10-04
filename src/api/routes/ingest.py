from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from src.api.deps import ContainerDep, RateLimited
from src.api.schemas import IngestResponse, JobResponse
from src.core.errors import NotFoundError
from src.core.kinds import parse_kinds
from src.core.types import JobSpec
from src.ingestion.hints import normalize_language_hint
from src.ingestion.storage import DataStore

router = APIRouter(prefix="/api/v1", tags=["ingest"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    dependencies=[RateLimited],
    responses={202: {"model": IngestResponse}},
)
async def ingest_document(
    request: Request,
    response: Response,
    container: ContainerDep,
    file: UploadFile = File(...),
    collection: str | None = Form(None, description="Target collection (default: the default collection)"),
    image_language: str | None = Form(None, description="OCR language hint: en, mr, hi or auto"),
    kinds: str | None = Form(None, description="Comma separated subset of text,table,image to index"),
    force: bool = False,
    wait: bool | None = None,
):
    """Store the upload and queue it for ingestion.

    With ``wait`` (default from ``INGEST_WAIT_DEFAULT``) the call returns the finished result, or
    ``202`` with a job id if it takes longer than the request timeout. Without it, it returns
    ``202`` immediately: poll ``status_url``.
    """
    settings = container.settings
    spec = container.config.collection(collection or container.config.default_collection)
    name = DataStore.safe_name(file.filename)
    container.parsers.for_path(Path(name), spec.parsers)  # 415 before anything is stored
    normalize_language_hint(image_language, settings.supported_ocr_languages)  # 400 before anything is stored
    selected = parse_kinds(kinds.split(",")) if kinds else None

    stored = await run_in_threadpool(container.store.save, spec, name, file.file)
    record = await container.jobs.submit(
        JobSpec(
            collection=spec.name, path=str(stored), force=force, image_language=image_language, kinds=selected
        )
    )
    should_wait = settings.ingest_wait_default if wait is None else wait
    if should_wait:
        record = await container.jobs.wait(record.id, max(1, settings.request_timeout_seconds - 5))

    status_url = str(request.url_for("get_job", job_id=record.id))
    if record.status == "failed":
        return JSONResponse(
            status_code=record.error_status or 500,
            content={
                "detail": record.error,
                "code": record.error_code,
                "job_id": record.id,
                "status_url": status_url,
            },
        )
    if record.status != "succeeded":
        response.status_code = 202
    return IngestResponse.of(record, status_url, str(stored))


@router.get("/jobs/{job_id}", response_model=JobResponse, name="get_job")
async def get_job(job_id: str, container: ContainerDep):
    record = await container.jobs.get(job_id)
    if record is None:
        raise NotFoundError(f"job '{job_id}' not found (unknown or expired)")
    return JobResponse.of(record)
