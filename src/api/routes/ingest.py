from __future__ import annotations

from fastapi import APIRouter, File, Form, Request, Response, UploadFile
from fastapi.responses import JSONResponse

from src.api.deps import RateLimited, ServiceDep
from src.contracts.models import IngestResponse, JobResponse

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
    service: ServiceDep,
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
    outcome = await service.ingest(
        file.filename or "",
        file.file,
        collection=collection,
        image_language=image_language,
        kinds=kinds.split(",") if kinds else None,
        force=force,
        wait=wait,
    )
    record = outcome.record
    status_url = str(request.url_for("get_job", job_id=record.id))
    if outcome.failed:
        return JSONResponse(
            status_code=record.error_status or 500,
            content={
                "detail": record.error,
                "code": record.error_code,
                "job_id": record.id,
                "status_url": status_url,
            },
        )
    if not outcome.finished:
        response.status_code = 202
    return outcome.response(status_url)


@router.get("/jobs/{job_id}", response_model=JobResponse, name="get_job")
async def get_job(job_id: str, service: ServiceDep):
    return await service.job(job_id)
