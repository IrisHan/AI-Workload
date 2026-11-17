"""Inference Designer HTTP contract, with domain-local error envelopes."""
# FastAPI evaluates dependency defaults as route declarations.
# ruff: noqa: B008
from typing import Annotated, Literal
from uuid import uuid4

from app.engines.inference.ir import PlanningError
from app.repositories.catalog import MissingResource
from fastapi import APIRouter, Depends, Header, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from .domain import canonical
from .repository import StoreError
from .responses import (
    CapabilitiesDTO,
    CompareDTO,
    DesignDTO,
    DetailPageDTO,
    GraphDTO,
    InspectionDTO,
    PageDTO,
    PlanDTO,
    PlansDTO,
    PreviewDTO,
    RevisionDTO,
    RunDTO,
)
from .schemas import Compare, CreateDesign, Preview, SaveRevision, Spec, SubmitRun
from .service import Service
from .sources import SourceAdapter


class WorkloadRoute(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()

        async def handler(request):
            try:
                length = request.headers.get("content-length", "0")
                if not length.isdecimal() or int(length) > 1024 * 1024:
                    raise StoreError(413, "REQUEST_TOO_LARGE", "Request exceeds 1 MiB")
                return await original(request)
            except (StoreError, PlanningError, MissingResource, RequestValidationError) as exc:
                status, path, details = 422, None, {}
                if isinstance(exc, StoreError):
                    status, code, details = exc.status, exc.code, exc.details
                elif isinstance(exc, PlanningError):
                    code, path, details = exc.code, exc.field_path, exc.details
                elif isinstance(exc, MissingResource):
                    status, code = 404, "SOURCE_NOT_FOUND"
                else:
                    code = "INVALID_CONSTRAINT"
                    details = {"errors": [{"field_path": ".".join(map(str, e["loc"])),
                                           "message": e["msg"], "type": e["type"]} for e in exc.errors()]}
                return JSONResponse(status_code=status, content={
                    "error": {"code": code, "message": str(exc)[:2000], "field_path": path,
                              "retryable": status in (429, 503), "details": details},
                    "request_id": uuid4().hex}, headers={"Retry-After": "2"} if status == 429 else {})
        return handler


router = APIRouter(prefix="/api/workloads", tags=["workload-designer"], route_class=WorkloadRoute)


def service(request: Request):
    return Service(request.app.state.sessions)


def key(value: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)]):
    return value


def available(request: Request):
    worker = getattr(request.app.state, "workload_worker", None)
    if worker is None or not worker.thread.is_alive():
        raise StoreError(503, "WORKER_UNAVAILABLE", "Workload worker is not running")


@router.get("/capabilities", response_model=CapabilitiesDTO)
def capabilities(request: Request, model_id: str | None = None, hardware_id: str | None = None):
    with request.app.state.sessions() as session:
        return SourceAdapter(session).capabilities(model_id, hardware_id)


@router.get("/designs", response_model=PageDTO)
def designs(cursor: str | None = None, limit: int = Query(20, ge=1, le=100),
            q: str = Query("", max_length=200), svc=Depends(service)):
    return svc.designs(cursor, limit, q)


@router.post("/designs", response_model=DesignDTO, status_code=201)
def create_design(body: CreateDesign, response: Response, idem=Depends(key), svc=Depends(service)):
    result, created = svc.create_design(body, idem)
    response.status_code = 201 if created else 200
    response.headers["ETag"] = result["etag"]
    return result


@router.get("/designs/{identifier}", response_model=DesignDTO)
def design(identifier: str, response: Response, svc=Depends(service)):
    result = svc.design(identifier)
    response.headers["ETag"] = result["etag"]
    return result


@router.post("/designs/{identifier}/revisions", response_model=DesignDTO, status_code=201)
def save_revision(identifier: str, body: SaveRevision, response: Response,
                  expected: str = Header(alias="If-Match"), svc=Depends(service)):
    result = svc.save(identifier, body, expected)
    response.headers["ETag"] = result["etag"]
    return result


@router.get("/designs/{identifier}/revisions/{revision}", response_model=RevisionDTO)
def revision(identifier: str, revision: int, svc=Depends(service)):
    if revision < 1:
        raise StoreError(422, "INVALID_CONSTRAINT", "Revision must be positive")
    return svc.revision(identifier, revision)


@router.delete("/designs/{identifier}", status_code=204)
def delete_design(identifier: str, expected: str = Header(alias="If-Match"), svc=Depends(service)):
    svc.delete(identifier, expected)
    return Response(status_code=204)


@router.post("/preview", response_model=PreviewDTO)
def preview(body: Preview, svc=Depends(service)):
    return svc.preview(body)


@router.post("/runs", response_model=RunDTO, status_code=202, dependencies=[Depends(available)])
def submit(body: SubmitRun, response: Response, idem=Depends(key), svc=Depends(service)):
    result = svc.submit(body, idem)
    response.headers["Location"] = "/api/workloads/runs/" + result["id"]
    return result


@router.get("/runs/{identifier}", response_model=RunDTO)
def run(identifier: str, svc=Depends(service)):
    return svc.run(identifier)


@router.post("/runs/{identifier}/cancel", response_model=RunDTO, status_code=202)
def cancel(identifier: str, body: Spec, response: Response, svc=Depends(service)):
    result = svc.cancel(identifier)
    response.status_code = 202 if result["status"] == "cancel_requested" else 200
    return result


@router.post("/runs/{identifier}/retry", response_model=RunDTO, status_code=202,
             dependencies=[Depends(available)])
def retry(identifier: str, body: Spec, response: Response, idem=Depends(key), svc=Depends(service)):
    result = svc.retry(identifier, idem)
    response.headers["Location"] = "/api/workloads/runs/" + result["id"]
    return result


@router.get("/runs/{identifier}/plans", response_model=PlansDTO)
def plans(identifier: str, cursor: str | None = None, limit: int = Query(20, ge=1, le=100), svc=Depends(service)):
    return svc.plans(identifier, cursor, limit)


@router.get("/runs/{identifier}/plans/{pid}", response_model=PlanDTO)
def plan(identifier: str, pid: str, svc=Depends(service)):
    return svc.plan(identifier, pid)


@router.get("/runs/{identifier}/plans/{pid}/layers", response_model=DetailPageDTO)
def layers(identifier: str, pid: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100),
           rank: int | None = Query(None, ge=0), svc=Depends(service)):
    return svc.details(identifier, pid, "layers", offset, limit, rank)


@router.get("/runs/{identifier}/plans/{pid}/ranks", response_model=DetailPageDTO)
def ranks(identifier: str, pid: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100),
          svc=Depends(service)):
    return svc.details(identifier, pid, "ranks", offset, limit)


@router.get("/runs/{identifier}/plans/{pid}/communication", response_model=DetailPageDTO)
def communication(identifier: str, pid: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=100),
                  svc=Depends(service)):
    return svc.details(identifier, pid, "communication", offset, limit)


@router.get("/runs/{identifier}/plans/{pid}/graph", response_model=GraphDTO)
def graph(identifier: str, pid: str, view: Literal["logical", "placement", "communication"] = "logical",
          layer: int | None = Query(None, ge=0), svc=Depends(service)):
    return svc.graph(identifier, pid, view, layer)


@router.post("/compare", response_model=CompareDTO)
def compare(body: Compare, svc=Depends(service)):
    return svc.compare(body.plan_refs)


@router.get("/runs/{identifier}/plans/{pid}/inspection", response_model=InspectionDTO)
def inspection(identifier: str, pid: str, layer: int = Query(ge=0), rank: int = Query(ge=0),
               phase: Literal["prefill", "decode"] = "prefill", svc=Depends(service)):
    return svc.inspection(identifier, pid, layer, rank, phase)


@router.get("/runs/{identifier}/export")
def export(identifier: str, svc=Depends(service)):
    return Response(canonical(svc.export(identifier)), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="workload-{identifier}.json"'})
