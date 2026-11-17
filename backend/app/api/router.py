from app.application.planner_service import PlannerService
from app.domain.specs import (
    ModelAnalysis,
    ModelMemoryInput,
    ModelSpec,
    StructureOptions,
    StructureSnapshot,
)
from app.repositories.catalog import CatalogRepository
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text

router = APIRouter(prefix="/api")


@router.api_route("/workloads", methods=["GET", "POST"], include_in_schema=False)
@router.get("/workloads/{identifier}", include_in_schema=False)
@router.post("/workloads/{identifier}/profile", include_in_schema=False)
@router.get("/workloads/{identifier}/profiles", include_in_schema=False)
def removed_workload(identifier: str | None = None):
    return JSONResponse(status_code=410, content={
        "error": {"code": "LEGACY_WORKLOAD_REMOVED",
                  "message": "Create an Inference Design at /api/workloads/designs."}})


def service(request: Request):
    with request.app.state.sessions() as session:
        yield PlannerService(CatalogRepository(session), request.app.state.importer)


class ImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_id: str = Field(min_length=3, max_length=193)
    revision: str = Field(default="main", min_length=1, max_length=96)


@router.get("/health")
def health(request: Request):
    with request.app.state.sessions() as session:
        session.execute(text("SELECT 1"))
    return {"status": "ok", "version": "0.1.0", "database": request.app.state.engine.dialect.name}


@router.get("/models", response_model=list[ModelSpec])
def models(svc: PlannerService = Depends(service)):
    return svc.repository.models()


@router.post("/models/import", response_model=ModelSpec)
def import_model(body: ImportRequest, svc: PlannerService = Depends(service)):
    return svc.import_model(body.repo_id, body.revision)


@router.get("/models/{identifier}", response_model=ModelSpec)
def model(identifier: str, svc: PlannerService = Depends(service)):
    return svc.repository.model(identifier)


@router.post("/models/{identifier}/analyze", response_model=ModelAnalysis)
def analyze(identifier: str, svc: PlannerService = Depends(service)):
    return svc.analyze_model(identifier)


@router.post("/models/{identifier}/memory")
def model_memory(identifier: str, body: ModelMemoryInput, svc: PlannerService = Depends(service)):
    return svc.memory_model(identifier, body)


@router.post("/models/{identifier}/structure", response_model=StructureSnapshot)
def create_structure(identifier: str, body: StructureOptions = StructureOptions(), svc: PlannerService = Depends(service)):
    return svc.structure_model(identifier, body)


@router.get("/models/{identifier}/structures", response_model=list[StructureSnapshot])
def structures(identifier: str, svc: PlannerService = Depends(service)):
    return svc.repository.structures(identifier)


@router.get("/models/{identifier}/structures/{snapshot_id}", response_model=StructureSnapshot)
def structure_snapshot(identifier: str, snapshot_id: str, svc: PlannerService = Depends(service)):
    return svc.repository.structure(identifier, snapshot_id)
