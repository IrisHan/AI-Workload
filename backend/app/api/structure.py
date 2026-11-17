"""Model-keyed routes around the transplanted library. No independent server."""

from app.api.router import service
from app.application.planner_service import PlannerService
from app.domain.specs import StructureCostInput
from app.lib.spec_studio.glossary import load_glossary
from fastapi import APIRouter, Depends, Query

router = APIRouter(prefix="/api")


@router.get("/structure/glossary")
def glossary():
    return load_glossary()


@router.get("/models/{model_id}/structures/{snapshot_id}/instances")
def instances(
    model_id: str,
    snapshot_id: str,
    offset: int = Query(0, ge=0, le=4096),
    limit: int = Query(20, ge=1, le=100),
    svc: PlannerService = Depends(service),
):
    return svc.structure_instances(model_id, snapshot_id, offset, limit)


@router.get("/models/{model_id}/structures/{snapshot_id}/templates/{template_id}")
def template(model_id: str, snapshot_id: str, template_id: str, svc: PlannerService = Depends(service)):
    return svc.structure_template(model_id, snapshot_id, template_id)


@router.post("/models/{model_id}/structures/{snapshot_id}/cost")
def cost(model_id: str, snapshot_id: str, body: StructureCostInput, svc: PlannerService = Depends(service)):
    return svc.structure_cost(model_id, snapshot_id, body)
