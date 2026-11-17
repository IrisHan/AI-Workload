from typing import Annotated, Literal

from app.hardware.architecture.schemas import ArchitectureDocument
from app.hardware.repository import HardwareRepository
from app.hardware.schemas import HardwareRecord, Precision, RooflineResult
from app.hardware.service import HardwareService
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/hardware", tags=["Hardware Center"])


def service(request: Request):
    with request.app.state.sessions() as session:
        yield HardwareService(HardwareRepository(session))


@router.get("", response_model=list[HardwareRecord])
def hardware(
    q: Annotated[str, Query(max_length=100)] = "",
    vendor: Literal["NVIDIA", "Huawei"] | None = None,
    form_factor: Literal["SXM", "PCIe", "Module"] | None = None,
    include_legacy: bool = False,
    planner_only: bool = False,
    svc: HardwareService = Depends(service),
):
    return svc.list(q, vendor, form_factor, include_legacy, planner_only)


@router.get("/{identifier}", response_model=HardwareRecord)
def detail(identifier: str, svc: HardwareService = Depends(service)):
    return svc.get(identifier)


@router.get("/{identifier}/roofline", response_model=RooflineResult)
def roofline(
    identifier: str,
    precision: Precision = "bf16",
    sparse: bool = False,
    intensity: Annotated[float, Query(gt=0, le=1e9, allow_inf_nan=False)] = 64,
    svc: HardwareService = Depends(service),
):
    return svc.roofline(identifier, precision, sparse, intensity)


@router.get("/{identifier}/export", response_model=HardwareRecord)
def export(identifier: str, svc: HardwareService = Depends(service)):
    record = svc.get(identifier)
    return JSONResponse(record.model_dump(), headers={
        "Content-Disposition": f'attachment; filename="{record.id}.json"',
    })


@router.get("/{identifier}/architecture", response_model=ArchitectureDocument)
def architecture(identifier: str, svc: HardwareService = Depends(service)):
    return svc.architecture(identifier)


@router.get("/{identifier}/architecture/export", response_model=ArchitectureDocument)
def architecture_export(identifier: str, svc: HardwareService = Depends(service)):
    document = svc.architecture(identifier)
    return JSONResponse(document.model_dump(), headers={
        "Content-Disposition": f'attachment; filename="{document.hardware_id}-architecture.json"',
    })
