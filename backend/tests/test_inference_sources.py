"""Adapter evidence, readonly capability discovery and immutable source checks."""
import pytest
from app.application.planner_service import seed
from app.engines.inference.ir import PlanningError
from app.hardware.repository import HardwareRepository
from app.infrastructure.database import database, migrate
from app.repositories.catalog import CatalogRepository
from app.workload.schemas import DesignInput
from app.workload.sources import SourceAdapter, runtime_assumptions, topology_snapshot


@pytest.fixture
def source_session(tmp_path):
    engine, sessions = database(f"sqlite:///{tmp_path}/sources.db")
    migrate(engine)
    with sessions() as session:
        seed(CatalogRepository(session))
        HardwareRepository(session).seed()
        yield session
    engine.dispose()


def test_capabilities_do_not_write_structure(source_session):
    from app.infrastructure.database import StructureRow
    from sqlalchemy import func, select

    adapter = SourceAdapter(source_session)
    model = adapter.catalog.models()[0]
    before = source_session.scalar(select(func.count()).select_from(StructureRow))
    result = adapter.capabilities(model.id, "nvidia-h100-sxm-80gb-20260913")
    assert result["supported"] is True
    assert result["precisions"] == ["bf16", "fp32"]
    assert source_session.scalar(select(func.count()).select_from(StructureRow)) == before


def test_freeze_defaults_and_source_hashes(source_session):
    adapter = SourceAdapter(source_session)
    model = adapter.catalog.models()[0]
    design = DesignInput(model_id=model.id, hardware_id="nvidia-h100-sxm-80gb-20260913")
    frozen = adapter.freeze(design)
    assert frozen.ir.config_sha256 == model.config_sha256
    assert frozen.topology["intra_injection_bytes_s"] == 450e9
    assert frozen.assumptions["safety_fraction"]["effective"] == 0.1
    assert adapter.catalog.structure(model.id, frozen.structure["id"]).id == frozen.structure["id"]
    updated = design.model_copy(update={"hardware_id": "nvidia-h20-sxm-96gb-20260913"})
    with pytest.raises(PlanningError, match="insufficient"):
        adapter.freeze(updated)


def test_fact_override_and_topology_conversion(source_session):
    hardware = HardwareRepository(source_session).get("nvidia-h100-sxm-80gb-20260913").model_dump()
    design = DesignInput.model_validate({
        "model_id": "mock", "hardware_id": hardware["id"],
        "overrides": [{"path": "safety_fraction", "value": 0.2, "reason": "scenario"}],
        "resources": {"node_count": 2, "gpus_per_node": 8,
                      "topology_template": "rdma_reference-v1",
                      "topology_overrides": {"nic_count": 2, "nic_bandwidth_bytes_s": 50e9,
                                             "oversubscription": 2.0}},
    })
    topology = topology_snapshot(design, hardware)
    assert topology["node_network_bytes_s"] == 50e9
    assert topology["intra_injection_bytes_s"] is None
    assert runtime_assumptions(design)["safety_fraction"] == {
        "engine_default": 0.1, "effective": 0.2, "user_override": 0.2,
        "reason": "scenario", "source_kind": "runtime_assumption",
    }
