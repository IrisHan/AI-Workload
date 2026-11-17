import hashlib
import json
from pathlib import Path

from app.domain.specs import (
    HardwareSpec,
    StructureCostInput,
    StructureOptions,
    StructureSnapshot,
    utcnow,
)
from app.engines.model_analyzer import analyze
from app.hardware.roofline import require_planner_precision
from app.integrations.huggingface import HuggingFaceImporter, normalize_model
from app.integrations.spec_studio import structure
from app.lib.spec_studio.cost import operator_cost
from app.lib.spec_studio.pagination import instance_page
from app.repositories.catalog import CatalogRepository, MissingResource

DATA = Path(__file__).resolve().parents[1] / "data"


class PlannerService:
    def __init__(self, repository: CatalogRepository, importer=None):
        self.repository = repository
        self.importer = importer or HuggingFaceImporter()

    def import_model(self, repo_id: str, revision: str):
        return self.repository.add_model(self.importer.import_model(repo_id, revision))

    def analyze_model(self, model_id: str):
        return analyze(self.repository.model(model_id))

    def memory_model(self, model_id, request):
        from app.engines.model_memory import memory_components

        model = self.repository.model(model_id)
        snapshot = self.structure_model(model_id)
        result = memory_components(snapshot.analysis, model.raw_config, **request.model_dump())
        result["input_snapshot"].update(model=model.model_dump(), structure_snapshot_id=snapshot.id)
        return result

    def structure_model(self, model_id: str, options: StructureOptions | None = None):
        model = self.repository.model(model_id)
        options = options or StructureOptions()
        result = structure(model, **options.model_dump())
        analysis = result["analysis"]
        identity = {
            "model_version": model.version,
            "model_id": model.id,
            "engine": analysis["engine_version"],
            "formula": analysis["formula_version"],
            "analysis": analysis,
        }
        identifier = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return self.repository.add_structure(
            StructureSnapshot(
                id=identifier,
                model_id=model.id,
                model_version=model.version,
                engine_version=analysis["engine_version"],
                formula_version=analysis["formula_version"],
                created_at=utcnow(),
                input_snapshot=model.model_dump(),
                **result,
            )
        )

    def structure_instances(self, model_id, snapshot_id, offset, limit):
        snapshot = self.repository.structure(model_id, snapshot_id)
        return instance_page(snapshot.analysis, offset, limit)

    def structure_template(self, model_id, snapshot_id, template_id):
        snapshot = self.repository.structure(model_id, snapshot_id)
        for template in snapshot.analysis["templates"] + snapshot.analysis.get("branches", []):
            if template["id"] == template_id:
                return template
        raise MissingResource("Template not found")

    def structure_cost(self, model_id, snapshot_id, request: StructureCostInput):
        snapshot = self.repository.structure(model_id, snapshot_id)
        gpu = self.repository.gpu(request.hardware_id)
        require_planner_precision(gpu, request.dtype)
        result = operator_cost(
            snapshot.analysis,
            **request.model_dump(exclude={"hardware_id"}),
            hardware={
                "peak_tflops": gpu.fp32_tflops if request.dtype == "fp32" else gpu.bf16_tflops,
                "bandwidth_bytes_s": gpu.memory_bandwidth_bytes_s,
                "memory_bytes": gpu.memory_bytes,
                "source": gpu.source_url,
                "version": gpu.version,
            },
        )
        result["input_snapshot"].update(
            model_id=model_id,
            structure_snapshot_id=snapshot_id,
            hardware_id=gpu.id,
            hardware=gpu.model_dump(),
        )
        return result

def seed(repository: CatalogRepository):
    for filename, repo, revision, expected, retrieved in [
        (
            "qwen-0.5b-config.json",
            "Qwen/Qwen2.5-0.5B",
            "060db6499f32faf8b98477b0a26969ef7d8b9987",
            494032768,
            "2026-09-10T19:35:55+08:00",
        ),
        (
            "qwen-1.5b-config.json",
            "Qwen/Qwen2.5-1.5B",
            "8faed761d45a263340a0528343f099c05c9a4323",
            1543714304,
            "2026-09-10T19:35:57+08:00",
        ),
    ]:
        model = normalize_model(repo, revision, (DATA / filename).read_bytes(), retrieved)
        if model.parameters != expected:
            raise RuntimeError(f"Seed parameter count mismatch for {repo}")
        repository.add_model(model)
    repository.add_gpu(
        HardwareSpec(
            id="nvidia-h100-sxm-80gb-v1",
            name="H100 SXM 80GB",
            vendor="NVIDIA",
            memory_bytes=80_000_000_000,
            memory_bandwidth_bytes_s=3.35e12,
            bf16_tflops=989.5,
            fp32_tflops=67,
            nvlink_gb_s=900,
            power_watts=700,
            source_url="https://www.nvidia.com/en-us/data-center/h100/",
            retrieved_at="2026-09-10T11:26:00+00:00",
            version="nvidia-h100-sxm-2026-09-10",
            notes="Official BF16 1979 TFLOPS is with sparsity; dense = 1979/2. "
            "Capacity uses SI GB, HBM BW uses SI TB/s. NVLink is aggregate bidirectional; "
            "700 W is maximum configurable TDP. Not a measured GPU inventory.",
        )
    )
