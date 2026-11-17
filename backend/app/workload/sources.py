"""The only Workload adapter allowed to resolve Model/Hardware source records."""
from app.application.planner_service import PlannerService
from app.domain.specs import StructureOptions
from app.engines.inference.ir import PlanningError
from app.engines.inference.normalize import capability_issues, normalize
from app.engines.inference.search import parallel_domains
from app.hardware.repository import HardwareRepository
from app.integrations.spec_studio import structure
from app.repositories.catalog import CatalogRepository

from .domain import FrozenSources, content_hash
from .schemas import OVERRIDE_DEFAULTS, DesignInput, TopologyOverrides


def runtime_assumptions(design: DesignInput):
    overrides = {o.path: o for o in design.overrides}
    return {
        key: {
            "engine_default": default, "effective": overrides[key].value if key in overrides else default,
            "user_override": overrides[key].value if key in overrides else None,
            "reason": overrides[key].reason if key in overrides else None,
            "source_kind": "runtime_assumption",
        }
        for key, (default, _, _) in OVERRIDE_DEFAULTS.items()
    }


def topology_defaults(hardware: dict):
    intra = inter = None
    if hardware["form_factor"] == "SXM":
        links = [link for link in hardware["interconnect"] if "nvlink" in link["name"].lower()
                 and link["direction"] != "unspecified" and link["bandwidth_gb_s"]]
        if links:
            link = links[0]
            intra = link["bandwidth_gb_s"] * 1e9 / (2 if link["direction"] == "bidirectional" else 1)
    if hardware["node"].get("nic_gb_s_per_gpu"):
        inter = hardware["node"]["nic_gb_s_per_gpu"] * 1e9
    return {template: {**TopologyOverrides().model_dump(),
                       "intra_bandwidth_bytes_s": intra if template == "switched_sxm_reference-v1" else None,
                       "inter_bandwidth_bytes_s": inter}
            for template in ("single_gpu-v1", "switched_sxm_reference-v1", "pcie_shared_root-v1", "rdma_reference-v1")}


def topology_snapshot(design: DesignInput, hardware: dict):
    resources = design.resources
    override = resources.topology_overrides
    template = resources.topology_template
    intra = override.intra_bandwidth_bytes_s
    inter = override.inter_bandwidth_bytes_s
    reference = hardware["node"]
    if inter is None and reference.get("nic_gb_s_per_gpu"):
        inter = reference["nic_gb_s_per_gpu"] * 1e9
    if template == "switched_sxm_reference-v1":
        if hardware["form_factor"] != "SXM":
            raise PlanningError("INVALID_TOPOLOGY", "SXM reference cannot be applied to PCIe/Module",
                                "resources.topology_template")
        if reference.get("accelerators") is None or resources.gpus_per_node > reference["accelerators"]:
            raise PlanningError("INVALID_TOPOLOGY", "GPU/node exceeds verified node reference")
        defaults = topology_defaults(hardware)[template]
        if intra is None:
            intra = defaults["intra_bandwidth_bytes_s"]
        if inter is None:
            inter = defaults["inter_bandwidth_bytes_s"]
    if template == "pcie_shared_root-v1" and hardware["form_factor"] != "PCIe":
        raise PlanningError("INVALID_TOPOLOGY", "PCIe template requires a PCIe GPU")
    node_nic = None
    if override.nic_count is not None and override.nic_bandwidth_bytes_s is not None:
        node_nic = override.nic_count * override.nic_bandwidth_bytes_s / override.oversubscription
        inter = min(inter, node_nic) if inter is not None else node_nic
    elif override.nic_count is not None or override.nic_bandwidth_bytes_s is not None:
        raise PlanningError("INVALID_TOPOLOGY", "NIC count and bandwidth must be provided together")
    elif inter is not None:
        node_nic = inter * resources.gpus_per_node / override.oversubscription
    result = {
        "template": template, "version": "2", "source_kind": "assumption_template",
        "fabric": "rail-optimized",
        "node_count": resources.node_count, "gpus_per_node": resources.gpus_per_node,
        "intra_injection_bytes_s": intra, "inter_injection_bytes_s": inter,
        "node_network_bytes_s": node_nic, "link_efficiency": override.link_efficiency,
        "inter_link_efficiency": override.inter_link_efficiency,
        "intra_latency_s": override.intra_latency_us / 1e6,
        "inter_latency_s": override.inter_latency_us / 1e6,
        "sharing_domains": ["gpu-injection", "node-network", "pcie-root"] if "pcie" in template
        else ["gpu-injection", "node-network"],
        "hardware_sha256": hardware["content_sha256"],
        "notes": ["Bandwidth is an injection limit, not an independent full-bandwidth GPU pair.",
                  "Rail-optimized inter-node network; default network efficiency is 1.",
                  "Latency, utilization and fabric connectivity are scenario assumptions."],
    }
    result["hash"] = content_hash(result)
    return result


class SourceAdapter:
    def __init__(self, session):
        self.catalog = CatalogRepository(session)
        self.hardware = HardwareRepository(session)
        self.planner = PlannerService(self.catalog)

    def model_structure(self, model_id, snapshot_id=None, *, persist=True):
        model = self.catalog.model(model_id)
        if not persist and snapshot_id is None:
            result = structure(model, include_mtp=False)
            analysis = result["analysis"]
            return model.model_dump(), {
                "id": "preview-" + content_hash(analysis)[:56],
                "model_id": model.id, "model_version": model.version,
                "engine_version": analysis["engine_version"], "formula_version": analysis["formula_version"],
                **result,
            }
        snapshot = (self.catalog.structure(model_id, snapshot_id) if snapshot_id else
                    self.planner.structure_model(model_id, StructureOptions(include_mtp=False)))
        return model.model_dump(), snapshot.model_dump()

    def capabilities(self, model_id=None, hardware_id=None):
        issues = []
        domains = None
        layouts = ["orthogonal-tp-ep-v1", "vllm-v0.10.2", "sglang-v0.5.3"]
        by_layout = {}
        model = snapshot = hardware = None
        if model_id:
            model, snapshot = self.model_structure(model_id, persist=False)
            issues = capability_issues(model, snapshot["analysis"])
            if not issues:
                try:
                    ir = normalize(model, snapshot)
                    by_layout = {layout: parallel_domains(ir, layout=layout) for layout in layouts}
                    domains = by_layout["orthogonal-tp-ep-v1"]
                except PlanningError as exc:
                    issues.append({**exc.as_dict(), "status": "unsupported"})
        if hardware_id:
            record = self.hardware.get(hardware_id)
            hardware = record.model_dump()
        return {
            "supported": not issues,
            "issues": issues, "model": model,
            "hardware": hardware,
            "topology_defaults": topology_defaults(hardware) if hardware else {},
            "parallel_domains": domains,
            "parallel_domains_by_layout": by_layout, "layouts": layouts,
            "structure_id": None,
            "precisions": hardware["planner_precisions"] if hardware else ["bf16", "fp32"],
            "overrides": {key: {"default": d, "minimum": low, "maximum": high}
                          for key, (d, low, high) in OVERRIDE_DEFAULTS.items()},
            "runtime_profiles": ["analytical-continuous-v1", "analytical-sequential-v1"],
            "layout": "orthogonal-tp-ep-v1", "etp": "derived_from_tp",
            "topology_templates": ["single_gpu-v1", "switched_sxm_reference-v1",
                                   "pcie_shared_root-v1", "rdma_reference-v1"],
            "limits": {"gpu_budget": 256, "concurrency": 4096},
        }

    def freeze(self, design: DesignInput, snapshot_id=None, *, persist=True):
        model, snapshot = self.model_structure(design.model_id, snapshot_id, persist=persist)
        ir = normalize(model, snapshot)
        hardware = self.hardware.get(design.hardware_id).model_dump()
        runtime = design.runtime
        for precision in {runtime.weight_dtype, runtime.activation_dtype, runtime.kv_dtype}:
            if precision not in hardware["planner_precisions"]:
                raise PlanningError("PRECISION_UNVERIFIED", "Hardware precision/memory evidence is insufficient",
                                    "runtime.weight_dtype")
        if design.load.prompt_tokens + design.load.output_tokens > ir.context_limit:
            raise PlanningError("CONTEXT_LIMIT", "Prompt + output exceeds model context", "load")
        assumptions = runtime_assumptions(design)
        if assumptions["runtime_fixed_bytes"]["effective"] >= hardware["memory_bytes"]:
            raise PlanningError("INVALID_OVERRIDE", "Runtime reserve must be smaller than HBM",
                                "overrides.runtime_fixed_bytes")
        return FrozenSources(model, snapshot, hardware, ir, topology_snapshot(design, hardware), assumptions)
