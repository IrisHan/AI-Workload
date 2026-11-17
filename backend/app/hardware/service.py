from app.hardware.architecture.assembler import assemble
from app.hardware.repository import HardwareRepository
from app.hardware.roofline import evaluate
from app.hardware.schemas import Precision


class HardwareService:
    def __init__(self, repository: HardwareRepository):
        self.repository = repository

    def list(self, q="", vendor=None, form_factor=None, include_legacy=False, planner_only=False):
        query = q.strip().casefold()
        return [
            r for r in self.repository.list(include_legacy)
            if (not query or query in f"{r.name} {r.architecture} {r.vendor} {r.sku}".casefold())
            and (vendor is None or r.vendor == vendor)
            and (form_factor is None or r.form_factor == form_factor)
            and (not planner_only or "bf16" in r.planner_precisions)
        ]

    def get(self, identifier):
        return self.repository.get(identifier)

    def roofline(self, identifier, precision: Precision, sparse: bool, intensity: float):
        return evaluate(self.get(identifier), precision, sparse, intensity)

    def architecture(self, identifier):
        return assemble(self.get(identifier), self.repository.list())
