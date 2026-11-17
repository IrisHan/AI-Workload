from app.domain.specs import HardwareSpec, ModelSpec, StructureSnapshot
from app.engines.model_capabilities import current_capabilities
from app.infrastructure.database import HardwareRow, ModelRow, StructureRow
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified


class MissingResource(Exception):
    pass


class CatalogRepository:
    def __init__(self, session: Session):
        self.session = session

    def _get(self, row_type, spec_type, identifier):
        row = self.session.get(row_type, identifier)
        if row is None:
            raise MissingResource(f"{spec_type.__name__} {identifier} not found")
        return spec_type.model_validate(row.payload)

    def models(self):
        return [
            current_capabilities(ModelSpec.model_validate(row.payload))
            for row in self.session.scalars(select(ModelRow).order_by(ModelRow.repo_id, ModelRow.revision))
        ]

    def model(self, identifier):
        return current_capabilities(self._get(ModelRow, ModelSpec, identifier))

    def hardware(self):
        from app.hardware.repository import HardwareRepository

        return HardwareRepository(self.session).list(include_legacy=True)

    def gpu(self, identifier):
        from app.hardware.repository import HardwareRepository

        return HardwareRepository(self.session).calculation_input(identifier)

    def add_model(self, spec: ModelSpec):
        existing = self.session.get(ModelRow, spec.id)
        if existing:
            stored = existing.payload
            incoming = spec.model_dump()
            if stored.get("workload_supported") != incoming.get("workload_supported") or stored.get("support_notes") != incoming.get("support_notes"):
                stored = {**stored, "workload_supported": incoming["workload_supported"], "support_notes": incoming["support_notes"]}
                existing.payload = stored
                flag_modified(existing, "payload")
                self.session.commit()
                return current_capabilities(ModelSpec.model_validate(stored))
            return current_capabilities(ModelSpec.model_validate(stored))
        self.session.add(
            ModelRow(id=spec.id, repo_id=spec.repo_id, revision=spec.revision, payload=spec.model_dump())
        )
        try:
            self.session.commit()
        except IntegrityError:
            self.session.rollback()
            return self.model(spec.id)
        return current_capabilities(spec)

    def add_gpu(self, spec: HardwareSpec):
        if not self.session.get(HardwareRow, spec.id):
            self.session.add(HardwareRow(id=spec.id, payload=spec.model_dump()))
            self.session.commit()

    def structures(self, model_id):
        self.model(model_id)
        rows = self.session.scalars(select(StructureRow).where(StructureRow.model_id == model_id))
        return sorted(
            [StructureSnapshot.model_validate(row.payload) for row in rows],
            key=lambda snapshot: snapshot.created_at, reverse=True,
        )

    def structure(self, model_id, identifier):
        snapshot = self._get(StructureRow, StructureSnapshot, identifier)
        if snapshot.model_id != model_id:
            raise MissingResource("Structure snapshot not found for model")
        return snapshot

    def add_structure(self, spec: StructureSnapshot):
        if self.session.get(StructureRow, spec.id):
            return self.structure(spec.model_id, spec.id)
        self.session.add(StructureRow(id=spec.id, model_id=spec.model_id, payload=spec.model_dump()))
        try:
            self.session.commit()
        except IntegrityError:
            self.session.rollback()
            return self.structure(spec.model_id, spec.id)
        return spec
