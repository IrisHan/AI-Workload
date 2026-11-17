"""Application orchestration: source freeze, immutable revisions and read models."""
import base64
import json
import time
from itertools import pairwise
from math import ceil

from app.engines.inference.ir import Candidate, PlanningError
from app.engines.inference.partition import make_candidate, rank_weights, ranks, validate_candidate
from app.infrastructure.database import HardwareRow, ModelRow
from sqlalchemy import and_, or_, select

from .domain import VERSIONS, FrozenSources, content_hash
from .repository import Repository, StoreError, etag
from .schemas import DesignInput
from .sources import SourceAdapter
from .tables import DesignRow, PlanRow, RevisionRow, RunRow


def revision_dto(row):
    return {"id": row.id, "design_id": row.design_id, "revision_no": row.revision_no,
            "input": row.payload, "input_hash": row.input_hash, "created_at": row.created_at}


def run_dto(row, repository):
    return {key: getattr(row, key) for key in (
        "id", "revision_id", "status", "input_hash", "versions", "queued_at", "started_at",
        "finished_at", "attempt", "progress", "error")} | {
        "design_id": repository.revision(row.revision_id).design_id,
        "design_input": row.frozen_input["design"], "structure_snapshot_id": row.structure_snapshot_id,
        "summary": row.result_manifest, "source_updates": [
            {"source": "engine", "pinned_version": row.versions["engine"],
             "available_version": VERSIONS["engine"],
             "change_kind": "unchanged" if row.versions == VERSIONS else "available"},
            {"source": "model", "pinned_version": row.frozen_input["sources"]["model"]["revision"],
             "available_version": None, "change_kind": "latest_not_queried"},
            {"source": "hardware", "pinned_version": row.frozen_input["sources"]["hardware"]["version"],
             "available_version": None, "change_kind": "latest_not_queried"}]}


def plan_dto(row):
    return {"id": row.id, "run_id": row.run_id, "rank": row.rank, "candidate_key": row.candidate_key,
            "result_hash": row.result_hash, "summary": row.summary}


def cursor_encode(values):
    return base64.urlsafe_b64encode(json.dumps(values).encode()).decode()


def cursor_decode(cursor, length):
    try:
        values = json.loads(base64.urlsafe_b64decode(cursor).decode())
        if not isinstance(values, list) or len(values) != length or any(not isinstance(v, str) for v in values):
            raise ValueError()
        return values
    except Exception as exc:
        raise StoreError(422, "INVALID_CURSOR", "Invalid pagination cursor") from exc


class Service:
    def __init__(self, sessions):
        self.sessions = sessions

    def design_dto(self, repository, row):
        head = repository.head(row)
        model = repository.session.get(ModelRow, head.model_id)
        hardware = repository.session.get(HardwareRow, head.hardware_id)
        revisions = select(RevisionRow.id).where(RevisionRow.design_id == row.id)
        runs = repository.session.scalars(select(RunRow).where(RunRow.revision_id.in_(revisions)).order_by(
            RunRow.queued_at.desc(), RunRow.id.desc()).limit(10)).all()
        return {"id": row.id, "name": row.name, "head_revision": row.head_revision,
                "created_at": row.created_at, "updated_at": row.updated_at, "etag": etag(row),
                "model_name": model.payload.get("name", head.model_id),
                "hardware_name": hardware.payload.get("name", head.hardware_id),
                "revision": revision_dto(head), "runs": [run_dto(r, repository) for r in runs]}

    def designs(self, cursor, limit, q=""):
        with self.sessions() as session:
            query = select(DesignRow)
            if q.strip():
                head = (select(RevisionRow.design_id).join(ModelRow, ModelRow.id == RevisionRow.model_id)
                        .join(HardwareRow, HardwareRow.id == RevisionRow.hardware_id)
                        .join(DesignRow, and_(DesignRow.id == RevisionRow.design_id,
                                             DesignRow.head_revision == RevisionRow.revision_no))
                        .where(or_(
                            DesignRow.name.icontains(q.strip(), autoescape=True),
                            RevisionRow.model_id.icontains(q.strip(), autoescape=True),
                            RevisionRow.hardware_id.icontains(q.strip(), autoescape=True),
                            ModelRow.payload["name"].as_string().icontains(q.strip(), autoescape=True),
                            HardwareRow.payload["name"].as_string().icontains(q.strip(), autoescape=True))))
                query = query.where(DesignRow.id.in_(head))
            if cursor:
                updated, identifier, search_query = cursor_decode(cursor, 3)
                if search_query != q.strip():
                    raise StoreError(422, "INVALID_CURSOR", "Cursor belongs to a different search")
                query = query.where(or_(DesignRow.updated_at < updated,
                                        and_(DesignRow.updated_at == updated, DesignRow.id < identifier)))
            rows = session.scalars(query.order_by(DesignRow.updated_at.desc(), DesignRow.id.desc())
                                   .limit(limit + 1)).all()
            return {"items": [self.design_dto(Repository(session), row) for row in rows[:limit]],
                    "next_cursor": cursor_encode([rows[limit - 1].updated_at, rows[limit - 1].id, q.strip()])
                    if len(rows) > limit else None}

    def design(self, identifier):
        with self.sessions() as session:
            repo = Repository(session)
            return self.design_dto(repo, repo.design(identifier))

    def validate_sources(self, design):
        # Save can preserve unsupported methods for editing, but IDs must exist.
        with self.sessions() as session:
            adapter = SourceAdapter(session)
            adapter.catalog.model(design.model_id)
            adapter.hardware.get(design.hardware_id)

    def create_design(self, body, key):
        self.validate_sources(body.input)
        with self.sessions() as session:
            repo = Repository(session)
            row, created = repo.create_design(body.model_dump(), key)
            return self.design_dto(repo, row), created

    def save(self, identifier, body, expected):
        self.validate_sources(body.input)
        with self.sessions() as session:
            repo = Repository(session)
            row, _ = repo.save_revision(identifier, body.input.model_dump(), expected)
            return self.design_dto(repo, row)

    def revision(self, identifier, number):
        with self.sessions() as session:
            Repository(session).design(identifier)
            row = session.scalar(select(RevisionRow).where(RevisionRow.design_id == identifier,
                                                         RevisionRow.revision_no == number))
            if not row:
                raise StoreError(404, "REVISION_NOT_FOUND", "Design revision does not exist")
            return revision_dto(row)

    def submit(self, body, key):
        digest = content_hash(body.model_dump())
        with self.sessions() as session:
            repo = Repository(session)
            existing = repo.idempotent(RunRow, key, digest)
            if existing:
                return run_dto(existing, repo)
            revision = repo.revision(body.design_revision_id)
            design = DesignInput.model_validate(revision.payload)
            sources = SourceAdapter(session).freeze(design, body.structure_snapshot_id)
            frozen = {"design": design.model_dump(), "search": body.search.model_dump(),
                      "versions": VERSIONS, "sources": {
                          name: getattr(sources, name) for name in
                          ("model", "structure", "hardware", "topology", "assumptions")}}
        with self.sessions() as session:
            row, _ = Repository(session).submit(body.design_revision_id, frozen, VERSIONS, key, digest)
            return run_dto(row, Repository(session))

    def retry(self, identifier, key):
        with self.sessions() as session:
            row = Repository(session).run(identifier)
            if row.versions != VERSIONS:
                raise StoreError(409, "ENGINE_VERSION_UNAVAILABLE", "Frozen engine version is unavailable")
            if row.status not in ("completed", "partial", "failed", "interrupted", "cancelled"):
                raise StoreError(409, "ACTIVE_RUN", "Run must end before retry")
            frozen, revision = row.frozen_input, row.revision_id
        with self.sessions() as session:
            row, _ = Repository(session).submit(revision, frozen, VERSIONS, key, content_hash({"retry": identifier}))
            return run_dto(row, Repository(session))

    def run(self, identifier):
        with self.sessions() as session:
            repo = Repository(session)
            return run_dto(repo.run(identifier), repo)

    def cancel(self, identifier):
        with self.sessions() as session:
            repo = Repository(session)
            return run_dto(repo.cancel(identifier), repo)

    def delete(self, identifier, expected):
        with self.sessions() as session:
            Repository(session).delete_design(identifier, expected)

    def preview(self, body):
        began = time.monotonic()
        diagnostics, lower = [], None
        coverage, status = "constraints_only", "unknown"
        try:
            with self.sessions() as session:
                sources = SourceAdapter(session).freeze(body.input, persist=False)
            candidate_spec = body.manual_candidate or body.input.parallelism.manual_candidate
            if candidate_spec:
                candidate = make_candidate(sources.ir, body.input, candidate_spec)
                validate_candidate(sources.ir, candidate, body.input)
                lower = 0
                for rank in ranks(candidate, body.input.resources.gpus_per_node):
                    if time.monotonic() - began > 0.45:
                        coverage, lower = "timeout", None
                        break
                    lower = max(lower, sum(s.bytes for s in rank_weights(
                        sources.ir, candidate, rank, body.input.runtime.weight_dtype)))
                else:
                    coverage = "static_lower_bound"
            else:
                # Logical source storage / entire GPU budget is only a lower bound.
                bytes_per = 4 if body.input.runtime.weight_dtype == "fp32" else 2
                lower = ceil(sources.ir.logical_numel * bytes_per / (
                    body.input.resources.node_count * body.input.resources.gpus_per_node))
                coverage = "static_lower_bound"
            if lower is not None and lower > sources.hardware["memory_bytes"]:
                status = "fail"
                diagnostics.append({"code": "WEIGHT_LOWER_BOUND_OOM", "observed": lower,
                                    "limit": sources.hardware["memory_bytes"]})
        except PlanningError as exc:
            diagnostics.append(exc.as_dict())
            status = "fail"
        elapsed = (time.monotonic() - began) * 1000
        if elapsed > 500:
            coverage = "timeout"
        return {"client_revision": body.client_revision, "coverage": coverage, "needs_full_run": True,
                "capacity_status": status, "lower_bound_bytes": lower, "diagnostics": diagnostics,
                "elapsed_ms": elapsed}

    def plans(self, identifier, cursor, limit):
        with self.sessions() as session:
            row = Repository(session).run(identifier)
            after = 0
            if cursor:
                run_id, position = cursor_decode(cursor, 2)
                if run_id != identifier or not position.isdecimal():
                    raise StoreError(422, "INVALID_CURSOR", "Cursor belongs to a different run")
                after = int(position)
            rows = session.scalars(select(PlanRow).where(PlanRow.run_id == identifier, PlanRow.rank > after)
                                   .order_by(PlanRow.rank).limit(limit + 1)).all()
            manifest = row.result_manifest or {}
            coverage = manifest.get("coverage", {})
            return {"items": [plan_dto(r) for r in rows[:limit]],
                    "next_cursor": cursor_encode([identifier, str(rows[limit - 1].rank)]) if len(rows) > limit else None,
                    "search_complete": row.status == "completed", "minimum_proven": coverage.get("minimum_proven", False),
                    "coverage": coverage}

    def plan(self, run_id, plan_id):
        with self.sessions() as session:
            return plan_dto(Repository(session).plan(run_id, plan_id))

    def details(self, run_id, plan_id, kind, offset, limit, rank=None):
        with self.sessions() as session:
            repo = Repository(session)
            row = repo.plan(run_id, plan_id)
            if rank is None:
                data = repo.details(plan_id, kind, offset, limit)
            else:
                # Bounded rank-scoped scan; output pagination is after filtering.
                all_rows = repo.details(plan_id, kind, 0, 100000)["items"]
                all_rows = [r for r in all_rows if r.get("rank") == rank]
                items = all_rows[offset:offset + limit]
                data = {"items": items, "total": len(all_rows),
                        "next_offset": offset + limit if offset + limit < len(all_rows) else None}
            return {**data, "result_hash": row.result_hash}

    def graph(self, run_id, plan_id, view, layer):
        with self.sessions() as session:
            repo = Repository(session)
            row, run = repo.plan(run_id, plan_id), repo.run(run_id)
            summary = row.summary
            nodes, edges = [], []
            network = None
            truncated = False
            if view == "placement":
                for rank in summary["placement"]:
                    nodes.append({"id": f'rank-{rank["rank"]}', "label": f'GPU {rank["rank"]}',
                                  "rank_refs": [rank["rank"]], **rank})
            elif view == "communication":
                from .network import network_graph
                events = repo.details(plan_id, "communication", 0, 100000)
                replay = summary.get("diagnostic_replay", {})
                nodes, edges, network = network_graph(
                    summary, run.frozen_input["sources"], events["items"],
                    events_complete=events["next_offset"] is None and replay.get("trace_complete", False))
            else:
                from app.engines.inference.normalize import normalize
                source = run.frozen_input["sources"]
                ir = normalize(source["model"], source["structure"])
                selected = ir.layers if layer is None else [r for r in ir.layers if r.index == layer]
                truncated = len(selected) > 200
                for item in selected[:200]:
                    candidate = Candidate(**{**summary["candidate"],
                                             "boundaries": tuple(summary["candidate"]["boundaries"])})
                    refs = [r["rank"] for r in summary["placement"] if r["p"] == candidate.stage(item.index)]
                    if layer is None:
                        nodes.append({"id": item.instance_id, "instance_id": item.instance_id,
                                      "source_node_id": item.template_id, "label": f"Layer {item.index}",
                                      "layer_index": item.index, "attention": item.attention,
                                      "ffn": item.ffn, "rank_refs": refs, "stage": candidate.stage(item.index)})
                    else:
                        template = next(t for t in source["structure"]["analysis"]["templates"]
                                        if t["id"] == item.template_id)
                        source_nodes = {n["id"]: n for n in template["nodes"]}
                        outputs = {op.output: op.id for op in item.operators}
                        for operator in item.operators:
                            original = source_nodes[operator.source_node_id]
                            nodes.append({"id": operator.id, "label": original["label"],
                                          "instance_id": item.instance_id, "source_node_id": operator.source_node_id,
                                          "module_id": original.get("module_id"), "category": original["category"],
                                          "weight_refs": list(operator.weight_refs), "layer_index": item.index,
                                          "rank_refs": refs, "stage": candidate.stage(item.index)})
                            for index, dependency in enumerate(operator.inputs):
                                edges.append({"id": f"{operator.id}/in/{index}",
                                              "source": outputs[dependency], "target": operator.id})
                if layer is None:
                    for a, b in pairwise(nodes):
                        edges.append({"id": a["id"] + ">" + b["id"], "source": a["id"], "target": b["id"]})
            return {"nodes": nodes[:1000], "edges": edges[:2000], "plan_id": plan_id,
                    "source_structure_id": run.structure_snapshot_id,
                    "scope": {"view": view, "layer": layer}, "layout_hint": "top-down", "network": network,
                    "truncated": truncated or len(nodes) > 1000 or len(edges) > 2000,
                    "drilldown": {"parameter": "layer", "page_size": 200} if truncated else None,
                    "result_hash": row.result_hash}

    def inspection(self, run_id, plan_id, layer_index, rank_index, phase):
        from dataclasses import asdict

        from app.engines.inference.cache import cache_allocation, request_quotas
        from app.engines.inference.compute import stage_program
        from app.engines.inference.lifetimes import activation_program, peak_ledger
        from app.engines.inference.normalize import normalize

        with self.sessions() as session:
            repo = Repository(session)
            plan, run = repo.plan(run_id, plan_id), repo.run(run_id)
            source = run.frozen_input["sources"]
            ir = normalize(source["model"], source["structure"])
            design = DesignInput.model_validate(run.frozen_input["design"])
            candidate = Candidate(**{**plan.summary["candidate"],
                                     "boundaries": tuple(plan.summary["candidate"]["boundaries"])})
            if layer_index >= len(ir.layers) or rank_index >= candidate.gpu_count:
                raise StoreError(404, "SCOPE_NOT_FOUND", "Layer or GPU is outside this plan")
            layer = ir.layers[layer_index]
            rank = ranks(candidate, design.resources.gpus_per_node)[rank_index]
            resident = candidate.stage(layer_index) == rank.p
            quotas = request_quotas(design.load.concurrency, candidate)[rank.d]
            chunk = min(design.load.prompt_tokens, design.runtime.prefill_chunk_tokens) if phase == "prefill" else 1
            tokens = [min(min(q, design.runtime.max_batch_sequences) * chunk,
                          design.runtime.max_batched_tokens) for q in quotas]
            expert_tokens = sum(tokens) * min(layer.top_k, layer.experts // candidate.ep) if layer.experts else None
            allocations = activation_program(layer, tokens[rank.e], candidate.tp,
                                             4 if design.runtime.activation_dtype == "fp32" else 2,
                                             expert_tokens=expert_tokens,
                                             expert_tp=candidate.expert_tp if layer.experts
                                             else candidate.tp) if resident else ()
            shards = [asdict(w) for w in rank_weights(ir, candidate, rank, design.runtime.weight_dtype)
                      if w.storage_id in {w.storage_id for w in layer.weights}]
            cache = cache_allocation(layer, [design.load.prompt_tokens + design.load.output_tokens] *
                                     (quotas[rank.e] if resident else 0), candidate.tp, design.runtime.kv_dtype)
            timing = {"operators": [], "reason": "Historical engine; run again for timing decomposition",
                      "scope": "representative_batch", "phase": phase}
            if run.versions == VERSIONS:
                batches = []
                context = 0 if phase == "prefill" else design.load.prompt_tokens
                for token_count in tokens:
                    remaining, group = token_count, []
                    while remaining:
                        query = min(chunk, remaining)
                        group.append((context, query, phase == "decode" or query == design.load.prompt_tokens))
                        remaining -= query
                    batches.append(group)
                sources = FrozenSources(source["model"], source["structure"], source["hardware"],
                                        ir, source["topology"], source["assumptions"])
                program = stage_program(sources, candidate, design, rank.d, rank.p, batches,
                                        details=True, layer_index=layer_index) if resident else None
                timing = {**timing, "reason": None, "context_tokens": context,
                          "tokens_per_group": tokens, "operators": [
                              row for row in program["operators"] if row["rank"] == rank_index
                          ] if program else [], "hardware": source["hardware"],
                          "assumptions": source["assumptions"]}
            result = {"result_hash": plan.result_hash, "source_structure_id": ir.structure_id,
                    "scope": {"layer": layer_index, "rank": rank_index, "phase": phase,
                              "kind": "reserved_admission_bound"},
                    "resident": resident, "placement": asdict(rank), "shards": shards,
                    "local_tokens": tokens[rank.e] if resident else 0,
                    "resident_request_quota": quotas[rank.e] if resident else 0,
                    "cache": cache, "allocations": [asdict(a) for a in allocations],
                    "activation_peak_bytes": peak_ledger({}, allocations)["peak_bytes"], "timing": timing}
            if run.versions == VERSIONS:
                from app.engines.inference.explain import inspection_explanation
                result["explanation"] = inspection_explanation(sources, candidate, design, result)
                result["explanation"]["scope"].update(run_id=run_id, plan_id=plan_id)
            return result

    def compare(self, refs):
        plans, inputs, versions, assumptions = [], [], [], []
        with self.sessions() as session:
            repo = Repository(session)
            for ref in refs:
                plans.append(plan_dto(repo.plan(ref.run_id, ref.plan_id)))
                run = repo.run(ref.run_id)
                inputs.append(run.frozen_input["design"])
                versions.append(run.versions)
                assumptions.append(run.frozen_input["sources"]["assumptions"])
        def flatten(value, prefix=""):
            result = {}
            for key, item in value.items():
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(item, dict):
                    result.update(flatten(item, path))
                else:
                    result[path] = item
            return result
        flat = [flatten(value) for value in inputs]
        differences = [{"path": key, "values": [row.get(key) for row in flat]}
                       for key in sorted(set().union(*(row.keys() for row in flat)))
                       if len({content_hash(row.get(key)) for row in flat}) > 1]
        return {"plans": plans, "input_differences": differences, "versions": versions, "assumptions": assumptions}

    def export(self, identifier):
        with self.sessions() as session:
            repo = Repository(session)
            run = repo.run(identifier)
            rows = session.scalars(select(PlanRow).where(PlanRow.run_id == identifier).order_by(PlanRow.rank)).all()
            plans = []
            for row in rows:
                plans.append({**plan_dto(row), "details": {kind: repo.details(row.id, kind, 0, 100000)["items"]
                              for kind in ("layers", "ranks", "groups", "communication", "events")}})
            payload = {"run": run_dto(run, repo), "frozen_input": run.frozen_input, "plans": plans,
                       "format": "workload-export-1.0"}
            return {**payload, "export_hash": content_hash(payload)}
