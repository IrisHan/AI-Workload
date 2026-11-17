"""Shape primitives shared by compressed attention and composite model adapters."""
from math import prod

from app.engines.operator_methods import feed_forward
from app.lib.spec_studio.inventory import annotate_graph, inventory_metrics, module_row, moe_layout

FORMULA = "components-1.1.0"


class Ledger:
    def __init__(self, source, dtype="bf16"):
        self.rows = []
        self.source = source
        self.dtype = dtype

    def tensor(self, name, shape, category, *, count=1, active_count=None, matrix=False,
               dtype=None, notes=None):
        n = prod(shape)
        active_count = count if active_count is None else active_count
        row = module_row(
            name, category, shape, n * count, active=n * active_count,
            matrix=n * active_count if matrix else 0, count=count,
            dtype=dtype or self.dtype, formula=f"{' * '.join(map(str, shape))} * {count}",
            source=self.source, notes=notes,
        )
        self.rows.append(row)
        return row

    def linear(self, name, inf, outf, category="attention", *, bias=False, count=1,
               active_count=None, dtype=None):
        row = self.tensor(name, [outf, inf], category, count=count,
                          active_count=active_count, matrix=True, dtype=dtype)
        row.update(input_shape=f"[B, S, {inf}]", output_shape=f"[B, S, {outf}]")
        if bias:
            self.tensor(name + ".bias", [outf], category, count=count,
                        active_count=active_count, dtype=dtype)

    def norm(self, name, dim, *, layernorm=False, category="norm"):
        self.tensor(name + ".weight", [dim], category)
        if layernorm:
            self.tensor(name + ".bias", [dim], category)

    def ffn(self, d, width, *, prefix="", gated=True, bias=False, count=1, active_count=None,
            dtype=None):
        feed_forward(self, d, width, prefix=prefix, gated=gated, bias=bias,
                     count=count, active_count=active_count, dtype=dtype)


def module_graph(rows, stages=None, links=None):
    """Module dependency graph; explicit stages keep parallel projections parallel."""
    names = {row["id"] for row in rows}
    stages = stages if stages is not None else ([] if links is not None else [[row["id"]] for row in rows])
    nodes = [
        dict(id=row["id"], label=row["label"], category=row["category"],
             module_id=row["id"], detail=row["notes"]) for row in rows
    ]
    edges = [dict(source=a, target=b, label=None) for a, b in (links or [])]
    for before, after in zip(stages, stages[1:]):
        edges.extend(dict(source=a, target=b, label=None) for a in before for b in after)
    for stage in stages + [[a, b] for a, b in (links or [])]:
        for name in stage:
            if name not in names:
                names.add(name)
                category = "residual" if name.startswith("hc_") else "attention"
                if any(word in name for word in ("expert", "swiglu", "hash", "top_k")):
                    category = "ffn"
                nodes.append(dict(id=name, label=name, category=category, module_id=None,
                                  detail="Tensor operation; parameters belong to connected modules, not counted twice."))
    return nodes, edges


def template(name, rows, backbone, *, attention, ffn="MOE", stages=None, links=None, role="main"):
    from app.lib.spec_studio.grouped import config_hash

    signature = config_hash(dict(rows=rows, attention=attention, ffn=ffn, formula=FORMULA))
    nodes, edges = module_graph(rows, stages, links)
    annotate_graph(nodes, rows)
    payload = dict(id="t-" + signature[:12], signature=signature, label=name,
                attn_kind=attention, ffn_kind=ffn, role=role, modules=rows,
                nodes=nodes, edges=edges, metrics=inventory_metrics(rows), backbone=backbone, mla=None)
    if ffn == "MOE":
        payload["moe"] = moe_layout(rows)
    return payload


def assemble(cfg, backbone, templates_by_layer, shared, *, options, branches=None,
             cache=None, diagnostics=None):
    from app.lib.spec_studio.grouped import config_hash
    from app.lib.spec_studio.validation import ensure_safe_numbers

    templates, groups, execution, known = [], [], [], {}
    for index, item in enumerate(templates_by_layer):
        if item["signature"] not in known:
            group = dict(id=f"g{len(groups)+1}", template_id=item["id"], count=0,
                         ranges=[], metrics={k: 0 for k in item["metrics"]})
            known[item["signature"]] = group
            groups.append(group)
            templates.append(item)
        group = known[item["signature"]]
        group["count"] += 1
        for key, value in item["metrics"].items():
            group["metrics"][key] += value
        if group["ranges"] and group["ranges"][-1][1] == index:
            group["ranges"][-1][1] += 1
        else:
            group["ranges"].append([index, index + 1])
        if execution and execution[-1]["group_id"] == group["id"]:
            execution[-1]["stop_exclusive"] = index + 1
        else:
            execution.append(dict(group_id=group["id"], start=index, stop_exclusive=index+1))
    totals = dict(total=0, active=0, active_body=0, embed=0, lm_head=0,
                  attn=0, ffn_total=0, ffn_active=0, other=0, mtp=0, bytes_by_dtype={})
    matrices = 0
    for rows, count in [(shared, 1)] + [(t["modules"], known[t["signature"]]["count"]) for t in templates]:
        for row in rows:
            n, a = row["params"] * count, row["active_params"] * count
            totals["total"] += n
            totals["active"] += a
            matrices += row["matrix_params"] * count
            category = row["category"]
            key = {"embedding": "embed", "head": "lm_head", "attention": "attn",
                   "ffn": "ffn_total", "mtp": "mtp"}.get(category, "other")
            totals[key] += n
            if category == "ffn":
                totals["ffn_active"] += a
            bpe = {"bf16": 2, "fp16": 2, "fp32": 4, "fp8": 1, "fp4": 0.5}.get(row["dtype"], 2)
            payload = row.get("payload_bytes", row["params"]*bpe)*count
            totals["bytes_by_dtype"][row["dtype"]] = totals["bytes_by_dtype"].get(row["dtype"], 0) + payload
    totals["active_body"] = totals["active"] - totals["embed"] - totals["lm_head"]
    weight = sum(totals["bytes_by_dtype"].values())
    cache = cache or dict(kind="unknown", formula="not modeled", bytes_per_token=None,
                         dtype_assumed=options.get("kv_dtype", "bf16"), notes="No generic cache estimate.")
    result = dict(
        schema_version="2", engine_version="planner-components-1.0.0", formula_version=FORMULA,
        status="supported", identity=dict(name=options.get("name", "model"), model_type=cfg["model_type"],
        repo_id=options.get("repo_id"), architectures=cfg.get("architectures", [])),
        source=dict(repo_id=options.get("repo_id"), requested_revision=options.get("revision"),
                    resolved_sha=options.get("sha"), config_sha256=options.get("config_sha256") or config_hash(cfg),
                    fetched_at=options.get("fetched_at"), kind="huggingface-config" if options.get("sha") else "local-config"),
        backbone=backbone, templates=templates, groups=groups, execution=execution,
        shared_modules=shared, branches=branches or [], totals=totals,
        serving=dict(nparams_total=totals["total"], nparams_active=totals["active"],
                     nparams_active_body=totals["active_body"], weight_bytes=weight,
                     weight_bytes_gib=weight/1024**3, weight_bytes_logical=weight,
                     weight_bytes_source="per-module-payload-estimate-excludes-scales",
                     flop_gemm_decode=2*matrices, kv_layout=cache,
                     include_mtp=options.get("include_mtp", False)),
        capabilities=[dict(id=t["attn_kind"].lower(), label=t["label"], supported=True,
                           evidence=[{"config_key": "model_type", "value": cfg["model_type"]}])
                      for t in templates],
        diagnostics=(diagnostics or []) + [dict(severity="warning", code="memory.payload",
            message="Weight payload estimate excludes quantization scales, packing, allocator and runtime buffers; not GPU peak residency.")],
    )
    ensure_safe_numbers(result)
    from app.engines.morphology import attach_morphology
    return attach_morphology(result, cfg)
