from __future__ import annotations

from typing import Any

from .accountant import account, cross_check_index
from .glossary import glossary_for, load_glossary
from .graph import build_graph
from .ir import InspectResult
from .matcher import match_config
from .validation import ensure_safe_numbers


def inspect_config(
    cfg: dict[str, Any],
    *,
    name: str = "model",
    include_mtp: bool = False,
    safetensors_index: dict[str, Any] | None = None,
    repo_id: str | None = None,
    revision: str | None = None,
    sha: str | None = None,
    hub_used_storage: int | None = None,
    hub_safetensors: dict[str, int] | None = None,
    kv_dtype: str = "bf16",
) -> InspectResult:
    match = match_config(cfg, name=name)
    if repo_id:
        match["identity"].repo_id = repo_id
    if revision:
        match["identity"].revision = revision
    if sha:
        match["identity"].sha = sha
    arch = account(match, include_mtp=include_mtp, kv_dtype=kv_dtype)
    cross_check_index(
        arch,
        safetensors_index,
        hub_used_storage=hub_used_storage,
        hub_safetensors=hub_safetensors,
    )
    graph = build_graph(arch)
    caps = [c.id for c in arch.capabilities]
    ensure_safe_numbers(arch.model_dump())
    return InspectResult(arch=arch, graph=graph, glossary=glossary_for(caps) or load_glossary())


def inspect_config_v2(cfg: dict[str, Any], **kwargs) -> dict[str, Any]:
    from .grouped import inspect_grouped

    return inspect_grouped(cfg, **kwargs)
