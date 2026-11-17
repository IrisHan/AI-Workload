from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

HF_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.)?huggingface\.co/([^/]+)/([^/]+)(?:/.*)?",
    re.IGNORECASE,
)


@dataclass
class RawBundle:
    repo_id: str
    revision: str
    sha: str | None
    config: dict[str, Any]
    tokenizer_config: dict[str, Any] | None = None
    safetensors_index: dict[str, Any] | None = None
    readme_excerpt: str | None = None
    used_storage: int | None = None
    safetensors_histogram: dict[str, int] | None = None
    config_sha256: str | None = None
    fetched_at: str | None = None


_CACHE: dict[str, RawBundle] = {}
_LOCK = Lock()
MAX_METADATA_BYTES = 4 * 1024 * 1024


def parse_repo_id(url_or_id: str) -> str:
    text = url_or_id.strip().rstrip("/")
    m = HF_URL_RE.match(text)
    if m:
        text = f"{m.group(1)}/{m.group(2)}"
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", text):
        from huggingface_hub.utils import validate_repo_id

        validate_repo_id(text)
        return text
    raise ValueError(f"not a Hugging Face repo id or URL: {url_or_id!r}")


def fetch_bundle(
    repo_id: str,
    *,
    revision: str = "main",
    token: str | None = None,
    cache_dir: Path | None = None,
) -> RawBundle:
    from huggingface_hub import hf_hub_download, model_info

    repo_id = parse_repo_id(repo_id)
    info = model_info(repo_id, revision=revision, token=token, timeout=15, files_metadata=True)
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Hub did not resolve an immutable SHA")
    used_storage, histogram = _hub_weight_stats(info)
    auth_scope = hashlib.sha256((token or "").encode()).hexdigest()
    cache_key = f"{repo_id}@{sha}:{auth_scope}"
    with _LOCK:
        if cache_key in _CACHE:
            cached = deepcopy(_CACHE[cache_key])
            cached.revision = revision
            return cached
    sizes = {item.rfilename: getattr(item, "size", None) for item in getattr(info, "siblings", []) or []}
    if (sizes.get("config.json") or 0) > MAX_METADATA_BYTES:
        raise ValueError("config.json exceeds metadata size limit")

    cfg_path = hf_hub_download(
        repo_id,
        filename="config.json",
        revision=sha,
        token=token,
        cache_dir=cache_dir,
        etag_timeout=10,
    )
    config = _read_json(cfg_path)
    bundle = RawBundle(
        repo_id=repo_id,
        revision=revision,
        sha=sha,
        config=config,
        used_storage=used_storage,
        safetensors_histogram=histogram,
        config_sha256=hashlib.sha256(Path(cfg_path).read_bytes()).hexdigest(),
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )
    for name in ("model.safetensors.index.json", "model.safetensors.index.fp8.json"):
        if (sizes.get(name) or 0) > MAX_METADATA_BYTES:
            continue
        idx = _optional_json(repo_id, name, sha, token, cache_dir)
        if idx:
            bundle.safetensors_index = idx
            break
    readme_path = (
        None
        if (sizes.get("README.md") or 0) > MAX_METADATA_BYTES
        else _optional_file(repo_id, "README.md", sha, token, cache_dir)
    )
    if readme_path:
        text = Path(readme_path).read_text(encoding="utf-8", errors="replace")
        bundle.readme_excerpt = text[:8000]
    with _LOCK:
        if len(_CACHE) >= 32:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[cache_key] = deepcopy(bundle)
    return bundle


def _hub_weight_stats(info: Any) -> tuple[int | None, dict[str, int] | None]:
    used = getattr(info, "usedStorage", None)
    if used is None:
        used = getattr(info, "used_storage", None)
    hist = None
    st = getattr(info, "safetensors", None)
    if st is not None:
        params = getattr(st, "parameters", None)
        if isinstance(params, dict):
            hist = {str(k): int(v) for k, v in params.items()}
        elif params is not None and hasattr(params, "items"):
            hist = {str(k): int(v) for k, v in params.items()}
    return (int(used) if used else None), hist


def _optional_json(repo_id, filename, revision, token, cache_dir) -> dict[str, Any] | None:
    path = _optional_file(repo_id, filename, revision, token, cache_dir)
    return _read_json(path) if path else None


def _optional_file(repo_id, filename, revision, token, cache_dir) -> str | None:
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id,
            filename=filename,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            etag_timeout=10,
        )
    except Exception:
        return None


def _read_json(path: str) -> dict[str, Any]:
    if Path(path).stat().st_size > MAX_METADATA_BYTES:
        raise ValueError("metadata exceeds size limit")
    return json.loads(Path(path).read_text())
