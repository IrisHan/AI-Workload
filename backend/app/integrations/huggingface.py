import hashlib
import json
import re
from urllib.parse import quote, urljoin, urlparse

import httpx
from app.domain.specs import ModelSpec, utcnow
from app.engines.layer_methods import is_hybrid_language
from app.engines.model_analyzer import validate_legacy_config
from app.engines.model_capabilities import CAPABILITY_VERSION
from app.engines.model_structure import inspect_model

REPO_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
REVISION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


class SourceUnavailable(Exception):
    pass


def hub_scope(cfg, logical, hub_total):
    """Classify Hub tensor totals against logical parameters. Never overwrite logical N."""
    notes = (
        "Checkpoint total is stored tensor elements, may include MTP, scales and integer buffers; "
        "never replaces logical main parameters."
    )
    if hub_total is None:
        return "unknown", None, notes
    delta = hub_total - logical
    inner = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    if cfg.get("quantization_config") or inner.get("quantization_config"):
        return "quant-scale", delta, notes + " Quantization scales and packing may differ from logical shapes."
    if cfg.get("vision_config") or inner.get("vision_config"):
        return "different-scope", delta, notes + " Vision and connector weights change Hub vs language comparison."
    if (
        cfg.get("num_nextn_predict_layers")
        or cfg.get("mtp_num_hidden_layers")
        or inner.get("mtp_num_hidden_layers")
        or cfg.get("model_type") in ("qwen3_5", "qwen3_5_moe")
    ):
        return "different-scope", delta, notes + " MTP or vision live outside the main decoder total."
    if delta == 0:
        return "same-scope", 0, notes
    vocab = cfg.get("vocab_size") or cfg.get("padded_vocab_size")
    hidden = cfg.get("hidden_size") or cfg.get("n_embd") or cfg.get("d_model")
    layers = cfg.get("num_hidden_layers") or cfg.get("n_layer") or cfg.get("num_layers")
    if (
        cfg.get("tie_word_embeddings") is True
        and type(vocab) is int
        and type(hidden) is int
        and delta == vocab * hidden
    ):
        return (
            "tied-lm-head-duplicate",
            delta,
            notes + " Hub stores a separate lm_head while embeddings are tied; logical count excludes the duplicate.",
        )
    if (
        cfg.get("model_type") == "qwen"
        and cfg.get("no_bias") is True
        and type(layers) is int
        and type(hidden) is int
        and delta == layers * 3 * hidden
    ):
        return (
            "qwen1-fused-qkv-bias",
            delta,
            notes + " Hub includes fused c_attn.bias (L×3×hidden); logical no_bias layout omits it.",
        )
    if is_hybrid_language(cfg) and delta != 0:
        return (
            "different-scope",
            delta,
            notes + " Hub may include MTP or other non-main tensors; never replaces logical main parameters.",
        )
    raise ValueError(f"Parameter count mismatch: config={logical}, source={hub_total}")


def normalize_model(repo_id: str, revision: str, raw: bytes, retrieved_at: str | None = None) -> ModelSpec:
    config = json.loads(raw)
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object")
    analysis = inspect_model(config)
    parameters = analysis["totals"]["total"]
    bb = analysis["backbone"]
    try:
        validate_legacy_config(config)
        workload_supported = True
        family_is_legacy = True
    except ValueError:
        workload_supported = bool((analysis.get("morphology") or {}).get("cost_supported"))
        family_is_legacy = False
    digest = hashlib.sha256(raw).hexdigest()
    identifier = hashlib.sha256(f"{repo_id}@{revision}".encode()).hexdigest()[:24]
    return ModelSpec(
        id=identifier,
        name=repo_id.split("/")[-1],
        provider=repo_id.split("/")[0],
        repo_id=repo_id,
        revision=revision,
        architecture=config["model_type"],
        parameters=parameters,
        layers=bb["num_layers"],
        hidden_size=bb["hidden_size"],
        attention_heads=bb["num_heads"],
        kv_heads=bb["num_kv_heads"],
        intermediate_size=bb["intermediate_size"],
        vocab_size=bb["vocab_size"],
        max_position_embeddings=bb["context_length"],
        tie_word_embeddings=config.get("tie_word_embeddings", False),
        dtype=config.get("torch_dtype", config.get("dtype", "unknown")),
        source_url=f"https://huggingface.co/{repo_id}/resolve/{revision}/config.json",
        retrieved_at=retrieved_at or utcnow(),
        config_sha256=digest,
        raw_config=config,
        version=f"{revision}:{digest[:12]}",
        is_moe=any(item["ffn_kind"] in ("MOE", "LATENT_MOE") for item in analysis["templates"]),
        active_parameters=analysis["totals"]["active"],
        analysis_formula=analysis["formula_version"],
        parameter_scope="main-with-vision" if (
            config["model_type"] == "multi_modality" or isinstance(config.get("vision_config"), dict)
        ) else "main",
        workload_supported=workload_supported,
        capability_version=CAPABILITY_VERSION,
        support_notes=(
            []
            if family_is_legacy
            else (
                ["Workload uses morphology-workload-1.1.0 with operator-cost-2.2.0; not the dense GQA closed form."]
                if workload_supported
                else [
                    "Structure and static weight/cache analysis supported; closed-form Workload waits on remaining layer methods."
                ]
            )
        ),
    )


class HuggingFaceImporter:
    def __init__(self, transport=None):
        self.transport = transport

    def _fetch(self, client: httpx.Client, url: str) -> bytes:
        for _ in range(4):
            parsed = urlparse(url)
            if parsed.scheme != "https" or parsed.hostname != "huggingface.co":
                raise SourceUnavailable("External redirect rejected; only huggingface.co is allowed")
            with client.stream("GET", url) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise SourceUnavailable("Source returned an invalid redirect")
                    url = urljoin(url, location)
                    continue
                response.raise_for_status()
                content = bytearray()
                for chunk in response.iter_bytes():
                    content.extend(chunk)
                    if len(content) > 2_000_000:
                        raise SourceUnavailable("Source JSON exceeds 2 MB limit")
                return bytes(content)
        raise SourceUnavailable("Source returned too many redirects")

    def import_model(self, repo_id: str, revision: str = "main") -> ModelSpec:
        if repo_id.startswith("https://"):
            parsed = urlparse(repo_id)
            if parsed.netloc != "huggingface.co" or parsed.query or parsed.fragment:
                raise ValueError("Use an HTTPS huggingface.co repository URL without query or fragment")
            repo_id = parsed.path.strip("/")
        if not REPO_PATTERN.fullmatch(repo_id) or ".." in repo_id:
            raise ValueError("repo_id must be a HuggingFace organization/model identifier")
        if not REVISION_PATTERN.fullmatch(revision) or ".." in revision:
            raise ValueError("revision must be a branch, tag, or commit identifier")
        try:
            with httpx.Client(timeout=httpx.Timeout(15, connect=5), transport=self.transport) as client:
                meta = json.loads(
                    self._fetch(
                        client, f"https://huggingface.co/api/models/{repo_id}/revision/{quote(revision)}"
                    )
                )
                sha = meta.get("sha") if isinstance(meta, dict) else None
                if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
                    raise SourceUnavailable("Source did not return an immutable commit SHA")
                raw = self._fetch(client, f"https://huggingface.co/{repo_id}/resolve/{sha}/config.json")
                model = normalize_model(repo_id, sha, raw)
                tensor_meta = meta.get("safetensors") or {}
                total = tensor_meta.get("total")
                comparison, delta, notes = hub_scope(model.raw_config, model.parameters, total if type(total) is int else None)
                return model.model_copy(update={"source_metadata": {
                    "checkpoint_safetensors": tensor_meta,
                    "comparison": comparison,
                    "checkpoint_minus_main": delta,
                    "notes": notes,
                }})
        except httpx.HTTPError as exc:
            raise SourceUnavailable(
                "HuggingFace unavailable or repository is private/gated; retry a public model"
            ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise SourceUnavailable("HuggingFace returned invalid JSON") from exc
