"""MC-01: checkpoint paths, layer-specific ignores and independent payload sums."""
import copy
import json
from pathlib import Path

from app.engines.model_memory import memory_components, module_payload_bytes
from app.engines.model_structure import inspect_model
from app.engines.morphology import module_ignored

ROOT = Path(__file__).resolve().parents[2] / "docs/reports"


def test_gpt_oss_full_checkpoint_attention_and_router_ignore():
    cfg = json.loads((ROOT / "2026-09-15-frontier-families/evidence/sources/openai--gpt-oss-120b/config.json").read_text())
    a = inspect_model(cfg)
    template = a["templates"][0]
    for name in ("q_proj", "router"):
        row = next(r for r in template["modules"] if r["id"] == name)
        assert module_payload_bytes(row, cfg, 2, layer=0) == row["params"] * 2
    assert any("model.layers.{layer}.self_attn.q_proj" in r["checkpoint_paths"]
               for r in template["modules"] if r["id"] == "q_proj")


def test_quant_layer_specific_ignore_is_not_multiplied_across_group():
    cfg = json.loads((ROOT / "2026-09-15-frontier-families/evidence/sources/openai--gpt-oss-20b/config.json").read_text())
    cfg["quantization_config"]["modules_to_not_convert"] = []
    cfg["quantization_config"]["ignore"] = []
    a = inspect_model(cfg)
    base = memory_components(a, cfg)["weight_bytes"]
    changed = copy.deepcopy(cfg)
    changed["quantization_config"]["ignore"] = ["model.layers.0.self_attn.q_proj"]
    one = memory_components(a, changed)["weight_bytes"]
    group = next(g for g in a["groups"] if any(start == 0 for start, _ in g["ranges"]))
    t = next(t for t in a["templates"] if t["id"] == group["template_id"])
    row = next(r for r in t["modules"] if r["id"] == "q_proj")
    expected_delta = row["shape"][0] * row["shape"][1] * 3 // 2
    assert one - base == expected_delta
    changed["quantization_config"]["ignore"] = ["model.layers.*.self_attn.q_proj"]
    assert memory_components(a, changed)["weight_bytes"] - base == expected_delta * cfg["num_hidden_layers"]


def test_path_boundaries_shared_vision_and_regex():
    assert module_ignored("router", "ffn", ["model.layers.1.mlp.router"],
                          paths=["model.layers.1.mlp.router"])
    assert not module_ignored("router", "ffn", ["model.layers.1.mlp.router"],
                              paths=["model.layers.10.mlp.router"])
    assert module_ignored("shared_expert.up_proj", "ffn", ["model.layers.*.mlp.shared_experts"],
                          paths=["model.layers.4.mlp.shared_experts.up_proj"])
    assert module_ignored("vision.downsample", "vision", ["visual.downsample"],
                          role="vision", paths=["model.visual.downsample"])
    assert not module_ignored("vision.patch", "vision", ["mm_projector"], role="vision")
    assert module_ignored("q_proj", "attention", [r"re:.*layers\.0\.self_attn.*"],
                          paths=["model.layers.0.self_attn.q_proj"])
    assert not module_ignored("q_proj", "attention", [])


def test_grouped_bias_is_not_quantized_with_matrix():
    cfg = dict(model_type="gpt_oss", quantization_config={"quant_method": "mxfp4"})
    row = dict(id="q_proj", category="attention", shape=[16, 8],
               params=144, matrix_params=128, count=1)
    assert module_payload_bytes(row, cfg, 2) == 128 // 2 + 16 * 2
