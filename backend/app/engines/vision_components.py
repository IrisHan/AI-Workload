"""Vision encoder and connector inventories from pinned implementation shapes."""
from app.engines.components import Ledger
from app.lib.spec_studio.validation import integer

VL_SOURCE = "https://github.com/deepseek-ai/DeepSeek-VL/tree/681bffb4519856ad27cc17531aacde31ddf6f1a7/deepseek_vl/models"
QWEN_SOURCE = "https://github.com/huggingface/transformers/blob/a2c15b30764b7c6cb0632ac6aed6eac227edc674/src/transformers/models/qwen3_5/modeling_qwen3_5.py"
MOONVIT_SOURCE = (
    "https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct/blob/"
    "398eede0903cd983a2bfa0cc634e9ac1d843f375/modeling_kimi_vl.py"
)


def vit_block(ledger, d, f, count, prefix):
    for norm in ("norm1", "norm2"):
        ledger.tensor(prefix + norm + ".weight", [d], "vision", count=count)
        ledger.tensor(prefix + norm + ".bias", [d], "vision", count=count)
    ledger.linear(prefix + "qkv", d, 3*d, "vision", bias=True, count=count)
    ledger.linear(prefix + "out", d, d, "vision", bias=True, count=count)
    ledger.linear(prefix + "mlp.up", d, f, "vision", bias=True, count=count)
    ledger.linear(prefix + "mlp.down", f, d, "vision", bias=True, count=count)


def convolution(ledger, name, inf, outf, kernel, *, bias=False):
    ledger.tensor(name, [outf, inf, *kernel], "vision")
    if bias:
        ledger.tensor(name + ".bias", [outf], "vision")


def deepseek_vision(cfg):
    vision = cfg.get("vision_config", {})
    params = vision.get("params", {})
    high, low = params.get("high_res_cfg", {}), params.get("low_res_cfg", {})
    if (vision.get("cls") != "HybridVisionTower" or high.get("model_name") != "sam_b_downsample"
            or low.get("model_name") != "siglip_large_patch16_384"
            or params.get("concat_type") != "tuple"):
        raise ValueError("unsupported vision tower composition; expected audited SAM + SigLIP")
    for item in (high, low):
        if item.get("output_dim", 1024) != 1024 or item.get("select_layer", -1) != -1:
            raise ValueError("unsupported vision feature selection")
        integer(item.get("image_size"), "vision.image_size", high=4096)
        if item["image_size"] % 16:
            raise ValueError("vision image must align to patch16")
    ledger = Ledger(VL_SOURCE, "fp16")
    d = 768
    grid = high["image_size"] // 16
    convolution(ledger, "sam.patch", 3, d, [16, 16], bias=True)
    ledger.tensor("sam.position", [grid, grid, d], "vision")
    vit_block(ledger, d, 4*d, 12, "sam.block.")
    ledger.tensor("sam.relative_window", [2, 27, d//12], "vision", count=8)
    ledger.tensor("sam.relative_global", [2, 2*grid-1, d//12], "vision", count=4)
    for prefix in ("sam.neck", "sam.neck_hd"):
        convolution(ledger, prefix + ".conv1", d, 256, [1, 1])
        ledger.norm(prefix + ".norm1", 256, layernorm=True, category="vision")
        convolution(ledger, prefix + ".conv2", 256, 256, [3, 3])
        ledger.norm(prefix + ".norm2", 256, layernorm=True, category="vision")
    convolution(ledger, "sam.downsample1", 256, 512, [3, 3])
    convolution(ledger, "sam.downsample2", 512, 1024, [3, 3])
    ledger.tensor("sam.hd_alpha", [1], "vision")
    d = 1024
    convolution(ledger, "siglip.patch", 3, d, [16, 16], bias=True)
    ledger.tensor("siglip.position", [(low["image_size"]//16)**2, d], "vision")
    vit_block(ledger, d, 4*d, 24, "siglip.block.")
    ledger.norm("siglip.norm", d, layernorm=True, category="vision")
    # Pool exists in checkpoint even though feature extraction skips its forward.
    start = len(ledger.rows)
    ledger.tensor("siglip.pool.latent", [1, d], "vision")
    ledger.linear("siglip.pool.q", d, d, "vision", bias=True)
    ledger.linear("siglip.pool.kv", d, 2*d, "vision", bias=True)
    ledger.linear("siglip.pool.out", d, d, "vision", bias=True)
    ledger.norm("siglip.pool.norm", d, layernorm=True, category="vision")
    ledger.linear("siglip.pool.up", d, 4*d, "vision", bias=True)
    ledger.linear("siglip.pool.down", 4*d, d, "vision", bias=True)
    for row in ledger.rows[start:]:
        row.update(active_params=0, matrix_params=0, notes="Checkpoint-resident pool skipped by feature-only forward.")
    for name in ("high", "low"):
        ledger.norm("hybrid." + name, d, layernorm=True, category="vision")
        for row in ledger.rows[-2:]:
            row.update(active_params=0, notes="Defined but unused in pinned HybridVisionTower.forward.")
    aligner = cfg.get("aligner_config", {})
    ap = aligner.get("params", {})
    if aligner.get("cls") != "MlpProjector" or ap.get("projector_type") != "low_high_hybrid_split_mlp_gelu":
        raise ValueError("unsupported vision connector")
    width = integer(ap.get("n_embed"), "connector width", high=65536)
    if width % 2 or ap.get("input_dim") != d:
        raise ValueError("connector and encoder dimensions mismatch")
    depth = integer(ap.get("depth"), "connector depth", high=16)
    ledger.linear("connector.high", d, width//2, "connector", bias=True)
    ledger.linear("connector.low", d, width//2, "connector", bias=True)
    for i in range(depth-1):
        ledger.linear(f"connector.gelu_linear{i}", width, width, "connector", bias=True)
    return ledger.rows


def qwen_vision(cfg):
    for key in ("hidden_size", "intermediate_size", "depth", "num_heads", "in_channels",
                "patch_size", "temporal_patch_size", "spatial_merge_size",
                "num_position_embeddings", "out_hidden_size"):
        integer(cfg.get(key), "vision." + key, high=65536)
    if cfg.get("deepstack_visual_indexes"):
        raise ValueError("additional deepstack vision mergers require a separate inventory")
    ledger = Ledger(QWEN_SOURCE)
    d, f = cfg["hidden_size"], cfg["intermediate_size"]
    convolution(ledger, "vision.patch", cfg["in_channels"], d,
                [cfg["temporal_patch_size"], cfg["patch_size"], cfg["patch_size"]], bias=True)
    ledger.tensor("vision.position", [cfg["num_position_embeddings"], d], "vision")
    vit_block(ledger, d, f, cfg["depth"], "vision.block.")
    merged = d * cfg["spatial_merge_size"]**2
    ledger.norm("vision.merger.norm", d, layernorm=True, category="connector")
    ledger.linear("vision.merger.fc1", merged, merged, "connector", bias=True)
    ledger.linear("vision.merger.fc2", merged, cfg["out_hidden_size"], "connector", bias=True)
    return ledger.rows


def is_moonvit(cfg):
    if not isinstance(cfg, dict):
        return False
    if cfg.get("model_type") in ("moonvit", "moonvit3d"):
        return True
    return cfg.get("vt_hidden_size") is not None or (
        cfg.get("patch_size") == 14 and isinstance(cfg.get("merge_kernel_size"), list)
    )


def moonvit(cfg, text_hidden):
    d = cfg.get("hidden_size") or cfg.get("vt_hidden_size")
    f = cfg.get("intermediate_size") or cfg.get("vt_intermediate_size")
    depth = cfg.get("num_hidden_layers") or cfg.get("vt_num_hidden_layers")
    heads = cfg.get("num_attention_heads") or cfg.get("vt_num_attention_heads")
    patch = cfg.get("patch_size")
    pos_h = cfg.get("init_pos_emb_height")
    pos_w = cfg.get("init_pos_emb_width")
    merge = cfg.get("merge_kernel_size")
    for name, value in (
        ("vision.hidden_size", d), ("vision.intermediate_size", f),
        ("vision.depth", depth), ("vision.heads", heads),
        ("vision.patch_size", patch), ("vision.pos_h", pos_h), ("vision.pos_w", pos_w),
    ):
        integer(value, name, high=65536)
    if not isinstance(merge, list) or len(merge) != 2 or any(type(item) is not int for item in merge):
        raise ValueError("MoonViT merge_kernel_size must be two integers")
    integer(merge[0], "vision.merge_h", high=16)
    integer(merge[1], "vision.merge_w", high=16)
    qkv = cfg.get("qkv_hidden_size") or d
    integer(qkv, "vision.qkv_hidden_size", high=65536)
    if qkv % heads:
        raise ValueError("MoonViT qkv hidden size must be divisible by heads")
    rms = cfg.get("norm_type") == "rmsnorm"
    attn_bias = cfg.get("attn_bias")
    if attn_bias is None:
        attn_bias = True
    linear_bias = cfg.get("linear_bias")
    if linear_bias is None:
        linear_bias = True
    patch_bias = cfg.get("patch_embed_proj_bias")
    if patch_bias is None:
        patch_bias = True
    text_out = cfg.get("text_hidden_size") or text_hidden
    integer(text_out, "vision.text_hidden_size", high=65536)
    if text_out != text_hidden:
        raise ValueError("MoonViT projector output does not match language hidden size")
    mm = cfg.get("mm_hidden_size") or d
    integer(mm, "vision.mm_hidden_size", high=65536)
    ledger = Ledger(MOONVIT_SOURCE)
    convolution(ledger, "vision.patch", 3, d, [patch, patch], bias=patch_bias)
    ledger.tensor("vision.position", [pos_h, pos_w, d], "vision")
    for name in ("norm0", "norm1"):
        ledger.tensor("vision.block." + name + ".weight", [d], "vision", count=depth)
        if not rms:
            ledger.tensor("vision.block." + name + ".bias", [d], "vision", count=depth)
    ledger.linear("vision.block.wqkv", d, 3 * qkv, "vision", bias=attn_bias, count=depth)
    ledger.linear("vision.block.wo", qkv, d, "vision", bias=attn_bias, count=depth)
    ledger.linear("vision.block.mlp.up", d, f, "vision", bias=linear_bias, count=depth)
    ledger.linear("vision.block.mlp.down", f, d, "vision", bias=linear_bias, count=depth)
    ledger.norm("vision.final_norm", d, layernorm=not rms, category="vision")
    merged = mm * merge[0] * merge[1]
    projector = cfg.get("mm_projector_type")
    if projector == "patchmergerv2":
        ledger.linear("vision.merger.fc1", merged, merged, "connector")
        ledger.linear("vision.merger.fc2", merged, text_out, "connector")
        ledger.norm("vision.merger.post_norm", text_out, category="connector")
    else:
        ledger.norm("vision.merger.norm", mm, layernorm=True, category="connector")
        ledger.linear("vision.merger.fc1", merged, merged, "connector", bias=True)
        ledger.linear("vision.merger.fc2", merged, text_out, "connector", bias=True)
    return ledger.rows


def clip_rope_vit(vision, text_hidden, parent=None):
    """CLIP-style encoder with Conv3d patch embed and 3D RoPE; MiniMax M3 VL tower."""
    parent = parent or {}
    compression = vision.get("img_token_compression_config") or parent.get("img_token_compression_config") or {}
    d = vision["hidden_size"]
    f = vision["intermediate_size"]
    depth = vision["num_hidden_layers"]
    heads = vision["num_attention_heads"]
    patch = vision["patch_size"]
    channels = vision.get("num_channels") or 3
    temporal = vision.get("temporal_patch_size") or compression.get("temporal_patch_size") or 1
    merge = vision.get("spatial_merge_size") or compression.get("spatial_merge_size") or 1
    for name, value in (
        ("vision.hidden_size", d), ("vision.intermediate_size", f), ("vision.depth", depth),
        ("vision.heads", heads), ("vision.patch_size", patch), ("vision.temporal", temporal),
        ("vision.merge", merge),
    ):
        integer(value, name, high=65536)
    projector_hidden = parent.get("projector_hidden_size") or vision.get("projection_dim") or text_hidden
    integer(projector_hidden, "vision.projector_hidden_size", high=65536)
    if d % heads:
        raise ValueError("CLIP vision hidden size must be divisible by heads")
    source = (
        "https://github.com/huggingface/transformers/blob/"
        "v5.13.0/src/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py"
    )
    ledger = Ledger(source)
    convolution(ledger, "vision.patch", channels, d, [temporal, patch, patch], bias=False)
    ledger.norm("vision.pre_ln", d, layernorm=True, category="vision")
    for name in ("norm1", "norm2"):
        ledger.tensor("vision.block." + name + ".weight", [d], "vision", count=depth)
        ledger.tensor("vision.block." + name + ".bias", [d], "vision", count=depth)
    for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
        ledger.linear("vision.block." + proj, d, d, "vision", bias=True, count=depth)
    ledger.linear("vision.block.mlp.fc1", d, f, "vision", bias=True, count=depth)
    ledger.linear("vision.block.mlp.fc2", f, d, "vision", bias=True, count=depth)
    ledger.linear("vision.projector.linear_1", d, projector_hidden, "connector", bias=True)
    ledger.linear("vision.projector.linear_2", projector_hidden, text_hidden, "connector", bias=True)
    merged = text_hidden * merge * merge
    ledger.linear("vision.projector.merge_linear_1", merged, projector_hidden, "connector", bias=True)
    ledger.linear("vision.projector.merge_linear_2", projector_hidden, text_hidden, "connector", bias=True)
    return ledger.rows


def intern_vit(vision, text_hidden, parent=None):
    """InternViT encoder + InternVL2.5 mlp1 projector (pixel-shuffle downsample)."""
    parent = parent or {}
    d = vision["hidden_size"]
    f = vision["intermediate_size"]
    depth = vision["num_hidden_layers"]
    heads = vision["num_attention_heads"]
    patch = vision["patch_size"]
    image = vision.get("image_size") or parent.get("force_image_size")
    channels = vision.get("num_channels") or 3
    downsample = parent.get("downsample_ratio") or 0.5
    for name, value in (
        ("vision.hidden_size", d), ("vision.intermediate_size", f), ("vision.depth", depth),
        ("vision.heads", heads), ("vision.patch_size", patch), ("vision.image_size", image),
    ):
        integer(value, name, high=65536)
    if image % patch:
        raise ValueError("InternViT image_size must align to patch_size")
    grid = image // patch
    merge = int(round(1 / downsample)) if downsample else 1
    integer(merge, "vision.pixel_shuffle", high=16)
    source = "InternViT + InternVL2.5 mlp1 projector; Qianfan-OCR template internvl2_5"
    ledger = Ledger(source)
    convolution(ledger, "vision.patch", channels, d, [patch, patch], bias=True)
    ledger.tensor("vision.class_embedding", [d], "vision")
    ledger.tensor("vision.position", [grid * grid + 1, d], "vision")
    for name in ("norm1", "norm2"):
        ledger.tensor("vision.block." + name + ".weight", [d], "vision", count=depth)
        ledger.tensor("vision.block." + name + ".bias", [d], "vision", count=depth)
    ledger.linear("vision.block.wqkv", d, 3 * d, "vision", bias=True, count=depth)
    ledger.linear("vision.block.wo", d, d, "vision", bias=True, count=depth)
    ledger.tensor("vision.block.ls1", [d], "vision", count=depth)
    ledger.tensor("vision.block.ls2", [d], "vision", count=depth)
    ledger.linear("vision.block.mlp.fc1", d, f, "vision", bias=True, count=depth)
    ledger.linear("vision.block.mlp.fc2", f, d, "vision", bias=True, count=depth)
    merged = d * merge * merge
    ledger.norm("vision.mlp1.ln", merged, layernorm=True, category="connector")
    ledger.linear("vision.mlp1.fc1", merged, text_hidden, "connector", bias=True)
    ledger.linear("vision.mlp1.fc2", text_hidden, text_hidden, "connector", bias=True)
    return ledger.rows


def glm_next_vision(cfg, text_hidden):
    """Qwen-style ViT extras: fused QKV, per-head QK-Norm, SwiGLU, RMSNorm, SwiGLU merger."""
    for key in ("hidden_size", "intermediate_size", "depth", "num_heads", "in_channels",
                "patch_size", "temporal_patch_size", "spatial_merge_size", "out_hidden_size"):
        integer(cfg.get(key), "vision." + key, high=65536)
    d, f, depth = cfg["hidden_size"], cfg["intermediate_size"], cfg["depth"]
    heads = cfg["num_heads"]
    out = cfg["out_hidden_size"]
    if out != text_hidden:
        raise ValueError("vision connector output does not match text hidden size")
    proj_w = cfg.get("projection_intermediate_size") or (d * cfg["spatial_merge_size"] ** 2)
    integer(proj_w, "vision.projection_intermediate_size", high=1048576)
    c = d // heads
    source = (
        "https://huggingface.co/zai-org/GLM-5.3-Flash/blob/"
        "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/model-00062-of-00062.safetensors"
        "#header: model.visual.* BF16 shapes"
    )
    ledger = Ledger(source)
    convolution(
        ledger, "vision.patch", cfg["in_channels"], d,
        [cfg["temporal_patch_size"], cfg["patch_size"], cfg["patch_size"]], bias=True,
    )
    ledger.norm("vision.post_layernorm", d, category="vision")
    ledger.linear("vision.block.qkv", d, 3 * d, "vision", bias=True, count=depth)
    for name, dim in (("q_norm", c), ("k_norm", c), ("norm1", d), ("norm2", d)):
        ledger.tensor("vision.block." + name + ".weight", [dim], "vision", count=depth)
    ledger.linear("vision.block.proj", d, d, "vision", bias=True, count=depth)
    ledger.linear("vision.block.mlp.gate_proj", d, f, "vision", bias=True, count=depth)
    ledger.linear("vision.block.mlp.up_proj", d, f, "vision", bias=True, count=depth)
    ledger.linear("vision.block.mlp.down_proj", f, d, "vision", bias=True, count=depth)
    merged = d * cfg["spatial_merge_size"] ** 2
    convolution(ledger, "vision.downsample", d, merged,
                [cfg["spatial_merge_size"], cfg["spatial_merge_size"]], bias=True)
    ledger.linear("vision.merger.proj", merged, merged, "connector")
    ledger.linear("vision.merger.gate_proj", merged, proj_w, "connector")
    ledger.linear("vision.merger.up_proj", merged, proj_w, "connector")
    ledger.linear("vision.merger.down_proj", proj_w, out, "connector")
    ledger.norm("vision.merger.post_projection_norm", out, layernorm=True, category="connector")
    return ledger.rows
