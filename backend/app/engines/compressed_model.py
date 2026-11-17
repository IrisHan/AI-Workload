"""Shared-KV compressed attention, expert FFNs and hyper-connection composition."""
from copy import deepcopy

from app.engines.components import Ledger, assemble, template
from app.engines.operator_methods import compressed_attention
from app.lib.spec_studio.inventory import inventory_metrics, module_row
from app.lib.spec_studio.validation import integer

SOURCE = "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py"


def validate(cfg):
    if cfg.get("architectures") != ["DeepseekV4ForCausalLM"]:
        raise ValueError("compressed attention architecture mismatch")
    for key in ("hidden_size", "num_attention_heads", "head_dim", "q_lora_rank", "o_lora_rank",
                "o_groups", "moe_intermediate_size", "n_routed_experts", "num_experts_per_tok",
                "index_head_dim", "index_n_heads", "index_topk", "sliding_window",
                "vocab_size", "max_position_embeddings", "qk_rope_head_dim"):
        integer(cfg.get(key), key, high=16777216)
    integer(cfg.get("num_hidden_layers"), "num_hidden_layers", high=4096)
    integer(cfg.get("hc_mult"), "hc_mult", high=16)
    integer(cfg.get("num_nextn_predict_layers", 0), "mtp", low=0, high=16)
    integer(cfg.get("num_hash_layers", 0), "num_hash_layers", low=0, high=cfg["num_hidden_layers"])
    if cfg.get("n_shared_experts") != 1 or cfg["num_experts_per_tok"] > cfg["n_routed_experts"]:
        raise ValueError("unsupported shared expert count or invalid top-k")
    if cfg["num_attention_heads"] % cfg["o_groups"] or cfg.get("num_key_value_heads") != 1:
        raise ValueError("invalid grouped output or shared-KV heads")
    if cfg["qk_rope_head_dim"] > min(cfg["head_dim"], cfg["index_head_dim"]):
        raise ValueError("RoPE dimension exceeds attention/index head")
    if cfg.get("attention_bias") or cfg.get("mlp_bias") or cfg.get("hidden_act") != "silu":
        raise ValueError("compressed attention requires bias-free SwiGLU")
    if cfg.get("tie_word_embeddings") is not False:
        raise ValueError("compressed attention tied embedding variant is unverified")
    if cfg.get("expert_dtype") not in ("fp4", "fp8"):
        raise ValueError("unsupported expert storage precision")
    ratios = cfg.get("compress_ratios")
    if not isinstance(ratios, list) or len(ratios) != cfg["num_hidden_layers"] + cfg.get("num_nextn_predict_layers", 0):
        raise ValueError("compress_ratios must include exactly main and MTP layers")
    if any(type(r) is not int or r not in (0, 4, 128) for r in ratios):
        raise ValueError("unsupported compression schedule")


def hyper_connection(ledger, prefix, d, n, *, head=False):
    mix = n if head else (n+2)*n
    ledger.linear(prefix + ".fn", n*d, mix, "residual", dtype="fp32")
    ledger.tensor(prefix + ".base", [mix], "residual", dtype="fp32")
    ledger.tensor(prefix + ".scale", [1 if head else 3], "residual", dtype="fp32")


def block(cfg, index):
    d, h, c, r = (cfg[k] for k in ("hidden_size", "num_attention_heads", "head_dim", "q_lora_rank"))
    g, o, n = cfg["o_groups"], cfg["o_lora_rank"], cfg["hc_mult"]
    ratio = cfg["compress_ratios"][index]
    ledger = Ledger(SOURCE, "fp8")
    ledger.norm("attn_norm", d)
    ledger.norm("ffn_norm", d)
    for prefix in ("hc_attn", "hc_ffn"):
        hyper_connection(ledger, prefix, d, n)
    compressed_attention(
        ledger, hidden=d, heads=h, head_dim=c, q_rank=r, output_groups=g,
        output_rank=o, ratio=ratio, index_heads=cfg["index_n_heads"],
        index_dim=cfg["index_head_dim"],
    )
    e, k, f = cfg["n_routed_experts"], cfg["num_experts_per_tok"], cfg["moe_intermediate_size"]
    ledger.linear("router", d, e, "ffn", dtype="fp32")
    if index >= cfg.get("num_hash_layers", 0):
        ledger.tensor("router_correction", [e], "ffn", dtype="fp32")
    ledger.ffn(d, f, prefix="routed_expert.", count=e, active_count=k, dtype=cfg["expert_dtype"])
    ledger.ffn(d, f, prefix="shared_expert.")
    for row in ledger.rows:
        if "norm" in row["id"]:
            row["dtype"] = "fp32"
    return ledger.rows


def block_links(cfg, index):
    links = []

    def chain(*names):
        links.extend(zip(names, names[1:]))

    for prefix, start, norm, end in (
        ("hc_attn", "input", "attn_norm.weight", "o_b_proj"),
        ("hc_ffn", "hc_attn_post", "ffn_norm.weight", "weighted_expert_sum"),
    ):
        chain(start, prefix + ".fn", prefix + "_sinkhorn", prefix + "_pre", norm)
        chain(prefix + ".base", prefix + "_sinkhorn")
        chain(prefix + ".scale", prefix + "_sinkhorn")
        chain(start, prefix + "_pre")
        chain(start, prefix + "_post")
        chain(prefix + "_sinkhorn", prefix + "_post")
        chain(end, prefix + "_post")
    chain("attn_norm.weight", "q_a_proj", "q_norm.weight", "q_b_proj", "rope_q", "qk_matmul")
    chain("attn_norm.weight", "kv_proj", "kv_norm.weight", "rope_k", "qk_matmul")
    chain("rope_k", "av_matmul")
    chain("qk_matmul", "causal_window", "mask_softmax_sink", "av_matmul",
          "inverse_rope", "o_a_proj", "o_b_proj")
    chain("attn_sink", "mask_softmax_sink")
    ratio = cfg["compress_ratios"][index]

    def compressor(prefix, target):
        chain("attn_norm.weight", prefix + ".kv", prefix + ".pool", prefix + ".norm.weight", target)
        chain("attn_norm.weight", prefix + ".gate", prefix + ".pool")
        chain(prefix + ".ape", prefix + ".pool")

    if ratio:
        compressor("compressor", "compressed_rope")
        chain("compressed_rope", "qk_matmul")
        chain("compressed_rope", "av_matmul")
    if ratio == 4:
        compressor("indexer.compressor", "indexer.score")
        chain("q_norm.weight", "indexer.q_proj", "indexer.score")
        chain("attn_norm.weight", "indexer.weights_proj", "indexer.score")
        chain("indexer.score", "sparse_select", "causal_window")
    chain("ffn_norm.weight", "router", "router_scoring")
    if index < cfg.get("num_hash_layers", 0):
        chain("token_ids", "hash_lookup", "expert_selection")
        chain("router_scoring", "expert_selection")
    else:
        chain("router_scoring", "top_k", "expert_selection")
        chain("router_correction", "top_k")
    for prefix in ("routed_expert", "shared_expert"):
        chain("ffn_norm.weight", prefix + ".gate_proj", prefix + ".clamped_swiglu", prefix + ".down_proj", "weighted_expert_sum")
        chain("ffn_norm.weight", prefix + ".up_proj", prefix + ".clamped_swiglu")
    chain("expert_selection", "routed_expert.gate_proj")
    chain("expert_selection", "weighted_expert_sum")
    chain("hc_ffn_post", "output")
    return links


def inspect_compressed(cfg, **options):
    validate(cfg)
    d, v, layers = cfg["hidden_size"], cfg["vocab_size"], cfg["num_hidden_layers"]
    bb = dict(hidden_size=d, vocab_size=v, num_layers=layers, num_heads=cfg["num_attention_heads"],
              num_kv_heads=1, head_dim=cfg["head_dim"], intermediate_size=cfg["moe_intermediate_size"],
              context_length=cfg["max_position_embeddings"], activation="silu",
              mtp_layers=cfg.get("num_nextn_predict_layers", 0), sliding_window=cfg["sliding_window"])
    shared = Ledger(SOURCE, "bf16")
    shared.tensor("embed", [v, d], "embedding")
    shared.norm("final_norm", d)
    hyper_connection(shared, "hc_head", d, cfg["hc_mult"], head=True)
    shared.linear("lm_head", d, v, "head")
    templates = []
    for i in range(layers):
        ratio = cfg["compress_ratios"][i]
        kind = {0: "SWA", 4: "CSA", 128: "HCA"}[ratio]
        rows = block(cfg, i)
        item = template(kind + " + MoE + mHC", rows, bb, attention=kind, links=block_links(cfg, i))
        item["compress_ratio"] = ratio
        templates.append(item)
    branches = []
    mtp_count = cfg.get("num_nextn_predict_layers", 0)
    for offset in range(mtp_count):
        rows = block(cfg, layers+offset)
        extra = Ledger(SOURCE, "bf16")
        extra.linear("e_proj", d, d, "fusion")
        extra.linear("h_proj", d, d, "fusion")
        for name in ("enorm", "hnorm", "norm"):
            extra.norm(name, d)
        hyper_connection(extra, "hc_head", d, cfg["hc_mult"], head=True)
        rows += extra.rows
        rows.append(module_row("shared_lm_head", "head", [v, d], 0, shared_with="lm_head",
                               source=SOURCE, formula="shared; 0 unique"))
        links = block_links(cfg, layers+offset) + [
            ("token_embeddings", "enorm.weight"), ("enorm.weight", "e_proj"),
            ("hidden_input", "hnorm.weight"), ("hnorm.weight", "h_proj"),
            ("e_proj", "input"), ("h_proj", "input"), ("output", "hc_head.fn"),
            ("hc_head.fn", "hc_head_collapse"), ("hc_head.base", "hc_head_collapse"),
            ("hc_head.scale", "hc_head_collapse"), ("hc_head_collapse", "norm.weight"),
            ("norm.weight", "shared_lm_head"),
        ]
        branch = template("MTP", rows, bb, attention={0: "SWA", 4: "CSA", 128: "HCA"}[
            cfg["compress_ratios"][layers+offset]], role="mtp", links=links)
        branch.update(id=f"mtp{offset}", count=1, included=options.get("include_mtp", False))
        branches.append(branch)
        metrics = inventory_metrics(rows)
        enabled = options.get("include_mtp", False)
        shared.rows.append(module_row(
            branch["id"], "mtp", [], metrics["params_total"] if enabled else 0,
            active=metrics["params_active"] if enabled else 0,
            matrix=metrics["matrix_params"] if enabled else 0, dtype="mixed",
            formula="block + e/h fusion + three norms + head collapse",
            source=SOURCE, notes="Shared main embed/head add zero unique parameters.",
        ))
        shared.rows[-1]["payload_bytes"] = sum(
            row["params"]*{"bf16": 2, "fp16": 2, "fp32": 4, "fp8": 1, "fp4": 0.5}[row["dtype"]]
            for row in rows) if enabled else 0
    elem = {"bf16": 2, "fp16": 2, "fp32": 4, "fp8": 1}[options.get("kv_dtype", "bf16")]
    cache = dict(kind="shared-KV-compressed", bytes_per_token=None,
                 dtype_assumed=options.get("kv_dtype", "bf16"),
                 formula="sum(B * (min(S,W)*c + floor(S/r)*c + CSA*floor(S/4)*ci) * bytes)",
                 notes="K=V counted once. Context-dependent payload; no constant bytes/token. Compressor state buffers and quantization scales are separate.")
    context = min(4096, cfg["max_position_embeddings"])
    ratios = cfg["compress_ratios"][:layers + (mtp_count if options.get("include_mtp") else 0)]
    cache["reference_context"] = context
    cache["reference_bytes"] = sum(
        (min(context, cfg["sliding_window"])*cfg["head_dim"]
         + (context//r*cfg["head_dim"] if r else 0)
         + (context//r*cfg["index_head_dim"] if r == 4 else 0))*elem for r in ratios)
    result = assemble(cfg, bb, templates, shared.rows, options=options, branches=branches, cache=cache)
    result["diagnostics"].append(dict(severity="info", code="router.integer_buffer",
        message=f"Hash tables excluded from floating parameter count: {cfg.get('num_hash_layers', 0)*v*cfg['num_experts_per_tok']} integer elements. Source checkpoint and runtime may use different integer dtypes."))
    return deepcopy(result)
