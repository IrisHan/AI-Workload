"""Repository-independent shape methods.

Inputs are resolved dimensions, never HuggingFace config dictionaries. A writer
provides linear/norm/tensor operations; Ledger and the grouped inventory adapter
preserve their own row IDs, bias representation, dtype and source attribution.
These methods account for weights, not runtime kernels or full FLOPs.
"""


def qkv_projections(writer, *, hidden, heads, kv_heads, head_dim, qkv_bias=False,
                    output_bias=False):
    """MHA/GQA/MQA share projections; only query/KV head multiplicity differs."""
    writer.linear("q_proj", hidden, heads * head_dim, bias=qkv_bias)
    writer.linear("k_proj", hidden, kv_heads * head_dim, bias=qkv_bias)
    writer.linear("v_proj", hidden, kv_heads * head_dim, bias=qkv_bias)
    writer.linear("o_proj", heads * head_dim, hidden, bias=output_bias)


def qk_norm(writer, *, query_dim, key_dim):
    """Per-head norms use c/c; fused norms use H*c/K*c."""
    writer.norm("q_norm", query_dim, category="attention")
    writer.norm("k_norm", key_dim, category="attention")


def mla_projections(writer, *, hidden, heads, q_rank, kv_rank, nope_dim, rope_dim, value_dim):
    """MLA Q/K/V projections. Output projection/gate ordering belongs to assembly."""
    if q_rank:
        writer.linear("q_a_proj", hidden, q_rank)
        writer.norm("q_a_norm", q_rank, category="attention")
        writer.linear("q_b_proj", q_rank, heads * (nope_dim + rope_dim))
    else:
        writer.linear("q_proj", hidden, heads * (nope_dim + rope_dim))
    writer.linear("kv_a_proj", hidden, kv_rank + rope_dim)
    writer.norm("kv_a_norm", kv_rank, category="attention")
    writer.linear("kv_b_proj", kv_rank, heads * (nope_dim + value_dim))


def dsa_projections(writer, *, hidden, q_rank, index_heads, index_dim):
    """DSA scores projected low-rank Q against normalized index keys."""
    if type(q_rank) is not int or q_rank <= 0:
        raise ValueError("DSA indexer requires a positive q_lora_rank")
    writer.linear("indexer.q_proj", q_rank, index_heads * index_dim)
    writer.linear("indexer.k_proj", hidden, index_dim)
    writer.norm("indexer.k_norm", index_dim, layernorm=True, category="attention")
    writer.linear("indexer.weights_proj", hidden, index_heads)


def msa_projections(writer, *, hidden, index_heads, index_dim):
    """MSA uses full hidden Q and Q/K RMS scales, without DSA weights_proj."""
    writer.linear("indexer.q_proj", hidden, index_heads * index_dim)
    writer.linear("indexer.k_proj", hidden, index_dim)
    writer.norm("indexer.q_norm", index_dim, category="attention")
    writer.norm("indexer.k_norm", index_dim, category="attention")


def gated_qkv_projections(writer, *, hidden, heads, kv_heads, head_dim):
    writer.linear("q_gate_proj", hidden, 2 * heads * head_dim)
    writer.linear("k_proj", hidden, kv_heads * head_dim)
    writer.linear("v_proj", hidden, kv_heads * head_dim)
    writer.linear("o_proj", heads * head_dim, hidden)
    qk_norm(writer, query_dim=head_dim, key_dim=head_dim)


def gated_delta_projections(writer, *, hidden, key_heads, key_dim, value_heads,
                            value_dim, kernel):
    key, value = key_heads * key_dim, value_heads * value_dim
    writer.linear("in_proj_qkv", hidden, 2 * key + value)
    writer.linear("in_proj_z", hidden, value)
    writer.linear("in_proj_b", hidden, value_heads)
    writer.linear("in_proj_a", hidden, value_heads)
    writer.tensor("depthwise_conv", [2 * key + value, 1, kernel], "attention")
    writer.tensor("dt_bias", [value_heads], "attention", dtype="fp32")
    writer.tensor("A_log", [value_heads], "attention", dtype="fp32")
    writer.norm("gated_norm", value_dim, category="attention")
    writer.linear("out_proj", value, hidden)


def kda_projections(writer, *, hidden, heads, head_dim, kernel, full_rank_gate=False):
    width = heads * head_dim
    writer.linear("q_proj", hidden, width)
    writer.linear("k_proj", hidden, width)
    writer.linear("v_proj", hidden, width)
    for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        writer.tensor(name, [width, 1, kernel], "attention")
    writer.tensor("A_log", [heads], "attention", dtype="fp32")
    writer.linear("f_a_proj", hidden, head_dim)
    writer.linear("f_b_proj", head_dim, width)
    writer.tensor("dt_bias", [width], "attention", dtype="fp32")
    writer.linear("b_proj", hidden, heads)
    if full_rank_gate:
        writer.linear("g_proj", hidden, width)
    else:
        writer.linear("g_a_proj", hidden, head_dim)
        writer.linear("g_b_proj", head_dim, width)
    writer.norm("o_norm", head_dim, category="attention")
    writer.linear("o_proj", width, hidden)


def feed_forward(writer, hidden, width, *, prefix="", gated=True, bias=False,
                 count=1, active_count=None, dtype=None):
    """SwiGLU: three projections; ungated GELU: two. Experts reuse the same method."""
    options = dict(bias=bias, count=count, active_count=active_count)
    if dtype is not None:
        options["dtype"] = dtype
    if gated:
        writer.linear(prefix + "gate_proj", hidden, width, "ffn", **options)
    writer.linear(prefix + "up_proj", hidden, width, "ffn", **options)
    writer.linear(prefix + "down_proj", width, hidden, "ffn", **options)


def kv_compressor(writer, *, prefix, hidden, dim, ratio):
    if ratio not in (4, 128):
        raise ValueError("unsupported KV compression ratio")
    overlap = 2 if ratio == 4 else 1
    writer.tensor(prefix + ".ape", [ratio, overlap * dim], "position", dtype="fp32")
    writer.linear(prefix + ".kv", hidden, overlap * dim, dtype="bf16")
    writer.linear(prefix + ".gate", hidden, overlap * dim, dtype="bf16")
    writer.norm(prefix + ".norm", dim, category="attention")


def compressed_attention(writer, *, hidden, heads, head_dim, q_rank, output_groups,
                         output_rank, ratio, index_heads, index_dim):
    """V4 shared K=V, grouped output and optional CSA/HCA compressors."""
    if ratio not in (0, 4, 128):
        raise ValueError("unsupported compression schedule")
    writer.linear("q_a_proj", hidden, q_rank)
    writer.norm("q_norm", q_rank, category="attention")
    writer.linear("q_b_proj", q_rank, heads * head_dim)
    writer.linear("kv_proj", hidden, head_dim)
    writer.norm("kv_norm", head_dim, category="attention")
    writer.linear("o_a_proj", heads * head_dim // output_groups,
                  output_groups * output_rank, dtype="bf16")
    writer.linear("o_b_proj", output_groups * output_rank, hidden)
    writer.tensor("attn_sink", [heads], "attention", dtype="fp32")
    if ratio:
        kv_compressor(writer, prefix="compressor", hidden=hidden, dim=head_dim, ratio=ratio)
    if ratio == 4:
        writer.linear("indexer.q_proj", q_rank, index_heads * index_dim)
        writer.linear("indexer.weights_proj", hidden, index_heads, dtype="bf16")
        kv_compressor(writer, prefix="indexer.compressor", hidden=hidden,
                      dim=index_dim, ratio=ratio)
