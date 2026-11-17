"""Parameter / byte / FLOP primitives. No model-specific logic."""

from __future__ import annotations

BYTES_PER_PARAM = {
    "fp32": 4.0,
    "float32": 4.0,
    "bf16": 2.0,
    "bfloat16": 2.0,
    "fp16": 2.0,
    "float16": 2.0,
    "fp8": 1.0,
    "e4m3": 1.0,
    "fp4": 0.5,
    "e2m1": 0.5,
}


def linear_params(in_f: int, out_f: int, *, bias: bool = False) -> int:
    return in_f * out_f + (out_f if bias else 0)


def embedding_params(vocab: int, dim: int) -> int:
    return vocab * dim


def rmsnorm_params(dim: int) -> int:
    return dim


def swiglu_params(d: int, d_ff: int, *, bias: bool = False) -> int:
    return (
        linear_params(d, d_ff, bias=bias)
        + linear_params(d, d_ff, bias=bias)
        + linear_params(d_ff, d, bias=bias)
    )


def dense_ffn_params(d: int, d_ff: int, *, activation: str, bias: bool = False) -> int:
    gated = activation.lower() in {"silu", "swiglu", "gelu_pytorch_tanh", "swish"}
    if gated:
        return swiglu_params(d, d_ff, bias=bias)
    return linear_params(d, d_ff, bias=bias) + linear_params(d_ff, d, bias=bias)


def gptq_linear_bytes(
    in_f: int,
    out_f: int,
    *,
    bits: int = 4,
    group_size: int = 128,
    has_g_idx: bool = True,
    dummy_bias: bool = True,
) -> int:
    """AutoGPTQ QuantLinear on-disk bytes for one (in, out) matrix. No model-specific logic."""
    gs = max(int(group_size) or 128, 1)
    qweight = in_f * out_f * bits / 8
    qzeros = in_f * out_f * bits / (8 * gs)
    scales = 2.0 * in_f * out_f / gs
    g_idx = 4 * in_f if has_g_idx else 0
    bias = 2 * out_f if dummy_bias else 0
    return int(qweight + qzeros + scales + g_idx + bias)


def bytes_for(params: int, dtype: str) -> float:
    return params * BYTES_PER_PARAM.get(dtype.lower(), 2.0)


def gib(nbytes: float) -> float:
    return nbytes / (1024**3)
