from app.lib.spec_studio.inspect import inspect_config


def test_qwen_exact_bias_inventory(qwen_cfg):
    arch = inspect_config(qwen_cfg).arch
    assert arch.totals.total == 1543714304
    assert arch.layers[0].attn_params == 5507072


def test_decode_gemm_excludes_non_matrix_parameters(qwen_cfg):
    arch = inspect_config(qwen_cfg).arch
    expected = 2 * (28 * (5505024 + 41287680) + 233373696)
    assert arch.serving.flop_gemm_decode == expected


def test_fp8_weights_do_not_imply_fp8_kv(v3_cfg):
    arch = inspect_config(v3_cfg).arch
    assert arch.dtype_weight == "fp8"
    assert arch.serving.kv_layout.bytes_per_token == 61 * (512 + 64) * 2
    assert arch.serving.kv_layout.dtype_assumed == "bf16"
