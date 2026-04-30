# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SplitD SM90 forward & backward kernel correctness tests.
#
# Tests SplitD kernels (head_dim == 512) against fp32 PyTorch reference,
# PyTorch SDPA, and flex_attention (eager mode).

# ---------------------------------------------------------------------------
# Pre-import stub
# ---------------------------------------------------------------------------
import os
import math
import gc
import random
from functools import wraps

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from src.testing import maybe_fake_tensor_mode, is_fake_mode
from src.interface import (
    split_flash_attn_func,
    split_flash_attn_varlen_func,
    _flash_attn_fwd_sm90,
    _flash_attn_bwd_sm90,
    _bwd_preprocess,
)


# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="SM90-only test requires Hopper GPU",
)

USE_FAKE_TENSOR = int(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", 0)) == 1

torch._dynamo.config.cache_size_limit = 1000


@pytest.fixture(autouse=True)
def _cuda_cleanup():
    """Synchronize and clean up CUDA state between tests.

    SplitD kernels use large SMEM (228KB) and TMA descriptors.  Without
    explicit synchronization, asynchronous CUDA errors from one test can
    poison subsequent tests.
    """
    if torch.cuda.is_available() and not is_fake_mode():
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass
    yield
    if torch.cuda.is_available() and not is_fake_mode():
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass
        gc.collect()
        torch.cuda.empty_cache()


def retry_on_oom(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except torch.OutOfMemoryError as e:
            if "out of memory" in str(e).lower():
                if hasattr(_flash_attn_fwd_sm90, "compile_cache"):
                    _flash_attn_fwd_sm90.compile_cache.clear()
                for attr in ["compile_cache_dkdv", "compile_cache_dq"]:
                    if hasattr(_flash_attn_bwd_sm90, attr):
                        getattr(_flash_attn_bwd_sm90, attr).clear()
                if hasattr(_bwd_preprocess, "compile_cache"):
                    _bwd_preprocess.compile_cache.clear()
                torch._dynamo.reset()
                gc.collect()
                torch.cuda.empty_cache()
                return func(*args, **kwargs)
            raise

    return wrapper


# ---------------------------------------------------------------------------
# Reference helpers
# ---------------------------------------------------------------------------


def sdpa_ref(q, k, v, causal=False, softcap=None, softmax_scale=None):
    """SDPA reference with softcap support.  Input/output: (B, S, H, D)."""
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_kv, _ = k.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    q_s = q.transpose(1, 2)
    k_s = k.transpose(1, 2)
    v_s = v.transpose(1, 2)

    if nheads != nheads_kv:
        rep = nheads // nheads_kv
        k_s = k_s.repeat_interleave(rep, dim=1)
        v_s = v_s.repeat_interleave(rep, dim=1)

    need_manual = (softcap is not None and softcap > 0.0) or causal
    if need_manual:
        scores = torch.matmul(q_s * softmax_scale, k_s.transpose(-2, -1))
        if softcap is not None and softcap > 0.0:
            scores = softcap * torch.tanh(scores / softcap)
        if causal:
            mask = torch.triu(
                torch.ones(seqlen_q, seqlen_k, device=q.device, dtype=torch.bool),
                diagonal=seqlen_k - seqlen_q + 1,
            )
            scores.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v_s)
    else:
        out = F.scaled_dot_product_attention(q_s, k_s, v_s, scale=softmax_scale)

    return out.transpose(1, 2)


def attention_ref(q, k, v, causal=False, softmax_scale=None):
    """Manual fp32 attention with autograd support.  Input/output: (B, S, H, D)."""
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_kv, d_v = v.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    qt = q.transpose(1, 2)
    kt = k.transpose(1, 2)
    vt = v.transpose(1, 2)

    if nheads != nheads_kv:
        rep = nheads // nheads_kv
        kt = kt.repeat_interleave(rep, dim=1)
        vt = vt.repeat_interleave(rep, dim=1)

    scores = torch.matmul(qt * softmax_scale, kt.transpose(-2, -1))
    if causal:
        mask = torch.triu(
            torch.ones(seqlen_q, seqlen_k, device=q.device, dtype=torch.bool),
            diagonal=seqlen_k - seqlen_q + 1,
        )
        scores.masked_fill_(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, vt).transpose(1, 2)


def attention_ref_pt(q, k, v, causal=False, softmax_scale=None):
    """Same-dtype attention for tolerance calibration (softmax in fp32)."""
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_kv, _ = v.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    qt = q.transpose(1, 2)
    kt = k.transpose(1, 2)
    vt = v.transpose(1, 2)

    if nheads != nheads_kv:
        rep = nheads // nheads_kv
        kt = kt.repeat_interleave(rep, dim=1)
        vt = vt.repeat_interleave(rep, dim=1)

    scores = torch.matmul(qt * softmax_scale, kt.transpose(-2, -1))
    if causal:
        mask = torch.triu(
            torch.ones(seqlen_q, seqlen_k, device=q.device, dtype=torch.bool),
            diagonal=seqlen_k - seqlen_q + 1,
        )
        scores.masked_fill_(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(attn, vt).transpose(1, 2)


def _flex_fwd(q, k, v, causal, softmax_scale=None):
    """Flex attention forward (eager mode).  Input/output: (B, S, H, D)."""
    batch, seqlen_q = q.shape[:2]
    seqlen_k = k.shape[1]
    enable_gqa = q.shape[2] != k.shape[2]

    q_fl = q.detach().transpose(1, 2).contiguous()
    k_fl = k.detach().transpose(1, 2).contiguous()
    v_fl = v.detach().transpose(1, 2).contiguous()

    kwargs = dict(enable_gqa=enable_gqa)
    if softmax_scale is not None:
        kwargs["scale"] = softmax_scale
    if causal:
        kwargs["block_mask"] = create_block_mask(
            lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
            B=batch,
            H=None,
            Q_LEN=seqlen_q,
            KV_LEN=seqlen_k,
            device=q_fl.device,
        )

    with torch.no_grad():
        out = flex_attention(q_fl, k_fl, v_fl, **kwargs)
    return out.transpose(1, 2)


def _flex_bwd(q, k, v, g, causal, nheads, nheads_kv, softmax_scale=None):
    """Flex attention backward (eager mode).  Input/output: (B, S, H, D)."""
    batch, seqlen_q = q.shape[:2]
    seqlen_k = k.shape[1]
    enable_gqa = nheads != nheads_kv

    q_fl = q.detach().transpose(1, 2).contiguous().requires_grad_(True)
    k_fl = k.detach().transpose(1, 2).contiguous().requires_grad_(True)
    v_fl = v.detach().transpose(1, 2).contiguous().requires_grad_(True)
    g_fl = g.detach().transpose(1, 2).contiguous()

    kwargs = dict(enable_gqa=enable_gqa)
    if softmax_scale is not None:
        kwargs["scale"] = softmax_scale
    if causal:
        kwargs["block_mask"] = create_block_mask(
            lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
            B=batch,
            H=None,
            Q_LEN=seqlen_q,
            KV_LEN=seqlen_k,
            device=q_fl.device,
        )

    out = flex_attention(q_fl, k_fl, v_fl, **kwargs)
    dq, dk, dv = torch.autograd.grad(out, (q_fl, k_fl, v_fl), g_fl)
    return dq.transpose(1, 2), dk.transpose(1, 2), dv.transpose(1, 2)


def _make_nonuniform_seqlens(seqlen, batch_size, tile_m=128):
    """Generate non-uniform sequence lengths, spread from 0.5x to 1.5x."""
    if batch_size <= 1:
        s_short = max(tile_m, round(seqlen * 0.25 / tile_m) * tile_m)
        s_long = max(tile_m, round(seqlen * 0.75 / tile_m) * tile_m)
        return [s_short, s_long]
    return [max(tile_m, round(seqlen * (0.5 + i / (batch_size - 1)) / tile_m) * tile_m) for i in range(batch_size)]


# ===========================================================================
# Forward correctness tests
# ===========================================================================


# ---------------------------------------------------------------------------
# test_splitd_fwd_correctness — Batched FWD against SDPA reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mha_type", ["mha", "gqa", "mqa"])
@pytest.mark.parametrize("softcap", [0.0, 15.0])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [512])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [(64, 128), (128, 128), (256, 256), (113, 203), (512, 256), (1024, 1024)],
)
@retry_on_oom
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_splitd_fwd_correctness(seqlen_q, seqlen_k, d, causal, softcap, mha_type, dtype):
    """Test SplitD forward kernel output against PyTorch SDPA reference."""
    if causal and seqlen_q > seqlen_k:
        pytest.skip("causal with seqlen_q > seqlen_k produces NaN for some rows")

    device = "cuda"
    torch.manual_seed(42)
    torch.cuda.empty_cache()

    batch_size = 2 if seqlen_k >= 1024 else 4
    nheads = 6
    nheads_kv = nheads if mha_type == "mha" else (3 if mha_type == "gqa" else 1)
    softmax_scale = 1.0 / math.sqrt(d)

    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype)
    if softcap > 0.0:
        q = q * softcap / 4

    with torch.no_grad():
        out_ref = sdpa_ref(
            q.float(), k.float(), v.float(), causal=causal, softcap=softcap, softmax_scale=softmax_scale
        ).to(dtype)
        out_sdpa = sdpa_ref(q, k, v, causal=causal, softcap=softcap, softmax_scale=softmax_scale)

    pack_gqa_vals = [False, True, None] if mha_type != "mha" else [False]
    qhead_per_kvhead = nheads // nheads_kv
    for pack_gqa in pack_gqa_vals:
        if pack_gqa is not False and qhead_per_kvhead > 1 and 64 % qhead_per_kvhead != 0:
            continue
        out_fa = split_flash_attn_func(
            q.detach().clone(),
            k.detach().clone(),
            v.detach().clone(),
            causal=causal,
            softcap=softcap,
            pack_gqa=pack_gqa,
        )
        if is_fake_mode():
            continue

        fa_vs_ref_max = (out_fa - out_ref).abs().max().item()
        sdpa_vs_ref_max = (out_sdpa - out_ref).abs().max().item()
        fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()

        print(
            f"\n[d={d} seq=({seqlen_q},{seqlen_k}) {mha_type} causal={causal} "
            f"softcap={softcap} {dtype} pack_gqa={pack_gqa}]"
        )
        print(f"  SDPA vs ref: max={sdpa_vs_ref_max:.6f}")
        print(f"  SplitD vs ref: max={fa_vs_ref_max:.6f}  atol={fwd_atol:.6f}")

        rtol = 2 if softcap == 0.0 else 3
        assert fa_vs_ref_max <= rtol * sdpa_vs_ref_max + fwd_atol, (
            f"SplitD output max diff {fa_vs_ref_max:.6f} exceeds tolerance {rtol * sdpa_vs_ref_max + fwd_atol:.6f}"
        )


# ---------------------------------------------------------------------------
# Interface contract and validation tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("varlen", [False, True], ids=["batched", "varlen"])
@pytest.mark.parametrize("return_lse", [False, True])
@retry_on_oom
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_splitd_public_api_return_contract(varlen, return_lse):
    device = "cuda"
    batch_size = 2
    seqlen = 64
    nheads = 4
    dtype = torch.bfloat16

    if varlen:
        total_tokens = batch_size * seqlen
        q = torch.randn(total_tokens, nheads, 512, device=device, dtype=dtype)
        k = torch.randn(total_tokens, nheads, 512, device=device, dtype=dtype)
        v = torch.randn(total_tokens, nheads, 512, device=device, dtype=dtype)
        cu_seqlens = torch.arange(0, total_tokens + 1, step=seqlen, device=device, dtype=torch.int32)
        result = split_flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seqlen,
            max_seqlen_k=seqlen,
            return_lse=return_lse,
        )
        expected_out_shape = (total_tokens, nheads, 512)
        expected_lse_shape = (nheads, total_tokens)
    else:
        q = torch.randn(batch_size, seqlen, nheads, 512, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, nheads, 512, device=device, dtype=dtype)
        v = torch.randn(batch_size, seqlen, nheads, 512, device=device, dtype=dtype)
        result = split_flash_attn_func(q, k, v, return_lse=return_lse)
        expected_out_shape = (batch_size, seqlen, nheads, 512)
        expected_lse_shape = (batch_size, nheads, seqlen)

    if return_lse:
        out, lse = result
        assert out.shape == expected_out_shape
        assert lse.shape == expected_lse_shape
        assert lse.dtype == torch.float32
    else:
        assert isinstance(result, torch.Tensor)
        assert result.shape == expected_out_shape


@pytest.mark.parametrize("head_dim,head_dim_v", [(384, 384), (512, 384)])
@retry_on_oom
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_splitd_rejects_non_512_head_dims(head_dim, head_dim_v):
    device = "cuda"
    dtype = torch.bfloat16
    q = torch.empty(1, 64, 4, head_dim, device=device, dtype=dtype)
    k = torch.empty(1, 64, 4, head_dim, device=device, dtype=dtype)
    v = torch.empty(1, 64, 4, head_dim_v, device=device, dtype=dtype)

    with pytest.raises(ValueError, match="head_dim == 512"):
        split_flash_attn_func(q, k, v)


def test_splitd_rejects_cpu_tensors():
    q = torch.empty(1, 64, 4, 512, dtype=torch.bfloat16)
    k = torch.empty(1, 64, 4, 512, dtype=torch.bfloat16)
    v = torch.empty(1, 64, 4, 512, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="CUDA"):
        split_flash_attn_func(q, k, v)


@retry_on_oom
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_splitd_rejects_bad_cu_seqlens_dtype():
    device = "cuda"
    dtype = torch.bfloat16
    total_tokens = 128
    q = torch.empty(total_tokens, 4, 512, device=device, dtype=dtype)
    k = torch.empty(total_tokens, 4, 512, device=device, dtype=dtype)
    v = torch.empty(total_tokens, 4, 512, device=device, dtype=dtype)
    cu_seqlens = torch.tensor([0, 64, 128], device=device, dtype=torch.int64)

    with pytest.raises(TypeError, match="cu_seqlens_q"):
        split_flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=64,
            max_seqlen_k=64,
        )


@retry_on_oom
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_splitd_rejects_fp16_training_inputs():
    device = "cuda"
    total_tokens = 128
    q = torch.empty(total_tokens, 4, 512, device=device, dtype=torch.float16, requires_grad=True)
    k = torch.empty(total_tokens, 4, 512, device=device, dtype=torch.float16, requires_grad=True)
    v = torch.empty(total_tokens, 4, 512, device=device, dtype=torch.float16, requires_grad=True)
    cu_seqlens = torch.tensor([0, 64, 128], device=device, dtype=torch.int32)

    with pytest.raises(NotImplementedError, match="bfloat16 only"):
        split_flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=64,
            max_seqlen_k=64,
        )


@retry_on_oom
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_splitd_rejects_missing_varlen_max_seqlen():
    device = "cuda"
    dtype = torch.bfloat16
    total_tokens = 128
    q = torch.empty(total_tokens, 4, 512, device=device, dtype=dtype)
    k = torch.empty(total_tokens, 4, 512, device=device, dtype=dtype)
    v = torch.empty(total_tokens, 4, 512, device=device, dtype=dtype)
    cu_seqlens = torch.tensor([0, 64, 128], device=device, dtype=torch.int32)

    with pytest.raises(ValueError, match="max_seqlen_q"):
        split_flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_k=64,
        )


@pytest.mark.parametrize(
    "cu_values,match",
    [
        ([1, 64, 128], r"cu_seqlens_q\[0\]"),
        ([0, 64, 127], r"cu_seqlens_q\[-1\]"),
        ([0, 128, 64, 128], "monotonically non-decreasing"),
    ],
    ids=["bad_start", "bad_end", "nonmonotonic"],
)
@retry_on_oom
def test_splitd_rejects_bad_cu_seqlens_values(cu_values, match):
    device = "cuda"
    dtype = torch.bfloat16
    total_tokens = 128
    q = torch.empty(total_tokens, 4, 512, device=device, dtype=dtype)
    k = torch.empty(total_tokens, 4, 512, device=device, dtype=dtype)
    v = torch.empty(total_tokens, 4, 512, device=device, dtype=dtype)
    cu_seqlens = torch.tensor(cu_values, device=device, dtype=torch.int32)

    with pytest.raises(ValueError, match=match):
        split_flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=128,
            max_seqlen_k=128,
        )


@retry_on_oom
def test_splitd_rejects_too_small_max_seqlen():
    device = "cuda"
    dtype = torch.bfloat16
    total_tokens = 128
    q = torch.empty(total_tokens, 4, 512, device=device, dtype=dtype)
    k = torch.empty(total_tokens, 4, 512, device=device, dtype=dtype)
    v = torch.empty(total_tokens, 4, 512, device=device, dtype=dtype)
    cu_seqlens = torch.tensor([0, 32, 128], device=device, dtype=torch.int32)

    with pytest.raises(ValueError, match="max_seqlen_q"):
        split_flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=64,
            max_seqlen_k=128,
        )


# ---------------------------------------------------------------------------
# test_splitd_lse — LSE correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [512])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [(128, 128), (256, 256), (113, 203)],
)
@retry_on_oom
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_splitd_lse(seqlen_q, seqlen_k, d, causal, dtype):
    """Test LSE output of SplitD kernel against manual computation."""
    device = "cuda"
    torch.manual_seed(0)
    torch.cuda.empty_cache()

    batch_size = 4
    nheads = 4
    softmax_scale = 1.0 / math.sqrt(d)

    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype)

    out_fa, lse_fa = split_flash_attn_func(q, k, v, causal=causal, return_lse=True)
    if is_fake_mode():
        return

    # Reference LSE in fp32
    q_f32 = q.float().transpose(1, 2)
    k_f32 = k.float().transpose(1, 2)
    scores = torch.matmul(q_f32 * softmax_scale, k_f32.transpose(-2, -1))
    if causal:
        mask = torch.triu(
            torch.ones(seqlen_q, seqlen_k, device=device, dtype=torch.bool),
            diagonal=seqlen_k - seqlen_q + 1,
        )
        scores.masked_fill_(mask, float("-inf"))
    lse_ref = torch.logsumexp(scores, dim=-1)

    lse_diff = (lse_fa - lse_ref).abs().max().item()
    print(f"\n[LSE d={d} seq=({seqlen_q},{seqlen_k}) causal={causal}]  diff={lse_diff:.6f}")
    assert lse_diff < 1e-2, f"LSE max diff {lse_diff} too large"

    # Forward output check
    with torch.no_grad():
        out_ref = sdpa_ref(q.float(), k.float(), v.float(), causal=causal, softmax_scale=softmax_scale).to(dtype)
        out_sdpa = sdpa_ref(q, k, v, causal=causal, softmax_scale=softmax_scale)
    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    assert (out_fa - out_ref).abs().max().item() <= 2 * (out_sdpa - out_ref).abs().max().item() + fwd_atol


# ---------------------------------------------------------------------------
# test_splitd_fwd_varlen — Varlen FWD correctness (GQA q=32, kv=4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("uniform", [True, False], ids=["uniform", "nonuniform"])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("d", [512])
@pytest.mark.parametrize(
    "seqlen_min,seqlen_max,batch_size",
    [
        (256, 512, 4),
        (512, 1024, 4),
        (1024, 2048, 2),
        (2048, 4096, 1),
        (4096, 8192, 1),
        (8192, 16384, 1),
        (16384, 32768, 1),
    ],
)
@retry_on_oom
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_splitd_fwd_varlen(seqlen_min, seqlen_max, batch_size, d, causal, dtype, uniform):
    """Test SplitD varlen forward (GQA q=32, kv=4) against per-sequence reference.

    Short seqlens (≤4096): fp32 manual reference with tight tolerance.
    Long seqlens (>4096): F.sdpa (bf16) reference with looser tolerance.
    """
    device = "cuda"
    torch.manual_seed(42)
    torch.cuda.empty_cache()

    nheads = 32
    nheads_kv = 4
    softmax_scale = 1.0 / math.sqrt(d)

    if uniform:
        random.seed(42)
        seqlens_list = [random.randint(seqlen_min, seqlen_max) for _ in range(batch_size)]
    else:
        avg_seqlen = (seqlen_min + seqlen_max) // 2
        seqlens_list = _make_nonuniform_seqlens(avg_seqlen, batch_size)
        batch_size = len(seqlens_list)

    total_tokens = sum(seqlens_list)
    max_seqlen = max(seqlens_list)
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for i, s in enumerate(seqlens_list):
        cu_seqlens[i + 1] = cu_seqlens[i] + s

    q = torch.randn(total_tokens, nheads, d, device=device, dtype=dtype)
    k = torch.randn(total_tokens, nheads_kv, d, device=device, dtype=dtype)
    v = torch.randn(total_tokens, nheads_kv, d, device=device, dtype=dtype)

    out_fa = split_flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=causal,
        softmax_scale=softmax_scale,
    )
    if is_fake_mode():
        return

    assert not out_fa.isnan().any(), "SplitD output contains NaN"
    assert not out_fa.isinf().any(), "SplitD output contains Inf"

    # Per-sequence reference
    use_fp32_ref = max_seqlen <= 4096
    out_ref = torch.zeros_like(out_fa)
    with torch.no_grad():
        for b in range(batch_size):
            start = cu_seqlens[b].item()
            end = cu_seqlens[b + 1].item()
            q_b = q[start:end].unsqueeze(0)
            k_b = k[start:end].unsqueeze(0)
            v_b = v[start:end].unsqueeze(0)
            if use_fp32_ref:
                ref_b = sdpa_ref(q_b.float(), k_b.float(), v_b.float(), causal=causal, softmax_scale=softmax_scale)
                out_ref[start:end] = ref_b.squeeze(0).to(dtype)
            else:
                # F.sdpa (bf16) — fp32 manual would OOM at large seqlens
                q_s = q_b.transpose(1, 2)
                k_s = k_b.transpose(1, 2)
                v_s = v_b.transpose(1, 2)
                rep = nheads // nheads_kv
                k_s = k_s.repeat_interleave(rep, dim=1)
                v_s = v_s.repeat_interleave(rep, dim=1)
                ref_b = F.scaled_dot_product_attention(
                    q_s,
                    k_s,
                    v_s,
                    scale=softmax_scale,
                    is_causal=causal,
                )
                out_ref[start:end] = ref_b.transpose(1, 2).squeeze(0)

    fa_vs_ref_max = (out_fa - out_ref).abs().max().item()
    fa_vs_ref_mean = (out_fa - out_ref).abs().mean().item()

    mode = "uniform" if uniform else "nonuniform"
    print(
        f"\n[VARLEN-FWD-{mode} d={d} seqlen=[{seqlen_min},{seqlen_max}] "
        f"B={batch_size} H={nheads} Hkv={nheads_kv} causal={causal}]"
    )
    print(f"  seqlens: {seqlens_list}")
    print(f"  SplitD vs ref: max={fa_vs_ref_max:.6f}  mean={fa_vs_ref_mean:.6f}")

    if use_fp32_ref:
        out_sdpa_dtype = torch.zeros_like(out_fa)
        with torch.no_grad():
            for b in range(batch_size):
                start = cu_seqlens[b].item()
                end = cu_seqlens[b + 1].item()
                sdpa_b = sdpa_ref(
                    q[start:end].unsqueeze(0),
                    k[start:end].unsqueeze(0),
                    v[start:end].unsqueeze(0),
                    causal=causal,
                    softmax_scale=softmax_scale,
                )
                out_sdpa_dtype[start:end] = sdpa_b.squeeze(0)
        sdpa_vs_ref_max = (out_sdpa_dtype - out_ref).abs().max().item()
        fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
        assert fa_vs_ref_max <= 2 * sdpa_vs_ref_max + fwd_atol, (
            f"SplitD varlen max diff {fa_vs_ref_max:.6f} exceeds tolerance {2 * sdpa_vs_ref_max + fwd_atol:.6f}"
        )
    else:
        assert fa_vs_ref_max < 0.05, f"SplitD output max diff {fa_vs_ref_max:.6f} against F.sdpa exceeds 0.05"


# ===========================================================================
# Backward correctness tests
# ===========================================================================


# ---------------------------------------------------------------------------
# test_splitd_bwd_correctness — Batched BWD against fp32 reference (MHA)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [512])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [(64, 128), (128, 128), (256, 256), (113, 203), (512, 256), (1024, 1024)],
)
@retry_on_oom
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_splitd_bwd_correctness(seqlen_q, seqlen_k, d, causal, dtype):
    """Test SplitD backward (dQ, dK, dV) against fp32 reference + flex.  MHA only."""
    if causal and seqlen_q > seqlen_k:
        pytest.skip("causal with seqlen_q > seqlen_k not well-defined for backward")

    device = "cuda"
    torch.manual_seed(42)
    torch.cuda.empty_cache()

    batch_size = 2 if seqlen_k >= 1024 else 4
    nheads = 4
    softmax_scale = 1.0 / math.sqrt(d)

    # fp32 reference
    q_ref = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=torch.float32).requires_grad_(True)
    k_ref = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=torch.float32).requires_grad_(True)
    v_ref = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=torch.float32).requires_grad_(True)

    out_ref = attention_ref(q_ref, k_ref, v_ref, causal=causal, softmax_scale=softmax_scale)
    g = torch.randn_like(out_ref)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), g)

    # Same-dtype baseline for tolerance
    q_pt = q_ref.detach().to(dtype).to(torch.float32).requires_grad_(True)
    k_pt = k_ref.detach().to(dtype).to(torch.float32).requires_grad_(True)
    v_pt = v_ref.detach().to(dtype).to(torch.float32).requires_grad_(True)
    out_pt = attention_ref(q_pt, k_pt, v_pt, causal=causal, softmax_scale=softmax_scale)
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_pt, k_pt, v_pt), g)

    # Flash SplitD
    q_fa = q_ref.detach().to(dtype).requires_grad_(True)
    k_fa = k_ref.detach().to(dtype).requires_grad_(True)
    v_fa = v_ref.detach().to(dtype).requires_grad_(True)
    out_fa, lse = split_flash_attn_func(q_fa, k_fa, v_fa, causal=causal, softmax_scale=softmax_scale, return_lse=True)
    if is_fake_mode():
        return
    dq_fa, dk_fa, dv_fa = torch.autograd.grad(out_fa, (q_fa, k_fa, v_fa), g.to(dtype))

    # Flex reference
    dq_fl, dk_fl, dv_fl = _flex_bwd(
        q_ref.detach().to(dtype),
        k_ref.detach().to(dtype),
        v_ref.detach().to(dtype),
        g.to(dtype),
        causal,
        nheads,
        nheads,
        softmax_scale,
    )

    # Metrics
    dq_err = (dq_fa.float() - dq_ref).abs().max().item()
    dk_err = (dk_fa.float() - dk_ref).abs().max().item()
    dv_err = (dv_fa.float() - dv_ref).abs().max().item()
    dq_pt_err = (dq_pt - dq_ref).abs().max().item()
    dk_pt_err = (dk_pt - dk_ref).abs().max().item()
    dv_pt_err = (dv_pt - dv_ref).abs().max().item()
    dq_atol = 2 * (dq_ref + 0.3 - 0.3 - dq_ref).abs().max().item()
    dk_atol = 2 * (dk_ref + 0.3 - 0.3 - dk_ref).abs().max().item()
    dv_atol = 2 * (dv_ref + 0.3 - 0.3 - dv_ref).abs().max().item()

    print(f"\n[BWD d={d} seq=({seqlen_q},{seqlen_k}) causal={causal} {dtype}]")
    print(f"  dQ: flash={dq_err:.6f}  pt={dq_pt_err:.6f}  flex={dq_fl.float().sub(dq_ref).abs().max().item():.6f}")
    print(f"  dK: flash={dk_err:.6f}  pt={dk_pt_err:.6f}  flex={dk_fl.float().sub(dk_ref).abs().max().item():.6f}")
    print(f"  dV: flash={dv_err:.6f}  pt={dv_pt_err:.6f}  flex={dv_fl.float().sub(dv_ref).abs().max().item():.6f}")

    rtol = 4
    assert dq_err <= rtol * dq_pt_err + dq_atol, f"dQ max diff {dq_err:.6f} exceeds tolerance"
    assert dk_err <= rtol * dk_pt_err + dk_atol, f"dK max diff {dk_err:.6f} exceeds tolerance"
    assert dv_err <= rtol * dv_pt_err + dv_atol, f"dV max diff {dv_err:.6f} exceeds tolerance"


# ---------------------------------------------------------------------------
# test_splitd_bwd_fwd_consistency — Forward output sanity check (BWD prerequisite)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [512])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [(128, 128), (256, 256), (113, 203)],
)
@retry_on_oom
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_splitd_bwd_fwd_consistency(seqlen_q, seqlen_k, d, causal, dtype):
    """Verify forward output and LSE correctness (prerequisite for backward)."""
    if causal and seqlen_q > seqlen_k:
        pytest.skip("causal with seqlen_q > seqlen_k not supported")

    device = "cuda"
    torch.manual_seed(0)
    torch.cuda.empty_cache()

    batch_size = 4
    nheads = 4
    softmax_scale = 1.0 / math.sqrt(d)

    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype)

    out_fa, lse_fa = split_flash_attn_func(q, k, v, causal=causal, softmax_scale=softmax_scale, return_lse=True)
    if is_fake_mode():
        return

    with torch.no_grad():
        out_ref = attention_ref(q.float(), k.float(), v.float(), causal=causal, softmax_scale=softmax_scale).to(dtype)
        out_sdpa = attention_ref_pt(q, k, v, causal=causal, softmax_scale=softmax_scale)

    out_err = (out_fa - out_ref).abs().max().item()
    sdpa_err = (out_sdpa - out_ref).abs().max().item()
    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()

    # LSE check
    q_f32 = q.float().transpose(1, 2)
    k_f32 = k.float().transpose(1, 2)
    scores = torch.matmul(q_f32 * softmax_scale, k_f32.transpose(-2, -1))
    if causal:
        mask = torch.triu(
            torch.ones(seqlen_q, seqlen_k, device=device, dtype=torch.bool), diagonal=seqlen_k - seqlen_q + 1
        )
        scores.masked_fill_(mask, float("-inf"))
    lse_err = (lse_fa - torch.logsumexp(scores, dim=-1)).abs().max().item()

    # Flex check
    out_flex = _flex_fwd(q, k, v, causal, softmax_scale=softmax_scale)
    flex_err = (out_flex.to(dtype) - out_ref).abs().max().item()

    print(f"\n[FWD-CONSISTENCY d={d} seq=({seqlen_q},{seqlen_k}) causal={causal}]")
    print(f"  out={out_err:.6f}  lse={lse_err:.6f}  flex={flex_err:.6f}  atol={fwd_atol:.6f}")

    assert out_err <= 2 * sdpa_err + fwd_atol
    assert lse_err < 1e-2


# ---------------------------------------------------------------------------
# test_splitd_bwd_gqa — Batched BWD with GQA (q=32, kv=4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [512])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [(128, 128), (256, 256), (113, 203)],
)
@retry_on_oom
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_splitd_bwd_gqa(seqlen_q, seqlen_k, d, causal, dtype):
    """Test SplitD backward with GQA (q_heads=32, kv_heads=4), non-varlen."""
    if causal and seqlen_q > seqlen_k:
        pytest.skip("causal with seqlen_q > seqlen_k not supported")

    device = "cuda"
    torch.manual_seed(42)
    torch.cuda.empty_cache()

    batch_size = 2
    nheads = 32
    nheads_kv = 4
    softmax_scale = 1.0 / math.sqrt(d)

    q_ref = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=torch.float32).requires_grad_(True)
    k_ref = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=torch.float32).requires_grad_(True)
    v_ref = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=torch.float32).requires_grad_(True)

    out_ref = attention_ref(q_ref, k_ref, v_ref, causal=causal, softmax_scale=softmax_scale)
    g = torch.randn_like(out_ref)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), g)

    q_fa = q_ref.detach().to(dtype).requires_grad_(True)
    k_fa = k_ref.detach().to(dtype).requires_grad_(True)
    v_fa = v_ref.detach().to(dtype).requires_grad_(True)
    out_fa, _ = split_flash_attn_func(q_fa, k_fa, v_fa, causal=causal, softmax_scale=softmax_scale, return_lse=True)
    if is_fake_mode():
        return
    dq_fa, dk_fa, dv_fa = torch.autograd.grad(out_fa, (q_fa, k_fa, v_fa), g.to(dtype))

    dq_fl, dk_fl, dv_fl = _flex_bwd(
        q_ref.detach().to(dtype),
        k_ref.detach().to(dtype),
        v_ref.detach().to(dtype),
        g.to(dtype),
        causal,
        nheads,
        nheads_kv,
        softmax_scale,
    )

    dq_err = (dq_fa.float() - dq_ref).abs().max().item()
    dk_err = (dk_fa.float() - dk_ref).abs().max().item()
    dv_err = (dv_fa.float() - dv_ref).abs().max().item()

    print(f"\n[BWD-GQA d={d} seq=({seqlen_q},{seqlen_k}) causal={causal} {dtype}]")
    print(f"  dQ: flash={dq_err:.6f}  flex={dq_fl.float().sub(dq_ref).abs().max().item():.6f}")
    print(f"  dK: flash={dk_err:.6f}  flex={dk_fl.float().sub(dk_ref).abs().max().item():.6f}")
    print(f"  dV: flash={dv_err:.6f}  flex={dv_fl.float().sub(dv_ref).abs().max().item():.6f}")

    gqa_atol = 0.06
    assert dq_err < gqa_atol, f"dQ max diff {dq_err:.6f} exceeds {gqa_atol}"
    assert dk_err < gqa_atol, f"dK max diff {dk_err:.6f} exceeds {gqa_atol}"
    assert dv_err < gqa_atol, f"dV max diff {dv_err:.6f} exceeds {gqa_atol}"


# ---------------------------------------------------------------------------
# test_splitd_bwd_varlen — Varlen BWD correctness (GQA + MHA)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("uniform", [True, False], ids=["uniform", "nonuniform"])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("d", [512])
@pytest.mark.parametrize(
    "nheads,nheads_kv,seqlen_min,seqlen_max,batch_size",
    [
        # GQA 32:4
        (32, 4, 256, 512, 4),
        (32, 4, 512, 1024, 4),
        (32, 4, 1024, 2048, 2),
        (32, 4, 2048, 4096, 1),
        # MHA 4:4
        (4, 4, 256, 512, 4),
        (4, 4, 512, 1024, 2),
    ],
    ids=[
        "gqa-256-512",
        "gqa-512-1024",
        "gqa-1024-2048",
        "gqa-2048-4096",
        "mha-256-512",
        "mha-512-1024",
    ],
)
@retry_on_oom
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_splitd_bwd_varlen(
    nheads,
    nheads_kv,
    seqlen_min,
    seqlen_max,
    batch_size,
    d,
    causal,
    dtype,
    uniform,
):
    """Test SplitD varlen backward against per-sequence fp32 reference + flex."""
    device = "cuda"
    torch.manual_seed(42)
    torch.cuda.empty_cache()

    tile_n = 64  # SplitD tile size — round seqlens for TMA alignment

    if uniform:
        random.seed(42)
        seqlens_list = [
            max(tile_n, (random.randint(seqlen_min, seqlen_max) + tile_n - 1) // tile_n * tile_n)
            for _ in range(batch_size)
        ]
    else:
        avg_seqlen = (seqlen_min + seqlen_max) // 2
        seqlens_list = _make_nonuniform_seqlens(avg_seqlen, batch_size, tile_m=tile_n)
        batch_size = len(seqlens_list)

    total_tokens = sum(seqlens_list)
    max_seqlen = max(seqlens_list)
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    for i, s in enumerate(seqlens_list):
        cu_seqlens[i + 1] = cu_seqlens[i] + s

    softmax_scale = 1.0 / math.sqrt(d)

    q = torch.randn(total_tokens, nheads, d, device=device, dtype=dtype).requires_grad_(True)
    k = torch.randn(total_tokens, nheads_kv, d, device=device, dtype=dtype).requires_grad_(True)
    v = torch.randn(total_tokens, nheads_kv, d, device=device, dtype=dtype).requires_grad_(True)

    out_fa, lse = split_flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=causal,
        softmax_scale=softmax_scale,
        return_lse=True,
    )
    if is_fake_mode():
        return

    g = torch.randn_like(out_fa)
    dq_fa, dk_fa, dv_fa = torch.autograd.grad(out_fa, (q, k, v), g)

    # Per-sequence fp32 reference + flex
    dq_ref = torch.zeros(total_tokens, nheads, d, device=device, dtype=torch.float32)
    dk_ref = torch.zeros(total_tokens, nheads_kv, d, device=device, dtype=torch.float32)
    dv_ref = torch.zeros(total_tokens, nheads_kv, d, device=device, dtype=torch.float32)
    dq_fl = torch.zeros(total_tokens, nheads, d, device=device, dtype=dtype)
    dk_fl = torch.zeros(total_tokens, nheads_kv, d, device=device, dtype=dtype)
    dv_fl = torch.zeros(total_tokens, nheads_kv, d, device=device, dtype=dtype)

    for b in range(batch_size):
        start = cu_seqlens[b].item()
        end = cu_seqlens[b + 1].item()

        # fp32 reference
        q_b = q[start:end].detach().float().unsqueeze(0).requires_grad_(True)
        k_b = k[start:end].detach().float().unsqueeze(0).requires_grad_(True)
        v_b = v[start:end].detach().float().unsqueeze(0).requires_grad_(True)
        g_b = g[start:end].float().unsqueeze(0)
        out_b = attention_ref(q_b, k_b, v_b, causal=causal, softmax_scale=softmax_scale)
        dq_b, dk_b, dv_b = torch.autograd.grad(out_b, (q_b, k_b, v_b), g_b)
        dq_ref[start:end] = dq_b.squeeze(0)
        dk_ref[start:end] = dk_b.squeeze(0)
        dv_ref[start:end] = dv_b.squeeze(0)

        # flex (per-sequence, test dtype)
        dq_f, dk_f, dv_f = _flex_bwd(
            q[start:end].detach().unsqueeze(0),
            k[start:end].detach().unsqueeze(0),
            v[start:end].detach().unsqueeze(0),
            g[start:end].unsqueeze(0),
            causal,
            nheads,
            nheads_kv,
            softmax_scale,
        )
        dq_fl[start:end] = dq_f.squeeze(0)
        dk_fl[start:end] = dk_f.squeeze(0)
        dv_fl[start:end] = dv_f.squeeze(0)

    dq_err = (dq_fa.float() - dq_ref).abs().max().item()
    dk_err = (dk_fa.float() - dk_ref).abs().max().item()
    dv_err = (dv_fa.float() - dv_ref).abs().max().item()

    head_tag = "GQA" if nheads != nheads_kv else "MHA"
    mode = "uniform" if uniform else "nonuniform"
    print(
        f"\n[BWD-VARLEN-{head_tag}-{mode} d={d} seqlen=[{seqlen_min},{seqlen_max}] "
        f"B={batch_size} H={nheads} Hkv={nheads_kv} causal={causal}]"
    )
    print(f"  seqlens: {seqlens_list}")
    print(f"  dQ: flash={dq_err:.6f}  flex={dq_fl.float().sub(dq_ref).abs().max().item():.6f}")
    print(f"  dK: flash={dk_err:.6f}  flex={dk_fl.float().sub(dk_ref).abs().max().item():.6f}")
    print(f"  dV: flash={dv_err:.6f}  flex={dv_fl.float().sub(dv_ref).abs().max().item():.6f}")

    assert dq_err < 0.05, f"dQ max diff {dq_err:.6f} exceeds 0.05"
    assert dk_err < 0.05, f"dK max diff {dk_err:.6f} exceeds 0.05"
    assert dv_err < 0.05, f"dV max diff {dv_err:.6f} exceeds 0.05"
