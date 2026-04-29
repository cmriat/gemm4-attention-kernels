# SplitD Flash Attention interface for head_dim == 512 on SM90 (Hopper).
# Based on https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/interface.py
# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
#
# This interface is specialized to the D=512 SplitD kernels.
# All non-SplitD code paths (head_dim <= 256) have been removed.
# Training-only build: page_table, learnable_sink, seqused_q/k, block_sparsity removed.

import os
import math
from functools import lru_cache
from typing import Optional, Tuple, Callable

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from quack.compile_utils import make_fake_tensor as fake_tensor
from .cache_utils import get_jit_cache
from .testing import is_fake_mode


if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
    from . import cute_dsl_ptxas  # noqa: F401

    cute_dsl_ptxas.patch()


from . import utils
from . import fa_logging
from .cute_dsl_utils import (
    to_cute_tensor,
    to_cute_aux_tensor,
    get_aux_tensor_metadata,
)
from .flash_fwd_sm90_d512 import (
    FlashAttentionForwardSm90TrainOnly as FlashAttentionForwardSplitD,
)
from .flash_bwd_preprocess import FlashAttentionBackwardPreprocess
from .fmha_dkdv_sm90_d512 import FlashBwdDKDV_SplitD_Sm90
from .fmha_dq_sm90_d512 import FlashBwdDQ_SplitD_Sm90


SUPPORTED_HEAD_DIM = 512
FWD_TILE_M = 64
FWD_TILE_N = 128
BWD_TILE_M = 64
BWD_TILE_N = 64


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _parse_arch_str(arch_str):
    """Parse arch string (e.g. 'sm_90a', '90') to int (e.g. 90)."""
    import re

    match = re.match(r"^(?:sm_?|SM_?)?(\d+)(\d)([af]?)$", arch_str)
    if not match:
        raise ValueError(f"Invalid arch format: {arch_str}")
    major, minor, _ = match.groups()
    return int(major) * 10 + int(minor)


@lru_cache(maxsize=None)
def _get_device_arch():
    """Cached device arch check. Override with FLASH_ATTENTION_ARCH env var."""
    arch_override = os.environ.get("FLASH_ATTENTION_ARCH", None)
    if arch_override is not None:
        return _parse_arch_str(arch_override)
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + int(minor)


def _validate_head_dims(head_dim: int, head_dim_v: int) -> None:
    """Validate SplitD head dimension constraints: head_dim == head_dim_v == 512."""
    if head_dim != SUPPORTED_HEAD_DIM or head_dim_v != SUPPORTED_HEAD_DIM:
        raise ValueError(
            f"(head_dim, head_dim_v)=({head_dim}, {head_dim_v}) is not supported. "
            f"This SplitD interface requires head_dim == {SUPPORTED_HEAD_DIM} and "
            f"head_dim_v == {SUPPORTED_HEAD_DIM}, matching the kernel's fixed "
            "8x64 D-slice layout."
        )


def maybe_contiguous(x):
    return x.contiguous() if x is not None and not x.is_contiguous() else x


def _validate_tensor(tensor, name, expected_shape, expected_dtype, expected_device):
    if tensor is None:
        raise ValueError(f"{name} must not be None")
    if tensor.shape != expected_shape:
        raise ValueError(f"{name} has shape {tensor.shape}, expected {expected_shape}")
    if tensor.dtype != expected_dtype:
        raise TypeError(f"{name} has dtype {tensor.dtype}, expected {expected_dtype}")
    if tensor.device != expected_device:
        raise RuntimeError(f"{name} is on {tensor.device}, expected {expected_device}")


def _validate_cu_seqlens(tensor, name, batch_size: Optional[int] = None) -> None:
    if tensor is None:
        return
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be a 1D tensor, got rank {tensor.ndim}")
    if batch_size is not None and tensor.shape != (batch_size + 1,):
        raise ValueError(f"{name} must have shape ({batch_size + 1},), got {tensor.shape}")
    if tensor.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
    if tensor.stride(0) != 1:
        raise ValueError(f"{name} must be contiguous")


def _ensure_cuda_tensors(*named_tensors) -> None:
    if is_fake_mode():
        return
    for name, tensor in named_tensors:
        if tensor is not None and not tensor.is_cuda:
            raise RuntimeError(f"{name} must be on a CUDA device, got {tensor.device}")


def _validate_qkv_common(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
):
    q_rank = 3 if cu_seqlens_q is not None else 4
    kv_rank = 3 if cu_seqlens_k is not None else 4
    if q.ndim != q_rank:
        raise ValueError(f"q must have rank {q_rank}, got rank {q.ndim}")
    if k.ndim != kv_rank:
        raise ValueError(f"k must have rank {kv_rank}, got rank {k.ndim}")
    if v.ndim != kv_rank:
        raise ValueError(f"v must have rank {kv_rank}, got rank {v.ndim}")

    num_head, head_dim = q.shape[-2:]
    seqlen_k = k.shape[-3]
    num_head_kv = k.shape[-2]
    head_dim_v = v.shape[-1]

    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        _validate_cu_seqlens(cu_seqlens_q, "cu_seqlens_q")
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q.shape[0]

    if k.shape[-1] != head_dim:
        raise ValueError(f"k head_dim is {k.shape[-1]}, expected {head_dim}")
    if v.shape[-3] != seqlen_k or v.shape[-2] != num_head_kv:
        raise ValueError(
            f"v has shape {v.shape}, expected matching seqlen/head dims (*, {seqlen_k}, {num_head_kv}, {head_dim_v})"
        )

    if cu_seqlens_k is None:
        expected_k_shape = (batch_size, seqlen_k, num_head_kv, head_dim)
        expected_v_shape = (batch_size, seqlen_k, num_head_kv, head_dim_v)
    else:
        _validate_cu_seqlens(cu_seqlens_k, "cu_seqlens_k", batch_size)
        expected_k_shape = (seqlen_k, num_head_kv, head_dim)
        expected_v_shape = (seqlen_k, num_head_kv, head_dim_v)
    if k.shape != expected_k_shape:
        raise ValueError(f"k has shape {k.shape}, expected {expected_k_shape}")
    if v.shape != expected_v_shape:
        raise ValueError(f"v has shape {v.shape}, expected {expected_v_shape}")
    if cu_seqlens_q is not None:
        _validate_cu_seqlens(cu_seqlens_q, "cu_seqlens_q", batch_size)

    if q.dtype not in torch2cute_dtype_map:
        raise TypeError("SM90 CuTe inputs must be torch.float16 or torch.bfloat16")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError(f"q, k, and v must have the same dtype, got {q.dtype}, {k.dtype}, {v.dtype}")
    _ensure_cuda_tensors(
        ("q", q),
        ("k", k),
        ("v", v),
        ("cu_seqlens_q", cu_seqlens_q),
        ("cu_seqlens_k", cu_seqlens_k),
    )
    if num_head % num_head_kv != 0:
        raise ValueError(f"num_head ({num_head}) must be divisible by num_head_kv ({num_head_kv})")
    _validate_head_dims(head_dim, head_dim_v)
    return batch_size, seqlen_q, total_q, seqlen_k, num_head, num_head_kv, head_dim, head_dim_v


def _unsupported_training_features(
    requires_grad: bool,
    softcap: Optional[float],
    local: bool,
    score_mod: Optional[Callable],
    mask_mod: Optional[Callable],
    aux_tensors: Optional[list[torch.Tensor]],
):
    if not requires_grad:
        return
    unsupported = []
    if softcap is not None:
        unsupported.append("softcap")
    if local:
        unsupported.append("local/window attention")
    if score_mod is not None:
        unsupported.append("score_mod")
    if mask_mod is not None:
        unsupported.append("mask_mod")
    if aux_tensors is not None:
        unsupported.append("aux_tensors")
    if unsupported:
        raise NotImplementedError("SplitD backward does not support training with " + ", ".join(unsupported) + ".")


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}


def _resolve_causal_local_window(causal, window_size_left, window_size_right, mask_mod=None):
    local = False
    if window_size_left is not None or window_size_right is not None:
        if causal:
            raise ValueError("causal and window_size are mutually exclusive")
        if mask_mod is not None:
            raise ValueError("mask_mod and window_size are mutually exclusive")
        if window_size_left is not None and window_size_right is not None:
            if window_size_left < 0 and window_size_right < 0:
                causal, local = False, False
                window_size_left, window_size_right = None, None
            elif window_size_right == 0 and window_size_left < 0:
                causal = True
                local = False
                window_size_left, window_size_right = None, None
            else:
                causal, local = False, True
        else:
            causal, local = False, True
    else:
        local = False
    return causal, local, window_size_left, window_size_right


# ---------------------------------------------------------------------------
# Forward pass — SplitD SM90 (training, head_dim == 512)
# ---------------------------------------------------------------------------


def _flash_attn_fwd_sm90(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: Optional[float] = None,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    pack_gqa: Optional[bool] = None,
    score_mod: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    aux_tensors: Optional[list[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SplitD SM90 forward pass for FlashAttention (training only, head_dim == 512)."""
    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    (
        batch_size,
        seqlen_q,
        total_q,
        seqlen_k,
        num_head,
        num_head_kv,
        head_dim,
        head_dim_v,
    ) = _validate_qkv_common(q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k)

    arch = _get_device_arch()
    if arch // 10 != 9:
        raise RuntimeError(
            f"This SM90-only interface requires Hopper (SM 9.x), got compute capability {arch}. "
            "Use the full interface.py for other architectures."
        )
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    if softcap == 0.0:
        softcap = None
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1

    device = q.device
    out_torch_dtype = q.dtype
    q_batch_seqlen_shape = (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q,)
    lse_shape = (batch_size, num_head, seqlen_q) if cu_seqlens_q is None else (num_head, total_q)
    requires_grad = q.requires_grad or k.requires_grad or v.requires_grad

    if out is None:
        out = torch.empty(*q_batch_seqlen_shape, num_head, head_dim_v, dtype=out_torch_dtype, device=device)
    else:
        _validate_tensor(out, "out", (*q_batch_seqlen_shape, num_head, head_dim_v), out_torch_dtype, device)

    if lse is None:
        lse = torch.empty(lse_shape, dtype=torch.float32, device=device) if requires_grad or return_lse else None
    elif lse is not None:
        _validate_tensor(lse, "lse", lse_shape, torch.float32, device)

    dtype = torch2cute_dtype_map[q.dtype]

    causal, local, window_size_left, window_size_right = _resolve_causal_local_window(
        causal, window_size_left, window_size_right, mask_mod
    )
    _unsupported_training_features(requires_grad, softcap, local, score_mod, mask_mod, aux_tensors)

    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # SplitD tile sizes (hardcoded)
    tile_m = FWD_TILE_M  # tile_m=64 required by num_wg_mma==1 for register headroom
    tile_n = FWD_TILE_N  # tile_n=128 with sO_spill for register pressure management

    # Auto-detect K=V: same data pointer means same tensor
    kv_same = k is v if is_fake_mode() else k.data_ptr() == v.data_ptr()

    if max_seqlen_q is None:
        max_seqlen_q = seqlen_q if cu_seqlens_q is None else total_q
    if max_seqlen_k is None:
        max_seqlen_k = seqlen_k

    if softcap is not None:
        if score_mod is not None:
            raise ValueError("softcap and score_mod cannot be used together")
        score_mod = utils.create_softcap_scoremod(softcap)

    score_mod_hash = utils.hash_callable(score_mod) if score_mod is not None else False
    mask_mod_hash = utils.hash_callable(mask_mod) if mask_mod is not None else False

    is_varlen = cu_seqlens_q is not None or cu_seqlens_k is not None

    if mask_mod is not None and is_varlen:
        raise NotImplementedError("mask_mod with aux_tensors is not yet supported for varlen sequences.")

    if aux_tensors is not None:
        aux_tensor_metadata = get_aux_tensor_metadata(aux_tensors)
    else:
        aux_tensor_metadata = None

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        score_mod_hash,
        mask_mod_hash,
        aux_tensor_metadata,
        lse is None,
        cu_seqlens_q is None,
        cu_seqlens_k is None,
        window_size_left is not None,
        window_size_right is not None,
        tile_m,
        tile_n,
        pack_gqa,
        arch,
        kv_same,
        fa_logging.get_fa_log_level(),
    )
    if compile_key not in _flash_attn_fwd_sm90.compile_cache:
        cu_seqlens_q_tensor, cu_seqlens_k_tensor = [
            to_cute_tensor(t, assumed_align=4, leading_dim=0) if t is not None else None
            for t in (cu_seqlens_q, cu_seqlens_k)
        ]
        q_tensor, k_tensor, v_tensor, o_tensor = [to_cute_tensor(t) for t in (q, k, v, out)]
        if lse is not None:
            lse_tensor = to_cute_tensor(lse, assumed_align=4)
        else:
            lse_tensor = None

        cute_aux_tensors = None
        if aux_tensors is not None:
            cute_aux_tensors = [to_cute_aux_tensor(buf) for buf in aux_tensors]

        fa_fwd = FlashAttentionForwardSplitD(
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            is_causal=causal,
            is_local=local,
            pack_gqa=pack_gqa,
            tile_m=tile_m,
            tile_n=tile_n,
            kv_same=kv_same,
            mask_mod=mask_mod,
            score_mod=score_mod,
            has_aux_tensors=aux_tensors is not None,
        )

        # Positional args must match FlashAttentionForwardSplitD.__call__ signature:
        # mQ, mK, mV, mO, mLSE, scale, cuseqlens_q, cuseqlens_k, wsl, wsr, aux, stream
        compile_args = [
            fa_fwd,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            window_size_left,
            window_size_right,
            cute_aux_tensors,
            current_stream,
        ]
        _flash_attn_fwd_sm90.compile_cache[compile_key] = cute.compile(
            *compile_args,
            options=("--enable-tvm-ffi --ptxas-options '--verbose --warn-on-spills --warn-on-local-memory-usage'"),
        )

    if not is_fake_mode():
        q_call, k_call, v_call = q.detach(), k.detach(), v.detach()
        call_args = [
            q_call,
            k_call,
            v_call,
            out.detach(),
            lse,
            softmax_scale,
            cu_seqlens_q,
            cu_seqlens_k,
            window_size_left,
            window_size_right,
            aux_tensors,
        ]
        _flash_attn_fwd_sm90.compile_cache[compile_key](*call_args)
    return out, lse


_flash_attn_fwd_sm90.compile_cache = get_jit_cache("fwd_sm90")


# ---------------------------------------------------------------------------
# Backward helpers
# ---------------------------------------------------------------------------


def make_fake_bwd_tensors(dtype, has_gqa, varlen_q, varlen_k):
    sym = cute.sym_int
    div = 128 // dtype.width  # 8 for fp16/bf16
    b, seqlen_q, seqlen_k, h_q, d, d_v = sym(), sym(), sym(), sym(), sym(), sym()
    h_kv = h_q if not has_gqa else sym()
    seqlen_q_rounded, seqlen_k_rounded = sym(), sym()
    seqlen_q_d_rounded, seqlen_k_d_rounded, seqlen_k_dv_rounded = sym(), sym(), sym()
    total_q, total_k, total_q_rounded, total_k_rounded = sym(), sym(), sym(), sym()
    total_q_d_rounded, total_k_d_rounded, total_k_dv_rounded = sym(), sym(), sym()
    b_seqlenq = (b, seqlen_q) if not varlen_q else (total_q,)
    b_seqlenk = (b, seqlen_k) if not varlen_k else (total_k,)
    mQ = fake_tensor(dtype, (*b_seqlenq, h_q, d), divisibility=div)
    mO = fake_tensor(dtype, (*b_seqlenq, h_q, d_v), divisibility=div)
    mdO = fake_tensor(dtype, (*b_seqlenq, h_q, d_v), divisibility=div)
    mK = fake_tensor(dtype, (*b_seqlenk, h_kv, d), divisibility=div)
    mV = fake_tensor(dtype, (*b_seqlenk, h_kv, d_v), divisibility=div)
    mdQ = fake_tensor(dtype, (*b_seqlenq, h_q, d), divisibility=div)
    mdK = fake_tensor(dtype, (*b_seqlenk, h_kv, d), divisibility=div)
    mdV = fake_tensor(dtype, (*b_seqlenk, h_kv, d_v), divisibility=div)
    if not varlen_q:
        mLSE = fake_tensor(Float32, (b, h_q, seqlen_q), divisibility=1)
        mLSElog2 = fake_tensor(Float32, (b, h_q, seqlen_q_rounded), divisibility=4)
        mPdPsum = fake_tensor(Float32, (b, h_q, seqlen_q_rounded), divisibility=4)
        dQaccum = fake_tensor(Float32, (b, h_q, seqlen_q_d_rounded), divisibility=4)
    else:
        mLSE = fake_tensor(Float32, (h_q, total_q), divisibility=1)
        mLSElog2 = fake_tensor(Float32, (h_q, total_q_rounded), divisibility=4)
        mPdPsum = fake_tensor(Float32, (h_q, total_q_rounded), divisibility=4)
        dQaccum = fake_tensor(Float32, (h_q, total_q_d_rounded), divisibility=4)
    if not has_gqa:
        mdKaccum, mdVaccum = None, None
    else:
        if not varlen_k:
            mdKaccum = fake_tensor(Float32, (b, h_kv, seqlen_k_rounded), divisibility=4)
            mdVaccum = fake_tensor(Float32, (b, h_kv, seqlen_k_dv_rounded), divisibility=4)
        else:
            mdKaccum = fake_tensor(Float32, (h_kv, total_k_rounded), divisibility=4)
            mdVaccum = fake_tensor(Float32, (h_kv, total_k_dv_rounded), divisibility=4)
    return mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, dQaccum, mdKaccum, mdVaccum


def _compile_bwd_preprocess(
    dtype,
    head_dim,
    head_dim_v,
    m_block_size,
    has_cuseqlens_q,
    has_seqused_q,
    has_dlse,
    has_dqaccum=True,
):
    """Compile bwd preprocess kernel using cute fake tensors."""
    mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, mdQaccum, mdKaccum, mdVaccum = make_fake_bwd_tensors(
        dtype, has_gqa=True, varlen_q=has_cuseqlens_q, varlen_k=False
    )
    batch = mQ.shape[0] if not has_cuseqlens_q else cute.sym_int()
    batchp1 = cute.sym_int()
    mCuSeqlensQ = fake_tensor(Int32, (batchp1,), divisibility=1) if has_cuseqlens_q else None
    mSequsedQ = fake_tensor(Int32, (batch,), divisibility=1) if has_seqused_q else None
    mdLSE = fake_tensor(Float32, mLSE.shape, divisibility=1) if has_dlse else None
    fa_bwd_pre = FlashAttentionBackwardPreprocess(dtype, head_dim, head_dim_v, m_block_size)
    return cute.compile(
        fa_bwd_pre,
        mO,
        mdO,
        mPdPsum,
        mLSE,
        mLSElog2,
        mdQaccum if has_dqaccum else None,
        mCuSeqlensQ,
        mSequsedQ,
        mdLSE,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _bwd_preprocess(
    out,
    dout,
    dpsum,
    lse,
    lse_log2,
    dq_accum,
    cu_seqlens_q,
    seqused_q,
    dlse,
    dtype,
    head_dim,
    head_dim_v,
    m_block_size,
):
    """Backward preprocess: compute (o * dout).sum(dim=-1) - dLSE, lse * log2_e, and zero out dq_accum."""
    is_varlen = cu_seqlens_q is not None
    has_dqaccum = dq_accum is not None
    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        m_block_size,
        is_varlen,
        seqused_q is not None,
        dlse is not None,
        has_dqaccum,
    )
    if compile_key not in _bwd_preprocess.compile_cache:
        _bwd_preprocess.compile_cache[compile_key] = _compile_bwd_preprocess(*compile_key)
    if not is_fake_mode():
        _bwd_preprocess.compile_cache[compile_key](
            out, dout, dpsum, lse, lse_log2, dq_accum, cu_seqlens_q, seqused_q, dlse
        )


_bwd_preprocess.compile_cache = get_jit_cache("bwd_pre_sm90")


# ---------------------------------------------------------------------------
# Backward pass — SplitD SM90 (training, head_dim == 512)
# ---------------------------------------------------------------------------


def _flash_attn_bwd_sm90(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: float = 0.0,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    dlse: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SplitD SM90 backward pass (training only, head_dim == 512)."""
    arch = _get_device_arch()
    if arch // 10 != 9:
        raise RuntimeError(f"This SM90-only interface requires Hopper (SM 9.x), got compute capability {arch}.")

    if softcap != 0.0:
        raise NotImplementedError("SplitD backward does not support softcap yet")

    causal, local, window_size_left, window_size_right = _resolve_causal_local_window(
        causal, window_size_left, window_size_right
    )
    if local:
        raise NotImplementedError("SplitD backward does not support local/window attention yet")

    # SplitD tile sizes (hardcoded)
    m_block_size = BWD_TILE_M
    n_block_size = BWD_TILE_N

    q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k = [
        maybe_contiguous(t) for t in (q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k)
    ]
    (
        batch_size,
        seqlen_q,
        total_q,
        seqlen_k,
        num_head,
        num_head_kv,
        head_dim,
        head_dim_v,
    ) = _validate_qkv_common(q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k)
    if q.dtype == torch.float16:
        raise NotImplementedError(
            "SplitD backward currently supports bfloat16 only; the fp16 dQ path has a known launch failure."
        )
    if cu_seqlens_q is None:
        seqlen_q_for_rounding = seqlen_q
    else:
        seqlen_q_for_rounding = max_seqlen_q if max_seqlen_q is not None else total_q

    seqlen_q_rounded = (seqlen_q_for_rounding + m_block_size - 1) // m_block_size * m_block_size
    device = q.device
    out_torch_dtype = q.dtype
    if cu_seqlens_q is not None:
        out_shape = (total_q, num_head, head_dim_v)
        lse_shape = (num_head, total_q)
    else:
        out_shape = (batch_size, seqlen_q, num_head, head_dim_v)
        lse_shape = (batch_size, num_head, seqlen_q)
    _validate_tensor(out, "out", out_shape, out_torch_dtype, device)
    _validate_tensor(dout, "dout", out_shape, out_torch_dtype, device)
    _validate_tensor(lse, "lse", lse_shape, torch.float32, device)
    if dlse is not None:
        dlse = maybe_contiguous(dlse)
        _validate_tensor(dlse, "dlse", lse_shape, torch.float32, device)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    qhead_per_kvhead = num_head // num_head_kv

    if dq is None:
        dq = torch.empty_like(q)
    else:
        _validate_tensor(dq, "dq", q.shape, out_torch_dtype, device)

    if dk is None:
        dk = torch.empty_like(k)
    else:
        _validate_tensor(dk, "dk", k.shape, out_torch_dtype, device)

    if dv is None:
        dv = torch.empty_like(v)
    else:
        _validate_tensor(dv, "dv", v.shape, out_torch_dtype, device)

    # SplitD writes dQ directly, no fp32 accumulator needed
    dq_accum = None

    if cu_seqlens_q is None:
        dpsum = torch.empty(batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=device)
        lse_log2 = torch.empty(batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=device)
    else:
        total_q_rounded_padded = (total_q + cu_seqlens_q.shape[0] * m_block_size - 1) // m_block_size * m_block_size
        dpsum = torch.empty(num_head, total_q_rounded_padded, dtype=torch.float32, device=device)
        lse_log2 = torch.empty(num_head, total_q_rounded_padded, dtype=torch.float32, device=device)

    dtype = torch2cute_dtype_map[q.dtype]
    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # (1) Preprocess (seqused_q=None since seqused is removed)
    _bwd_preprocess(
        out,
        dout,
        dpsum,
        lse,
        lse_log2,
        dq_accum,
        cu_seqlens_q,
        None,
        dlse,
        dtype,
        head_dim,
        head_dim_v,
        m_block_size,
    )

    # (2) Compile and execute SplitD dKdV and dQ kernels
    dkdv_key = (
        dtype,
        head_dim,
        head_dim_v,
        causal,
        m_block_size,
        n_block_size,
        cu_seqlens_q is not None,
        cu_seqlens_k is not None,
        qhead_per_kvhead,
    )
    if dkdv_key not in _flash_attn_bwd_sm90.compile_cache_dkdv:
        q_t, k_t, v_t, do_t = [to_cute_tensor(t) for t in (q, k, v, dout)]
        dk_t, dv_t = [to_cute_tensor(t) for t in (dk, dv)]
        lse_log2_t = to_cute_tensor(lse_log2, assumed_align=4)
        dpsum_t = to_cute_tensor(dpsum, assumed_align=4)
        cu_seqlens_q_t = (
            to_cute_tensor(cu_seqlens_q, assumed_align=4, leading_dim=0) if cu_seqlens_q is not None else None
        )
        cu_seqlens_k_t = (
            to_cute_tensor(cu_seqlens_k, assumed_align=4, leading_dim=0) if cu_seqlens_k is not None else None
        )

        fa_dkdv = FlashBwdDKDV_SplitD_Sm90(
            dtype,
            head_dim,
            head_dim_v=head_dim_v,
            is_causal=causal,
            qhead_per_kvhead=qhead_per_kvhead,
            tile_m=m_block_size,
            tile_n=n_block_size,
        )
        _flash_attn_bwd_sm90.compile_cache_dkdv[dkdv_key] = cute.compile(
            fa_dkdv,
            q_t,
            k_t,
            v_t,
            do_t,
            lse_log2_t,
            dpsum_t,
            dk_t,
            dv_t,
            softmax_scale,
            cu_seqlens_q_t,
            cu_seqlens_k_t,
            current_stream,
            options=("--enable-tvm-ffi --ptxas-options '--verbose --warn-on-spills --warn-on-local-memory-usage'"),
        )

    dq_key = (
        dtype,
        head_dim,
        head_dim_v,
        causal,
        m_block_size,
        n_block_size,
        cu_seqlens_q is not None,
        cu_seqlens_k is not None,
        qhead_per_kvhead,
    )
    if dq_key not in _flash_attn_bwd_sm90.compile_cache_dq:
        q_t2, k_t2, v_t2, do_t2 = [to_cute_tensor(t) for t in (q, k, v, dout)]
        dq_t = to_cute_tensor(dq)
        lse_log2_t2 = to_cute_tensor(lse_log2, assumed_align=4)
        dpsum_t2 = to_cute_tensor(dpsum, assumed_align=4)
        cu_seqlens_q_t2 = (
            to_cute_tensor(cu_seqlens_q, assumed_align=4, leading_dim=0) if cu_seqlens_q is not None else None
        )
        cu_seqlens_k_t2 = (
            to_cute_tensor(cu_seqlens_k, assumed_align=4, leading_dim=0) if cu_seqlens_k is not None else None
        )

        fa_dq = FlashBwdDQ_SplitD_Sm90(
            dtype,
            head_dim,
            head_dim_v=head_dim_v,
            is_causal=causal,
            qhead_per_kvhead=qhead_per_kvhead,
            tile_m=m_block_size,
            tile_n=n_block_size,
        )
        _flash_attn_bwd_sm90.compile_cache_dq[dq_key] = cute.compile(
            fa_dq,
            q_t2,
            k_t2,
            v_t2,
            do_t2,
            lse_log2_t2,
            dpsum_t2,
            dq_t,
            softmax_scale,
            cu_seqlens_q_t2,
            cu_seqlens_k_t2,
            current_stream,
            options=("--enable-tvm-ffi --ptxas-options '--verbose --warn-on-spills --warn-on-local-memory-usage'"),
        )

    # Execute dKdV and dQ kernels
    if not is_fake_mode():
        _flash_attn_bwd_sm90.compile_cache_dkdv[dkdv_key](
            q.detach(),
            k.detach(),
            v.detach(),
            dout,
            lse_log2,
            dpsum,
            dk,
            dv,
            softmax_scale,
            cu_seqlens_q,
            cu_seqlens_k,
        )
        _flash_attn_bwd_sm90.compile_cache_dq[dq_key](
            q.detach(),
            k.detach(),
            v.detach(),
            dout,
            lse_log2,
            dpsum,
            dq,
            softmax_scale,
            cu_seqlens_q,
            cu_seqlens_k,
        )

    return dq, dk, dv


_flash_attn_bwd_sm90.compile_cache_dkdv = get_jit_cache("bwd_splitd_dkdv_sm90")
_flash_attn_bwd_sm90.compile_cache_dq = get_jit_cache("bwd_splitd_dq_sm90")


# ---------------------------------------------------------------------------
# Autograd classes
# ---------------------------------------------------------------------------


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        softcap: float = 0.0,
        pack_gqa: Optional[bool] = None,
        return_lse: bool = False,
    ):
        out, lse = _flash_attn_fwd_sm90(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            pack_gqa=pack_gqa,
            return_lse=return_lse,
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.return_lse = return_lse
        ctx.set_materialize_grads(False)
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        q, k, v, out, lse = ctx.saved_tensors
        if not ctx.return_lse:
            dlse = None
        if dout is None:
            dout = torch.zeros_like(out)
        dq, dk, dv = _flash_attn_bwd_sm90(
            q,
            k,
            v,
            out,
            dout,
            lse,
            ctx.softmax_scale,
            ctx.causal,
            ctx.softcap,
            window_size_left=ctx.window_size[0],
            window_size_right=ctx.window_size[1],
            dlse=dlse,
        )
        return dq, dk, dv, *((None,) * 6)


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        softcap: float = 0.0,
        pack_gqa: Optional[bool] = None,
        score_mod: Optional[Callable] = None,
        aux_tensors: Optional[list] = None,
        return_lse: bool = False,
    ):
        out, lse = _flash_attn_fwd_sm90(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=softcap,
            pack_gqa=pack_gqa,
            score_mod=score_mod,
            aux_tensors=aux_tensors,
            return_lse=return_lse,
        )
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.return_lse = return_lse
        ctx.set_materialize_grads(False)
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        if not ctx.return_lse:
            dlse = None
        if dout is None:
            dout = torch.zeros_like(out)
        dq, dk, dv = _flash_attn_bwd_sm90(
            q,
            k,
            v,
            out,
            dout,
            lse,
            ctx.softmax_scale,
            ctx.causal,
            ctx.softcap,
            window_size_left=ctx.window_size[0],
            window_size_right=ctx.window_size[1],
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=ctx.max_seqlen_q,
            max_seqlen_k=ctx.max_seqlen_k,
            dlse=dlse,
        )
        return dq, dk, dv, *((None,) * 12)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def split_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    softcap: float = 0.0,
    pack_gqa: Optional[bool] = None,
    return_lse: bool = False,
):
    out, lse = FlashAttnFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        window_size,
        softcap,
        pack_gqa,
        return_lse,
    )
    return (out, lse) if return_lse else out


def split_flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    softcap: float = 0.0,
    pack_gqa: Optional[bool] = None,
    score_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    return_lse: bool = False,
):
    out, lse = FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size,
        softcap,
        pack_gqa,
        score_mod,
        aux_tensors,
        return_lse,
    )
    return (out, lse) if return_lse else out


__all__ = [
    "split_flash_attn_func",
    "split_flash_attn_varlen_func",
]
