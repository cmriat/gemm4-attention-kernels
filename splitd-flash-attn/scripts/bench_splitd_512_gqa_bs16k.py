# Copyright (c) 2026.
# Fixed-token SplitD benchmark for D=512 GQA (Q heads=32, KV heads=4).
#
# This script follows the FlashAttention benchmark convention of keeping
# total tokens fixed while sweeping sequence length:
#
#   batch_size = total_tokens / seqlen
#
# It benchmarks batched and true non-uniform varlen inputs, causal and
# non-causal masks, forward and backward passes, then writes metrics and plots.

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except ImportError:  # pragma: no cover - depends on the local PyTorch build.
    create_block_mask = None
    flex_attention = None

from src.bench_utils import bandwidth_bwd_bytes, bandwidth_fwd_bytes, flops
from src.interface import (
    _flash_attn_bwd_sm90,
    _flash_attn_fwd_sm90,
    _bwd_preprocess,
    split_flash_attn_func,
    split_flash_attn_varlen_func,
)


HEAD_DIM = 512
NHEADS_Q = 32
NHEADS_KV = 4
DTYPE = torch.bfloat16
DTYPE_NAME = "bf16"
BWD_FLOPS_MULTIPLIER = 2.5
DEFAULT_TOTAL_TOKENS = 16_384
DEFAULT_SEQLENS = [512, 1024, 2048, 4096, 8192, 16384]
DEFAULT_WARMUP = 5
DEFAULT_REPEATS = 20
TILE_SIZE = 128
PLOT_ORDER = ["SplitD", "SDPA", "Flex"]

_COMPILED_FLEX_ATTENTION = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Benchmark SplitD D=512 GQA with Q heads=32, KV heads=4, fixed total tokens, and per-mode plots.")
    )
    parser.add_argument("--total-tokens", type=int, default=DEFAULT_TOTAL_TOKENS)
    parser.add_argument("--seqlens", type=int, nargs="+", default=DEFAULT_SEQLENS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_PROJECT_ROOT / "assets" / "bench_splitd_512_gqa_bs16k",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Only run SplitD. Batched SDPA and Flex backward baselines are skipped.",
    )
    return parser.parse_args()


def _require_sm90() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    major, minor = torch.cuda.get_device_capability()
    if major != 9:
        raise SystemExit(f"SplitD D=512 kernels require SM90/Hopper, got sm_{major}{minor}.")


def _configure_torch_compile() -> None:
    try:
        torch._dynamo.config.cache_size_limit = 1000
    except Exception:
        pass
    try:
        torch._functorch.config.donated_buffer = False
    except Exception:
        pass


def _get_compiled_flex_attention():
    global _COMPILED_FLEX_ATTENTION
    if flex_attention is None:
        raise RuntimeError("torch.nn.attention.flex_attention is not available.")
    if _COMPILED_FLEX_ATTENTION is None:
        _COMPILED_FLEX_ATTENTION = torch.compile(flex_attention)
    return _COMPILED_FLEX_ATTENTION


def _clear_compile_caches() -> None:
    if hasattr(_flash_attn_fwd_sm90, "compile_cache"):
        _flash_attn_fwd_sm90.compile_cache.clear()
    for cache_attr in ("compile_cache_dkdv", "compile_cache_dq"):
        if hasattr(_flash_attn_bwd_sm90, cache_attr):
            getattr(_flash_attn_bwd_sm90, cache_attr).clear()
    if hasattr(_bwd_preprocess, "compile_cache"):
        _bwd_preprocess.compile_cache.clear()
    try:
        torch._dynamo.reset()
    except Exception:
        pass


def _cleanup_cuda(clear_compile_cache: bool = False) -> None:
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass
    if clear_compile_cache:
        _clear_compile_caches()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _benchmark_fn(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def _benchmark_with_retry(fn, warmup: int, repeats: int) -> float:
    try:
        return _benchmark_fn(fn, warmup, repeats)
    except torch.OutOfMemoryError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        _cleanup_cuda(clear_compile_cache=True)
        return _benchmark_fn(fn, warmup, repeats)


def _batch_size_for(seqlen: int, total_tokens: int) -> int:
    if seqlen <= 0:
        raise ValueError(f"seqlen must be positive, got {seqlen}.")
    if total_tokens % seqlen != 0:
        raise ValueError(
            f"seqlen={seqlen} does not divide total_tokens={total_tokens}; "
            "fixed-token benchmark requires an integer batch size."
        )
    batch_size = total_tokens // seqlen
    if batch_size < 1:
        raise ValueError(f"seqlen={seqlen} is larger than total_tokens={total_tokens}; batch_size would be zero.")
    return batch_size


def _round_to_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def _adjust_lengths_to_total(lengths: list[int], total_tokens: int, tile_size: int) -> list[int]:
    diff = total_tokens - sum(lengths)
    if diff % tile_size != 0:
        raise ValueError(f"Cannot adjust lengths by {diff}; total and lengths must share tile={tile_size}.")
    if diff > 0:
        for _ in range(diff // tile_size):
            idx = min(range(len(lengths)), key=lengths.__getitem__)
            lengths[idx] += tile_size
    elif diff < 0:
        for _ in range((-diff) // tile_size):
            candidates = [i for i, length in enumerate(lengths) if length > tile_size]
            if not candidates:
                raise ValueError("Cannot reduce sequence lengths without dropping below tile size.")
            idx = max(candidates, key=lengths.__getitem__)
            lengths[idx] -= tile_size
    if sum(lengths) != total_tokens:
        raise AssertionError("sequence length adjustment failed")
    return lengths


def _make_nonuniform_varlen_seqlens(
    seqlen: int,
    total_tokens: int,
    tile_size: int = TILE_SIZE,
) -> list[int]:
    """Create true non-uniform varlen lengths with exact total token count."""
    nominal_batch = _batch_size_for(seqlen, total_tokens)
    if nominal_batch == 1:
        first = _round_to_multiple(total_tokens * 0.25, tile_size)
        first = min(max(tile_size, first), total_tokens - tile_size)
        return [first, total_tokens - first]

    lengths = []
    for i in range(nominal_batch):
        ratio = 0.5 + i / (nominal_batch - 1)
        lengths.append(_round_to_multiple(seqlen * ratio, tile_size))
    return _adjust_lengths_to_total(lengths, total_tokens, tile_size)


def _cu_seqlens(seq_lengths: list[int], device: str) -> torch.Tensor:
    offsets = [0]
    for length in seq_lengths:
        offsets.append(offsets[-1] + length)
    return torch.tensor(offsets, device=device, dtype=torch.int32)


def _split_output(out):
    return out[0] if isinstance(out, tuple) else out


def _make_bwd_fn(fwd_fn, inputs: list[torch.Tensor]):
    out = _split_output(fwd_fn())
    grad = torch.randn_like(out)

    def bwd_fn():
        for x in inputs:
            x.grad = None
        out.backward(grad, retain_graph=True)

    return bwd_fn


def _prepare_sdpa_fwd(q, k, v, causal: bool):
    q_sdpa = q.transpose(1, 2).contiguous()
    k_sdpa = k.transpose(1, 2).contiguous()
    v_sdpa = v.transpose(1, 2).contiguous()

    repeat_factor = NHEADS_Q // NHEADS_KV
    k_sdpa = k_sdpa.repeat_interleave(repeat_factor, dim=1)
    v_sdpa = v_sdpa.repeat_interleave(repeat_factor, dim=1)

    def fn():
        F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=causal)

    return fn


def _prepare_sdpa_bwd(q, k, v, causal: bool):
    q_sdpa = q.detach().transpose(1, 2).contiguous().requires_grad_(True)
    k_sdpa = k.detach().transpose(1, 2).contiguous().requires_grad_(True)
    v_sdpa = v.detach().transpose(1, 2).contiguous().requires_grad_(True)

    repeat_factor = NHEADS_Q // NHEADS_KV
    k_sdpa = k_sdpa.repeat_interleave(repeat_factor, dim=1).detach().requires_grad_(True)
    v_sdpa = v_sdpa.repeat_interleave(repeat_factor, dim=1).detach().requires_grad_(True)

    out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, is_causal=causal)
    grad = torch.randn_like(out)

    def bwd_fn():
        q_sdpa.grad = None
        k_sdpa.grad = None
        v_sdpa.grad = None
        out.backward(grad, retain_graph=True)

    return bwd_fn


def _prepare_flex_bwd(q, k, v, causal: bool, seqlen: int):
    if create_block_mask is None:
        raise RuntimeError("torch.nn.attention.flex_attention.create_block_mask is not available.")

    q_flex = q.detach().transpose(1, 2).contiguous().requires_grad_(True)
    k_flex = k.detach().transpose(1, 2).contiguous().requires_grad_(True)
    v_flex = v.detach().transpose(1, 2).contiguous().requires_grad_(True)

    if causal:

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal_mask,
            B=q_flex.shape[0],
            H=None,
            Q_LEN=seqlen,
            KV_LEN=seqlen,
            device=q_flex.device,
        )
    else:
        block_mask = None

    flex_kwargs: dict[str, Any] = {
        "enable_gqa": True,
        "kernel_options": {"BLOCK_M": 32, "BLOCK_N": 32, "num_stages": 1, "num_warps": 4},
    }
    if block_mask is not None:
        flex_kwargs["block_mask"] = block_mask

    out = _get_compiled_flex_attention()(q_flex, k_flex, v_flex, **flex_kwargs)
    grad = torch.randn_like(out)

    def bwd_fn():
        q_flex.grad = None
        k_flex.grad = None
        v_flex.grad = None
        out.backward(grad, retain_graph=True)

    return bwd_fn


def _metric_row(
    *,
    direction: str,
    input_mode: str,
    causal: bool,
    seqlen: int,
    batch_size: int,
    num_seqs: int,
    total_tokens: int,
    max_seqlen: int,
    seq_lengths: list[int],
    impl: str,
    ms: float,
    flop_count: float,
    byte_count: float,
) -> dict[str, Any]:
    return {
        "direction": direction,
        "input_mode": input_mode,
        "causal": causal,
        "seqlen": seqlen,
        "batch_size": batch_size,
        "num_seqs": num_seqs,
        "total_tokens": total_tokens,
        "max_seqlen": max_seqlen,
        "dtype": DTYPE_NAME,
        "head_dim": HEAD_DIM,
        "nheads_q": NHEADS_Q,
        "nheads_kv": NHEADS_KV,
        "impl": impl,
        "ms": ms,
        "tflops": flop_count / (ms * 1e-3) / 1e12,
        "gbps": byte_count / (ms * 1e-3) / 1e9,
        "seq_lengths": seq_lengths,
    }


def _print_row(row: dict[str, Any]) -> None:
    causal = "causal" if row["causal"] else "noncausal"
    print(
        f"  {row['impl']:<6} {row['direction']:<8} {row['input_mode']:<7} {causal:<9} "
        f"S={row['seqlen']:<5} B={row['batch_size']:<2} "
        f"{row['ms']:8.3f} ms  {row['tflops']:7.1f} TFLOPS  {row['gbps']:8.1f} GB/s",
        flush=True,
    )


def _benchmark_batched_forward(
    seqlen: int,
    causal: bool,
    total_tokens: int,
    warmup: int,
    repeats: int,
    skip_baselines: bool,
) -> list[dict[str, Any]]:
    _cleanup_cuda()
    torch.manual_seed(42)
    device = "cuda"
    batch_size = _batch_size_for(seqlen, total_tokens)

    q = torch.randn(batch_size, seqlen, NHEADS_Q, HEAD_DIM, device=device, dtype=DTYPE)
    k = torch.randn(batch_size, seqlen, NHEADS_KV, HEAD_DIM, device=device, dtype=DTYPE)
    v = torch.randn(batch_size, seqlen, NHEADS_KV, HEAD_DIM, device=device, dtype=DTYPE)

    flop_count = flops(batch_size, NHEADS_Q, seqlen, seqlen, HEAD_DIM, HEAD_DIM, causal=causal)
    byte_count = bandwidth_fwd_bytes(batch_size, NHEADS_Q, NHEADS_KV, seqlen, seqlen, HEAD_DIM, HEAD_DIM)

    rows = []
    splitd_fn = partial(split_flash_attn_func, q, k, v, causal=causal)
    splitd_ms = _benchmark_with_retry(splitd_fn, warmup, repeats)
    rows.append(
        _metric_row(
            direction="forward",
            input_mode="batched",
            causal=causal,
            seqlen=seqlen,
            batch_size=batch_size,
            num_seqs=batch_size,
            total_tokens=total_tokens,
            max_seqlen=seqlen,
            seq_lengths=[seqlen] * batch_size,
            impl="SplitD",
            ms=splitd_ms,
            flop_count=flop_count,
            byte_count=byte_count,
        )
    )
    _print_row(rows[-1])

    if not skip_baselines:
        sdpa_fn = _prepare_sdpa_fwd(q, k, v, causal)
        sdpa_ms = _benchmark_with_retry(sdpa_fn, warmup, repeats)
        rows.append(
            _metric_row(
                direction="forward",
                input_mode="batched",
                causal=causal,
                seqlen=seqlen,
                batch_size=batch_size,
                num_seqs=batch_size,
                total_tokens=total_tokens,
                max_seqlen=seqlen,
                seq_lengths=[seqlen] * batch_size,
                impl="SDPA",
                ms=sdpa_ms,
                flop_count=flop_count,
                byte_count=byte_count,
            )
        )
        _print_row(rows[-1])
        del sdpa_fn

    del q, k, v, splitd_fn
    _cleanup_cuda()
    return rows


def _benchmark_varlen_forward(
    seqlen: int,
    causal: bool,
    total_tokens: int,
    warmup: int,
    repeats: int,
) -> list[dict[str, Any]]:
    _cleanup_cuda()
    torch.manual_seed(42)
    device = "cuda"
    nominal_batch = _batch_size_for(seqlen, total_tokens)
    seq_lengths = _make_nonuniform_varlen_seqlens(seqlen, total_tokens)
    num_seqs = len(seq_lengths)
    max_seqlen = max(seq_lengths)
    cu_seqlens = _cu_seqlens(seq_lengths, device)

    q = torch.randn(total_tokens, NHEADS_Q, HEAD_DIM, device=device, dtype=DTYPE)
    k = torch.randn(total_tokens, NHEADS_KV, HEAD_DIM, device=device, dtype=DTYPE)
    v = torch.randn(total_tokens, NHEADS_KV, HEAD_DIM, device=device, dtype=DTYPE)

    flop_count = sum(flops(1, NHEADS_Q, sl, sl, HEAD_DIM, HEAD_DIM, causal=causal) for sl in seq_lengths)
    byte_count = sum(bandwidth_fwd_bytes(1, NHEADS_Q, NHEADS_KV, sl, sl, HEAD_DIM, HEAD_DIM) for sl in seq_lengths)

    splitd_fn = partial(
        split_flash_attn_varlen_func,
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=causal,
    )
    splitd_ms = _benchmark_with_retry(splitd_fn, warmup, repeats)
    row = _metric_row(
        direction="forward",
        input_mode="varlen",
        causal=causal,
        seqlen=seqlen,
        batch_size=nominal_batch,
        num_seqs=num_seqs,
        total_tokens=total_tokens,
        max_seqlen=max_seqlen,
        seq_lengths=seq_lengths,
        impl="SplitD",
        ms=splitd_ms,
        flop_count=flop_count,
        byte_count=byte_count,
    )
    _print_row(row)

    del q, k, v, cu_seqlens, splitd_fn
    _cleanup_cuda()
    return [row]


def _benchmark_batched_backward(
    seqlen: int,
    causal: bool,
    total_tokens: int,
    warmup: int,
    repeats: int,
    skip_baselines: bool,
) -> list[dict[str, Any]]:
    _cleanup_cuda()
    torch.manual_seed(42)
    device = "cuda"
    batch_size = _batch_size_for(seqlen, total_tokens)

    q = torch.randn(batch_size, seqlen, NHEADS_Q, HEAD_DIM, device=device, dtype=DTYPE).requires_grad_(True)
    k = torch.randn(batch_size, seqlen, NHEADS_KV, HEAD_DIM, device=device, dtype=DTYPE).requires_grad_(True)
    v = torch.randn(batch_size, seqlen, NHEADS_KV, HEAD_DIM, device=device, dtype=DTYPE).requires_grad_(True)

    fwd_flops = flops(batch_size, NHEADS_Q, seqlen, seqlen, HEAD_DIM, HEAD_DIM, causal=causal)
    flop_count = fwd_flops * BWD_FLOPS_MULTIPLIER
    byte_count = bandwidth_bwd_bytes(batch_size, NHEADS_Q, NHEADS_KV, seqlen, seqlen, HEAD_DIM, HEAD_DIM)

    rows = []
    splitd_bwd_fn = _make_bwd_fn(partial(split_flash_attn_func, q, k, v, causal=causal), [q, k, v])
    splitd_ms = _benchmark_with_retry(splitd_bwd_fn, warmup, repeats)
    rows.append(
        _metric_row(
            direction="backward",
            input_mode="batched",
            causal=causal,
            seqlen=seqlen,
            batch_size=batch_size,
            num_seqs=batch_size,
            total_tokens=total_tokens,
            max_seqlen=seqlen,
            seq_lengths=[seqlen] * batch_size,
            impl="SplitD",
            ms=splitd_ms,
            flop_count=flop_count,
            byte_count=byte_count,
        )
    )
    _print_row(rows[-1])
    del splitd_bwd_fn
    _cleanup_cuda()

    if not skip_baselines:
        sdpa_bwd_fn = _prepare_sdpa_bwd(q, k, v, causal)
        sdpa_ms = _benchmark_with_retry(sdpa_bwd_fn, warmup, repeats)
        rows.append(
            _metric_row(
                direction="backward",
                input_mode="batched",
                causal=causal,
                seqlen=seqlen,
                batch_size=batch_size,
                num_seqs=batch_size,
                total_tokens=total_tokens,
                max_seqlen=seqlen,
                seq_lengths=[seqlen] * batch_size,
                impl="SDPA",
                ms=sdpa_ms,
                flop_count=flop_count,
                byte_count=byte_count,
            )
        )
        _print_row(rows[-1])
        del sdpa_bwd_fn
        _cleanup_cuda()

        flex_bwd_fn = _prepare_flex_bwd(q, k, v, causal, seqlen)
        flex_ms = _benchmark_with_retry(flex_bwd_fn, warmup, repeats)
        rows.append(
            _metric_row(
                direction="backward",
                input_mode="batched",
                causal=causal,
                seqlen=seqlen,
                batch_size=batch_size,
                num_seqs=batch_size,
                total_tokens=total_tokens,
                max_seqlen=seqlen,
                seq_lengths=[seqlen] * batch_size,
                impl="Flex",
                ms=flex_ms,
                flop_count=flop_count,
                byte_count=byte_count,
            )
        )
        _print_row(rows[-1])
        del flex_bwd_fn

    del q, k, v
    _cleanup_cuda()
    return rows


def _benchmark_varlen_backward(
    seqlen: int,
    causal: bool,
    total_tokens: int,
    warmup: int,
    repeats: int,
) -> list[dict[str, Any]]:
    _cleanup_cuda()
    torch.manual_seed(42)
    device = "cuda"
    nominal_batch = _batch_size_for(seqlen, total_tokens)
    seq_lengths = _make_nonuniform_varlen_seqlens(seqlen, total_tokens)
    num_seqs = len(seq_lengths)
    max_seqlen = max(seq_lengths)
    cu_seqlens = _cu_seqlens(seq_lengths, device)

    q = torch.randn(total_tokens, NHEADS_Q, HEAD_DIM, device=device, dtype=DTYPE).requires_grad_(True)
    k = torch.randn(total_tokens, NHEADS_KV, HEAD_DIM, device=device, dtype=DTYPE).requires_grad_(True)
    v = torch.randn(total_tokens, NHEADS_KV, HEAD_DIM, device=device, dtype=DTYPE).requires_grad_(True)

    fwd_flops = sum(flops(1, NHEADS_Q, sl, sl, HEAD_DIM, HEAD_DIM, causal=causal) for sl in seq_lengths)
    flop_count = fwd_flops * BWD_FLOPS_MULTIPLIER
    byte_count = sum(bandwidth_bwd_bytes(1, NHEADS_Q, NHEADS_KV, sl, sl, HEAD_DIM, HEAD_DIM) for sl in seq_lengths)

    fwd_fn = partial(
        split_flash_attn_varlen_func,
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=causal,
    )
    splitd_bwd_fn = _make_bwd_fn(fwd_fn, [q, k, v])
    splitd_ms = _benchmark_with_retry(splitd_bwd_fn, warmup, repeats)
    row = _metric_row(
        direction="backward",
        input_mode="varlen",
        causal=causal,
        seqlen=seqlen,
        batch_size=nominal_batch,
        num_seqs=num_seqs,
        total_tokens=total_tokens,
        max_seqlen=max_seqlen,
        seq_lengths=seq_lengths,
        impl="SplitD",
        ms=splitd_ms,
        flop_count=flop_count,
        byte_count=byte_count,
    )
    _print_row(row)

    del q, k, v, cu_seqlens, fwd_fn, splitd_bwd_fn
    _cleanup_cuda()
    return [row]


def _write_results(rows: list[dict[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics.csv"
    json_path = out_dir / "metrics.json"
    fieldnames = [
        "direction",
        "input_mode",
        "causal",
        "seqlen",
        "batch_size",
        "num_seqs",
        "total_tokens",
        "max_seqlen",
        "dtype",
        "head_dim",
        "nheads_q",
        "nheads_kv",
        "impl",
        "ms",
        "tflops",
        "gbps",
        "seq_lengths",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = row.copy()
            csv_row["seq_lengths"] = ";".join(str(x) for x in row["seq_lengths"])
            writer.writerow(csv_row)
    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)
    return csv_path, json_path


def _plot_one(
    rows: list[dict[str, Any]],
    *,
    direction: str,
    input_mode: str,
    causal: bool,
    out_dir: Path,
) -> Path:
    selected = [
        row
        for row in rows
        if row["direction"] == direction and row["input_mode"] == input_mode and row["causal"] == causal
    ]
    causal_label = "causal" if causal else "noncausal"
    filename = f"{direction}_{input_mode}_{causal_label}.png"
    out_path = out_dir / filename
    tick_values = sorted({row["seqlen"] for row in selected}) or DEFAULT_SEQLENS

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for impl in PLOT_ORDER:
        impl_rows = sorted(
            (row for row in selected if row["impl"] == impl),
            key=lambda r: r["seqlen"],
        )
        if not impl_rows:
            continue
        xs = [row["seqlen"] for row in impl_rows]
        axes[0].plot(xs, [row["ms"] for row in impl_rows], marker="o", label=impl)
        axes[1].plot(xs, [row["tflops"] for row in impl_rows], marker="o", label=impl)

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(tick_values)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(True, which="major", linestyle="--", alpha=0.35)
        ax.set_xlabel("Sequence length")
        ax.legend()
    axes[0].set_ylabel("Latency (ms)")
    axes[1].set_ylabel("Throughput (TFLOPs/s)")

    if input_mode == "batched":
        subtitle = "batched, B*S fixed"
    else:
        subtitle = "true non-uniform varlen, total tokens fixed"
    fig.suptitle(
        f"SplitD D512 GQA {direction} {input_mode} {causal_label} ({subtitle})",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _plot_results(rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for direction in ("forward", "backward"):
        for input_mode in ("batched", "varlen"):
            for causal in (False, True):
                paths.append(
                    _plot_one(
                        rows,
                        direction=direction,
                        input_mode=input_mode,
                        causal=causal,
                        out_dir=out_dir,
                    )
                )
    return paths


def _ordered_impls(rows: list[dict[str, Any]]) -> list[str]:
    present = {row["impl"] for row in rows}
    ordered = [impl for impl in PLOT_ORDER if impl in present]
    ordered.extend(sorted(present - set(PLOT_ORDER)))
    return ordered


def _bar_panel(
    ax,
    rows: list[dict[str, Any]],
    seqlens: list[int],
    impls: list[str],
    metric: str,
    ylabel: str,
) -> None:
    x_positions = list(range(len(seqlens)))
    width = 0.5 if len(impls) <= 1 else min(0.8 / len(impls), 0.28)
    offset_start = -0.5 * width * (len(impls) - 1)

    for impl_idx, impl in enumerate(impls):
        values = []
        for seqlen in seqlens:
            row = next(
                (candidate for candidate in rows if candidate["impl"] == impl and candidate["seqlen"] == seqlen),
                None,
            )
            values.append(float("nan") if row is None else row[metric])
        offsets = [x + offset_start + impl_idx * width for x in x_positions]
        ax.bar(offsets, values, width=width, label=impl)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(seqlen) for seqlen in seqlens], rotation=30, ha="right")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    if impls:
        ax.legend()


def _plot_bar_one(
    rows: list[dict[str, Any]],
    *,
    direction: str,
    input_mode: str,
    causal: bool,
    out_dir: Path,
) -> Path:
    selected = [
        row
        for row in rows
        if row["direction"] == direction and row["input_mode"] == input_mode and row["causal"] == causal
    ]
    seqlens = sorted({row["seqlen"] for row in selected}) or DEFAULT_SEQLENS
    impls = _ordered_impls(selected)
    causal_label = "causal" if causal else "noncausal"
    filename = f"{direction}_{input_mode}_{causal_label}_bar.png"
    out_path = out_dir / filename

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    _bar_panel(axes[0], selected, seqlens, impls, "ms", "Latency (ms)")
    _bar_panel(axes[1], selected, seqlens, impls, "tflops", "Throughput (TFLOPs/s)")

    if input_mode == "batched":
        subtitle = "batched, B*S fixed"
    else:
        subtitle = "true non-uniform varlen, total tokens fixed"
    fig.suptitle(
        f"SplitD D512 GQA {direction} {input_mode} {causal_label} bars ({subtitle})",
        fontsize=12,
    )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _plot_bar_results(rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for direction in ("forward", "backward"):
        for input_mode in ("batched", "varlen"):
            for causal in (False, True):
                paths.append(
                    _plot_bar_one(
                        rows,
                        direction=direction,
                        input_mode=input_mode,
                        causal=causal,
                        out_dir=out_dir,
                    )
                )
    return paths


def _validate_args(args: argparse.Namespace) -> None:
    if args.total_tokens <= 0:
        raise ValueError("--total-tokens must be positive.")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative.")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive.")
    for seqlen in args.seqlens:
        _batch_size_for(seqlen, args.total_tokens)
    if args.total_tokens % TILE_SIZE != 0:
        raise ValueError(f"--total-tokens must be a multiple of {TILE_SIZE} for varlen packing.")


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    _require_sm90()
    _configure_torch_compile()

    os.environ.setdefault("MPLBACKEND", "Agg")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("SplitD D512 GQA fixed-token benchmark", flush=True)
    print(
        f"Config: total_tokens={args.total_tokens}, Hq={NHEADS_Q}, Hkv={NHEADS_KV}, "
        f"D={HEAD_DIM}, dtype={DTYPE_NAME}, warmup={args.warmup}, repeats={args.repeats}",
        flush=True,
    )
    print(f"Output directory: {args.out_dir}", flush=True)

    all_rows: list[dict[str, Any]] = []
    for causal in (False, True):
        for seqlen in args.seqlens:
            print(f"\n[forward batched] causal={causal} seqlen={seqlen}", flush=True)
            all_rows.extend(
                _benchmark_batched_forward(
                    seqlen,
                    causal,
                    args.total_tokens,
                    args.warmup,
                    args.repeats,
                    args.skip_baselines,
                )
            )
            print(f"[forward varlen] causal={causal} seqlen={seqlen}", flush=True)
            all_rows.extend(_benchmark_varlen_forward(seqlen, causal, args.total_tokens, args.warmup, args.repeats))
            print(f"[backward batched] causal={causal} seqlen={seqlen}", flush=True)
            all_rows.extend(
                _benchmark_batched_backward(
                    seqlen,
                    causal,
                    args.total_tokens,
                    args.warmup,
                    args.repeats,
                    args.skip_baselines,
                )
            )
            print(f"[backward varlen] causal={causal} seqlen={seqlen}", flush=True)
            all_rows.extend(_benchmark_varlen_backward(seqlen, causal, args.total_tokens, args.warmup, args.repeats))

    csv_path, json_path = _write_results(all_rows, args.out_dir)
    plot_paths = _plot_results(all_rows, args.out_dir)
    bar_plot_paths = _plot_bar_results(all_rows, args.out_dir)

    print("\nWrote results:", flush=True)
    print(f"  {csv_path}", flush=True)
    print(f"  {json_path}", flush=True)
    for path in plot_paths:
        print(f"  {path}", flush=True)
    for path in bar_plot_paths:
        print(f"  {path}", flush=True)


if __name__ == "__main__":
    main()
