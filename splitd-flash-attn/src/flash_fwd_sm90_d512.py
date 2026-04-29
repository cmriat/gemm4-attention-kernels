# Copyright (c) 2026 China Merchants Research Institute Of Advanced Technology
# SM90 (Hopper) forward pass for flash attention — SplitD for large head_dim (512+).
#
# Based on flash_fwd_sm90.py, with SplitD to handle head_dim > 256.
# SplitD splits head_dim into chunks so SMEM usage is O(chunk) instead of O(D).
#
# Design:
#   - QK-GEMM: accumulates across D/chunk_size D-chunks in registers → complete S
#   - Softmax: standard online softmax on complete S (unchanged)
#   - PV-GEMM: produces D/chunk_size output slices, each with its own accumulator
#   - Persistent Q: all Q chunks reside in SMEM for the duration of one work_tile
#   - K/V: 2-stage pipelined across D-chunks within each n_block
#   - Multi-group PV: D-slices processed in groups of max_live_pv_slices to fit
#     within 255-reg hardware limit; QK+softmax recomputed per group (identical)
#
# Simplifications vs flash_fwd_sm90_train_only.py:
#   - intra_wg_overlap = False (no K/V ping-pong)
#   - num_wg_mma = 1 (single warpgroup MMA, 256 regs/thread for R_D pressure)
#   - mma_pv_is_rs = True always (P in registers, no sP)
#   - No warp scheduler barriers
#   - No paged KV / learnable_sink / seqused / block_sparsity

from typing import Callable, Optional, Type
from functools import partial
import math

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.base_dsl.arch import Arch
from cutlass.base_dsl import BaseDSL

from quack import copy_utils
from quack import layout_utils
from quack import sm90_utils

from .cute_dsl_utils import assume_tensor_aligned
from . import utils
from .mask import AttentionMask
from .softmax import Softmax, apply_score_mod_inner
from .seqlen_info import SeqlenInfoQK
from .block_info import BlockInfo
from . import pipeline as pipeline_custom
from .pack_gqa import PackGQA, pack_gqa_layout, make_packgqa_tiled_tma_atom
from .named_barrier import NamedBarrierFwd
from quack.cute_dsl_utils import ParamsBase
from .tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
    SingleTileLPTScheduler,
    SingleTileVarlenScheduler,
)
from cutlass.cute import FastDivmodDivisor


class FlashAttentionForwardSm90TrainOnly:
    """SM90 forward kernel with SplitD for large head_dim (512+).

    SplitD splits head_dim into 64-wide chunks so that SMEM usage is
    independent of head_dim. Phase 2: Q pipelined (4-stage), K pipelined
    (2-stage) with delayed release, max_live=8 for zero QK recomputation.

    Compared to the standard SM90 forward:
    - Single warpgroup MMA (num_wg_mma=1) for register headroom
    - No intra-warpgroup overlap / scheduler barriers
    - mma_pv_is_rs always True (P stays in registers)
    """

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        is_local: bool = False,
        pack_gqa: bool = True,
        tile_m: int = 64,
        tile_n: int = 128,
        score_mod: Optional[cutlass.Constexpr] = None,
        mask_mod: Optional[cutlass.Constexpr] = None,
        has_aux_tensors: bool = False,
        kv_same: bool = False,
    ):
        self.dtype = dtype
        hdim_multiple_of = 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.tile_hdimv = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_causal = is_causal
        self.is_local = is_local
        self.pack_gqa = pack_gqa
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.score_mod = score_mod
        self.mask_mod = mask_mod
        self.qk_acc_dtype = Float32
        self.vec_size: cutlass.Constexpr = getattr(
            score_mod, "__vec_size__", 1 if cutlass.const_expr(has_aux_tensors) else 2
        )
        self.arch = BaseDSL._get_dsl().get_arch_enum()

        # ── SM90 SplitD specific ──
        self.mma_pv_is_rs = True
        self.buffer_align_bytes = 1024
        self.use_tma_KV = True
        self.cluster_shape_mn = (1, 1)
        assert self.arch >= Arch.sm_90 and self.arch <= Arch.sm_90a, "Only SM 9.x is supported"
        assert self.tile_m <= 64, f"SplitD requires tile_m <= 64 for num_wg_mma==1 (256 regs/thread), got {self.tile_m}"
        assert not self.pack_gqa or self.tile_m % self.qhead_per_kvhead == 0, (
            f"SplitD requires tile_m ({self.tile_m}) divisible by qhead_per_kvhead "
            f"({self.qhead_per_kvhead}) when pack_gqa=True. "
            f"Use pack_gqa=False for this head configuration."
        )

        # ── SplitD parameters ──
        self.tile_hdim_chunk = 64
        self.tile_hdimv_chunk = 64
        self.num_d_chunks = self.tile_hdim // self.tile_hdim_chunk
        self.num_d_slices = self.tile_hdimv // self.tile_hdimv_chunk

        # K-persistence: 4-stage K pipeline, last 4 chunks persist for PV Resident.
        self.num_stages_q = 3
        self.num_stages_k = 4

        # ── K/V mode ──
        self.kv_same = kv_same

        # ── Phase 2: max_live=8, zero QK recomputation ──
        self.max_live_pv_slices = 8
        self.num_pv_groups = (self.num_d_slices + self.max_live_pv_slices - 1) // self.max_live_pv_slices

        # K-persistence requires exactly 8 D-chunks/slices (D=512, chunk=64)
        assert self.num_d_chunks == 8 and self.num_d_slices == 8, (
            f"K-persistence requires num_d_chunks==num_d_slices==8 (D=512), got {self.num_d_chunks}/{self.num_d_slices}"
        )

    def _get_tiled_mma(self):
        # QK-GEMM: S[tile_m, tile_n] = Q[tile_m, chunk] @ K[chunk, tile_n]
        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, 1, 1),
            tiler_mn=(64, self.tile_n),
        )
        # PV-GEMM: O[tile_m, chunk] = P[tile_m, tile_n] @ V[tile_n, chunk]
        tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, 1, 1),
            tiler_mn=(64, self.tile_hdimv_chunk),
            a_source=warpgroup.OperandSource.RMEM,
        )
        return tiled_mma_qk, tiled_mma_pv

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sO_struct = [
            cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(layout)], self.buffer_align_bytes]
            for layout in (self.sQ_layout, self.sK_layout, self.sO_layout)
        ]
        # sO_spill: fp32 accumulator spill buffer for single-acc_O rotation
        sO_spill_struct = cute.struct.Align[
            cute.struct.MemRange[Float32, cute.cosize(self.sO_spill_layout)],
            self.buffer_align_bytes,
        ]
        mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages_q * 2]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages_k * 2]

        @cute.struct
        class SharedStorageKV:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            sQ: sQ_struct
            sK: sK_struct
            sO_spill: sO_spill_struct  # fp32 acc_O spill for single-acc_O rotation
            sO: sO_struct

        return SharedStorageKV

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        aux_tensors: Optional[list] = None,
        stream: cuda.CUstream = None,
    ):
        # Type check (inlined, SplitD has no mSeqUsedQ/K)
        if const_expr(not (mQ.element_type == mK.element_type == mV.element_type == mO.element_type)):
            raise TypeError("All tensors must have the same data type")
        if const_expr(mQ.element_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if const_expr(mLSE is not None and mLSE.element_type != Float32):
            raise TypeError("LSE tensor must be Float32")
        if const_expr(mCuSeqlensQ is not None and mCuSeqlensQ.element_type != Int32):
            raise TypeError("cu_seqlens_q tensor must be Int32")
        if const_expr(mCuSeqlensK is not None and mCuSeqlensK.element_type != Int32):
            raise TypeError("cu_seqlens_k tensor must be Int32")
        assert mQ.element_type == self.dtype
        self.varlen_q = mCuSeqlensQ is not None

        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        QO_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mQ, mO = [layout_utils.select(t, QO_layout_transpose) for t in (mQ, mO)]
        KV_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mK, mV = [layout_utils.select(t, KV_layout_transpose) for t in (mK, mV)]
        LSE_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mLSE = layout_utils.select(mLSE, LSE_layout_transpose) if const_expr(mLSE is not None) else None

        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_qk.size
        self.num_threads_per_warp_group = 128
        self.num_wg_mma = self.num_mma_threads // self.num_threads_per_warp_group
        assert self.num_wg_mma == 1, "SplitD requires num_wg_mma == 1"
        self.num_threads = self.num_threads_per_warp_group * 2
        self.num_producer_threads = 32
        self.num_Q_load_threads = self.num_threads_per_warp_group
        self.num_epilogue_threads = self.num_mma_threads
        self.num_mma_regs = 256
        self.num_producer_regs = 56

        self.use_tma_Q = self.arch >= Arch.sm_90 and not (self.pack_gqa and self.tile_m % self.qhead_per_kvhead != 0)
        self.use_tma_O = self.use_tma_Q

        # ═══ SMEM layouts — chunk-sized with staging ═══
        # sQ: 4-stage pipeline (Phase 2: reloaded per n_block, not persistent)
        self.sQ_layout = sm90_utils.make_smem_layout(
            mQ.element_type,
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, self.tile_hdim_chunk),
            stage=self.num_stages_q,
        )
        # sK: 2-stage pipeline with delayed release for TMA-WGMMA overlap
        self.sK_layout = sm90_utils.make_smem_layout(
            mK.element_type,
            LayoutEnum.ROW_MAJOR,
            (self.tile_n, self.tile_hdim_chunk),
            stage=self.num_stages_k,
        )
        # sO_spill: fp32 accumulator spill buffer for single-acc_O rotation
        # Each slot = tile_m × tile_hdimv_chunk × 4B = 16KB
        self.sO_spill_layout = sm90_utils.make_smem_layout(
            Float32,
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, self.tile_hdimv_chunk),
            stage=self.max_live_pv_slices,
        )
        # sO: per-slice epilogue write
        self.sO_layout = sm90_utils.make_smem_layout(
            mO.element_type,
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, self.tile_hdimv_chunk),
        )
        self.sP_layout = None

        SharedStorage = self._get_shared_storage_cls()

        mQ_og, mO_og = mQ, mO
        if const_expr(self.pack_gqa):
            nheads_kv = mK.shape[2]
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1)

        gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_KV = cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_O = cpasync.CopyBulkTensorTileS2GOp()

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [
                ("Q_chunk", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
            ]
        }

        make_tiled_tma_atom_fn = (
            partial(make_packgqa_tiled_tma_atom, qhead_per_kvhead=self.qhead_per_kvhead, head_idx=2)
            if const_expr(self.pack_gqa)
            else cpasync.make_tiled_tma_atom
        )

        tma_atom_Q, tma_tensor_Q = None, None
        if const_expr(self.use_tma_Q):
            tma_atom_Q, tma_tensor_Q = make_tiled_tma_atom_fn(
                gmem_tiled_copy_Q,
                mQ_og if const_expr(self.pack_gqa) else mQ,
                cute.select(self.sQ_layout, mode=[0, 1]),
                (self.tile_m, self.tile_hdim_chunk),
            )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_KV,
            mK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim_chunk),
            1,
        )
        # V TMA atom: when kv_same=False, V is loaded from mV into sK (time-multiplexed).
        tma_atom_V, tma_tensor_V = None, None
        if const_expr(not self.kv_same):
            tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mV,
                cute.select(self.sK_layout, mode=[0, 1]),  # V stored in sK buffer
                (self.tile_n, self.tile_hdimv_chunk),
                1,
            )
        tma_atom_O, tma_tensor_O = None, None
        if const_expr(self.use_tma_O):
            mO_tma = mO_og if const_expr(self.pack_gqa) else mO
            if const_expr(self.varlen_q):
                mO_tma = copy_utils.create_ragged_tensor_for_tma(mO_tma, ragged_dim=0, ptr_shift=True)
            tma_atom_O, tma_tensor_O = make_tiled_tma_atom_fn(
                gmem_tiled_copy_O,
                mO_tma,
                self.sO_layout,
                (self.tile_m, self.tile_hdimv_chunk),
            )

        if const_expr(mCuSeqlensQ is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = (
                SingleTileScheduler if const_expr(not self.is_causal or self.is_local) else SingleTileLPTScheduler
            )
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]) if const_expr(mCuSeqlensQ is None) else cute.size(mCuSeqlensQ.shape[0] - 1),
            1,
            cute.size(mK.shape[0]),
            mQ.shape[1],
            mV.shape[1],
            total_q=cute.size(mQ.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.tile_m, self.tile_n),
            mCuSeqlensQ=mCuSeqlensQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            element_size=self.dtype.width // 8,
            lpt=self.is_causal or self.is_local,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(softmax_scale, self.score_mod)
        window_size_left = Int32(window_size_left) if window_size_left is not None else None
        window_size_right = Int32(window_size_right) if window_size_right is not None else None
        fastdiv_mods = utils.compute_fastdiv_mods(mQ, mK, self.qhead_per_kvhead, self.pack_gqa, aux_tensors, None)

        self.kernel(
            tma_tensor_Q if const_expr(self.use_tma_Q) else mQ,
            tma_tensor_K,
            tma_tensor_V if const_expr(not self.kv_same) else None,
            tma_tensor_O if const_expr(self.use_tma_O) else mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            window_size_left,
            window_size_right,
            self.sQ_layout,
            self.sK_layout,
            self.sO_spill_layout,
            self.sO_layout,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            aux_tensors,
            fastdiv_mods,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: Optional[cute.Tensor],
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sO_spill_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        aux_tensors=Optional[list[cute.Tensor]],
        fastdiv_mods=None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        mbar_ptr_Q = storage.mbar_ptr_Q.data_ptr()
        ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        tma_warp = ThreadCooperativeGroup(1)
        load_threads = ThreadCooperativeGroup(self.num_threads_per_warp_group)
        mma_warps = ThreadCooperativeGroup(self.num_mma_threads // cute.arch.WARP_SIZE)

        # pipeline_q: 3-stage per-chunk pipeline (SplitD guarantees use_tma_Q=True)
        pipeline_q = pipeline_custom.PipelineTmaAsync.create(
            barrier_storage=mbar_ptr_Q,
            num_stages=self.num_stages_q,
            producer_group=tma_warp,
            consumer_group=mma_warps,
            tx_count=self.tma_copy_bytes["Q_chunk"],
            defer_sync=True,
        )

        # pipeline_k: 4-stage pipeline with delayed release for TMA-WGMMA overlap.
        # kv_same=False: sK time-multiplexed (QK then PV), all 8 V slices loaded.
        pipeline_k = pipeline_custom.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_K.data_ptr(),
            num_stages=self.num_stages_k,
            producer_group=tma_warp,
            consumer_group=mma_warps,
            tx_count=self.tma_copy_bytes["K"],
            defer_sync=True,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # ═══ SMEM tensors ═══
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sKVt = layout_utils.transpose_view(sK)
        # sO_spill: fp32 accumulator spill buffer for single-acc_O rotation
        sO_spill = storage.sO_spill.get_tensor(
            sO_spill_layout.outer,
            swizzle=sO_spill_layout.inner,
            dtype=Float32,
        )
        sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype)

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            False,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        if warp_idx < 4:  # Producer warpgroup
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            self.load(
                mQ,
                mK,
                mV,
                sQ,
                sK,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_k,
                pipeline_q,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
            )
        else:  # Consumer warpgroup
            cute.arch.setmaxregister_increase(self.num_mma_regs)
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                mO,
                mLSE,
                sQ,
                sK,
                sKVt,
                sO,
                sO_spill,
                pipeline_k,
                pipeline_q,
                tma_atom_O,
                tidx,
                softmax_scale_log2,
                softmax_scale,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                aux_tensors,
                fastdiv_mods,
            )

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: Optional[cute.CopyAtom],
        pipeline_k: pipeline.PipelineAsync,
        pipeline_q: pipeline.PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tidx, _, _ = cute.arch.thread_idx()
        is_load_warp = warp_idx_in_wg == 0 or const_expr(not self.use_tma_Q)
        is_kv_load_warp = warp_idx_in_wg == 0

        if is_load_warp:
            q_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages_q)
            k_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages_k)
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()

            while work_tile.is_valid_tile:
                m_block, head_idx, batch_idx, _ = work_tile.tile_idx
                seqlen = SeqlenInfoCls(batch_idx)
                mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
                head_idx_kv = head_idx // self.qhead_per_kvhead if const_expr(not self.pack_gqa) else head_idx
                mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx_kv]
                if const_expr(not self.kv_same):
                    mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx_kv]

                n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)

                if is_kv_load_warp:
                    for i_n in cutlass.range(n_block_max - n_block_min, unroll=1):
                        n_block = n_block_max - 1 - i_n  # inverse order

                        # ── Phase A: 8 interleaved (Q_d, K_d) pairs for QK-GEMM ──
                        # Q reloaded per n_block through 4-stage pipeline.
                        # K chunks cycle through 2-stage pipeline.
                        for d in cutlass.range_constexpr(self.num_d_chunks):
                            # Load Q chunk d
                            if const_expr(self.use_tma_Q):
                                gQ_d = cute.local_tile(
                                    mQ_cur,
                                    (self.tile_m, self.tile_hdim_chunk),
                                    (None, d),
                                )
                                load_Q_d, _, _ = copy_utils.tma_get_copy_fn(
                                    tma_atom_Q,
                                    0,
                                    cute.make_layout(1),
                                    gQ_d,
                                    sQ,
                                )
                                pipeline_q.producer_acquire(q_producer_state)
                                load_Q_d(
                                    src_idx=m_block,
                                    dst_idx=q_producer_state.index,
                                    tma_bar_ptr=pipeline_q.producer_get_barrier(q_producer_state),
                                )
                                pipeline_q.producer_commit(q_producer_state)
                                q_producer_state.advance()

                            # Load K chunk d
                            gK_d = cute.local_tile(
                                mK_cur,
                                (self.tile_n, self.tile_hdim_chunk),
                                (None, d),
                            )
                            tma_fn_K_d, _, _ = copy_utils.tma_get_copy_fn(
                                tma_atom_K,
                                0,
                                cute.make_layout(1),
                                gK_d,
                                sK,
                            )
                            pipeline_k.producer_acquire(k_producer_state)
                            tma_fn_K_d(
                                src_idx=n_block,
                                dst_idx=k_producer_state.index,
                                tma_bar_ptr=pipeline_k.producer_get_barrier(k_producer_state),
                            )
                            pipeline_k.producer_commit(k_producer_state)
                            k_producer_state.advance()

                        # ── Phase B: V slice loading via pipeline_k ──
                        if const_expr(self.kv_same):
                            for j in cutlass.range_constexpr(self.num_stages_k):
                                gKV_j = cute.local_tile(
                                    mK_cur,
                                    (self.tile_n, self.tile_hdim_chunk),
                                    (None, j),
                                )
                                tma_fn_KV_j, _, _ = copy_utils.tma_get_copy_fn(
                                    tma_atom_K,
                                    0,
                                    cute.make_layout(1),
                                    gKV_j,
                                    sK,
                                )
                                pipeline_k.producer_acquire(k_producer_state)
                                tma_fn_KV_j(
                                    src_idx=n_block,
                                    dst_idx=k_producer_state.index,
                                    tma_bar_ptr=pipeline_k.producer_get_barrier(k_producer_state),
                                )
                                pipeline_k.producer_commit(k_producer_state)
                                k_producer_state.advance()
                        else:
                            # K≠V: load ALL 8 V slices from mV into sK
                            # (sK time-multiplexed, K fully consumed by QK).
                            for j in cutlass.range_constexpr(self.num_d_slices):
                                gV_j = cute.local_tile(
                                    mV_cur,
                                    (self.tile_n, self.tile_hdimv_chunk),
                                    (None, j),
                                )
                                tma_fn_V_j, _, _ = copy_utils.tma_get_copy_fn(
                                    tma_atom_V,
                                    0,
                                    cute.make_layout(1),
                                    gV_j,
                                    sK,
                                )
                                pipeline_k.producer_acquire(k_producer_state)
                                tma_fn_V_j(
                                    src_idx=n_block,
                                    dst_idx=k_producer_state.index,
                                    tma_bar_ptr=pipeline_k.producer_get_barrier(k_producer_state),
                                )
                                pipeline_k.producer_commit(k_producer_state)
                                k_producer_state.advance()

                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

            # Producer tail to avoid early exit of blocks in cluster.
            if is_kv_load_warp:
                pipeline_k.producer_tail(k_producer_state)
            if const_expr(self.use_tma_Q):
                if warp_idx_in_wg == 0:
                    pipeline_q.producer_tail(q_producer_state)

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sKVt: cute.Tensor,
        sO: cute.Tensor,
        sO_spill: cute.Tensor,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_q: pipeline.PipelineAsync,
        tma_atom_O: cute.CopyAtom,
        tidx: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        aux_tensors: Optional[list],
        fastdiv_mods=None,
    ):
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        warp_group_thread_layout = cute.make_layout(self.num_wg_mma, stride=self.num_threads_per_warp_group)
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx))

        # ═══ QK-GEMM fragments: K-dim = tile_hdim_chunk ═══
        _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
            wg_mma_qk,
            (self.tile_m, self.tile_n, self.tile_hdim_chunk),
            sQ,
            sK,
        )

        # ═══ PV-GEMM fragments: K=V, B-operand = transpose_view(sK) ═══
        # tOrKVt mode-3 = num_stages_k (pipeline stages, transposed view)
        _, tOrP, tOrKVt = sm90_utils.partition_fragment_ABC(
            wg_mma_pv,
            (self.tile_m, self.tile_hdimv_chunk, self.tile_n),
            None,
            sKVt,
        )

        q_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            self.num_stages_q,
        )
        k_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            self.num_stages_k,
        )

        # ═══ SMEM spill copy setup (fp32 acc_O ↔ sO_spill) ═══
        smem_copy_atom_fp32 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            Float32,
            num_bits_per_copy=2 * Float32.width,
        )
        smem_thr_copy_spill = cute.make_tiled_copy_C(
            smem_copy_atom_fp32,
            tiled_mma_pv,
        ).get_slice(tidx)
        # Partition sO_spill: mode-3 = staging dimension (max_live slots)
        # retile(acc_O) gives the register-side partition
        # partition_D(sO_spill) gives the SMEM-side partition with mode-3 = slot index
        taccOsO_spill = smem_thr_copy_spill.partition_D(sO_spill)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()

        # softmax num_rows from QK acc shape
        qk_acc_shape = tiled_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
        softmax_num_rows = cute.size(qk_acc_shape[0][0]) * cute.size(qk_acc_shape[1])

        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            if cutlass.const_expr(fastdiv_mods is not None):
                recompute_q = cutlass.const_expr(aux_tensors is not None and seqlen.has_cu_seqlens_q)
                recompute_k = cutlass.const_expr(aux_tensors is not None and seqlen.has_cu_seqlens_k)
                seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                fastdiv_mods = (
                    seqlen_q_divmod if not recompute_q else FastDivmodDivisor(seqlen.seqlen_q),
                    seqlen_k_divmod if not recompute_k else FastDivmodDivisor(seqlen.seqlen_k),
                )

            mask = AttentionMaskCls(seqlen)
            mask_fn = partial(
                mask.apply_mask,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=m_block,
                thr_mma=thr_mma_qk,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                aux_tensors=aux_tensors,
                fastdiv_mods=fastdiv_mods,
            )
            score_mod_fn = None
            if const_expr(self.score_mod is not None):
                score_mod_fn = partial(
                    self.apply_score_mod,
                    thr_mma_qk,
                    batch_idx,
                    head_idx,
                    m_block,
                    softmax_scale=softmax_scale,
                    aux_tensors=aux_tensors,
                    fastdiv_mods=fastdiv_mods,
                )

            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)

            # ══════════════════════════════════════════════════════════
            # Phase 2: single group, all 8 slices, ZERO QK recomputation
            # ══════════════════════════════════════════════════════════
            acc_O = cute.make_rmem_tensor(
                tiled_mma_pv.partition_shape_C((self.tile_m, self.tile_hdimv_chunk)),
                Float32,
            )

            softmax = Softmax.create(
                softmax_scale_log2,
                num_rows=softmax_num_rows,
                softmax_scale=softmax_scale,
            )

            # Pre-allocate QK accumulator (reused across n_blocks)
            acc_S = cute.make_rmem_tensor(
                tiled_mma_qk.partition_shape_C((self.tile_m, self.tile_n)),
                Float32,
            )

            mma_one_n_block = partial(
                self.mma_one_n_block_splitd,
                acc_S=acc_S,
                acc_O=acc_O,
                taccOsO_spill=taccOsO_spill,
                smem_thr_copy_spill=smem_thr_copy_spill,
                smem_copy_atom_fp32=smem_copy_atom_fp32,
                tSrQ=tSrQ,
                tSrK=tSrK,
                tOrP=tOrP,
                tOrKVt=tOrKVt,
                tiled_mma_qk=tiled_mma_qk,
                tiled_mma_pv=tiled_mma_pv,
                pipeline_k=pipeline_k,
                pipeline_q=pipeline_q,
                softmax=softmax,
                seqlen=seqlen,
                score_mod_fn=score_mod_fn,
                mask_fn=mask_fn,
                slice_start=0,
            )

            # ── First n-block: mask_seqlen=True, is_first=True ──
            q_consumer_state, k_consumer_state = mma_one_n_block(
                n_block=n_block_max - 1,
                q_consumer_state=q_consumer_state,
                k_consumer_state=k_consumer_state,
                mask_seqlen=True,
                is_first_n_block=True,
            )

            # ── Remaining n-blocks ──
            for i_n in cutlass.range(n_block_max - 1 - n_block_min, unroll=1):
                n_block = n_block_max - 2 - i_n
                q_consumer_state, k_consumer_state = mma_one_n_block(
                    n_block=n_block,
                    q_consumer_state=q_consumer_state,
                    k_consumer_state=k_consumer_state,
                    mask_seqlen=False,
                    is_first_n_block=False,
                )

            # ── Finalize softmax + epilogue (all 8 slices) ──
            row_scale = softmax.finalize(sink_val=None)
            taccOrO_fin = smem_thr_copy_spill.retile(acc_O)
            for _j in cutlass.range_constexpr(self.max_live_pv_slices):
                cute.autovec_copy(taccOsO_spill[None, None, None, _j], taccOrO_fin)
                softmax.rescale_O(acc_O, row_scale)
                self.epilogue_slice(
                    acc_O,
                    softmax.row_sum,
                    mO,
                    mLSE,
                    sO,
                    seqlen,
                    tma_atom_O,
                    tiled_mma_pv,
                    tidx,
                    m_block,
                    head_idx,
                    batch_idx,
                    d_slice_idx=_j,
                    write_lse=(_j == 0),
                )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def mma_one_n_block_splitd(
        self,
        n_block: Int32,
        acc_S: cute.Tensor,
        acc_O: cute.Tensor,
        taccOsO_spill: cute.Tensor,
        smem_thr_copy_spill: cute.TiledCopy,
        smem_copy_atom_fp32: cute.CopyAtom,
        tSrQ: cute.Tensor,
        tSrK: cute.Tensor,
        tOrP: cute.Tensor,
        tOrKVt: cute.Tensor,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_q: pipeline.PipelineAsync,
        q_consumer_state: pipeline.PipelineState,
        k_consumer_state: pipeline.PipelineState,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        slice_start: cutlass.Constexpr[int] = 0,
        score_mod_fn: Optional[Callable] = None,
        mask_fn: Optional[Callable] = None,
        mask_seqlen: cutlass.Constexpr[bool] = False,
        is_first_n_block: cutlass.Constexpr[bool] = False,
    ):
        # ── QK-GEMM with Q (3-stage) + K (4-stage) pipeline + delayed release ──
        # Q and K consumed per D-chunk. Delayed release (wait_group(2)):
        #   - Q released 2 iterations behind (all 8 Q stages released)
        #   - kv_same=False: ALL K chunks released (no persistence, sK freed for V).
        k_release_state = k_consumer_state.clone()
        q_release_state = q_consumer_state.clone()

        for d in cutlass.range_constexpr(self.num_d_chunks):
            pipeline_q.consumer_wait(
                q_consumer_state,
                pipeline_q.consumer_try_wait(q_consumer_state),
            )
            pipeline_k.consumer_wait(
                k_consumer_state,
                pipeline_k.consumer_try_wait(k_consumer_state),
            )

            if d < self.num_d_chunks - 1:
                # Non-last chunk: issue WGMMA async, then delayed release
                sm90_utils.gemm_w_idx(
                    tiled_mma_qk,
                    acc_S,
                    tSrQ,
                    tSrK,
                    zero_init=(d == 0),
                    A_idx=q_consumer_state.index,
                    B_idx=k_consumer_state.index,
                    wg_wait=-1,
                )
                if d >= 2:
                    # wait_group(2): at most 2 outstanding → group d-2 done
                    warpgroup.wait_group(2)
                    # Q: always release (all 8 Q stages freed)
                    pipeline_q.consumer_release(q_release_state)
                    q_release_state.advance()
                    # K release: d is from range_constexpr → condition is compile-time.
                    if const_expr(self.kv_same):
                        # K=V: only release chunks 0..3 (d-2 < num_stages_k).
                        if d - 2 < self.num_stages_k:
                            pipeline_k.consumer_release(k_release_state)
                            k_release_state.advance()
                    else:
                        # K≠V: release ALL K chunks (no persistence)
                        pipeline_k.consumer_release(k_release_state)
                        k_release_state.advance()
            else:
                # Last chunk (d=7): drain all groups for softmax
                sm90_utils.gemm_w_idx(
                    tiled_mma_qk,
                    acc_S,
                    tSrQ,
                    tSrK,
                    zero_init=(d == 0),
                    A_idx=q_consumer_state.index,
                    B_idx=k_consumer_state.index,
                    wg_wait=0,
                )
                # Q: release Q_{d-2} (all groups complete after wg_wait=0)
                pipeline_q.consumer_release(q_release_state)
                q_release_state.advance()
                if const_expr(not self.kv_same):
                    # K≠V: release K chunk d-2=5 (no persistence)
                    pipeline_k.consumer_release(k_release_state)
                    k_release_state.advance()

            q_consumer_state.advance()
            k_consumer_state.advance()

        # Trailing Q releases: release Q_6, Q_7
        for _ in cutlass.range_constexpr(2):
            pipeline_q.consumer_release(q_release_state)
            q_release_state.advance()
        if const_expr(not self.kv_same):
            # K≠V: trailing K releases for chunks 6, 7 (no persistence)
            for _ in cutlass.range_constexpr(2):
                pipeline_k.consumer_release(k_release_state)
                k_release_state.advance()
        # acc_S now holds complete S = Q @ K^T
        # kv_same=False: all K stages released, sK free for V loading

        if const_expr(score_mod_fn is not None):
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
        partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=mask_seqlen)(
            acc_S=acc_S,
            n_block=n_block,
        )
        row_scale = softmax.online_softmax(
            acc_S,
            is_first=is_first_n_block,
            check_inf=True,
        )

        # acc_S → P (convert to fp16 for PV-GEMM A operand)
        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP.store(tOrP_acc.load().to(self.dtype))

        # ── Phase 3: PV-GEMM ──
        # Two acc_O registers (A=acc_O, B=acc_O_B) alternate in pairs:
        # while WGMMA writes A async, B's restore+rescale+dispatch overlaps.
        acc_O_B = cute.make_rmem_tensor(
            tiled_mma_pv.partition_shape_C((self.tile_m, self.tile_hdimv_chunk)),
            Float32,
        )
        taccOrO_A = smem_thr_copy_spill.retile(acc_O)
        taccOrO_B = smem_thr_copy_spill.retile(acc_O_B)

        if const_expr(self.kv_same):
            # ═══ K=V: Resident (slices 4..7) + Reload (slices 0..3) ═══
            num_resident_pairs = self.num_stages_k // 2
            for j_pair in cutlass.range(num_resident_pairs, unroll=1):
                j_a = self.num_stages_k + j_pair * 2  # sO_spill index: 4, 6
                j_b = self.num_stages_k + j_pair * 2 + 1  # sO_spill index: 5, 7
                stage_a = j_pair * 2  # stage: 0, 2 (runtime Int32)
                stage_b = j_pair * 2 + 1  # stage: 1, 3 (runtime Int32)

                # ── Slice j_a: restore A + issue WGMMA A (async, no wait) ──
                if not is_first_n_block:
                    cute.autovec_copy(taccOsO_spill[None, None, None, j_a], taccOrO_A)
                    softmax.rescale_O(acc_O, row_scale)

                sm90_utils.gemm_w_idx(
                    tiled_mma_pv,
                    acc_O,
                    tOrP,
                    tOrKVt,
                    zero_init=is_first_n_block,
                    B_idx=stage_a,
                    wg_wait=-1,
                )

                # ── Slice j_b: restore B + issue WGMMA B (async, no wait) ──
                if not is_first_n_block:
                    cute.autovec_copy(taccOsO_spill[None, None, None, j_b], taccOrO_B)
                    softmax.rescale_O(acc_O_B, row_scale)

                sm90_utils.gemm_w_idx(
                    tiled_mma_pv,
                    acc_O_B,
                    tOrP,
                    tOrKVt,
                    zero_init=is_first_n_block,
                    B_idx=stage_b,
                    wg_wait=-1,
                )

                # ── Drain A: wait_group(1) → A done, release persistent stage ──
                warpgroup.wait_group(1)
                pipeline_k.consumer_release(k_release_state)
                k_release_state.advance()
                cute.copy(smem_copy_atom_fp32, taccOrO_A, taccOsO_spill[None, None, None, j_a])

                # ── Drain B: wait_group(0) → all done, release persistent stage ──
                warpgroup.wait_group(0)
                pipeline_k.consumer_release(k_release_state)
                k_release_state.advance()
                cute.copy(smem_copy_atom_fp32, taccOrO_B, taccOsO_spill[None, None, None, j_b])

            # Sub-phase B (Reload): slices 0..3 loaded by producer.
            # Standard consumer_wait + double-buffered WGMMA.
            num_reload_pairs = self.num_stages_k // 2
            for j_pair in cutlass.range(num_reload_pairs, unroll=1):
                j_a = j_pair * 2  # sO_spill index: 0, 2
                j_b = j_pair * 2 + 1  # sO_spill index: 1, 3

                # ── Slice j_a: restore A + wait + issue WGMMA A (async) ──
                if not is_first_n_block:
                    cute.autovec_copy(taccOsO_spill[None, None, None, j_a], taccOrO_A)
                    softmax.rescale_O(acc_O, row_scale)

                pipeline_k.consumer_wait(
                    k_consumer_state,
                    pipeline_k.consumer_try_wait(k_consumer_state),
                )
                sm90_utils.gemm_w_idx(
                    tiled_mma_pv,
                    acc_O,
                    tOrP,
                    tOrKVt,
                    zero_init=is_first_n_block,
                    B_idx=k_consumer_state.index,
                    wg_wait=-1,
                )
                k_state_a = k_consumer_state.clone()
                k_consumer_state.advance()

                # ── Slice j_b: restore B + wait + issue WGMMA B (async) ──
                if not is_first_n_block:
                    cute.autovec_copy(taccOsO_spill[None, None, None, j_b], taccOrO_B)
                    softmax.rescale_O(acc_O_B, row_scale)

                pipeline_k.consumer_wait(
                    k_consumer_state,
                    pipeline_k.consumer_try_wait(k_consumer_state),
                )
                sm90_utils.gemm_w_idx(
                    tiled_mma_pv,
                    acc_O_B,
                    tOrP,
                    tOrKVt,
                    zero_init=is_first_n_block,
                    B_idx=k_consumer_state.index,
                    wg_wait=-1,
                )

                # ── Drain A: wait_group(1) → A done, B still in flight ──
                warpgroup.wait_group(1)
                pipeline_k.consumer_release(k_state_a)
                cute.copy(smem_copy_atom_fp32, taccOrO_A, taccOsO_spill[None, None, None, j_a])

                # ── Drain B: wait_group(0) → all done ──
                warpgroup.wait_group(0)
                pipeline_k.consumer_release(k_consumer_state)
                k_consumer_state.advance()
                cute.copy(smem_copy_atom_fp32, taccOrO_B, taccOsO_spill[None, None, None, j_b])
        else:
            # ═══ K≠V: Uniform loop, all 8 V slices via consumer_wait ═══
            # All K stages released after QK. Producer loads 8 V slices
            # into sK (time-multiplexed). Double-buffered WGMMA in pairs.
            num_v_pairs = self.num_d_slices // 2  # = 4 pairs = 8 slices
            for j_pair in cutlass.range(num_v_pairs, unroll=1):
                j_a = j_pair * 2  # sO_spill index: 0, 2, 4, 6
                j_b = j_pair * 2 + 1  # sO_spill index: 1, 3, 5, 7

                # ── Slice j_a: restore A + wait V + issue WGMMA A (async) ──
                if not is_first_n_block:
                    cute.autovec_copy(taccOsO_spill[None, None, None, j_a], taccOrO_A)
                    softmax.rescale_O(acc_O, row_scale)

                pipeline_k.consumer_wait(
                    k_consumer_state,
                    pipeline_k.consumer_try_wait(k_consumer_state),
                )
                sm90_utils.gemm_w_idx(
                    tiled_mma_pv,
                    acc_O,
                    tOrP,
                    tOrKVt,
                    zero_init=is_first_n_block,
                    B_idx=k_consumer_state.index,
                    wg_wait=-1,
                )
                k_state_a = k_consumer_state.clone()
                k_consumer_state.advance()

                # ── Slice j_b: restore B + wait V + issue WGMMA B (async) ──
                if not is_first_n_block:
                    cute.autovec_copy(taccOsO_spill[None, None, None, j_b], taccOrO_B)
                    softmax.rescale_O(acc_O_B, row_scale)

                pipeline_k.consumer_wait(
                    k_consumer_state,
                    pipeline_k.consumer_try_wait(k_consumer_state),
                )
                sm90_utils.gemm_w_idx(
                    tiled_mma_pv,
                    acc_O_B,
                    tOrP,
                    tOrKVt,
                    zero_init=is_first_n_block,
                    B_idx=k_consumer_state.index,
                    wg_wait=-1,
                )

                # ── Drain A: wait_group(1) → A done, release V stage ──
                warpgroup.wait_group(1)
                pipeline_k.consumer_release(k_state_a)
                cute.copy(smem_copy_atom_fp32, taccOrO_A, taccOsO_spill[None, None, None, j_a])

                # ── Drain B: wait_group(0) → all done, release V stage ──
                warpgroup.wait_group(0)
                pipeline_k.consumer_release(k_consumer_state)
                k_consumer_state.advance()
                cute.copy(smem_copy_atom_fp32, taccOrO_B, taccOsO_spill[None, None, None, j_b])

        return q_consumer_state, k_consumer_state

    @cute.jit
    def epilogue_slice(
        self,
        acc_O: cute.Tensor,
        lse: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sO: cute.Tensor,
        seqlen: SeqlenInfoQK,
        tma_atom_O: cute.CopyAtom,
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        d_slice_idx: cutlass.Constexpr[int] = 0,
        write_lse: cutlass.Constexpr[bool] = False,
    ):
        rO = cute.make_fragment_like(acc_O, self.dtype)
        rO.store(acc_O.load().to(self.dtype))

        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue),
            number_of_threads=self.num_epilogue_threads,
        )
        smem_copy_atom_O = utils.get_smem_store_atom(
            self.arch.major * 10 + self.arch.minor,
            self.dtype,
        )
        smem_thr_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma).get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

        # Write LSE once (first slice only)
        if const_expr(write_lse and mLSE is not None):
            cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv_chunk))
            mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2)[None, head_idx]
            if const_expr(not self.pack_gqa):
                gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
                gLSE_expanded_layout = cute.append(gLSE.layout, cute.make_layout((self.tile_hdimv_chunk,), stride=(0,)))
                gLSE_expanded = cute.make_tensor(gLSE.iterator, gLSE_expanded_layout)
                thr_mma = tiled_mma.get_slice(tidx)
                taccOgLSE = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(gLSE_expanded))
                assert cute.size(taccOgLSE, mode=[0]) == cute.size(lse)
                taccOcO = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cO))
                t0accOcO = layout_utils.reshape_acc_to_mn(tiled_mma.get_slice(0).partition_C(cO))
                if taccOcO[0][1] == 0:
                    for m in cutlass.range(cute.size(taccOgLSE.shape[1]), unroll_full=True):
                        if t0accOcO[m, 0][0] < seqlen.seqlen_q - m_block * self.tile_m - taccOcO[0][0]:
                            taccOgLSE[m, 0] = lse[m]
            else:
                pack_gqa = PackGQA(
                    self.tile_m,
                    self.tile_hdimv_chunk,
                    self.check_hdim_v_oob,
                    self.qhead_per_kvhead,
                )
                pack_gqa.store_LSE(mLSE_cur, lse, tiled_mma, tidx, m_block, seqlen.seqlen_q)

        # sO → gO (global memory at D-slice offset)
        ragged = seqlen.has_cu_seqlens_q
        mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3, ragged=ragged)[None, None, head_idx]

        cute.arch.fence_view_async_shared()
        cute.arch.barrier_arrive(
            barrier_id=int(NamedBarrierFwd.Epilogue),
            number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE,
        )
        gO = cute.local_tile(
            mO_cur,
            (self.tile_m, self.tile_hdimv_chunk),
            (m_block, d_slice_idx),
        )
        store_O, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_O,
            0,
            cute.make_layout(1),
            sO,
            gO,
            single_stage=True,
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 4:
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE,
            )
            store_O()
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def apply_score_mod(
        self,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        acc_S,
        n_block,
        softmax_scale,
        seqlen,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=None,
    ):
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        cS = cute.domain_offset((m_block * self.tile_m, n_block * self.tile_n), cS)
        tScS = thr_mma_qk.partition_C(cS)
        apply_score_mod_inner(
            acc_S,
            tScS,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax_scale,
            self.vec_size,
            self.qk_acc_dtype,
            aux_tensors,
            fastdiv_mods,
            seqlen_info=seqlen,
            constant_q_idx=None,
            qhead_per_kvhead=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
