# Copyright (c) 2026 China Merchants Research Institute Of Advanced Technology
# SM90 Backward dKdV Kernel with 4-Pass D-Split for head_dim=512.
# Correctness baseline — Plan A from SplitD_origin_plan1.md.
#
# Architecture:
#   - 2 WGs: 1 producer (TMA, 128 threads), 1 consumer (MMA, 128 threads)
#   - tile_m=64, tile_n=64, d_chunk=128, num_d_passes=4
#   - K/V persistent SMEM: pre-loaded once per d_pass (4 K + 4 V chunks)
#   - Q/dO streaming via pipeline_A (3 stages)
#   - Per d_pass, 6 phases per (n_block, m_block):
#       Phase 1: S = Q @ K^T (4 × d_inner=128 reduction, K from persistent SMEM)
#       Phase 2: P = exp2(S * scale_log2 - LSE)
#       Phase 3: dP = dO @ V^T (4 × d_inner=128 reduction, V from persistent SMEM)
#       Phase 4: dS = P * (dP - dPsum)
#       Phase 5: dV += P^T @ dO_d_pass  (via SMEM)
#       Phase 6: dK += dS^T @ Q_d_pass  (via SMEM)
#   - mma_dkv_is_rs = False (P/dS through SMEM for simplicity)

import math
from typing import Callable, Optional, Type
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute import FastDivmodDivisor
from cutlass import Float32, Int32, Boolean, const_expr
from cutlass.utils import LayoutEnum

from quack import copy_utils
from quack import layout_utils
from quack import sm90_utils
from quack.sm90_utils import gemm_w_idx

from .cute_dsl_utils import assume_tensor_aligned
from . import utils
from .mask import AttentionMask
from .seqlen_info import SeqlenInfoQK
from .block_info import BlockInfo
from . import pipeline
from quack.cute_dsl_utils import ParamsBase
from .tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
    SingleTileVarlenScheduler,
)
from .named_barrier import NamedBarrierBwd


class FlashBwdDKDV_SplitD_Sm90:
    """SM90 backward dKdV kernel with 4-Pass D-Split for large head_dim.

    Computes only dK and dV (no dQ). dQ is handled by a separate kernel.
    Each CTA has 2 warp groups: 1 TMA producer + 1 MMA consumer.
    """

    arch = 90

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        is_causal: bool = False,
        qhead_per_kvhead: int = 1,
        tile_m: int = 64,
        tile_n: int = 64,
    ):
        self.dtype = dtype
        hdim_multiple_of = 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.tile_hdimv = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv

        self.is_causal = is_causal
        self.qhead_per_kvhead = qhead_per_kvhead
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.qk_acc_dtype = Float32
        self.buffer_align_bytes = 1024

        # ── SplitD parameters ──
        self.d_chunk = 128  # output slice width for dK/dV
        self.num_d_passes = self.tile_hdim // self.d_chunk
        self.num_d_inner = self.tile_hdim // self.d_chunk  # inner reduction chunks
        assert self.tile_hdim % self.d_chunk == 0
        assert self.tile_hdimv % self.d_chunk == 0

        self.num_wg_mma = 1  # single consumer WG for register headroom
        self.num_threads = 256  # 1 producer WG + 1 consumer WG
        self.num_threads_per_warp_group = 128
        self.num_producer_threads = 32
        self.num_mma_regs = 256
        self.num_producer_regs = 56

        self.A_stage = 3  # sA pipeline stages: must be > release_lag(2) to avoid deadlock
        self.PdS_stage = 2  # sP/sdS stages (double-buffered: eliminates barrier 1)

        # ── K/V persistence: pre-load all chunks once per d_pass ──
        self.K_persist_chunks = self.num_d_inner  # 4 chunks of K
        self.V_persist_chunks = self.num_d_inner  # 4 chunks of V
        self.KV_preload_stages = self.K_persist_chunks + self.V_persist_chunks  # 8

        # ── MMA layout configuration ──
        self.SdP_swapAB = False
        self.dKV_swapAB = False
        self.AtomLayoutMSdP = 1
        self.AtomLayoutNdKV = 1
        self.mma_dkv_is_rs = False  # P/dS through SMEM for baseline

    def _setup_attributes(self):
        # sA: (tile_m, d_chunk) — holds Q_chunk, dO_chunk, dO_d_pass, Q_d_pass
        self.sA_layout = sm90_utils.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, self.d_chunk),
            stage=self.A_stage,
            major_mode_size=self.d_chunk,  # accommodate sA^T
        )
        # sK_persist: (tile_n, d_chunk) × num_d_inner — persistent K across m_blocks
        self.sK_persist_layout = sm90_utils.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_n, self.d_chunk),
            stage=self.K_persist_chunks,
        )
        # sV_persist: (tile_n, d_chunk) × num_d_inner — persistent V across m_blocks
        self.sV_persist_layout = sm90_utils.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_n, self.d_chunk),
            stage=self.V_persist_chunks,
        )
        # sB_epi_layout: per-chunk layout for epilogue TMA store (same as K/V per-chunk)
        self.sB_epi_layout = cute.select(self.sK_persist_layout, mode=[0, 1])
        # sP: (tile_m, tile_n) — P for dV GEMM A operand
        wg_n_SdP = self.num_wg_mma // self.AtomLayoutMSdP
        wg_n_dKV = self.AtomLayoutNdKV
        self.sPdS_layout = sm90_utils.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, self.tile_n),
            stage=self.PdS_stage,
            major_mode_size=math.gcd(self.tile_n // wg_n_SdP, self.tile_n // wg_n_dKV),
        )

    def _get_tiled_mma(self):
        # ── SdP: S = Q @ K^T, dP = dO @ V^T ──
        # shape_mnk: (tile_m, tile_n, d_chunk) = (64, 64, 128)
        atom_layout_SdP = (self.AtomLayoutMSdP, self.num_wg_mma // self.AtomLayoutMSdP, 1)
        tiler_mn_SdP = (self.tile_m // atom_layout_SdP[0], self.tile_n // atom_layout_SdP[1])
        tiled_mma_SdP = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            self.qk_acc_dtype,
            atom_layout_mnk=atom_layout_SdP,
            tiler_mn=tiler_mn_SdP,
        )

        # ── dKV: dV = P^T @ dO_d, dK = dS^T @ Q_d ──
        # shape_mnk: (tile_n, d_chunk, tile_m) = (64, 128, 64)
        atom_layout_dKV = (self.AtomLayoutNdKV, self.num_wg_mma // self.AtomLayoutNdKV, 1)
        # dV: M=tile_n, N=d_chunk
        tiler_mn_dV = (self.tile_n // atom_layout_dKV[0], self.d_chunk // atom_layout_dKV[1])
        tiled_mma_dV = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.MN,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=atom_layout_dKV,
            tiler_mn=tiler_mn_dV,
        )
        # dK: same shape as dV
        tiler_mn_dK = (self.tile_n // atom_layout_dKV[0], self.d_chunk // atom_layout_dKV[1])
        tiled_mma_dK = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.MN,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=atom_layout_dKV,
            tiler_mn=tiler_mn_dK,
        )
        return tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV

    def _get_shared_storage_cls(self):
        sA_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sA_layout)],
            self.buffer_align_bytes,
        ]
        sK_persist_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sK_persist_layout)],
            self.buffer_align_bytes,
        ]
        sV_persist_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sV_persist_layout)],
            self.buffer_align_bytes,
        ]
        # Dedicated single-stage epilogue buffer for TMA dK/dV store.
        sEpi_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sB_epi_layout)],
            self.buffer_align_bytes,
        ]

        @cute.struct
        class SharedStorageDKDV:
            mbar_ptr_A: cute.struct.MemRange[cutlass.Int64, self.A_stage * 2]
            mbar_ptr_KV: cute.struct.MemRange[cutlass.Int64, self.KV_preload_stages * 2]
            sLSE: cute.struct.MemRange[
                Float32,
                cute.round_up(self.tile_m, 64) * self.A_stage,
            ]
            sdPsum: cute.struct.MemRange[
                Float32,
                cute.round_up(self.tile_m, 64) * self.A_stage,
            ]
            sA: sA_struct
            sK_persist: sK_persist_struct
            sV_persist: sV_persist_struct
            sEpi: sEpi_struct
            sP: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(self.sPdS_layout)], 1024]
            sdS: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(self.sPdS_layout)], 1024]

        return SharedStorageDKDV

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        stream: cuda.CUstream = None,
    ):
        mQ, mK, mV, mdO, mdK, mdV = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mdO, mdK, mdV)]
        mLSE, mdPsum = [assume_tensor_aligned(t) for t in (mLSE, mdPsum)]

        # Transpose: (b, s, n, h) → (s, h, n, b)
        def _qkv_transpose(t):
            return layout_utils.select(t, [1, 3, 2, 0] if cute.rank(t.shape) == 4 else [0, 2, 1])

        mQ, mK, mV, mdO, mdK, mdV = [_qkv_transpose(t) for t in (mQ, mK, mV, mdO, mdK, mdV)]
        # Stats: (b, n, s) → (s, n, b)
        LSE_transpose = [2, 1, 0] if cute.rank(mLSE.shape) == 3 else [1, 0]
        mLSE = layout_utils.select(mLSE, LSE_transpose)
        mdPsum = layout_utils.select(mdPsum, LSE_transpose)

        tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_SdP.size
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()

        sK_layout_sel = cute.select(self.sK_persist_layout, mode=[0, 1])
        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [
                ("A", mQ, self.sA_layout),
                ("KV", mK, self.sK_persist_layout),
            ]
        }
        self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8

        sA_layout_sel = cute.select(self.sA_layout, mode=[0, 1])
        gmem_tiled_copy_g2s = cpasync.CopyBulkTensorTileG2SOp()
        # Q: tile shape (tile_m, d_chunk) = (64, 128)
        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_g2s,
            mQ,
            sA_layout_sel,
            (self.tile_m, self.d_chunk),
        )
        # dO: tile shape (tile_m, d_chunk) = (64, 128)
        tma_atom_dO, tma_tensor_dO = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_g2s,
            mdO,
            sA_layout_sel,
            (self.tile_m, self.d_chunk),
        )
        # K: tile shape (tile_n, d_chunk) = (64, 128), persistent in SMEM
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_g2s,
            mK,
            sK_layout_sel,
            (self.tile_n, self.d_chunk),
        )
        # V: tile shape (tile_n, d_chunk) = (64, 128), persistent in SMEM
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_g2s,
            mV,
            sK_layout_sel,
            (self.tile_n, self.d_chunk),
        )
        # dK/dV: store atoms + TMA tensors.
        self.varlen_k = mCuSeqlensK is not None
        self.is_varlen_q = mCuSeqlensQ is not None
        gmem_tiled_copy_s2g = cpasync.CopyBulkTensorTileS2GOp()
        sB_epi_sel = cute.select(self.sK_persist_layout, mode=[0, 1])
        # Varlen K: create ragged TMA tensors for dK/dV output writes
        mdK_tma = copy_utils.create_ragged_tensor_for_tma(mdK, ragged_dim=0, ptr_shift=True) if self.varlen_k else mdK
        mdV_tma = copy_utils.create_ragged_tensor_for_tma(mdV, ragged_dim=0, ptr_shift=True) if self.varlen_k else mdV
        tma_atom_dK, tma_tensor_dK = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_s2g,
            mdK_tma,
            sB_epi_sel,
            (self.tile_n, self.d_chunk),
        )
        tma_atom_dV, tma_tensor_dV = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_s2g,
            mdV_tma,
            sB_epi_sel,
            (self.tile_n, self.d_chunk),
        )

        # ── Tile scheduler (backward: iterate over n_blocks) ──
        if const_expr(mCuSeqlensK is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = SingleTileScheduler
        num_n_blocks = cute.ceil_div(cute.size(mK.shape[0]), self.tile_n)
        num_batch = cute.size(mK.shape[3]) if cute.rank(mK.shape) == 4 else cute.size(mCuSeqlensK.shape[0] - 1)
        tile_sched_args = TileSchedulerArguments(
            num_n_blocks,  # num_m_blocks (but backward swaps M/N)
            cute.size(mK.shape[2]),  # num_heads
            num_batch,
            1,  # cluster_size
            cute.size(mQ.shape[0]),  # seqlen_k → actually seqlen_q for m_block range
            mQ.shape[1],  # head_dim_qk
            mV.shape[1],  # head_dim_v
            total_q=cute.size(mK.shape[0]),
            tile_shape_mn=(self.tile_n, self.tile_m),  # swapped for backward
            mCuSeqlensQ=mCuSeqlensK,
            qhead_per_kvhead_packgqa=1,
            element_size=self.dtype.width // 8,
            lpt=self.is_causal,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        softmax_scale_log2 = softmax_scale * math.log2(math.e)

        qhead_per_kvhead_divmod = None
        if const_expr(self.qhead_per_kvhead > 1):
            qhead_per_kvhead_divmod = FastDivmodDivisor(self.qhead_per_kvhead)

        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_dO,
            tma_tensor_dK,
            tma_tensor_dV,
            tma_atom_Q,
            tma_atom_dO,
            tma_atom_K,
            tma_atom_V,
            tma_atom_dK,
            tma_atom_dV,
            mLSE,
            mdPsum,
            mCuSeqlensQ,
            mCuSeqlensK,
            softmax_scale_log2,
            softmax_scale,
            self.sA_layout,
            self.sK_persist_layout,
            self.sV_persist_layout,
            self.sPdS_layout,
            tiled_mma_SdP,
            tiled_mma_dK,
            tiled_mma_dV,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            qhead_per_kvhead_divmod,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        sA_layout: cute.ComposedLayout,
        sK_persist_layout: cute.ComposedLayout,
        sV_persist_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisor] = None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == 0:
            for atom in [tma_atom_Q, tma_atom_dO, tma_atom_K, tma_atom_V, tma_atom_dK, tma_atom_dV]:
                cpasync.prefetch_descriptor(atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        pipeline_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread)
        pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            self.num_mma_threads // cute.arch.WARP_SIZE,
        )
        # pipeline_KV: 8-stage pipeline for K/V pre-loading (4 K + 4 V chunks)
        pipeline_KV = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_KV.data_ptr(),
            num_stages=self.KV_preload_stages,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["KV"],
            defer_sync=True,
        )
        # pipeline_A: Q/dO streaming (last pipeline → triggers sync)
        # Base tx_count covers sA only; LSE/dPsum added via extra_tx_count
        # on the specific items that need them (Phase 1 last Q, Phase 3 last dO).
        pipeline_A = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_A.data_ptr(),
            num_stages=self.A_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["A"],
            defer_sync=False,
        )

        sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
        sK_persist = storage.sK_persist.get_tensor(sK_persist_layout.outer, swizzle=sK_persist_layout.inner)
        sV_persist = storage.sV_persist.get_tensor(sV_persist_layout.outer, swizzle=sV_persist_layout.inner)
        sP = storage.sP.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sdS = storage.sdS.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        # Single-stage SMEM view for epilogue TMA store.
        sB_epi_layout_sel = cute.select(sK_persist_layout, mode=[0, 1])
        sB_epi = storage.sEpi.get_tensor(sB_epi_layout_sel.outer, swizzle=sB_epi_layout_sel.inner)
        sLSE = storage.sLSE.get_tensor(
            cute.make_layout(
                (self.tile_m, self.A_stage),
                stride=(1, cute.round_up(self.tile_m, 64)),
            )
        )
        sdPsum = storage.sdPsum.get_tensor(
            cute.make_layout(
                (self.tile_m, self.A_stage),
                stride=(1, cute.round_up(self.tile_m, 64)),
            )
        )

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            False,  # is_local
            False,  # is_split_kv
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        )
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        if warp_idx < 4:
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            if warp_idx == 0:
                self.load(
                    mQ,
                    mK,
                    mV,
                    mdO,
                    mLSE,
                    mdPsum,
                    sA,
                    sK_persist,
                    sV_persist,
                    sLSE,
                    sdPsum,
                    tma_atom_Q,
                    tma_atom_dO,
                    tma_atom_K,
                    tma_atom_V,
                    pipeline_A,
                    pipeline_KV,
                    block_info,
                    SeqlenInfoCls,
                    TileSchedulerCls,
                    qhead_per_kvhead_divmod,
                )
        else:
            cute.arch.setmaxregister_increase(self.num_mma_regs)
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            self.mma(
                tiled_mma_SdP,
                tiled_mma_dK,
                tiled_mma_dV,
                mdK,
                mdV,
                sA,
                sK_persist,
                sV_persist,
                sP,
                sdS,
                sLSE,
                sdPsum,
                sB_epi,
                pipeline_A,
                pipeline_KV,
                tidx,
                tma_atom_dK,
                tma_atom_dV,
                softmax_scale_log2,
                softmax_scale,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                qhead_per_kvhead_divmod,
            )

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        sA: cute.Tensor,
        sK_persist: cute.Tensor,
        sV_persist: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_A: pipeline.PipelineAsync,
        pipeline_KV: pipeline.PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisor] = None,
    ):
        producer_state_A = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.A_stage
        )
        producer_state_KV = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.KV_preload_stages
        )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()

        while work_tile.is_valid_tile:
            n_block, head_idx_kv, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            # K/V slicing — invariant across Q heads in GQA group
            mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx_kv]
            mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx_kv]

            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)

            # ── Outer d_pass loop ──
            for d_pass in cutlass.range_constexpr(self.num_d_passes):
                # ═══ K/V pre-load: load all 4 K + 4 V chunks once per d_pass ═══
                # K/V are n_block-stationary, no need to reload per m_block.
                for d_inner in cutlass.range_constexpr(self.num_d_inner):
                    gK_d = cute.local_tile(mK_cur, (self.tile_n, self.d_chunk), (None, d_inner))
                    load_K_d, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_K,
                        0,
                        cute.make_layout(1),
                        gK_d,
                        sK_persist,
                    )
                    pipeline_KV.producer_acquire(producer_state_KV)
                    load_K_d(
                        src_idx=n_block,
                        dst_idx=d_inner,
                        tma_bar_ptr=pipeline_KV.producer_get_barrier(producer_state_KV),
                    )
                    pipeline_KV.producer_commit(producer_state_KV)
                    producer_state_KV.advance()

                for d_inner in cutlass.range_constexpr(self.num_d_inner):
                    gV_d = cute.local_tile(mV_cur, (self.tile_n, self.d_chunk), (None, d_inner))
                    load_V_d, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_V,
                        0,
                        cute.make_layout(1),
                        gV_d,
                        sV_persist,
                    )
                    pipeline_KV.producer_acquire(producer_state_KV)
                    load_V_d(
                        src_idx=n_block,
                        dst_idx=d_inner,
                        tma_bar_ptr=pipeline_KV.producer_get_barrier(producer_state_KV),
                    )
                    pipeline_KV.producer_commit(producer_state_KV)
                    producer_state_KV.advance()

                # ── GQA Q-head loop: iterate over all Q heads in this KV group ──
                for q_head_offset in cutlass.range(self.qhead_per_kvhead, unroll=1):
                    head_idx_q = head_idx_kv * self.qhead_per_kvhead + q_head_offset

                    # Q/dO/LSE/dPsum slicing — per Q head
                    mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx_q]
                    mdO_cur = seqlen.offset_batch_Q(mdO, batch_idx, dim=3)[None, None, head_idx_q]
                    mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2, padded=True)[None, head_idx_q]
                    mdPsum_cur = seqlen.offset_batch_Q(mdPsum, batch_idx, dim=2, padded=True)[None, head_idx_q]

                    gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (None,))
                    gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (None,))
                    # Wait for bwd preprocess to finish writing LSE and dPsum
                    cute.arch.griddepcontrol_wait()
                    load_LSE = copy_utils.cpasync_bulk_get_copy_fn(gLSE, sLSE)
                    load_dPsum = copy_utils.cpasync_bulk_get_copy_fn(gdPsum, sdPsum)

                    for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
                        # ═══ Phase 1: S reduction — Q_chunk × 4, K from persistent SMEM ═══
                        # Only the LAST Q chunk (d_inner=3) carries LSE.
                        for d_inner in cutlass.range_constexpr(self.num_d_inner):
                            gQ_d = cute.local_tile(mQ_cur, (self.tile_m, self.d_chunk), (None, d_inner))
                            load_Q_d, _, _ = copy_utils.tma_get_copy_fn(
                                tma_atom_Q,
                                0,
                                cute.make_layout(1),
                                gQ_d,
                                sA,
                            )
                            if cutlass.const_expr(d_inner == self.num_d_inner - 1):
                                pipeline_A.producer_acquire(
                                    producer_state_A,
                                    extra_tx_count=self.tma_copy_bytes["LSE"],
                                )
                            else:
                                pipeline_A.producer_acquire(producer_state_A)
                            load_Q_d(
                                src_idx=m_block,
                                dst_idx=producer_state_A.index,
                                tma_bar_ptr=pipeline_A.producer_get_barrier(producer_state_A),
                            )
                            if cutlass.const_expr(d_inner == self.num_d_inner - 1):
                                load_LSE(
                                    src_idx=m_block,
                                    dst_idx=producer_state_A.index,
                                    tma_bar_ptr=pipeline_A.producer_get_barrier(producer_state_A),
                                )
                            pipeline_A.producer_commit(producer_state_A)
                            producer_state_A.advance()

                        # ═══ Phase 3: dP reduction — dO_chunk × 4, V from persistent SMEM ═══
                        # Only the LAST dO chunk (d_inner=3) carries dPsum.
                        for d_inner in cutlass.range_constexpr(self.num_d_inner):
                            gdO_d = cute.local_tile(mdO_cur, (self.tile_m, self.d_chunk), (None, d_inner))
                            load_dO_d, _, _ = copy_utils.tma_get_copy_fn(
                                tma_atom_dO,
                                0,
                                cute.make_layout(1),
                                gdO_d,
                                sA,
                            )
                            if cutlass.const_expr(d_inner == self.num_d_inner - 1):
                                pipeline_A.producer_acquire(
                                    producer_state_A,
                                    extra_tx_count=self.tma_copy_bytes["dPsum"],
                                )
                            else:
                                pipeline_A.producer_acquire(producer_state_A)
                            load_dO_d(
                                src_idx=m_block,
                                dst_idx=producer_state_A.index,
                                tma_bar_ptr=pipeline_A.producer_get_barrier(producer_state_A),
                            )
                            if cutlass.const_expr(d_inner == self.num_d_inner - 1):
                                load_dPsum(
                                    src_idx=m_block,
                                    dst_idx=producer_state_A.index,
                                    tma_bar_ptr=pipeline_A.producer_get_barrier(producer_state_A),
                                )
                            pipeline_A.producer_commit(producer_state_A)
                            producer_state_A.advance()

                        # ═══ Phase 5: Load dO_d_pass → sA (for dV GEMM) ═══
                        gdO_pass = cute.local_tile(mdO_cur, (self.tile_m, self.d_chunk), (None, d_pass))
                        load_dO_pass, _, _ = copy_utils.tma_get_copy_fn(
                            tma_atom_dO,
                            0,
                            cute.make_layout(1),
                            gdO_pass,
                            sA,
                        )
                        pipeline_A.producer_acquire(producer_state_A)
                        load_dO_pass(
                            src_idx=m_block,
                            dst_idx=producer_state_A.index,
                            tma_bar_ptr=pipeline_A.producer_get_barrier(producer_state_A),
                        )
                        pipeline_A.producer_commit(producer_state_A)
                        producer_state_A.advance()

                        # ═══ Phase 6: Load Q_d_pass → sA (for dK GEMM) ═══
                        gQ_pass = cute.local_tile(mQ_cur, (self.tile_m, self.d_chunk), (None, d_pass))
                        load_Q_pass, _, _ = copy_utils.tma_get_copy_fn(
                            tma_atom_Q,
                            0,
                            cute.make_layout(1),
                            gQ_pass,
                            sA,
                        )
                        pipeline_A.producer_acquire(producer_state_A)
                        load_Q_pass(
                            src_idx=m_block,
                            dst_idx=producer_state_A.index,
                            tma_bar_ptr=pipeline_A.producer_get_barrier(producer_state_A),
                        )
                        pipeline_A.producer_commit(producer_state_A)
                        producer_state_A.advance()

                cute.arch.barrier(
                    barrier_id=int(NamedBarrierBwd.WarpSchedulerWG1),
                    number_of_threads=self.num_producer_threads + self.num_mma_threads,
                )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        pipeline_A.producer_tail(producer_state_A)
        pipeline_KV.producer_tail(producer_state_KV)

    @cute.jit
    def mma(
        self,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        sA: cute.Tensor,
        sK_persist: cute.Tensor,
        sV_persist: cute.Tensor,
        sP: cute.Tensor,
        sdS: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        sB_epi: cute.Tensor,
        pipeline_A: pipeline.PipelineAsync,
        pipeline_KV: pipeline.PipelineAsync,
        tidx: Int32,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisor] = None,
    ):
        thr_mma_SdP = tiled_mma_SdP.get_slice(tidx)
        wg_mma_SdP = tiled_mma_SdP.get_slice(0)
        wg_mma_dK = tiled_mma_dK.get_slice(0)
        wg_mma_dV = tiled_mma_dV.get_slice(0)

        # ── SdP GEMM fragments ──
        shape_mnk_SdP = (self.tile_m, self.tile_n, self.d_chunk)
        _, tSrA, tSrB_K = sm90_utils.partition_fragment_ABC(wg_mma_SdP, shape_mnk_SdP, sA, sK_persist, swap_AB=False)
        _, _, tSrB_V = sm90_utils.partition_fragment_ABC(wg_mma_SdP, shape_mnk_SdP, sA, sV_persist, swap_AB=False)

        # ── dV GEMM fragments: dV = P^T @ dO_d^T ──
        sPt = layout_utils.transpose_view(sP)
        sAt = layout_utils.transpose_view(sA)
        shape_mnk_dV = (self.tile_n, self.d_chunk, self.tile_m)
        acc_dV, tdVrPt, tdVrdOt = sm90_utils.partition_fragment_ABC(wg_mma_dV, shape_mnk_dV, sPt, sAt, swap_AB=False)
        mma_dV_fn = partial(gemm_w_idx, tiled_mma_dV, acc_dV, tdVrPt, tdVrdOt, swap_AB=False)

        # ── dK GEMM fragments: dK = dS^T @ Q_d^T ──
        sdSt = layout_utils.transpose_view(sdS)
        shape_mnk_dK = (self.tile_n, self.d_chunk, self.tile_m)
        acc_dK, tdKrdSt, tdKrQt = sm90_utils.partition_fragment_ABC(wg_mma_dK, shape_mnk_dK, sdSt, sAt, swap_AB=False)
        mma_dK_fn = partial(gemm_w_idx, tiled_mma_dK, acc_dK, tdKrdSt, tdKrQt, swap_AB=False)

        # ── P/dS R2S copy setup ──
        sP_cpy = sP
        copy_P_r2s, _, _ = copy_utils.get_smem_store_C(
            tiled_mma_SdP,
            sP_cpy,
            tidx,
            self.arch,
            transpose=False,
            position_independent=True,
        )
        sdS_cpy = sdS
        copy_dS_r2s, _, _ = copy_utils.get_smem_store_C(
            tiled_mma_SdP,
            sdS_cpy,
            tidx,
            self.arch,
            transpose=False,
            position_independent=True,
        )

        # ── LSE/dPsum partitioning (per-thread row mapping via MMA C partition) ──
        tLSEsLSE = layout_utils.mma_partition_C_vec(sLSE, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=True)
        tLSEsdPsum = layout_utils.mma_partition_C_vec(sdPsum, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=True)

        # ── Barriers ──
        PdS_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwd.PdS),
            num_threads=self.num_mma_threads,
        )

        # ── Consumer pipeline states ──
        consumer_state_A = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.A_stage
        )
        consumer_state_KV = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.KV_preload_stages
        )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()

        while work_tile.is_valid_tile:
            n_block, head_idx_kv, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)

            # ── Tile validity check (Cf. flash_bwd_sm90.py:918-921) ──
            process_tile = const_expr(not self.is_varlen_q) or m_block_min < m_block_max

            # ── Outer d_pass loop ──
            for d_pass in cutlass.range_constexpr(self.num_d_passes):
                # ═══ Wait for K/V pre-load to complete ═══
                # Producer has loaded 4 K + 4 V chunks into persistent SMEM.
                # Consumer waits on all 8 stages, then releases
                # producer won't re-acquire until next d_pass barrier
                for _kv in cutlass.range_constexpr(self.KV_preload_stages):
                    pipeline_KV.consumer_wait(
                        consumer_state_KV,
                        pipeline_KV.consumer_try_wait(consumer_state_KV),
                    )
                    pipeline_KV.consumer_release(consumer_state_KV)
                    consumer_state_KV.advance()

                dKV_accumulate = False
                pds_idx = Int32(0)  # PdS double-buffer index, toggles per m_block

                # ── GQA Q-head loop: accumulate dK/dV across all Q heads ──
                for q_head_offset in cutlass.range(self.qhead_per_kvhead, unroll=1):
                    head_idx_q = head_idx_kv * self.qhead_per_kvhead + q_head_offset

                    mask = AttentionMask(
                        self.tile_m,
                        self.tile_n,
                        seqlen,
                    )
                    mask_fn = partial(
                        mask.apply_mask,
                        batch_idx=batch_idx,
                        head_idx=head_idx_q,
                        n_block=n_block,
                        thr_mma=thr_mma_SdP,
                        mask_seqlen=True,
                        mask_causal=self.is_causal,
                    )

                    for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
                        consumer_state_A, pds_idx = self.mma_one_m_block_splitd(
                            m_block,
                            consumer_state_A,
                            tiled_mma_SdP,
                            tSrA,
                            tSrB_K,
                            tSrB_V,
                            shape_mnk_SdP,
                            mma_dV_fn,
                            mma_dK_fn,
                            copy_P_r2s,
                            copy_dS_r2s,
                            pipeline_A,
                            tLSEsLSE,
                            tLSEsdPsum,
                            softmax_scale_log2,
                            softmax_scale,
                            mask_fn,
                            PdS_barrier,
                            pds_idx,
                            dKV_accumulate=dKV_accumulate,
                        )
                        dKV_accumulate = True

                if process_tile:
                    self.epilogue_dKV_slice(
                        acc_dV,
                        mdV,
                        sB_epi,
                        acc_dK,
                        mdK,
                        sB_epi,
                        seqlen,
                        tma_atom_dK,
                        tma_atom_dV,
                        tiled_mma_dK,
                        tiled_mma_dV,
                        tidx,
                        n_block,
                        head_idx_kv,
                        batch_idx,
                        d_pass,
                    )

                # sync with producer before next d_pass.
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierBwd.WarpSchedulerWG1),
                    number_of_threads=self.num_producer_threads + self.num_mma_threads,
                )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 4:
            cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def mma_one_m_block_splitd(
        self,
        m_block: Int32,
        consumer_state_A: cutlass.pipeline.PipelineState,
        tiled_mma_SdP: cute.TiledMma,
        tSrA: cute.Tensor,
        tSrB_K: cute.Tensor,
        tSrB_V: cute.Tensor,
        shape_mnk_SdP: cute.Shape,
        mma_dV_fn: Callable,
        mma_dK_fn: Callable,
        copy_P_r2s: Callable,
        copy_dS_r2s: Callable,
        pipeline_A: pipeline.PipelineAsync,
        tLSEsLSE: cute.Tensor,
        tLSEsdPsum: cute.Tensor,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        mask_fn: Callable,
        PdS_barrier: cutlass.pipeline.NamedBarrier,
        pds_idx: Int32,
        dKV_accumulate: Boolean = True,
    ):
        # ═══════════════════════════════════════════════════════════════
        # Phase 1: S = Q @ K^T  (4 × d_inner=128 reduction)
        # K from persistent SMEM (sK_persist), Q from pipeline_A.
        # Uses delayed release: release_state trails consumer_state by 2
        # ═══════════════════════════════════════════════════════════════
        release_state_A_p1 = consumer_state_A.clone()

        acc_S = cute.make_rmem_tensor(tiled_mma_SdP.partition_shape_C(shape_mnk_SdP[:2]), Float32)
        for d_inner in cutlass.range_constexpr(self.num_d_inner):
            pipeline_A.consumer_wait(
                consumer_state_A,
                pipeline_A.consumer_try_wait(consumer_state_A),
            )

            if cutlass.const_expr(d_inner < self.num_d_inner - 1):
                gemm_w_idx(
                    tiled_mma_SdP,
                    acc_S,
                    tSrA,
                    tSrB_K,
                    zero_init=(d_inner == 0),
                    A_idx=consumer_state_A.index,
                    B_idx=d_inner,  # persistent K chunk index
                    wg_wait=-1,
                )
                if cutlass.const_expr(d_inner >= 2):
                    warpgroup.wait_group(2)
                    pipeline_A.consumer_release(release_state_A_p1)
                    release_state_A_p1.advance()
            else:
                gemm_w_idx(
                    tiled_mma_SdP,
                    acc_S,
                    tSrA,
                    tSrB_K,
                    zero_init=(d_inner == 0),
                    A_idx=consumer_state_A.index,
                    B_idx=d_inner,
                    wg_wait=0,
                )
                # Read LSE now — sLSE still valid (stage not yet released)
                tLSErLSE = copy_utils.load_s2r(tLSEsLSE[None, consumer_state_A.index])
                pipeline_A.consumer_release(release_state_A_p1)
                release_state_A_p1.advance()

            consumer_state_A.advance()

        # Trailing releases: 2 remaining stages (d_inner=2 and d_inner=3)
        for _ in cutlass.range_constexpr(2):
            pipeline_A.consumer_release(release_state_A_p1)
            release_state_A_p1.advance()

        # ═══════════════════════════════════════════════════════════════
        # Phase 2: P = exp2(S * scale_log2 - LSE)
        # ═══════════════════════════════════════════════════════════════
        mask_fn(acc_S, m_block=m_block)
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=False)
        for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
            lse_val = tLSErLSE[r]
            for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
                acc_S_mn[r, c] = cute.math.exp2(
                    acc_S_mn[r, c] * softmax_scale_log2 - lse_val,
                    fastmath=True,
                )

        tdVrP = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_S), self.dtype)
        if const_expr(self.PdS_stage == 1):
            PdS_barrier.arrive_and_wait()
        copy_P_r2s(tdVrP, dst_idx=pds_idx)

        # ═══════════════════════════════════════════════════════════════
        # Phase 3: dP = dO @ V^T  (4 × d_inner=128 reduction)
        # V from persistent SMEM (sV_persist), dO from pipeline_A.
        # ═══════════════════════════════════════════════════════════════
        release_state_A_p3 = consumer_state_A.clone()

        acc_dP = cute.make_rmem_tensor(tiled_mma_SdP.partition_shape_C(shape_mnk_SdP[:2]), Float32)
        for d_inner in cutlass.range_constexpr(self.num_d_inner):
            pipeline_A.consumer_wait(
                consumer_state_A,
                pipeline_A.consumer_try_wait(consumer_state_A),
            )

            if cutlass.const_expr(d_inner < self.num_d_inner - 1):
                gemm_w_idx(
                    tiled_mma_SdP,
                    acc_dP,
                    tSrA,
                    tSrB_V,
                    zero_init=(d_inner == 0),
                    A_idx=consumer_state_A.index,
                    B_idx=d_inner,  # persistent V chunk index
                    wg_wait=-1,
                )
                if cutlass.const_expr(d_inner >= 2):
                    warpgroup.wait_group(2)
                    pipeline_A.consumer_release(release_state_A_p3)
                    release_state_A_p3.advance()
            else:
                gemm_w_idx(
                    tiled_mma_SdP,
                    acc_dP,
                    tSrA,
                    tSrB_V,
                    zero_init=(d_inner == 0),
                    A_idx=consumer_state_A.index,
                    B_idx=d_inner,
                    wg_wait=0,
                )
                # Read dPsum now — sdPsum still valid (stage not yet released)
                tLSErdPsum = copy_utils.load_s2r(tLSEsdPsum[None, consumer_state_A.index])
                pipeline_A.consumer_release(release_state_A_p3)
                release_state_A_p3.advance()

            consumer_state_A.advance()

        # Trailing releases: 2 remaining stages
        for _ in cutlass.range_constexpr(2):
            pipeline_A.consumer_release(release_state_A_p3)
            release_state_A_p3.advance()

        # ═══════════════════════════════════════════════════════════════
        # Phase 4: dS = P * (dP - dPsum)
        # ═══════════════════════════════════════════════════════════════
        acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP, transpose=False)
        for r in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[0])):
            dpsum_val = tLSErdPsum[r]
            for c in cutlass.range(cute.size(acc_dP_mn, mode=[1]), unroll_full=True):
                acc_dP_mn[r, c] = acc_S_mn[r, c] * (acc_dP_mn[r, c] - dpsum_val) * softmax_scale

        # dS_scaled fp32 → fp16, write to sdS[pds_idx]
        tdKrdS = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_dP), self.dtype)
        cute.arch.fence_view_async_shared()
        PdS_barrier.arrive_and_wait()
        copy_dS_r2s(tdKrdS, dst_idx=pds_idx)

        # ═══════════════════════════════════════════════════════════════
        # Phase 5: dV += P^T @ dO_d_pass
        # ═══════════════════════════════════════════════════════════════
        cute.arch.fence_view_async_shared()
        PdS_barrier.arrive_and_wait()

        pipeline_A.consumer_wait(
            consumer_state_A,
            pipeline_A.consumer_try_wait(consumer_state_A),
        )
        smem_idx_dO_pass = consumer_state_A.index
        mma_dV_fn(
            A_idx=pds_idx,  # sP stage (double-buffered)
            B_idx=smem_idx_dO_pass,
            zero_init=not dKV_accumulate,
            wg_wait=0,
        )
        pipeline_A.consumer_release(consumer_state_A)
        consumer_state_A.advance()

        # ═══════════════════════════════════════════════════════════════
        # Phase 6: dK += dS^T @ Q_d_pass
        # ═══════════════════════════════════════════════════════════════
        pipeline_A.consumer_wait(
            consumer_state_A,
            pipeline_A.consumer_try_wait(consumer_state_A),
        )
        smem_idx_Q_pass = consumer_state_A.index
        mma_dK_fn(
            A_idx=pds_idx,  # sdS stage (double-buffered)
            B_idx=smem_idx_Q_pass,
            zero_init=not dKV_accumulate,
            wg_wait=0,
        )
        pipeline_A.consumer_release(consumer_state_A)
        consumer_state_A.advance()

        # Toggle PdS double-buffer index for next m_block
        pds_idx = 1 - pds_idx
        return consumer_state_A, pds_idx

    @cute.jit
    def epilogue_dKV_slice(
        self,
        acc_dV: cute.Tensor,
        mdV: cute.Tensor,
        sV_buf: cute.Tensor,  # sEpi epilogue buffer
        acc_dK: cute.Tensor,
        mdK: cute.Tensor,
        sK_buf: cute.Tensor,  # sEpi epilogue buffer (shared, serialized writes)
        seqlen: SeqlenInfoQK,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tidx: Int32,
        n_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        d_pass: Int32,
    ):
        epi_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwd.Epilogue),
            num_threads=self.num_mma_threads,
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Global tile addresses for current d_pass slice.
        # mdK/mdV are the TMA tensors (from make_tiled_tma_atom's 2nd return),
        mdK_cur = seqlen.offset_batch_K(mdK, batch_idx, dim=3, ragged=self.varlen_k)[None, None, head_idx]
        mdV_cur = seqlen.offset_batch_K(mdV, batch_idx, dim=3, ragged=self.varlen_k)[None, None, head_idx]
        gdK = cute.local_tile(mdK_cur, (self.tile_n, self.d_chunk), (n_block, d_pass))
        gdV = cute.local_tile(mdV_cur, (self.tile_n, self.d_chunk), (n_block, d_pass))

        # TMA store functions (sK_buf/sV_buf are single-stage epilogue views)
        store_dK, _, _ = copy_utils.tma_get_copy_fn(tma_atom_dK, 0, cute.make_layout(1), sK_buf, gdK, single_stage=True)
        store_dV, _, _ = copy_utils.tma_get_copy_fn(tma_atom_dV, 0, cute.make_layout(1), sV_buf, gdV, single_stage=True)

        # R2S copy for dV
        copy_dV_r2s, _, _ = copy_utils.get_smem_store_C(
            tiled_mma_dV,
            sV_buf,
            tidx,
            self.arch,
            transpose=False,
            position_independent=True,
        )
        # R2S copy for dK
        copy_dK_r2s, _, _ = copy_utils.get_smem_store_C(
            tiled_mma_dK,
            sK_buf,
            tidx,
            self.arch,
            transpose=False,
            position_independent=True,
        )

        # ── Write dV ──
        cute.arch.cp_async_bulk_wait_group(1, read=True)
        epi_barrier.arrive_and_wait()
        copy_dV_r2s(acc_dV, dst_idx=None)
        cute.arch.fence_view_async_shared()
        epi_barrier.arrive_and_wait()
        if warp_idx == 4:
            store_dV()
            cute.arch.cp_async_bulk_commit_group()

        # ── Write dK ──
        cute.arch.cp_async_bulk_wait_group(0, read=True)
        epi_barrier.arrive_and_wait()
        copy_dK_r2s(acc_dK, dst_idx=None)
        cute.arch.fence_view_async_shared()
        epi_barrier.arrive_and_wait()
        if warp_idx == 4:
            store_dK()
            cute.arch.cp_async_bulk_commit_group()
