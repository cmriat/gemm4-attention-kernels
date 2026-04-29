# Copyright (c) 2026 China Merchants Research Institute Of Advanced Technology
# SM90 Backward dQ Kernel with 4-Pass D-Split for head_dim=512.
#
# Architecture:
#   - 2 WGs: 1 producer (TMA, 128 threads), 1 consumer (MMA, 128 threads)
#   - tile_m=64, tile_n=64, d_chunk=128, num_d_passes=4
#   - Q-stationary outer loop, K/V streaming inner loop
#   - Q/dO persistent in SMEM (loaded once per m_block, invariant across n_blocks and d_passes)
#   - Per d_pass, 5 phases per (m_block, n_block):
#       Phase A: S = Q @ K^T (4 x d_inner=128 reduction)
#       Phase B: P = exp2(S * scale_log2 - LSE)
#       Phase C: dP = dO @ V^T (4 x d_inner=128 reduction)
#       Phase D: dS = P * (dP - dPsum), write dS -> sPdS
#       Phase E: dQ_acc += dS @ K_d_pass
#   - 3 pipelines: QdO (Q/dO preload per m_block), B (K/V streaming), Kt (K_d_pass MN_MAJOR)

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


class FlashBwdDQ_SplitD_Sm90:
    """SM90 backward dQ kernel with 4-Pass D-Split for large head_dim.

    Computes only dQ (no dK/dV). dK/dV is handled by fmha_dkdv_d256_sm90.py.
    Q-stationary: outer loop over m_blocks, inner loop over n_blocks.
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

        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_causal = is_causal
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.qk_acc_dtype = Float32
        self.buffer_align_bytes = 1024

        # -- SplitD parameters --
        self.d_chunk = 128  # output slice width for dQ
        self.num_d_passes = self.tile_hdim // self.d_chunk
        self.num_d_inner = self.tile_hdim // self.d_chunk  # inner reduction chunks
        assert self.tile_hdim % self.d_chunk == 0
        assert self.tile_hdimv % self.d_chunk == 0

        self.num_wg_mma = 1
        self.num_threads = 256
        self.num_threads_per_warp_group = 128
        self.num_producer_threads = 32
        self.num_mma_regs = 256
        self.num_producer_regs = 56

        # Q/dO persistence: load once per m_block into persistent SMEM
        self.Q_persist_chunks = self.num_d_inner  # 4 Q chunks
        self.dO_persist_chunks = self.num_d_inner  # 4 dO chunks
        self.QdO_preload_stages = self.Q_persist_chunks + self.dO_persist_chunks  # 8
        self.LSE_stage = 1  # single-stage LSE/dPsum (loaded once per m_block)
        self.B_stage = 3  # sB pipeline stages: must be > release_lag(2) to avoid deadlock
        self.Kt_stage = 1  # sKt pipeline stages (K_d_pass, MN_MAJOR)
        self.PdS_stage = 1  # sPdS stages

        self.SdP_swapAB = False
        self.AtomLayoutMSdP = 1
        self.AtomLayoutMdQ = 1

    def _setup_attributes(self):
        # sQ_persist: (tile_m, d_chunk) x Q_persist_chunks -- persistent Q across n_blocks
        self.sQ_persist_layout = sm90_utils.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, self.d_chunk),
            stage=self.Q_persist_chunks,
            major_mode_size=self.d_chunk,
        )
        # sdO_persist: (tile_m, d_chunk) x dO_persist_chunks -- persistent dO
        self.sdO_persist_layout = sm90_utils.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, self.d_chunk),
            stage=self.dO_persist_chunks,
            major_mode_size=self.d_chunk,
        )
        # sA_epi_layout: per-stage layout for epilogue TMA store (same swizzle as Q persist)
        self.sA_epi_layout = cute.select(self.sQ_persist_layout, mode=[0, 1])
        # sB: (tile_n, d_chunk) -- holds K_chunk, V_chunk
        self.sB_layout = sm90_utils.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_n, self.d_chunk),
            stage=self.B_stage,
        )
        # sKt: (tile_n, d_chunk) -- K_d_pass with MN_MAJOR layout for dQ GEMM B operand
        self.sKt_layout = sm90_utils.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_n, self.d_chunk),
            stage=self.Kt_stage,
            major_mode_size=self.d_chunk,  # MN_MAJOR: accommodate access along d_chunk (N dim)
        )
        # sPdS: (tile_m, tile_n) -- dS for dQ GEMM A operand
        self.sPdS_layout = sm90_utils.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, self.tile_n),
            stage=self.PdS_stage,
        )

    def _get_tiled_mma(self):
        """Create tiled MMA objects for SdP and dQ GEMMs."""
        # -- SdP: S = Q @ K^T, dP = dO @ V^T --
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

        # -- dQ: dQ = dS @ K, shape_mnk: (tile_m, d_chunk, tile_n) = (64, 128, 64) --
        # A from SMEM (sPdS, dS, K_MAJOR), B from SMEM (sKt, K_d_pass MN_MAJOR)
        atom_layout_dQ = (self.AtomLayoutMdQ, self.num_wg_mma // self.AtomLayoutMdQ, 1)
        tiler_mn_dQ = (self.tile_m // atom_layout_dQ[0], self.d_chunk // atom_layout_dQ[1])
        tiled_mma_dQ = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=atom_layout_dQ,
            tiler_mn=tiler_mn_dQ,
        )
        return tiled_mma_SdP, tiled_mma_dQ

    def _get_shared_storage_cls(self):
        sQ_persist_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sQ_persist_layout)],
            self.buffer_align_bytes,
        ]
        sdO_persist_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sdO_persist_layout)],
            self.buffer_align_bytes,
        ]
        sB_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sB_layout)],
            self.buffer_align_bytes,
        ]
        sKt_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sKt_layout)],
            self.buffer_align_bytes,
        ]
        # Dedicated single-stage epilogue buffer for TMA dQ store
        sEpi_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sA_epi_layout)],
            self.buffer_align_bytes,
        ]

        @cute.struct
        class SharedStorageDQ:
            mbar_ptr_QdO: cute.struct.MemRange[cutlass.Int64, self.QdO_preload_stages * 2]
            mbar_ptr_B: cute.struct.MemRange[cutlass.Int64, self.B_stage * 2]
            mbar_ptr_Kt: cute.struct.MemRange[cutlass.Int64, self.Kt_stage * 2]
            sLSE: cute.struct.MemRange[
                Float32,
                cute.round_up(self.tile_m, 64) * self.LSE_stage,
            ]
            sdPsum: cute.struct.MemRange[
                Float32,
                cute.round_up(self.tile_m, 64) * self.LSE_stage,
            ]
            sQ_persist: sQ_persist_struct
            sdO_persist: sdO_persist_struct
            sB: sB_struct
            sKt: sKt_struct
            sEpi: sEpi_struct
            sPdS: cute.struct.MemRange[
                self.dtype,
                cute.cosize(self.sPdS_layout),
            ]

        return SharedStorageDQ

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQ: cute.Tensor,
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        stream: cuda.CUstream = None,
    ):
        mQ, mK, mV, mdO, mdQ = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mdO, mdQ)]
        mLSE, mdPsum = [assume_tensor_aligned(t) for t in (mLSE, mdPsum)]

        # Transpose: (b, s, n, h) -> (s, h, n, b)
        def _qkv_transpose(t):
            return layout_utils.select(t, [1, 3, 2, 0] if cute.rank(t.shape) == 4 else [0, 2, 1])

        mQ, mK, mV, mdO, mdQ = [_qkv_transpose(t) for t in (mQ, mK, mV, mdO, mdQ)]
        # Stats: (b, n, s) -> (s, n, b)
        LSE_transpose = [2, 1, 0] if cute.rank(mLSE.shape) == 3 else [1, 0]
        mLSE = layout_utils.select(mLSE, LSE_transpose)
        mdPsum = layout_utils.select(mdPsum, LSE_transpose)

        tiled_mma_SdP, tiled_mma_dQ = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_SdP.size
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()

        sQ_layout_sel = cute.select(self.sQ_persist_layout, mode=[0, 1])
        sB_layout_sel = cute.select(self.sB_layout, mode=[0, 1])
        sKt_layout_sel = cute.select(self.sKt_layout, mode=[0, 1])
        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, layout_sel)
            for name, mX, layout_sel in [
                ("QdO", mQ, sQ_layout_sel),
                ("B", mK, sB_layout_sel),
                ("Kt", mK, sKt_layout_sel),
            ]
        }
        self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8

        gmem_tiled_copy_g2s = cpasync.CopyBulkTensorTileG2SOp()
        # Q: tile shape (tile_m, d_chunk) = (64, 128) → sQ_persist
        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_g2s,
            mQ,
            sQ_layout_sel,
            (self.tile_m, self.d_chunk),
        )
        # dO: tile shape (tile_m, d_chunk) = (64, 128) → sdO_persist
        tma_atom_dO, tma_tensor_dO = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_g2s,
            mdO,
            sQ_layout_sel,
            (self.tile_m, self.d_chunk),
        )
        # K: tile shape (tile_n, d_chunk) = (64, 128), K_MAJOR for QK GEMM
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_g2s,
            mK,
            sB_layout_sel,
            (self.tile_n, self.d_chunk),
        )
        # V: tile shape (tile_n, d_chunk) = (64, 128)
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_g2s,
            mV,
            sB_layout_sel,
            (self.tile_n, self.d_chunk),
        )
        # Kt: K_d_pass with MN_MAJOR layout for dQ GEMM B operand
        tma_atom_Kt, tma_tensor_Kt = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_g2s,
            mK,
            sKt_layout_sel,
            (self.tile_n, self.d_chunk),
        )
        # dQ: store atom (tile_m, d_chunk) — uses sQ_layout_sel (same swizzle as G2S)
        # Varlen Q: use ragged TMA tensor for dQ output writes
        self.varlen_q = mCuSeqlensQ is not None
        self.is_varlen_k = mCuSeqlensK is not None
        gmem_tiled_copy_s2g = cpasync.CopyBulkTensorTileS2GOp()
        mdQ_tma = copy_utils.create_ragged_tensor_for_tma(mdQ, ragged_dim=0, ptr_shift=True) if self.varlen_q else mdQ
        tma_atom_dQ, tma_tensor_dQ = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_s2g,
            mdQ_tma,
            sQ_layout_sel,
            (self.tile_m, self.d_chunk),
        )

        if const_expr(mCuSeqlensQ is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = SingleTileScheduler
        num_m_blocks = cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m)
        num_batch = cute.size(mQ.shape[3]) if cute.rank(mQ.shape) == 4 else cute.size(mCuSeqlensQ.shape[0] - 1)
        tile_sched_args = TileSchedulerArguments(
            num_m_blocks,
            cute.size(mQ.shape[2]),  # num_heads
            num_batch,
            1,  # cluster_size
            cute.size(mK.shape[0]),  # seqlen_k for n_block range
            mQ.shape[1],  # head_dim_qk
            mV.shape[1],  # head_dim_v
            total_q=cute.size(mQ.shape[0]),
            tile_shape_mn=(self.tile_m, self.tile_n),
            # Varlen: Q plays the "Q" role for the scheduler (dQ iterates Q blocks)
            mCuSeqlensQ=mCuSeqlensQ,
            qhead_per_kvhead_packgqa=1,
            element_size=self.dtype.width // 8,
            lpt=self.is_causal,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        softmax_scale_log2 = softmax_scale * math.log2(math.e)

        # GQA: FastDivmodDivisor for efficient head_idx → head_idx_kv mapping
        qhead_per_kvhead_divmod = None
        if const_expr(self.qhead_per_kvhead > 1):
            qhead_per_kvhead_divmod = FastDivmodDivisor(self.qhead_per_kvhead)

        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_dO,
            tma_tensor_Kt,
            tma_tensor_dQ,
            tma_atom_Q,
            tma_atom_dO,
            tma_atom_K,
            tma_atom_V,
            tma_atom_Kt,
            tma_atom_dQ,
            mLSE,
            mdPsum,
            mCuSeqlensQ,
            mCuSeqlensK,
            softmax_scale_log2,
            softmax_scale,
            self.sQ_persist_layout,
            self.sdO_persist_layout,
            self.sB_layout,
            self.sKt_layout,
            self.sPdS_layout,
            tiled_mma_SdP,
            tiled_mma_dQ,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            qhead_per_kvhead_divmod,
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
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mKt: cute.Tensor,  # same GMEM as mK, but TMA uses sKt layout
        mdQ: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_Kt: cute.CopyAtom,
        tma_atom_dQ: cute.CopyAtom,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        sQ_persist_layout: cute.ComposedLayout,
        sdO_persist_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sKt_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisor] = None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == 0:
            for atom in [tma_atom_Q, tma_atom_dO, tma_atom_K, tma_atom_V, tma_atom_Kt, tma_atom_dQ]:
                cpasync.prefetch_descriptor(atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        pipeline_producer_group = cutlass.pipeline.CooperativeGroup(cutlass.pipeline.Agent.Thread)
        pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            self.num_mma_threads // cute.arch.WARP_SIZE,
        )
        # Q/dO preload pipeline: 8 stages (4 Q + 4 dO), loaded once per m_block
        pipeline_QdO = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_QdO.data_ptr(),
            num_stages=self.QdO_preload_stages,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["QdO"],
            defer_sync=True,
        )
        pipeline_B = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_B.data_ptr(),
            num_stages=self.B_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["B"],
            defer_sync=True,
        )
        pipeline_Kt = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_Kt.data_ptr(),
            num_stages=self.Kt_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["Kt"],
            defer_sync=False,  # last pipeline: triggers mbarrier_init_fence + syncthreads
        )

        sQ_persist = storage.sQ_persist.get_tensor(sQ_persist_layout.outer, swizzle=sQ_persist_layout.inner)
        sdO_persist = storage.sdO_persist.get_tensor(sdO_persist_layout.outer, swizzle=sdO_persist_layout.inner)
        sB = storage.sB.get_tensor(sB_layout.outer, swizzle=sB_layout.inner)
        sKt = storage.sKt.get_tensor(sKt_layout.outer, swizzle=sKt_layout.inner)
        # Single-stage SMEM view for epilogue TMA store.
        sA_epi_layout_sel = cute.select(sQ_persist_layout, mode=[0, 1])
        sA_epi = storage.sEpi.get_tensor(sA_epi_layout_sel.outer, swizzle=sA_epi_layout_sel.inner)
        sPdS = storage.sPdS.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sLSE = storage.sLSE.get_tensor(
            cute.make_layout(
                (self.tile_m, self.LSE_stage),
                stride=(1, cute.round_up(self.tile_m, 64)),
            )
        )
        sdPsum = storage.sdPsum.get_tensor(
            cute.make_layout(
                (self.tile_m, self.LSE_stage),
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
                    mKt,
                    mLSE,
                    mdPsum,
                    sQ_persist,
                    sdO_persist,
                    sB,
                    sKt,
                    sLSE,
                    sdPsum,
                    tma_atom_Q,
                    tma_atom_dO,
                    tma_atom_K,
                    tma_atom_V,
                    tma_atom_Kt,
                    pipeline_QdO,
                    pipeline_B,
                    pipeline_Kt,
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
                tiled_mma_dQ,
                mdQ,
                sQ_persist,
                sdO_persist,
                sB,
                sKt,
                sPdS,
                sLSE,
                sdPsum,
                sA_epi,
                pipeline_QdO,
                pipeline_B,
                pipeline_Kt,
                tidx,
                tma_atom_dQ,
                softmax_scale_log2,
                softmax_scale,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
            )

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mKt: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        sQ_persist: cute.Tensor,
        sdO_persist: cute.Tensor,
        sB: cute.Tensor,
        sKt: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_Kt: cute.CopyAtom,
        pipeline_QdO: pipeline.PipelineAsync,
        pipeline_B: pipeline.PipelineAsync,
        pipeline_Kt: pipeline.PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisor] = None,
    ):
        producer_state_QdO = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.QdO_preload_stages
        )
        producer_state_B = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.B_stage
        )
        producer_state_Kt = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.Kt_stage
        )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()

        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            # GQA: map Q head index to KV head index for K/V loading
            head_idx_kv = head_idx if const_expr(self.qhead_per_kvhead == 1) else head_idx // qhead_per_kvhead_divmod

            # Slice global tensors for current head/batch
            mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
            mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx_kv]
            mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx_kv]
            mdO_cur = seqlen.offset_batch_Q(mdO, batch_idx, dim=3)[None, None, head_idx]
            mKt_cur = seqlen.offset_batch_K(mKt, batch_idx, dim=3)[None, None, head_idx_kv]
            mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2, padded=True)[None, head_idx]
            mdPsum_cur = seqlen.offset_batch_Q(mdPsum, batch_idx, dim=2, padded=True)[None, head_idx]

            gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (None,))
            gdPsum = cute.local_tile(mdPsum_cur, (self.tile_m,), (None,))
            load_LSE = copy_utils.cpasync_bulk_get_copy_fn(gLSE, sLSE)
            load_dPsum = copy_utils.cpasync_bulk_get_copy_fn(gdPsum, sdPsum)

            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)

            # ═══ Q/dO PRELOAD: once per m_block (invariant across d_passes and n_blocks) ═══
            for d_inner in cutlass.range_constexpr(self.num_d_inner):
                gQ_d = cute.local_tile(mQ_cur, (self.tile_m, self.d_chunk), (None, d_inner))
                load_Q_d, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Q,
                    0,
                    cute.make_layout(1),
                    gQ_d,
                    sQ_persist,
                )
                if cutlass.const_expr(d_inner == self.num_d_inner - 1):
                    pipeline_QdO.producer_acquire(
                        producer_state_QdO,
                        extra_tx_count=self.tma_copy_bytes["LSE"],
                    )
                else:
                    pipeline_QdO.producer_acquire(producer_state_QdO)
                load_Q_d(
                    src_idx=m_block,
                    dst_idx=d_inner,
                    tma_bar_ptr=pipeline_QdO.producer_get_barrier(producer_state_QdO),
                )
                if cutlass.const_expr(d_inner == self.num_d_inner - 1):
                    load_LSE(
                        src_idx=m_block,
                        dst_idx=0,
                        tma_bar_ptr=pipeline_QdO.producer_get_barrier(producer_state_QdO),
                    )
                pipeline_QdO.producer_commit(producer_state_QdO)
                producer_state_QdO.advance()

            for d_inner in cutlass.range_constexpr(self.num_d_inner):
                gdO_d = cute.local_tile(mdO_cur, (self.tile_m, self.d_chunk), (None, d_inner))
                load_dO_d, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_dO,
                    0,
                    cute.make_layout(1),
                    gdO_d,
                    sdO_persist,
                )
                if cutlass.const_expr(d_inner == self.num_d_inner - 1):
                    pipeline_QdO.producer_acquire(
                        producer_state_QdO,
                        extra_tx_count=self.tma_copy_bytes["dPsum"],
                    )
                else:
                    pipeline_QdO.producer_acquire(producer_state_QdO)
                load_dO_d(
                    src_idx=m_block,
                    dst_idx=d_inner,
                    tma_bar_ptr=pipeline_QdO.producer_get_barrier(producer_state_QdO),
                )
                if cutlass.const_expr(d_inner == self.num_d_inner - 1):
                    load_dPsum(
                        src_idx=m_block,
                        dst_idx=0,
                        tma_bar_ptr=pipeline_QdO.producer_get_barrier(producer_state_QdO),
                    )
                pipeline_QdO.producer_commit(producer_state_QdO)
                producer_state_QdO.advance()

            # ═══ d_pass loop: only stream K/V/Kt (Q/dO persistent in SMEM) ═══
            for d_pass in cutlass.range_constexpr(self.num_d_passes):
                for i_n in cutlass.range(n_block_max - n_block_min, unroll=1):
                    n_block = n_block_max - 1 - i_n  # inverse order for causal

                    # === Phase A: stream K chunks → sB ===
                    for d_inner in cutlass.range_constexpr(self.num_d_inner):
                        gK_d = cute.local_tile(mK_cur, (self.tile_n, self.d_chunk), (None, d_inner))
                        load_K_d, _, _ = copy_utils.tma_get_copy_fn(
                            tma_atom_K,
                            0,
                            cute.make_layout(1),
                            gK_d,
                            sB,
                        )
                        pipeline_B.producer_acquire(producer_state_B)
                        load_K_d(
                            src_idx=n_block,
                            dst_idx=producer_state_B.index,
                            tma_bar_ptr=pipeline_B.producer_get_barrier(producer_state_B),
                        )
                        pipeline_B.producer_commit(producer_state_B)
                        producer_state_B.advance()

                    # === Phase C: stream V chunks → sB ===
                    for d_inner in cutlass.range_constexpr(self.num_d_inner):
                        gV_d = cute.local_tile(mV_cur, (self.tile_n, self.d_chunk), (None, d_inner))
                        load_V_d, _, _ = copy_utils.tma_get_copy_fn(
                            tma_atom_V,
                            0,
                            cute.make_layout(1),
                            gV_d,
                            sB,
                        )
                        pipeline_B.producer_acquire(producer_state_B)
                        load_V_d(
                            src_idx=n_block,
                            dst_idx=producer_state_B.index,
                            tma_bar_ptr=pipeline_B.producer_get_barrier(producer_state_B),
                        )
                        pipeline_B.producer_commit(producer_state_B)
                        producer_state_B.advance()

                    # === Phase E: Load K_d_pass → sKt (MN_MAJOR) ===
                    gKt_pass = cute.local_tile(mKt_cur, (self.tile_n, self.d_chunk), (None, d_pass))
                    load_Kt_pass, _, _ = copy_utils.tma_get_copy_fn(
                        tma_atom_Kt,
                        0,
                        cute.make_layout(1),
                        gKt_pass,
                        sKt,
                    )
                    pipeline_Kt.producer_acquire(producer_state_Kt)
                    load_Kt_pass(
                        src_idx=n_block,
                        dst_idx=producer_state_Kt.index,
                        tma_bar_ptr=pipeline_Kt.producer_get_barrier(producer_state_Kt),
                    )
                    pipeline_Kt.producer_commit(producer_state_Kt)
                    producer_state_Kt.advance()

                # d_pass barrier: wait for consumer epilogue before next d_pass
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierBwd.WarpSchedulerWG1),
                    number_of_threads=self.num_producer_threads + self.num_mma_threads,
                )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        pipeline_QdO.producer_tail(producer_state_QdO)
        pipeline_B.producer_tail(producer_state_B)
        pipeline_Kt.producer_tail(producer_state_Kt)

    @cute.jit
    def mma(
        self,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        mdQ: cute.Tensor,
        sQ_persist: cute.Tensor,
        sdO_persist: cute.Tensor,
        sB: cute.Tensor,
        sKt: cute.Tensor,
        sPdS: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        sA_epi: cute.Tensor,
        pipeline_QdO: pipeline.PipelineAsync,
        pipeline_B: pipeline.PipelineAsync,
        pipeline_Kt: pipeline.PipelineAsync,
        tidx: Int32,
        tma_atom_dQ: cute.CopyAtom,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        wg_mma_SdP = tiled_mma_SdP.get_slice(0)
        wg_mma_dQ = tiled_mma_dQ.get_slice(0)

        # -- SdP GEMM fragments --
        # Phase A: S = Q @ K^T → A=sQ_persist, B=sB
        # Phase C: dP = dO @ V^T → A=sdO_persist, B=sB
        shape_mnk_SdP = (self.tile_m, self.tile_n, self.d_chunk)
        _, tSrA_Q, tSrB = sm90_utils.partition_fragment_ABC(wg_mma_SdP, shape_mnk_SdP, sQ_persist, sB, swap_AB=False)
        _, tSrA_dO, _ = sm90_utils.partition_fragment_ABC(wg_mma_SdP, shape_mnk_SdP, sdO_persist, sB, swap_AB=False)

        # -- dQ GEMM fragments: dQ = dS @ K_d_pass --
        sKt_t = layout_utils.transpose_view(sKt)
        shape_mnk_dQ = (self.tile_m, self.d_chunk, self.tile_n)
        acc_dQ, tDQrA, tDQrB = sm90_utils.partition_fragment_ABC(wg_mma_dQ, shape_mnk_dQ, sPdS, sKt_t, swap_AB=False)
        mma_dQ_fn = partial(gemm_w_idx, tiled_mma_dQ, acc_dQ, tDQrA, tDQrB, swap_AB=False)

        # -- dS R2S copy setup --
        copy_dS_r2s, _, _ = copy_utils.get_smem_store_C(
            tiled_mma_SdP,
            sPdS,
            tidx,
            self.arch,
            transpose=False,
            position_independent=True,
        )

        # -- LSE/dPsum partitioning (single-stage, per-thread row mapping) --
        thr_mma_SdP = tiled_mma_SdP.get_slice(tidx)
        tLSEsLSE = layout_utils.mma_partition_C_vec(sLSE, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=True)
        tLSEsdPsum = layout_utils.mma_partition_C_vec(sdPsum, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=True)

        PdS_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwd.PdS),
            num_threads=self.num_mma_threads,
        )

        consumer_state_QdO = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.QdO_preload_stages
        )
        consumer_state_B = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.B_stage
        )
        consumer_state_Kt = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Kt_stage
        )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()

        # Pre-allocate acc_S (reused across n_blocks)
        acc_S = cute.make_rmem_tensor(tiled_mma_SdP.partition_shape_C(shape_mnk_SdP[:2]), Float32)

        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)

            process_tile = const_expr(not self.is_varlen_k) or n_block_min < n_block_max

            mask = AttentionMask(
                self.tile_m,
                self.tile_n,
                seqlen,
            )

            # ═══ Wait for Q/dO preload (all 8 stages) ═══
            for _qdo in cutlass.range_constexpr(self.QdO_preload_stages):
                pipeline_QdO.consumer_wait(
                    consumer_state_QdO,
                    pipeline_QdO.consumer_try_wait(consumer_state_QdO),
                )
                pipeline_QdO.consumer_release(consumer_state_QdO)
                consumer_state_QdO.advance()

            # Read LSE and dPsum into registers (single-stage, always index 0)
            tLSErLSE = copy_utils.load_s2r(tLSEsLSE[None, 0])
            tLSErdPsum = copy_utils.load_s2r(tLSEsdPsum[None, 0])

            # -- Outer d_pass loop --
            for d_pass in cutlass.range_constexpr(self.num_d_passes):
                dQ_accumulate = False

                for i_n in cutlass.range(n_block_max - n_block_min, unroll=1):
                    n_block = n_block_max - 1 - i_n  # inverse order

                    mask_fn = partial(
                        mask.apply_mask,
                        batch_idx=batch_idx,
                        head_idx=head_idx,
                        m_block=m_block,
                        n_block=n_block,
                        thr_mma=thr_mma_SdP,
                        mask_seqlen=True,
                        mask_causal=self.is_causal,
                    )

                    consumer_state_B, consumer_state_Kt = self.mma_one_n_block_splitd(
                        consumer_state_B,
                        consumer_state_Kt,
                        tiled_mma_SdP,
                        tSrA_Q,
                        tSrA_dO,
                        tSrB,
                        shape_mnk_SdP,
                        acc_S,
                        mma_dQ_fn,
                        copy_dS_r2s,
                        pipeline_B,
                        pipeline_Kt,
                        tLSErLSE,
                        tLSErdPsum,
                        softmax_scale_log2,
                        softmax_scale,
                        mask_fn,
                        PdS_barrier,
                        dQ_accumulate=dQ_accumulate,
                    )
                    dQ_accumulate = True

                # -- dQ epilogue (scale already absorbed into dS) --
                if process_tile:
                    self.epilogue_dQ_slice(
                        acc_dQ,
                        mdQ,
                        sA_epi,
                        seqlen,
                        tma_atom_dQ,
                        tiled_mma_dQ,
                        tidx,
                        m_block,
                        head_idx,
                        batch_idx,
                        d_pass,
                    )

                # d_pass barrier: sync with producer before next d_pass
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
    def mma_one_n_block_splitd(
        self,
        consumer_state_B: cutlass.pipeline.PipelineState,
        consumer_state_Kt: cutlass.pipeline.PipelineState,
        tiled_mma_SdP: cute.TiledMma,
        tSrA_Q: cute.Tensor,
        tSrA_dO: cute.Tensor,
        tSrB: cute.Tensor,
        shape_mnk_SdP: cute.Shape,
        acc_S: cute.Tensor,
        mma_dQ_fn: Callable,
        copy_dS_r2s: Callable,
        pipeline_B: pipeline.PipelineAsync,
        pipeline_Kt: pipeline.PipelineAsync,
        tLSErLSE: cute.Tensor,
        tLSErdPsum: cute.Tensor,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        mask_fn: Callable,
        PdS_barrier: cutlass.pipeline.NamedBarrier,
        dQ_accumulate: Boolean = True,
    ):
        # =============================================================
        # Phase A: S = Q @ K^T  (4 x d_inner=128 reduction)
        # Q is persistent (A_idx=d_inner), only K streams via pipeline_B.
        # Delayed release for pipeline_B: release trails consumer by 2.
        # =============================================================
        release_state_B_pA = consumer_state_B.clone()

        for d_inner in cutlass.range_constexpr(self.num_d_inner):
            pipeline_B.consumer_wait(
                consumer_state_B,
                pipeline_B.consumer_try_wait(consumer_state_B),
            )

            if cutlass.const_expr(d_inner < self.num_d_inner - 1):
                gemm_w_idx(
                    tiled_mma_SdP,
                    acc_S,
                    tSrA_Q,
                    tSrB,
                    zero_init=(d_inner == 0),
                    A_idx=d_inner,
                    B_idx=consumer_state_B.index,
                    wg_wait=-1,
                )
                if cutlass.const_expr(d_inner >= 2):
                    warpgroup.wait_group(2)
                    pipeline_B.consumer_release(release_state_B_pA)
                    release_state_B_pA.advance()
            else:
                gemm_w_idx(
                    tiled_mma_SdP,
                    acc_S,
                    tSrA_Q,
                    tSrB,
                    zero_init=(d_inner == 0),
                    A_idx=d_inner,
                    B_idx=consumer_state_B.index,
                    wg_wait=0,
                )
                pipeline_B.consumer_release(release_state_B_pA)
                release_state_B_pA.advance()

            consumer_state_B.advance()

        # Trailing releases for the last 2 iterations
        for _ in cutlass.range_constexpr(2):
            pipeline_B.consumer_release(release_state_B_pA)
            release_state_B_pA.advance()

        # =============================================================
        # Phase B: P = exp2(S * scale_log2 - LSE)
        # =============================================================
        mask_fn(acc_S)
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=False)
        for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
            lse_val = tLSErLSE[r]
            for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
                acc_S_mn[r, c] = cute.math.exp2(
                    acc_S_mn[r, c] * softmax_scale_log2 - lse_val,
                    fastmath=True,
                )
        # P fp32 -> fp16, keep in registers for Phase D
        tdQrP = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_S), self.dtype)

        # =============================================================
        # Phase C: dP = dO @ V^T  (4 x d_inner=128 reduction)
        # dO is persistent (A_idx=d_inner), only V streams via pipeline_B.
        # =============================================================
        release_state_B_pC = consumer_state_B.clone()

        acc_dP = cute.make_rmem_tensor(tiled_mma_SdP.partition_shape_C(shape_mnk_SdP[:2]), Float32)
        for d_inner in cutlass.range_constexpr(self.num_d_inner):
            pipeline_B.consumer_wait(
                consumer_state_B,
                pipeline_B.consumer_try_wait(consumer_state_B),
            )

            if cutlass.const_expr(d_inner < self.num_d_inner - 1):
                gemm_w_idx(
                    tiled_mma_SdP,
                    acc_dP,
                    tSrA_dO,
                    tSrB,
                    zero_init=(d_inner == 0),
                    A_idx=d_inner,
                    B_idx=consumer_state_B.index,
                    wg_wait=-1,
                )
                if cutlass.const_expr(d_inner >= 2):
                    warpgroup.wait_group(2)
                    pipeline_B.consumer_release(release_state_B_pC)
                    release_state_B_pC.advance()
            else:
                gemm_w_idx(
                    tiled_mma_SdP,
                    acc_dP,
                    tSrA_dO,
                    tSrB,
                    zero_init=(d_inner == 0),
                    A_idx=d_inner,
                    B_idx=consumer_state_B.index,
                    wg_wait=0,
                )
                pipeline_B.consumer_release(release_state_B_pC)
                release_state_B_pC.advance()

            consumer_state_B.advance()

        # Trailing releases for the last 2 iterations
        for _ in cutlass.range_constexpr(2):
            pipeline_B.consumer_release(release_state_B_pC)
            release_state_B_pC.advance()

        # =============================================================
        # Phase D: dS = P * (dP - dPsum), write dS -> sPdS
        # =============================================================
        acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP, transpose=False)
        for r in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[0])):
            dpsum_val = tLSErdPsum[r]
            for c in cutlass.range(cute.size(acc_dP_mn, mode=[1]), unroll_full=True):
                acc_dP_mn[r, c] = acc_S_mn[r, c] * (acc_dP_mn[r, c] - dpsum_val) * softmax_scale

        # dS_scaled fp32 -> fp16, write to sPdS
        tdQrdS = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_dP), self.dtype)
        cute.arch.fence_view_async_shared()
        PdS_barrier.arrive_and_wait()
        copy_dS_r2s(tdQrdS, dst_idx=0)

        # =============================================================
        # Phase E: dQ_acc += dS @ K_d_pass
        # =============================================================
        cute.arch.fence_view_async_shared()
        PdS_barrier.arrive_and_wait()

        pipeline_Kt.consumer_wait(
            consumer_state_Kt,
            pipeline_Kt.consumer_try_wait(consumer_state_Kt),
        )
        mma_dQ_fn(
            A_idx=0,  # sPdS single stage
            B_idx=consumer_state_Kt.index,
            zero_init=not dQ_accumulate,
            wg_wait=0,
        )
        pipeline_Kt.consumer_release(consumer_state_Kt)
        consumer_state_Kt.advance()

        return consumer_state_B, consumer_state_Kt

    # ------------------------------------------------------------------
    # Epilogue: write dQ slice to global memory
    # ------------------------------------------------------------------
    @cute.jit
    def epilogue_dQ_slice(
        self,
        acc_dQ: cute.Tensor,
        mdQ: cute.Tensor,
        sA_buf: cute.Tensor,  # dedicated epilogue staging buffer (sEpi)
        seqlen: SeqlenInfoQK,
        tma_atom_dQ: cute.CopyAtom,
        tiled_mma_dQ: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        d_pass: Int32,
    ):
        epi_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwd.Epilogue),
            num_threads=self.num_mma_threads,
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Global tile address for current d_pass slice
        mdQ_cur = seqlen.offset_batch_Q(mdQ, batch_idx, dim=3, ragged=self.varlen_q)[None, None, head_idx]
        gdQ = cute.local_tile(mdQ_cur, (self.tile_m, self.d_chunk), (m_block, d_pass))

        # TMA store function (sA_buf is single-stage epilogue view)
        store_dQ, _, _ = copy_utils.tma_get_copy_fn(tma_atom_dQ, 0, cute.make_layout(1), sA_buf, gdQ, single_stage=True)

        # R2S copy for dQ (acc from tiled_mma_dQ -> sA_buf)
        copy_dQ_r2s, _, _ = copy_utils.get_smem_store_C(
            tiled_mma_dQ,
            sA_buf,
            tidx,
            self.arch,
            transpose=False,
            position_independent=True,
        )

        # -- Write dQ --
        cute.arch.cp_async_bulk_wait_group(1, read=True)
        epi_barrier.arrive_and_wait()
        copy_dQ_r2s(acc_dQ, dst_idx=None)
        cute.arch.fence_view_async_shared()
        epi_barrier.arrive_and_wait()
        if warp_idx == 4:
            store_dQ()
            cute.arch.cp_async_bulk_commit_group()
