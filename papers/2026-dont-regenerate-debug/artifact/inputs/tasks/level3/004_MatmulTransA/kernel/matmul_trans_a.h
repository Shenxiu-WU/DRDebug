/**
 * @file matmul_trans_a.h
 *
 * Persistent cube wrapper around `MatmulKernel` for the MatmulTransA op.
 *
 * Layout:
 *   - host already supplied A.T (shape M x K) and B (shape K x N) padded
 *     to multiples of `baseM/baseN/baseK`, where M=padded_n, N=padded_k,
 *     K=padded_m.
 *   - the cube schedules (M / baseM) x (N / baseN) baseM x baseN output
 *     tiles round-robin across `usedCoreNum` AIC cores; AIV is idle (we
 *     still ask for KERNEL_TYPE_MIX_AIC_1_2 so the runtime accepts the
 *     launch on a Mix-architecture device).
 *   - each tile is computed end-to-end on one AIC; Fixpipe writes the
 *     baseM x baseN result straight to GM at `(mIdx*baseM, nIdx*baseN)`,
 *     casting from float accumulator to `outType` via `quantPre`.
 */
#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"
#include "kernel_common.h"
#include "matmul_trans_a_tiling.h"
#include "matmul.h"

template <typename aType, typename outType>
class MatmulTransAKernel {
public:
    __aicore__ inline MatmulTransAKernel() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tilingGM,
                                AscendC::TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline QuantMode_t SelectQuantPre();

    MatmulKernel<aType, outType> mm_;
    AscendC::GlobalTensor<aType>   aGM_;
    AscendC::GlobalTensor<aType>   bGM_;
    AscendC::GlobalTensor<outType> cGM_;
    MatmulTransATiling tiling_;
    BlockScheduler sched_;
};

template <typename aType, typename outType>
__aicore__ inline QuantMode_t
MatmulTransAKernel<aType, outType>::SelectQuantPre()
{
    if constexpr (std::is_same<outType, half>::value) {
        return QuantMode_t::F322F16;
    } else if constexpr (std::is_same<outType, bfloat16_t>::value) {
        return QuantMode_t::F322BF16;
    } else {
        // float -> float, no cast
        return QuantMode_t::NoQuant;
    }
}

template <typename aType, typename outType>
__aicore__ inline void MatmulTransAKernel<aType, outType>::Init(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tilingGM, AscendC::TPipe *pipe)
{
    CopyTiling(&tiling_, tilingGM);

    aGM_.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a),
                         static_cast<uint64_t>(tiling_.M) * tiling_.K);
    bGM_.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(b),
                         static_cast<uint64_t>(tiling_.K) * tiling_.N);
    cGM_.SetGlobalBuffer(reinterpret_cast<__gm__ outType *>(c),
                         static_cast<uint64_t>(tiling_.M) * tiling_.N);

    int coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
    sched_.Init(tiling_.M, tiling_.N, tiling_.baseM, tiling_.baseN,
                AscendC::GetBlockNum(), coreIdx);

    if ASCEND_IS_AIC {
        mm_.Init(tiling_.K,            // reduction dim
                 tiling_.K,            // lda  (A row stride = K)
                 tiling_.N,            // ldb  (B row stride = N)
                 tiling_.baseM, tiling_.baseN, tiling_.baseK,
                 tiling_.l1Prefetch,
                 *pipe);
    }
}

template <typename aType, typename outType>
__aicore__ inline void MatmulTransAKernel<aType, outType>::Process()
{
    if ASCEND_IS_AIV {
        return; // pure cube kernel
    }

    QuantMode_t quantPre = SelectQuantPre();

    int mIdx, nIdx;
    while (sched_.HasNext()) {
        sched_.Next(mIdx, nIdx);
        auto aBlock = aGM_[mIdx * tiling_.baseM * tiling_.K];
        auto bBlock = bGM_[nIdx * tiling_.baseN];
        auto cBlock = cGM_[mIdx * tiling_.baseM * tiling_.N + nIdx * tiling_.baseN];
        mm_.ComputeBlock(aBlock, bBlock, cBlock, tiling_.N, quantPre);
    }
}
