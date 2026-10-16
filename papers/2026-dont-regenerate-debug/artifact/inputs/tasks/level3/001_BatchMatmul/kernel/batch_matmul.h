#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"
#include "kernel_common.h"
#include "matmul.h"
#include "batch_matmul_tiling.h"

using namespace AscendC;

template <typename aType, typename bType, typename cType>
class BatchMatmulKernel {
public:
    __aicore__ inline BatchMatmulKernel() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tiling, AscendC::TPipe *pipe);
    __aicore__ inline void Process();

    MatmulKernel<aType, bType, cType> mm_;
    AscendC::GlobalTensor<aType> aGM_;
    AscendC::GlobalTensor<bType> bGM_;
    AscendC::GlobalTensor<cType> cGM_;
    BatchMatmulTiling tiling_;
    BatchBlockScheduler sched_;
};

template <typename aType, typename bType, typename cType>
__aicore__ inline void BatchMatmulKernel<aType, bType, cType>::Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tilingGM, AscendC::TPipe *pipe)
{
    CopyTiling(&this->tiling_, tilingGM);

    aGM_.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), tiling_.B * tiling_.M * tiling_.K);
    bGM_.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), tiling_.B * tiling_.K * tiling_.N);
    cGM_.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), tiling_.B * tiling_.M * tiling_.N);

    int coreIdx = AscendC::GetBlockIdx();
    sched_.Init(tiling_.B, tiling_.M, tiling_.N, tiling_.baseM, tiling_.baseN,
                AscendC::GetBlockNum(), coreIdx);

    if ASCEND_IS_AIC {
        mm_.Init(tiling_.K, tiling_.K, tiling_.N,
                 tiling_.baseM, tiling_.baseN, tiling_.baseK, 1,
                 *pipe);
    }
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void BatchMatmulKernel<aType, bType, cType>::Process()
{
    if ASCEND_IS_AIV {
        return;
    }
    int bIdx, mIdx, nIdx;
    while (sched_.HasNext()) {
        sched_.Next(bIdx, mIdx, nIdx);
        if ASCEND_IS_AIC {
            auto aBlock = aGM_[bIdx * tiling_.M * tiling_.K + mIdx * tiling_.baseM * tiling_.K];
            auto bBlock = bGM_[bIdx * tiling_.K * tiling_.N + nIdx * tiling_.baseN];
            auto cBlock = cGM_[bIdx * tiling_.M * tiling_.N + mIdx * tiling_.baseM * tiling_.N + nIdx * tiling_.baseN];
            mm_.ComputeBlock(aBlock, bBlock, cBlock, tiling_.N);
        }
    }
}
