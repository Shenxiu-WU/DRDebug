/**
 * @file matmul.h
 *
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef MATMUL_CUSTOM_H
#define MATMUL_CUSTOM_H

#include "kernel_operator.h"
#include "matmul_tile.h"
#include "matmul_leakyrelu_tiling.h"

template <typename aType, typename bType, typename cType>
class MatmulKernel {
    static_assert(std::is_same<aType, bType>::value, "aType and bType must be the same type");

    // Nz fractal c0 size: 32 bytes / element size
    static constexpr uint32_t C0 = 32 / sizeof(aType);

public:
    __aicore__ inline MatmulKernel() {}
    __aicore__ inline void Init(uint32_t k, uint32_t lda, uint32_t ldb,
                                uint32_t baseM, uint32_t baseN, uint32_t baseK,
                                uint32_t l1Prefetch,
                                AscendC::TPipe &pipe);
    __aicore__ inline void ComputeBlock(const AscendC::GlobalTensor<aType> &aBlock,
                                        const AscendC::GlobalTensor<bType> &bBlock,
                                        const AscendC::GlobalTensor<cType> &cBlock);

private:
    __aicore__ inline void CopyA(const AscendC::GlobalTensor<aType> &A, uint32_t kLen);
    __aicore__ inline void CopyB(const AscendC::GlobalTensor<bType> &B, uint32_t kLen);
    __aicore__ inline void SplitA(const AscendC::LocalTensor<aType> &a1Local,
                                  uint32_t offset, uint32_t colC0Stride);
    __aicore__ inline void SplitB(const AscendC::LocalTensor<bType> &b1Local,
                                  uint32_t offset, uint32_t colC0Stride);
    __aicore__ inline void Compute(const AscendC::LocalTensor<cType> &c1Local, bool cmatrixInitVal);
    __aicore__ inline void CopyOut(const AscendC::GlobalTensor<cType> &C);

private:
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
    AscendC::TQue<AscendC::TPosition::A2, 1> inQueueA2;
    AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1;
    AscendC::TQue<AscendC::TPosition::B2, 1> inQueueB2;
    AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1;

    uint32_t l1Prefetch_;
    uint32_t baseM_, baseN_, baseK_;
    uint32_t baseMK_, baseKN_, baseMN_;
    uint32_t k;
    uint32_t aDValue, bDValue;
};

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernel<aType, bType, cType>::Init(
    uint32_t k, uint32_t lda, uint32_t ldb,
    uint32_t baseM, uint32_t baseN, uint32_t baseK,
    uint32_t l1Prefetch,
    AscendC::TPipe &pipe)
{
    ASSERT(k % baseK == 0);

    l1Prefetch_ = l1Prefetch;
    baseM_ = baseM;
    baseN_ = baseN;
    baseK_ = baseK;
    baseMK_ = baseM_ * baseK_;
    baseKN_ = baseK_ * baseN_;
    baseMN_ = baseM_ * baseN_;
    this->k = k;
    aDValue = lda;
    bDValue = ldb;

    // L1 buffers: each holds l1Prefetch_ k-tiles
    pipe.InitBuffer(inQueueA1, 2, baseMK_ * l1Prefetch_ * sizeof(aType));
    pipe.InitBuffer(inQueueA2, 2, baseMK_ * sizeof(aType));
    pipe.InitBuffer(inQueueB1, 2, baseKN_ * l1Prefetch_ * sizeof(bType));
    pipe.InitBuffer(inQueueB2, 2, baseKN_ * sizeof(bType));
    pipe.InitBuffer(outQueueCO1, 1, baseMN_ * sizeof(cType));
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernel<aType, bType, cType>::ComputeBlock(const AscendC::GlobalTensor<aType> &aBlock,
                                                                        const AscendC::GlobalTensor<bType> &bBlock,
                                                                        const AscendC::GlobalTensor<cType> &cBlock)
{
    AscendC::LocalTensor<cType> c1Local = outQueueCO1.AllocTensor<cType>();
    uint32_t kTiles = k / baseK_;

    for (uint32_t outer = 0; outer < kTiles; outer += l1Prefetch_) {
        uint32_t count = (kTiles - outer < l1Prefetch_) ? (kTiles - outer) : l1Prefetch_;
        uint32_t kLen = count * baseK_;

        // One DMA per matrix: GM -> L1
        CopyA(aBlock[outer * baseK_], kLen);
        CopyB(bBlock[outer * baseK_ * bDValue], kLen);

        // Wait for L1 ready (one sync per batch)
        AscendC::LocalTensor<aType> a1Local = inQueueA1.DeQue<aType>();
        AscendC::LocalTensor<bType> b1Local = inQueueB1.DeQue<bType>();

        // Inner loop: L1 -> L0 -> MMAD
        // A Nz [kLen/c0, baseM/16, 16, c0]: K outermost, sub-blocks contiguous
        //   offset = i * baseMK, colC0Stride = baseM (constant)
        // B Nz [baseN/c0, kLen/16, 16, c0]: N outermost, sub-blocks strided
        //   offset = i * baseK * c0, colC0Stride = kLen (varies with batch)
        for (uint32_t i = 0; i < count; i++) {
            SplitA(a1Local, i * baseMK_, baseM_);
            SplitB(b1Local, i * baseK_ * C0, kLen);
            Compute(c1Local, (outer + i == 0));
        }

        // Release L1 buffers (one sync per batch)
        inQueueA1.FreeTensor(a1Local);
        inQueueB1.FreeTensor(b1Local);
    }

    outQueueCO1.EnQue(c1Local);
    CopyOut(cBlock);
}

// GM -> L1: one DMA loads [baseM, kLen] for A
template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernel<aType, bType, cType>::CopyA(
    const AscendC::GlobalTensor<aType> &A, uint32_t kLen)
{
    AscendC::LocalTensor<aType> a1Local = inQueueA1.AllocTensor<aType>();
    LoadNdGmToNzL1(a1Local, A, baseM_, kLen, aDValue);
    inQueueA1.EnQue(a1Local);
}

// GM -> L1: one DMA loads [kLen, baseN] for B
template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernel<aType, bType, cType>::CopyB(
    const AscendC::GlobalTensor<bType> &B, uint32_t kLen)
{
    AscendC::LocalTensor<bType> b1Local = inQueueB1.AllocTensor<bType>();
    LoadNdGmToNzL1(b1Local, B, kLen, baseN_, bDValue);
    inQueueB1.EnQue(b1Local);
}

// L1 -> L0A: extract [baseM, baseK] sub-block from big Nz buffer
template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernel<aType, bType, cType>::SplitA(
    const AscendC::LocalTensor<aType> &a1Local, uint32_t offset, uint32_t colC0Stride)
{
    AscendC::LocalTensor<aType> a2Local = inQueueA2.AllocTensor<aType>();
    LoadNzL1ToZzL0A(a2Local, a1Local[offset], baseM_, baseK_, colC0Stride);
    inQueueA2.EnQue(a2Local);
}

// L1 -> L0B: extract [baseK, baseN] sub-block from big Nz buffer
template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernel<aType, bType, cType>::SplitB(
    const AscendC::LocalTensor<bType> &b1Local, uint32_t offset, uint32_t colC0Stride)
{
    AscendC::LocalTensor<bType> b2Local = inQueueB2.AllocTensor<bType>();
    LoadNzL1ToZnL0B(b2Local, b1Local[offset], baseK_, baseN_, colC0Stride);
    inQueueB2.EnQue(b2Local);
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernel<aType, bType, cType>::Compute(
    const AscendC::LocalTensor<cType> &c1Local, bool cmatrixInitVal)
{
    AscendC::LocalTensor<aType> a2Local = inQueueA2.DeQue<aType>();
    AscendC::LocalTensor<bType> b2Local = inQueueB2.DeQue<bType>();
    AscendC::MmadParams mmadParams;
    mmadParams.m = baseM_;
    mmadParams.n = baseN_;
    mmadParams.k = baseK_;
    mmadParams.cmatrixInitVal = cmatrixInitVal;
    AscendC::Mmad(c1Local, a2Local, b2Local, mmadParams);
    inQueueA2.FreeTensor(a2Local);
    inQueueB2.FreeTensor(b2Local);
}

// Nz->Nd
template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernel<aType, bType, cType>::CopyOut(const AscendC::GlobalTensor<cType> &C)
{
    auto c1Local = outQueueCO1.DeQue<cType>();
    FixpipeNzL0cToNdGm(C, c1Local, baseM_, baseN_);
    outQueueCO1.FreeTensor(c1Local);
}

#endif // MATMUL_CUSTOM_H
