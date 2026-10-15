/**
 * @file matmul.h
 *
 * Templated cube-only GEMM kernel used by MatmulTransA.
 *
 * Logical contract:
 *   C[M, N] = A[M, K] @ B[K, N]
 *
 * Caller is responsible for transposing the original A, padding to
 * multiples of `baseM/baseN/baseK`, and dispatching the right launcher
 * for each input dtype.
 *
 * `aType` is the dtype of A and B in GM. `outType` is the dtype of C
 * in GM. The cube always accumulates in `float`; if `outType` differs
 * from `float` the Fixpipe `quantPre` argument carries the down-cast.
 */
#ifndef MATMUL_CUSTOM_H
#define MATMUL_CUSTOM_H

#include "kernel_operator.h"
#include "matmul_tile.h"
#include "matmul_trans_a_tiling.h"

template <typename aType, typename outType>
class MatmulKernel {
    using accType = float;
    static constexpr uint32_t C0 = 32 / sizeof(aType);

public:
    __aicore__ inline MatmulKernel() {}
    __aicore__ inline void Init(uint32_t k,
                                uint32_t lda, uint32_t ldb,
                                uint32_t baseM, uint32_t baseN, uint32_t baseK,
                                uint32_t l1Prefetch,
                                AscendC::TPipe &pipe);

    __aicore__ inline void ComputeBlock(const AscendC::GlobalTensor<aType> &aBlock,
                                        const AscendC::GlobalTensor<aType> &bBlock,
                                        const AscendC::GlobalTensor<outType> &cBlock,
                                        uint32_t ldc,
                                        QuantMode_t quantPre);

private:
    __aicore__ inline void CopyA(const AscendC::GlobalTensor<aType> &A, uint32_t kLen);
    __aicore__ inline void CopyB(const AscendC::GlobalTensor<aType> &B, uint32_t kLen);
    __aicore__ inline void SplitA(const AscendC::LocalTensor<aType> &a1Local,
                                  uint32_t offset, uint32_t colC0Stride);
    __aicore__ inline void SplitB(const AscendC::LocalTensor<aType> &b1Local,
                                  uint32_t offset, uint32_t colC0Stride);
    __aicore__ inline void Compute(const AscendC::LocalTensor<accType> &c1Local,
                                   bool cmatrixInitVal);
    __aicore__ inline void CopyOut(const AscendC::GlobalTensor<outType> &C,
                                   uint32_t ldc,
                                   QuantMode_t quantPre);

private:
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
    AscendC::TQue<AscendC::TPosition::A2, 1> inQueueA2;
    AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1;
    AscendC::TQue<AscendC::TPosition::B2, 1> inQueueB2;
    AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1;

    uint32_t l1Prefetch_;
    uint32_t baseM_, baseN_, baseK_;
    uint32_t baseMK_, baseKN_, baseMN_;
    uint32_t k_;
    uint32_t aDValue_, bDValue_;
};

template <typename aType, typename outType>
__aicore__ inline void MatmulKernel<aType, outType>::Init(
    uint32_t k, uint32_t lda, uint32_t ldb,
    uint32_t baseM, uint32_t baseN, uint32_t baseK,
    uint32_t l1Prefetch, AscendC::TPipe &pipe)
{
    ASSERT(k % baseK == 0);

    l1Prefetch_ = l1Prefetch;
    baseM_ = baseM;
    baseN_ = baseN;
    baseK_ = baseK;
    baseMK_ = baseM_ * baseK_;
    baseKN_ = baseK_ * baseN_;
    baseMN_ = baseM_ * baseN_;
    k_ = k;
    aDValue_ = lda;
    bDValue_ = ldb;

    pipe.InitBuffer(inQueueA1, 2, baseMK_ * l1Prefetch_ * sizeof(aType));
    pipe.InitBuffer(inQueueA2, 2, baseMK_ * sizeof(aType));
    pipe.InitBuffer(inQueueB1, 2, baseKN_ * l1Prefetch_ * sizeof(aType));
    pipe.InitBuffer(inQueueB2, 2, baseKN_ * sizeof(aType));
    pipe.InitBuffer(outQueueCO1, 1, baseMN_ * sizeof(accType));
}

template <typename aType, typename outType>
__aicore__ inline void MatmulKernel<aType, outType>::ComputeBlock(
    const AscendC::GlobalTensor<aType> &aBlock,
    const AscendC::GlobalTensor<aType> &bBlock,
    const AscendC::GlobalTensor<outType> &cBlock,
    uint32_t ldc,
    QuantMode_t quantPre)
{
    AscendC::LocalTensor<accType> c1Local = outQueueCO1.AllocTensor<accType>();
    uint32_t kTiles = k_ / baseK_;

    for (uint32_t outer = 0; outer < kTiles; outer += l1Prefetch_) {
        uint32_t count = (kTiles - outer < l1Prefetch_) ? (kTiles - outer) : l1Prefetch_;
        uint32_t kLen = count * baseK_;

        CopyA(aBlock[outer * baseK_], kLen);
        CopyB(bBlock[outer * baseK_ * bDValue_], kLen);

        AscendC::LocalTensor<aType> a1Local = inQueueA1.DeQue<aType>();
        AscendC::LocalTensor<aType> b1Local = inQueueB1.DeQue<aType>();

        for (uint32_t i = 0; i < count; i++) {
            SplitA(a1Local, i * baseMK_, baseM_);
            SplitB(b1Local, i * baseK_ * C0, kLen);
            Compute(c1Local, (outer + i == 0));
        }

        inQueueA1.FreeTensor(a1Local);
        inQueueB1.FreeTensor(b1Local);
    }

    // Drain all pipes (notably cube) so the final Mmad has finished
    // writing L0C before Fixpipe is allowed to read it.
    AscendC::PipeBarrier<PIPE_ALL>();
    outQueueCO1.EnQue(c1Local);
    CopyOut(cBlock, ldc, quantPre);
}

template <typename aType, typename outType>
__aicore__ inline void MatmulKernel<aType, outType>::CopyA(
    const AscendC::GlobalTensor<aType> &A, uint32_t kLen)
{
    AscendC::LocalTensor<aType> a1Local = inQueueA1.AllocTensor<aType>();
    LoadNdGmToNzL1(a1Local, A, baseM_, kLen, aDValue_);
    inQueueA1.EnQue(a1Local);
}

template <typename aType, typename outType>
__aicore__ inline void MatmulKernel<aType, outType>::CopyB(
    const AscendC::GlobalTensor<aType> &B, uint32_t kLen)
{
    AscendC::LocalTensor<aType> b1Local = inQueueB1.AllocTensor<aType>();
    LoadNdGmToNzL1(b1Local, B, kLen, baseN_, bDValue_);
    inQueueB1.EnQue(b1Local);
}

template <typename aType, typename outType>
__aicore__ inline void MatmulKernel<aType, outType>::SplitA(
    const AscendC::LocalTensor<aType> &a1Local, uint32_t offset, uint32_t colC0Stride)
{
    AscendC::LocalTensor<aType> a2Local = inQueueA2.AllocTensor<aType>();
    LoadNzL1ToZzL0A(a2Local, a1Local[offset], baseM_, baseK_, colC0Stride);
    inQueueA2.EnQue(a2Local);
}

template <typename aType, typename outType>
__aicore__ inline void MatmulKernel<aType, outType>::SplitB(
    const AscendC::LocalTensor<aType> &b1Local, uint32_t offset, uint32_t colC0Stride)
{
    AscendC::LocalTensor<aType> b2Local = inQueueB2.AllocTensor<aType>();
    LoadNzL1ToZnL0B(b2Local, b1Local[offset], baseK_, baseN_, colC0Stride);
    inQueueB2.EnQue(b2Local);
}

template <typename aType, typename outType>
__aicore__ inline void MatmulKernel<aType, outType>::Compute(
    const AscendC::LocalTensor<accType> &c1Local, bool cmatrixInitVal)
{
    AscendC::LocalTensor<aType> a2Local = inQueueA2.DeQue<aType>();
    AscendC::LocalTensor<aType> b2Local = inQueueB2.DeQue<aType>();
    AscendC::MmadParams mmadParams;
    mmadParams.m = baseM_;
    mmadParams.n = baseN_;
    mmadParams.k = baseK_;
    mmadParams.cmatrixInitVal = cmatrixInitVal;
    AscendC::Mmad(c1Local, a2Local, b2Local, mmadParams);
    // Mmad runs on the cube pipe while the next iteration's LoadData
    // (L1->L0A/L0B) runs on MTE1; without this barrier the L0A/L0B
    // buffers can be reused while Mmad is still reading them.
    AscendC::PipeBarrier<PIPE_ALL>();
    inQueueA2.FreeTensor(a2Local);
    inQueueB2.FreeTensor(b2Local);
}

template <typename aType, typename outType>
__aicore__ inline void MatmulKernel<aType, outType>::CopyOut(
    const AscendC::GlobalTensor<outType> &C, uint32_t ldc,
    QuantMode_t quantPre)
{
    auto c1Local = outQueueCO1.DeQue<accType>();
    FixpipeNzL0cToNdGm(C, c1Local, baseM_, baseN_, ldc, quantPre);
    // Fixpipe runs on the FIX pipe; make sure it has finished reading
    // L0C before the (depth-1) CO1 buffer is released, otherwise the
    // next tile's Mmad(cmatrixInitVal=true) can overwrite L0C mid-read.
    AscendC::PipeBarrier<PIPE_FIX>();
    outQueueCO1.FreeTensor(c1Local);
}

#endif // MATMUL_CUSTOM_H
