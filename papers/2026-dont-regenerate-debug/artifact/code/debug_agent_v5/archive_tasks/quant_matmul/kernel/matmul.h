#ifndef MATMUL_H
#define MATMUL_H

#include "kernel_operator.h"
#include "matmul_tile.h"
#include "int8_matmul_scale_tiling.h"

constexpr uint32_t baseM = DEFAULT_BASE_M;
constexpr uint32_t baseN = DEFAULT_BASE_N;
constexpr uint32_t baseK = DEFAULT_BASE_K;

constexpr uint32_t baseMK = baseM * baseK;
constexpr uint32_t baseKN = baseK * baseN;
constexpr uint32_t baseMN = baseM * baseN;

constexpr uint32_t L1_PREFETCH = 3;

template <typename aType, typename bType, typename cType>
class MatmulKernel {
    static_assert(std::is_same<aType, bType>::value, "aType and bType must be the same type");
    static constexpr uint32_t C0 = 32 / sizeof(aType);

public:
    __aicore__ inline MatmulKernel() {}
    __aicore__ inline void Init(uint32_t k, uint32_t lda, uint32_t ldb, AscendC::TPipe &pipe);
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

    uint32_t k;
    uint32_t aDValue, bDValue;
};

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernel<aType, bType, cType>::Init(uint32_t k, uint32_t lda, uint32_t ldb, AscendC::TPipe &pipe)
{
    ASSERT(k % baseK == 0);
    this->k = k;
    aDValue = lda;
    bDValue = ldb;

    pipe.InitBuffer(inQueueA1, 2, baseMK * L1_PREFETCH * sizeof(aType));
    pipe.InitBuffer(inQueueA2, 2, baseMK * sizeof(aType));
    pipe.InitBuffer(inQueueB1, 2, baseKN * L1_PREFETCH * sizeof(bType));
    pipe.InitBuffer(inQueueB2, 2, baseKN * sizeof(bType));
    pipe.InitBuffer(outQueueCO1, 1, baseMN * sizeof(cType));
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernel<aType, bType, cType>::ComputeBlock(const AscendC::GlobalTensor<aType> &aBlock,
                                                                        const AscendC::GlobalTensor<bType> &bBlock,
                                                                        const AscendC::GlobalTensor<cType> &cBlock)
{
    AscendC::LocalTensor<cType> c1Local = outQueueCO1.AllocTensor<cType>();
    uint32_t kTiles = k / baseK;

    for (uint32_t outer = 0; outer < kTiles; outer += L1_PREFETCH) {
        uint32_t count = (kTiles - outer < L1_PREFETCH) ? (kTiles - outer) : L1_PREFETCH;
        uint32_t kLen = count * baseK;

        CopyA(aBlock[outer * baseK], kLen);
        CopyB(bBlock[outer * baseK * bDValue], kLen);

        AscendC::LocalTensor<aType> a1Local = inQueueA1.DeQue<aType>();
        AscendC::LocalTensor<bType> b1Local = inQueueB1.DeQue<bType>();

        for (uint32_t i = 0; i < count; i++) {
            SplitA(a1Local, i * baseMK, baseM);
            SplitB(b1Local, i * baseK * C0, kLen);
            Compute(c1Local, (outer + i == 0));
        }

        inQueueA1.FreeTensor(a1Local);
        inQueueB1.FreeTensor(b1Local);
    }

    outQueueCO1.EnQue(c1Local);
    CopyOut(cBlock);
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernel<aType, bType, cType>::CopyA(
    const AscendC::GlobalTensor<aType> &A, uint32_t kLen)
{
    AscendC::LocalTensor<aType> a1Local = inQueueA1.AllocTensor<aType>();
    LoadNdGmToNzL1(a1Local, A, baseM, kLen, aDValue);
    inQueueA1.EnQue(a1Local);
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernel<aType, bType, cType>::CopyB(
    const AscendC::GlobalTensor<bType> &B, uint32_t kLen)
{
    AscendC::LocalTensor<bType> b1Local = inQueueB1.AllocTensor<bType>();
    LoadNdGmToNzL1(b1Local, B, kLen, baseN, bDValue);
    inQueueB1.EnQue(b1Local);
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernel<aType, bType, cType>::SplitA(
    const AscendC::LocalTensor<aType> &a1Local, uint32_t offset, uint32_t colC0Stride)
{
    AscendC::LocalTensor<aType> a2Local = inQueueA2.AllocTensor<aType>();
    LoadNzL1ToZzL0A(a2Local, a1Local[offset], baseM, baseK, colC0Stride);
    inQueueA2.EnQue(a2Local);
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernel<aType, bType, cType>::SplitB(
    const AscendC::LocalTensor<bType> &b1Local, uint32_t offset, uint32_t colC0Stride)
{
    AscendC::LocalTensor<bType> b2Local = inQueueB2.AllocTensor<bType>();
    LoadNzL1ToZnL0B(b2Local, b1Local[offset], baseK, baseN, colC0Stride);
    inQueueB2.EnQue(b2Local);
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernel<aType, bType, cType>::Compute(
    const AscendC::LocalTensor<cType> &c1Local, bool cmatrixInitVal)
{
    AscendC::LocalTensor<aType> a2Local = inQueueA2.DeQue<aType>();
    AscendC::LocalTensor<bType> b2Local = inQueueB2.DeQue<bType>();
    AscendC::MmadParams mmadParams;
    mmadParams.m = baseM;
    mmadParams.n = baseN;
    mmadParams.k = baseK;
    mmadParams.cmatrixInitVal = cmatrixInitVal;
    AscendC::Mmad(c1Local, a2Local, b2Local, mmadParams);
    inQueueA2.FreeTensor(a2Local);
    inQueueB2.FreeTensor(b2Local);
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernel<aType, bType, cType>::CopyOut(const AscendC::GlobalTensor<cType> &C)
{
    auto c1Local = outQueueCO1.DeQue<cType>();
    FixpipeNzL0cToNdGm(C, c1Local, baseM, baseN);
    outQueueCO1.FreeTensor(c1Local);
}

#endif // MATMUL_H
