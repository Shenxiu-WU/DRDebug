#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"

#include "kernel_common.h"
#include "scatter_elements_v2_tiling.h"

__aicore__ inline half AddHalf(half a, half b)
{
    return static_cast<half>(static_cast<float>(a) + static_cast<float>(b));
}

__aicore__ inline half MulHalf(half a, half b)
{
    return static_cast<half>(static_cast<float>(a) * static_cast<float>(b));
}

template <typename T>
__aicore__ inline T AddScalar(T a, T b)
{
    return a + b;
}

template <>
__aicore__ inline half AddScalar<half>(half a, half b)
{
    return AddHalf(a, b);
}

template <typename T>
__aicore__ inline T MulScalar(T a, T b)
{
    return a * b;
}

template <>
__aicore__ inline half MulScalar<half>(half a, half b)
{
    return MulHalf(a, b);
}

template <typename DataType>
class ScatterElementsV2CommonKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR selfGM,
        GM_ADDR indexGM,
        GM_ADDR updatesGM,
        GM_ADDR outGM,
        GM_ADDR tilingGM,
        AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);
        selfGM_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(selfGM), tiling_.XRows * tiling_.XStride);
        indexGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(indexGM), tiling_.M * tiling_.YStride);
        updatesGM_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(updatesGM), tiling_.M * tiling_.YStride);
        outGM_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(outGM), tiling_.XRows * tiling_.XStride);

        if ASCEND_IS_AIV {
            pipe_ = pipe;
            subBlockRows_ = tiling_.subBlockM;
            pipe->InitBuffer(indexInQueue_, 1, AlignedIndexRowBytes());
            pipe->InitBuffer(updatesInQueue_, 1, AlignedUpdatesRowBytes());
            pipe->InitBuffer(outInQueue_, 1, AlignedOutRowBytes());
        }
    }

protected:
    __aicore__ inline const ScatterElementsV2KernelTiling &Tiling() const
    {
        return tiling_;
    }

    __aicore__ inline int32_t SubBlockRows() const
    {
        return subBlockRows_;
    }

    __aicore__ inline int32_t BlockCount() const
    {
        return (tiling_.M + tiling_.blockM - 1) / tiling_.blockM;
    }

    __aicore__ inline uint32_t OutRowBytes() const
    {
        return static_cast<uint32_t>(tiling_.XStride * sizeof(DataType));
    }

    __aicore__ inline uint32_t IndexRowBytes() const
    {
        return static_cast<uint32_t>(tiling_.YStride * sizeof(int32_t));
    }

    __aicore__ inline uint32_t UpdatesRowBytes() const
    {
        return static_cast<uint32_t>(tiling_.YStride * sizeof(DataType));
    }

    __aicore__ inline uint32_t AlignBytes(uint32_t bytes) const
    {
        return ((bytes + 31U) / 32U) * 32U;
    }

    __aicore__ inline uint32_t AlignedOutRowBytes() const
    {
        return AlignBytes(OutRowBytes());
    }

    __aicore__ inline uint32_t AlignedIndexRowBytes() const
    {
        return AlignBytes(IndexRowBytes());
    }

    __aicore__ inline uint32_t AlignedUpdatesRowBytes() const
    {
        return AlignBytes(UpdatesRowBytes());
    }

    __aicore__ inline void CopyOutRowToUb(AscendC::LocalTensor<DataType> &dst, AscendC::GlobalTensor<DataType> src)
    {
        const uint32_t padElems = (AlignedOutRowBytes() - OutRowBytes()) / sizeof(DataType);
        AscendC::DataCopyExtParams copyParams{1, OutRowBytes(), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DataType> padParams{
            true, 0, static_cast<uint8_t>(padElems), static_cast<DataType>(0)};
        AscendC::DataCopyPad(dst, src, copyParams, padParams);
    }

    __aicore__ inline void CopyIndexRowToUb(AscendC::LocalTensor<int32_t> &dst, AscendC::GlobalTensor<int32_t> src)
    {
        const uint32_t padElems = (AlignedIndexRowBytes() - IndexRowBytes()) / sizeof(int32_t);
        AscendC::DataCopyExtParams copyParams{1, IndexRowBytes(), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> padParams{true, 0, static_cast<uint8_t>(padElems), 0};
        AscendC::DataCopyPad(dst, src, copyParams, padParams);
    }

    __aicore__ inline void CopyUpdatesRowToUb(AscendC::LocalTensor<DataType> &dst, AscendC::GlobalTensor<DataType> src)
    {
        const uint32_t padElems = (AlignedUpdatesRowBytes() - UpdatesRowBytes()) / sizeof(DataType);
        AscendC::DataCopyExtParams copyParams{1, UpdatesRowBytes(), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DataType> padParams{
            true, 0, static_cast<uint8_t>(padElems), static_cast<DataType>(0)};
        AscendC::DataCopyPad(dst, src, copyParams, padParams);
    }

    __aicore__ inline void CopyOutRowToGm(AscendC::GlobalTensor<DataType> dst, AscendC::LocalTensor<DataType> &src)
    {
        AscendC::DataCopyExtParams copyParams{1, OutRowBytes(), 0, 0, 0};
        AscendC::DataCopyPad(dst, src, copyParams);
    }

    __aicore__ inline void ProcessScatterRow(int rowIdx)
    {
        // Allocate tensors
        outInQueue_.AllocTensor<DataType>(outInLocal_);
        indexInQueue_.AllocTensor<int32_t>(indexLocal_);
        updatesInQueue_.AllocTensor<DataType>(updatesLocal_);

        // Copy data to UB
        CopyOutRowToUb(outInLocal_, outGM_[rowIdx * tiling_.XStride]);
        CopyIndexRowToUb(indexLocal_, indexGM_[rowIdx * tiling_.YStride]);
        CopyUpdatesRowToUb(updatesLocal_, updatesGM_[rowIdx * tiling_.YStride]);

        // EnQue VECIN
        outInQueue_.EnQue(outInLocal_);
        indexInQueue_.EnQue(indexLocal_);
        updatesInQueue_.EnQue(updatesLocal_);

        // DeQue VECIN
        outInQueue_.DeQue<DataType>(outInLocal_);
        indexInQueue_.DeQue<int32_t>(indexLocal_);
        updatesInQueue_.DeQue<DataType>(updatesLocal_);

        AscendC::PipeBarrier<PIPE_ALL>();

        // Scatter compute
        for (int col = 0; col < tiling_.IG; ++col) {
            const int32_t scatterIdx = indexLocal_.GetValue(col);
            const DataType updateVal = updatesLocal_.GetValue(col);
            const DataType currentVal = outInLocal_.GetValue(scatterIdx);
            DataType newVal;
            if (tiling_.reductionMode == 0) {
                newVal = updateVal;
            } else if (tiling_.reductionMode == 1) {
                newVal = AddScalar(currentVal, updateVal);
            } else {
                newVal = MulScalar(currentVal, updateVal);
            }
            outInLocal_.SetValue(scatterIdx, newVal);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // Write back directly from VECIN buffer to GM
        CopyOutRowToGm(outGM_[rowIdx * tiling_.XStride], outInLocal_);

        // Free VECIN tensors
        outInQueue_.FreeTensor(outInLocal_);
        indexInQueue_.FreeTensor(indexLocal_);
        updatesInQueue_.FreeTensor(updatesLocal_);
    }

private:
    ScatterElementsV2KernelTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};
    int subBlockRows_{0};

    AscendC::GlobalTensor<DataType> selfGM_;
    AscendC::GlobalTensor<int32_t> indexGM_;
    AscendC::GlobalTensor<DataType> updatesGM_;
    AscendC::GlobalTensor<DataType> outGM_;

    AscendC::TQue<AscendC::TPosition::VECIN, 0> indexInQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> updatesInQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> outInQueue_;

    AscendC::LocalTensor<int32_t> indexLocal_;
    AscendC::LocalTensor<DataType> updatesLocal_;
    AscendC::LocalTensor<DataType> outInLocal_;
};
