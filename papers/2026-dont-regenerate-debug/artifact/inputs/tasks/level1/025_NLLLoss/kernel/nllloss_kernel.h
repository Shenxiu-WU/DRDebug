#pragma once

#include "kernel_operator.h"
#include "kernel_common.h"
#include "nllloss_tiling.h"

template <typename DataType>
class NLLLossKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR input,
        GM_ADDR target,
        GM_ADDR weight,
        GM_ADDR out,
        GM_ADDR partialSum,
        GM_ADDR partialCount,
        GM_ADDR tilingGM,
        AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);
        inputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(input), tiling_.numPos * tiling_.C);
        targetGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(target), tiling_.numPos);
        if (tiling_.hasWeight != 0) {
            weightGM_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(weight), tiling_.C);
        }

        if (tiling_.reduction == REDUCTION_NONE) {
            outGM_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(out), tiling_.numPos);
        } else {
            outGM_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(out), 1);
            const int32_t totalSlots = tiling_.usedCoreNum * tiling_.slotsPerCore;
            partialSumGM_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(partialSum), totalSlots);
            partialCountGM_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(partialCount), totalSlots);
        }

        if ASCEND_IS_AIV {
            pipe_ = pipe;
            subBlockN_ = tiling_.subBlockN;
            pipe_->InitBuffer(targetInQueue_, 1, AlignBytes(sizeof(int32_t)));
            pipe_->InitBuffer(inputRowInQueue_, 1, AlignBytes(tiling_.C * sizeof(DataType)));
            pipe_->InitBuffer(weightInQueue_, 1, AlignBytes(tiling_.C * sizeof(DataType)));
            pipe_->InitBuffer(outQueue_, 1, AlignBytes(sizeof(DataType)));
            pipe_->InitBuffer(sumQueue_, 1, AlignBytes(sizeof(DataType)));
            pipe_->InitBuffer(countQueue_, 1, AlignBytes(sizeof(DataType)));
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
            const int subBlockIdx = AscendC::GetSubBlockIdx();
            const int nNum = (tiling_.numPos + tiling_.blockN - 1) / tiling_.blockN;

            float localSum = 0.0f;
            float localCount = 0.0f;

            for (int localIdx = 0; localIdx < tiling_.tasksPerCore; ++localIdx) {
                const int bx = coreIdx * tiling_.tasksPerCore + localIdx;
                if (bx >= nNum) {
                    continue;
                }
                const int rowBase = bx * tiling_.blockN + subBlockIdx * tiling_.subBlockN;
                for (int row = 0; row < subBlockN_; ++row) {
                    const int rowIdx = rowBase + row;
                    if (rowIdx >= tiling_.numPos) {
                        continue;
                    }

                    // Read target
                    targetInQueue_.AllocTensor<int32_t>(targetLocal_);
                    {
                        const uint32_t actualBytes = static_cast<uint32_t>(sizeof(int32_t));
                        const uint32_t alignedBytes = AlignBytes(actualBytes);
                        const uint8_t padElems = static_cast<uint8_t>((alignedBytes - actualBytes) / sizeof(int32_t));
                        AscendC::DataCopyExtParams copyParams{1, actualBytes, 0, 0, 0};
                        AscendC::DataCopyPadExtParams<int32_t> padParams{true, 0, padElems, 0};
                        AscendC::DataCopyPad(targetLocal_, targetGM_[rowIdx], copyParams, padParams);
                    }
                    targetInQueue_.EnQue(targetLocal_);
                    targetInQueue_.DeQue<int32_t>(targetLocal_);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    const int32_t targetVal = targetLocal_.GetValue(0);
                    targetInQueue_.FreeTensor(targetLocal_);

                    float val = 0.0f;
                    if (targetVal != tiling_.ignoreIndex) {
                        // Read input row
                        inputRowInQueue_.AllocTensor<DataType>(inputRowLocal_);
                        {
                            const uint32_t actualBytes = static_cast<uint32_t>(tiling_.C * sizeof(DataType));
                            const uint32_t alignedBytes = AlignBytes(actualBytes);
                            const uint8_t padElems = static_cast<uint8_t>((alignedBytes - actualBytes) / sizeof(DataType));
                            AscendC::DataCopyExtParams inputCopyParams{1, actualBytes, 0, 0, 0};
                            AscendC::DataCopyPadExtParams<DataType> inputPadParams{true, 0, padElems, static_cast<DataType>(0)};
                            AscendC::DataCopyPad(inputRowLocal_, inputGM_[rowIdx * tiling_.C], inputCopyParams, inputPadParams);
                        }
                        inputRowInQueue_.EnQue(inputRowLocal_);
                        inputRowInQueue_.DeQue<DataType>(inputRowLocal_);
                        AscendC::PipeBarrier<PIPE_ALL>();

                        const int idx = static_cast<int>(targetVal);
                        val = static_cast<float>(inputRowLocal_.GetValue(idx));
                        inputRowInQueue_.FreeTensor(inputRowLocal_);

                        if (tiling_.hasWeight != 0) {
                            weightInQueue_.AllocTensor<DataType>(weightLocal_);
                            {
                                const uint32_t actualBytes = static_cast<uint32_t>(tiling_.C * sizeof(DataType));
                                const uint32_t alignedBytes = AlignBytes(actualBytes);
                                const uint8_t padElems = static_cast<uint8_t>((alignedBytes - actualBytes) / sizeof(DataType));
                                AscendC::DataCopyExtParams weightCopyParams{1, actualBytes, 0, 0, 0};
                                AscendC::DataCopyPadExtParams<DataType> weightPadParams{true, 0, padElems, static_cast<DataType>(0)};
                                AscendC::DataCopyPad(weightLocal_, weightGM_[0], weightCopyParams, weightPadParams);
                            }
                            weightInQueue_.EnQue(weightLocal_);
                            weightInQueue_.DeQue<DataType>(weightLocal_);
                            AscendC::PipeBarrier<PIPE_ALL>();
                            val = val * static_cast<float>(weightLocal_.GetValue(idx));
                            weightInQueue_.FreeTensor(weightLocal_);
                        }

                        val = -1.0f * val;
                    }

                    if (tiling_.reduction == REDUCTION_NONE) {
                        outQueue_.AllocTensor<DataType>(outLocal_);
                        outLocal_.SetValue(0, static_cast<DataType>(val));
                        AscendC::PipeBarrier<PIPE_ALL>();
                        outQueue_.EnQue(outLocal_);
                        outQueue_.DeQue<DataType>(outLocal_);
                        AscendC::PipeBarrier<PIPE_ALL>();
                        AscendC::DataCopyExtParams outCopyParams{1, static_cast<uint32_t>(sizeof(DataType)), 0, 0, 0};
                        AscendC::DataCopyPad(outGM_[rowIdx], outLocal_, outCopyParams);
                        outQueue_.FreeTensor(outLocal_);
                    } else {
                        localSum = localSum + val;
                        if (targetVal != tiling_.ignoreIndex) {
                            localCount = localCount + 1.0f;
                        }
                    }
                }
            }

            if (tiling_.reduction != REDUCTION_NONE) {
                // Write partial sum/count to GM
                sumQueue_.AllocTensor<DataType>(sumLocal_);
                sumLocal_.SetValue(0, static_cast<DataType>(localSum));
                AscendC::PipeBarrier<PIPE_ALL>();
                sumQueue_.EnQue(sumLocal_);
                sumQueue_.DeQue<DataType>(sumLocal_);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopyExtParams sumCopyParams{1, static_cast<uint32_t>(sizeof(DataType)), 0, 0, 0};
                AscendC::DataCopyPad(partialSumGM_[coreIdx * tiling_.slotsPerCore + subBlockIdx], sumLocal_, sumCopyParams);
                sumQueue_.FreeTensor(sumLocal_);

                countQueue_.AllocTensor<DataType>(countLocal_);
                countLocal_.SetValue(0, static_cast<DataType>(localCount));
                AscendC::PipeBarrier<PIPE_ALL>();
                countQueue_.EnQue(countLocal_);
                countQueue_.DeQue<DataType>(countLocal_);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::DataCopyExtParams countCopyParams{1, static_cast<uint32_t>(sizeof(DataType)), 0, 0, 0};
                AscendC::DataCopyPad(partialCountGM_[coreIdx * tiling_.slotsPerCore + subBlockIdx], countLocal_, countCopyParams);
                countQueue_.FreeTensor(countLocal_);

                // Core 0 sub-block 0 does final reduction
                if (coreIdx == 0 && subBlockIdx == 0) {
                    float totalSum = 0.0f;
                    float totalCount = 0.0f;
                    const int totalSlots = tiling_.usedCoreNum * tiling_.slotsPerCore;
                    for (int c = 0; c < totalSlots; ++c) {
                        inputRowInQueue_.AllocTensor<DataType>(sumLocal_);
                        {
                            const uint32_t actualBytes = static_cast<uint32_t>(sizeof(DataType));
                            const uint32_t alignedBytes = AlignBytes(actualBytes);
                            const uint8_t padElems = static_cast<uint8_t>((alignedBytes - actualBytes) / sizeof(DataType));
                            AscendC::DataCopyExtParams readSumParams{1, actualBytes, 0, 0, 0};
                            AscendC::DataCopyPadExtParams<DataType> readSumPad{true, 0, padElems, static_cast<DataType>(0)};
                            AscendC::DataCopyPad(sumLocal_, partialSumGM_[c], readSumParams, readSumPad);
                        }
                        inputRowInQueue_.EnQue(sumLocal_);
                        inputRowInQueue_.DeQue<DataType>(sumLocal_);
                        AscendC::PipeBarrier<PIPE_ALL>();
                        totalSum = totalSum + static_cast<float>(sumLocal_.GetValue(0));
                        inputRowInQueue_.FreeTensor(sumLocal_);

                        inputRowInQueue_.AllocTensor<DataType>(countLocal_);
                        {
                            const uint32_t actualBytes = static_cast<uint32_t>(sizeof(DataType));
                            const uint32_t alignedBytes = AlignBytes(actualBytes);
                            const uint8_t padElems = static_cast<uint8_t>((alignedBytes - actualBytes) / sizeof(DataType));
                            AscendC::DataCopyExtParams readCountParams{1, actualBytes, 0, 0, 0};
                            AscendC::DataCopyPadExtParams<DataType> readCountPad{true, 0, padElems, static_cast<DataType>(0)};
                            AscendC::DataCopyPad(countLocal_, partialCountGM_[c], readCountParams, readCountPad);
                        }
                        inputRowInQueue_.EnQue(countLocal_);
                        inputRowInQueue_.DeQue<DataType>(countLocal_);
                        AscendC::PipeBarrier<PIPE_ALL>();
                        totalCount = totalCount + static_cast<float>(countLocal_.GetValue(0));
                        inputRowInQueue_.FreeTensor(countLocal_);
                    }

                    if (tiling_.reduction == REDUCTION_MEAN && totalCount > 0.0f) {
                        totalSum = totalSum / totalCount;
                    }

                    outQueue_.AllocTensor<DataType>(outLocal_);
                    outLocal_.SetValue(0, static_cast<DataType>(totalSum));
                    AscendC::PipeBarrier<PIPE_ALL>();
                    outQueue_.EnQue(outLocal_);
                    outQueue_.DeQue<DataType>(outLocal_);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::DataCopyExtParams finalOutParams{1, static_cast<uint32_t>(sizeof(DataType)), 0, 0, 0};
                    AscendC::DataCopyPad(outGM_[0], outLocal_, finalOutParams);
                    outQueue_.FreeTensor(outLocal_);
                }
            }
        }
    }

private:
    __aicore__ inline uint32_t AlignBytes(uint32_t bytes) const
    {
        return ((bytes + 31U) / 32U) * 32U;
    }

    NLLLossTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};
    int32_t subBlockN_{0};

    AscendC::GlobalTensor<DataType> inputGM_;
    AscendC::GlobalTensor<int32_t> targetGM_;
    AscendC::GlobalTensor<DataType> weightGM_;
    AscendC::GlobalTensor<DataType> outGM_;
    AscendC::GlobalTensor<DataType> partialSumGM_;
    AscendC::GlobalTensor<DataType> partialCountGM_;

    AscendC::TQue<AscendC::TPosition::VECIN, 0> targetInQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> inputRowInQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> weightInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> outQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> sumQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> countQueue_;

    AscendC::LocalTensor<int32_t> targetLocal_;
    AscendC::LocalTensor<DataType> inputRowLocal_;
    AscendC::LocalTensor<DataType> weightLocal_;
    AscendC::LocalTensor<DataType> outLocal_;
    AscendC::LocalTensor<DataType> sumLocal_;
    AscendC::LocalTensor<DataType> countLocal_;
};
