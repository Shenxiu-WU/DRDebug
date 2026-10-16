#pragma once

#include "kernel_operator.h"
#include "kernel_common.h"
#include "unfold_backward_tiling.h"

template <typename DataType>
class UnfoldBackwardKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR grad_output,
        GM_ADDR grad_input,
        GM_ADDR tilingGM,
        AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);
        int64_t gradOutElems = static_cast<int64_t>(tiling_.B) * tiling_.nWindows * tiling_.size;
        int64_t gradInElems = static_cast<int64_t>(tiling_.B) * tiling_.D;
        gradOutputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(grad_output), gradOutElems);
        gradInputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(grad_input), gradInElems);
        pipe_ = pipe;
        subBlockB_ = tiling_.subBlockB;
        pipe_->InitBuffer(gradInQueue_, 1, AlignBytes(tiling_.D * sizeof(DataType)));
        pipe_->InitBuffer(gradOutQueue_, 1, AlignBytes(tiling_.size * sizeof(DataType)));
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
            const int subBlockIdx = AscendC::GetSubBlockIdx();
            for (int localIdx = 0; localIdx < tiling_.tasksPerCore; ++localIdx) {
                int bx = coreIdx * tiling_.tasksPerCore + localIdx;
                if (bx >= BlockCount()) {
                    continue;
                }
                int bBase = bx * tiling_.blockB + subBlockIdx * tiling_.subBlockB;
                for (int b = 0; b < subBlockB_; ++b) {
                    int bIdx = bBase + b;
                    if (bIdx < tiling_.B) {
                        ProcessRow(bIdx);
                    }
                }
            }
        }
    }

protected:
    __aicore__ inline int32_t BlockCount() const
    {
        return (tiling_.B + tiling_.blockB - 1) / tiling_.blockB;
    }

    __aicore__ inline uint32_t AlignBytes(uint32_t bytes) const
    {
        return ((bytes + 31U) / 32U) * 32U;
    }

    __aicore__ inline uint32_t GradInRowBytes() const
    {
        return static_cast<uint32_t>(tiling_.D * sizeof(DataType));
    }

    __aicore__ inline uint32_t GradOutSliceBytes() const
    {
        return static_cast<uint32_t>(tiling_.size * sizeof(DataType));
    }

    __aicore__ inline uint32_t AlignedGradInRowBytes() const
    {
        return AlignBytes(GradInRowBytes());
    }

    __aicore__ inline uint32_t AlignedGradOutSliceBytes() const
    {
        return AlignBytes(GradOutSliceBytes());
    }

    __aicore__ inline void ProcessRow(int32_t bIdx)
    {
        gradInQueue_.AllocTensor<DataType>(gradInLocal_);
        AscendC::Duplicate(gradInLocal_, static_cast<DataType>(0), tiling_.D);
        gradInQueue_.EnQue(gradInLocal_);

        for (int w = 0; w < tiling_.nWindows; ++w) {
            gradOutQueue_.AllocTensor<DataType>(gradOutLocal_);
            int64_t srcOffset = (static_cast<int64_t>(bIdx) * tiling_.nWindows + w) * tiling_.size;
            uint32_t copyBytes = GradOutSliceBytes();
            uint32_t padElems = (AlignedGradOutSliceBytes() - copyBytes) / sizeof(DataType);
            AscendC::DataCopyExtParams copyParams{1, copyBytes, 0, 0, 0};
            AscendC::DataCopyPadExtParams<DataType> padParams{
                true, 0, static_cast<uint8_t>(padElems), static_cast<DataType>(0)};
            AscendC::DataCopyPad(gradOutLocal_, gradOutputGM_[srcOffset], copyParams, padParams);
            gradOutQueue_.EnQue(gradOutLocal_);
            gradOutQueue_.DeQue<DataType>(gradOutLocal_);

            gradInQueue_.DeQue<DataType>(gradInLocal_);
            for (int s = 0; s < tiling_.size; ++s) {
                int dstPos = w * tiling_.step + s;
                if (dstPos < tiling_.D) {
                    float val_f = static_cast<float>(gradInLocal_.GetValue(dstPos)) + static_cast<float>(gradOutLocal_.GetValue(s));
                    gradInLocal_.SetValue(dstPos, static_cast<DataType>(val_f));
                }
            }
            gradInQueue_.EnQue(gradInLocal_);
            gradOutQueue_.FreeTensor(gradOutLocal_);
        }

        gradInQueue_.DeQue<DataType>(gradInLocal_);
        int64_t dstOffset = static_cast<int64_t>(bIdx) * tiling_.D;
        uint32_t outBytes = GradInRowBytes();
        AscendC::DataCopyExtParams copyParams{1, outBytes, 0, 0, 0};
        AscendC::DataCopyPad(gradInputGM_[dstOffset], gradInLocal_, copyParams);
        gradInQueue_.FreeTensor(gradInLocal_);
    }

private:
    UnfoldBackwardTiling tiling_;
    AscendC::TPipe *pipe_;
    int32_t subBlockB_;
    AscendC::GlobalTensor<DataType> gradOutputGM_;
    AscendC::GlobalTensor<DataType> gradInputGM_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> gradInQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> gradOutQueue_;
    AscendC::LocalTensor<DataType> gradInLocal_;
    AscendC::LocalTensor<DataType> gradOutLocal_;
};
