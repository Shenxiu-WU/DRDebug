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
        pipe_->InitBuffer(gradInBuf_, AlignBytes(tiling_.D * sizeof(DataType)));
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int coreIdx = AscendC::GetBlockIdx();
            for (int localIdx = 0; localIdx < tiling_.tasksPerCore; ++localIdx) {
                int bx = coreIdx * tiling_.tasksPerCore + localIdx;
                if (bx >= BlockCount()) {
                    continue;
                }
                int bBase = bx * tiling_.blockB;
                for (int b = 0; b < tiling_.blockB; ++b) {
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
        // Scalar implementation: row starts / lengths (D, size) are generally
        // not 32B-aligned for this op's shapes, so block DataCopyPad is unsafe.
        AscendC::LocalTensor<DataType> gradInLocal_ = gradInBuf_.Get<DataType>();
        for (int32_t i = 0; i < tiling_.D; ++i) {
            gradInLocal_.SetValue(i, static_cast<DataType>(0));
        }

        const int64_t srcBase = static_cast<int64_t>(bIdx) * tiling_.nWindows * tiling_.size;
        for (int32_t w = 0; w < tiling_.nWindows; ++w) {
            const int64_t winBase = srcBase + static_cast<int64_t>(w) * tiling_.size;
            for (int32_t s = 0; s < tiling_.size; ++s) {
                const int32_t dstPos = w * tiling_.step + s;
                if (dstPos < tiling_.D) {
                    float val_f = static_cast<float>(gradInLocal_.GetValue(dstPos)) +
                                  static_cast<float>(gradOutputGM_[winBase + s].GetValue(0));
                    gradInLocal_.SetValue(dstPos, static_cast<DataType>(val_f));
                }
            }
        }

        const int64_t dstOffset = static_cast<int64_t>(bIdx) * tiling_.D;
        for (int32_t i = 0; i < tiling_.D; ++i) {
            gradInputGM_[dstOffset + i].SetValue(0, gradInLocal_.GetValue(i));
        }
    }

private:
    UnfoldBackwardTiling tiling_;
    AscendC::TPipe *pipe_;
    int32_t subBlockB_;
    AscendC::GlobalTensor<DataType> gradOutputGM_;
    AscendC::GlobalTensor<DataType> gradInputGM_;
    AscendC::TBuf<AscendC::TPosition::VECOUT> gradInBuf_;
};
