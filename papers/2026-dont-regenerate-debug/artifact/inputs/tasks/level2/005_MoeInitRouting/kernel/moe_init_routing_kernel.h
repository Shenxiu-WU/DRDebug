#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <type_traits>

#include "kernel_operator.h"

#include "kernel_common.h"
#include "moe_init_routing_tiling.h"

template <typename DataType>
class MoeInitRoutingKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR expandedRowIdx,
        GM_ADDR y,
        GM_ADDR tilingGM,
        AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);

        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(x),
                             tiling_.N * tiling_.XStride);
        rowIdxGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(expandedRowIdx),
                                  tiling_.M);
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(y),
                             tiling_.M * tiling_.YStride);

        if ASCEND_IS_AIV {
            pipe_ = pipe;
            subBlockRows_ = tiling_.subBlockM;
            pipe_->InitBuffer(xInQueue_, 1, AlignedXRowBytes());
            pipe_->InitBuffer(yOutQueue_, 1, AlignedYRowBytes());
            pipe_->InitBuffer(rowIdxInQueue_, 1, AlignBytes(sizeof(int32_t)));
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
            const int subBlockIdx = AscendC::GetSubBlockIdx();

            for (int localIdx = 0; localIdx < tiling_.tasksPerCore; ++localIdx) {
                const int bx = coreIdx * tiling_.tasksPerCore + localIdx;
                if (bx >= BlockCount()) {
                    continue;
                }
                const int rowBase = bx * tiling_.blockM + subBlockIdx * tiling_.subBlockM;
                for (int row = 0; row < subBlockRows_; ++row) {
                    const int outRow = rowBase + row;
                    if (outRow < tiling_.M) {
                        const int32_t srcRow = ResolveSourceRow(outRow);
                        ProcessRow(outRow, srcRow);
                    }
                }
            }
        }
    }

private:
    __aicore__ inline int32_t BlockCount() const
    {
        return (tiling_.M + tiling_.blockM - 1) / tiling_.blockM;
    }

    __aicore__ inline uint32_t XRowBytes() const
    {
        return static_cast<uint32_t>(tiling_.H * sizeof(DataType));
    }

    __aicore__ inline uint32_t YRowBytes() const
    {
        return static_cast<uint32_t>(tiling_.H * sizeof(DataType));
    }

    __aicore__ inline uint32_t AlignBytes(uint32_t bytes) const
    {
        return ((bytes + 31U) / 32U) * 32U;
    }

    __aicore__ inline uint32_t AlignedXRowBytes() const
    {
        return AlignBytes(XRowBytes());
    }

    __aicore__ inline uint32_t AlignedYRowBytes() const
    {
        return AlignBytes(YRowBytes());
    }

    __aicore__ inline int32_t ResolveSourceRow(int rowIdx)
    {
        AscendC::LocalTensor<int32_t> rowMapLocal;
        rowIdxInQueue_.AllocTensor<int32_t>(rowMapLocal);
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
        AscendC::DataCopyPad(rowMapLocal, rowIdxGM_[rowIdx], copyParams, padParams);
        rowIdxInQueue_.EnQue(rowMapLocal);
        rowIdxInQueue_.DeQue<int32_t>(rowMapLocal);
        const int32_t mappedRow = rowMapLocal.GetValue(0);
        rowIdxInQueue_.FreeTensor(rowMapLocal);
        return mappedRow;
    }

    __aicore__ inline void ProcessRow(int outRow, int32_t srcRow)
    {
        AscendC::LocalTensor<DataType> xLocal;
        AscendC::LocalTensor<DataType> yLocal;
        xInQueue_.AllocTensor<DataType>(xLocal);
        yOutQueue_.AllocTensor<DataType>(yLocal);

        // GM -> UB: pull H elements from X[srcRow, 0..H-1].
        AscendC::DataCopyExtParams copyInParams{1, XRowBytes(), 0, 0, 0};
        const uint32_t padInElems = (AlignedXRowBytes() - XRowBytes()) / sizeof(DataType);
        AscendC::DataCopyPadExtParams<DataType> padInParams{
            true, 0, static_cast<uint8_t>(padInElems), static_cast<DataType>(0)};
        AscendC::DataCopyPad(xLocal, xGM_[srcRow * tiling_.XStride], copyInParams, padInParams);
        xInQueue_.EnQue(xLocal);

        xInQueue_.DeQue<DataType>(xLocal);
        // Block-level UB→UB copy via Adds with 0 on the binary
        // representation of the row. Reinterpreting fp16 / bf16 as int16 (and
        // fp32 as int32) lets us reuse the integer Adds path which is
        // supported for all three element widths and avoids the bf16
        // restriction of the floating-point Adds intrinsic.
        const uint32_t copyElems = AlignedXRowBytes() / sizeof(DataType);
        if constexpr (sizeof(DataType) == 4) {
            auto xInt = xLocal.template ReinterpretCast<int32_t>();
            auto yInt = yLocal.template ReinterpretCast<int32_t>();
            AscendC::Adds(yInt, xInt, static_cast<int32_t>(0), copyElems);
        } else {
            auto xInt = xLocal.template ReinterpretCast<int16_t>();
            auto yInt = yLocal.template ReinterpretCast<int16_t>();
            AscendC::Adds(yInt, xInt, static_cast<int16_t>(0), copyElems);
        }
        AscendC::PipeBarrier<PIPE_V>();

        xInQueue_.FreeTensor(xLocal);
        yOutQueue_.EnQue(yLocal);

        yOutQueue_.DeQue<DataType>(yLocal);
        AscendC::DataCopyExtParams copyOutParams{1, YRowBytes(), 0, 0, 0};
        AscendC::DataCopyPad(yGM_[outRow * tiling_.YStride], yLocal, copyOutParams);
        yOutQueue_.FreeTensor(yLocal);
    }

private:
    MoeInitRoutingKernelTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};
    int subBlockRows_{0};

    AscendC::GlobalTensor<DataType> xGM_;
    AscendC::GlobalTensor<int32_t> rowIdxGM_;
    AscendC::GlobalTensor<DataType> yGM_;

    AscendC::TQue<AscendC::TPosition::VECIN, 0> xInQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> rowIdxInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> yOutQueue_;
};
