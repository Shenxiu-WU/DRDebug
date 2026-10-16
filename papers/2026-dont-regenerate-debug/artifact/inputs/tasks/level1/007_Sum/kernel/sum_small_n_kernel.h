#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <type_traits>
#include "kernel_operator.h"
#include "kernel_common.h"
#include "sum_tiling.h"
#include "vector_tile.h"

template <typename dataType>
class SumSmallNKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR tilingGM, AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);
        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(x), tiling_.M * tiling_.N);
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(y), tiling_.M * 1);

        if ASCEND_IS_AIV {
            pipe_ = pipe;
            subBlockRows_ = tiling_.blockM / AscendC::GetSubBlockNum();
            // Allocate buffers
            pipe_->InitBuffer(xInQueue_, 1, tiling_.N * sizeof(dataType));
            pipe_->InitBuffer(yOutQueue_, 1, sizeof(dataType));
            pipe_->InitBuffer(reduceBuf_, tiling_.N * sizeof(float));
            // Allocate enough space for ReduceSum workspace
            pipe_->InitBuffer(sumBuf_, tiling_.N * sizeof(float));
            if constexpr (!std::is_same_v<dataType, float>) {
                pipe_->InitBuffer(xCastBuf_, tiling_.N * sizeof(float));
                pipe_->InitBuffer(yCastBuf_, sizeof(float));
            }

            // Pre-load nothing (no reusable data)
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
            const int subBlockIdx = AscendC::GetSubBlockIdx();

            for (int localIdx = 0; localIdx < tiling_.tasksPerCore; ++localIdx) {
                const int bx = coreIdx * tiling_.tasksPerCore + localIdx;
                if (bx >= BlockCount()) continue;

                for (int row = 0; row < subBlockRows_; ++row) {
                    const int rowIdx = bx * tiling_.blockM + subBlockIdx * subBlockRows_ + row;
                    if (rowIdx < tiling_.M) {
                        ProcessRow(rowIdx);
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

    __aicore__ inline AscendC::RoundMode OutputRoundMode() const
    {
        if constexpr (std::is_same_v<dataType, bfloat16_t>) {
            return AscendC::RoundMode::CAST_ROUND;
        }
        return AscendC::RoundMode::CAST_NONE;
    }

    __aicore__ inline void CopyInX(int32_t rowIdx)
    {
        xInQueue_.AllocTensor<dataType>(xInLocal_);
        LoadGmToUb(xInLocal_, xGM_[rowIdx * tiling_.N], static_cast<uint32_t>(tiling_.N));
        xInQueue_.EnQue(xInLocal_);
    }

    __aicore__ inline void CopyOutY(int32_t rowIdx)
    {
        yOutQueue_.DeQue<dataType>(yOutLocal_);
        StoreUbToGm(yGM_[rowIdx], yOutLocal_, 1);
        yOutQueue_.FreeTensor(yOutLocal_);
    }

    __aicore__ inline void ComputeRow()
    {
        yOutQueue_.AllocTensor<dataType>(yOutLocal_);
        reduceLocal_ = reduceBuf_.Get<float>();
        sumLocal_ = sumBuf_.Get<float>();

        xInQueue_.DeQue<dataType>(xInLocal_);

        // Cast input to float32 if needed
        AscendC::LocalTensor<float> xFp32;
        if constexpr (std::is_same_v<dataType, float>) {
            xFp32 = xInLocal_.template ReinterpretCast<float>();
        } else {
            xFp32 = xCastBuf_.Get<float>();
            AscendC::Cast(xFp32, xInLocal_, AscendC::RoundMode::CAST_NONE, tiling_.N);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // Reduce sum along last dimension
        AscendC::ReduceSum<float>(sumLocal_, xFp32, reduceLocal_, tiling_.N);
        AscendC::PipeBarrier<PIPE_ALL>();

        // Copy result to output
        if constexpr (std::is_same_v<dataType, float>) {
            // For float, reinterpret cast sumLocal_ → yOutLocal_ (same buffer after enqueue)
            AscendC::LocalTensor<float> yFp32 = yOutLocal_.template ReinterpretCast<float>();
            AscendC::DataCopy(yFp32, sumLocal_, 1);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::Cast(yOutLocal_, sumLocal_, OutputRoundMode(), 1);
            AscendC::PipeBarrier<PIPE_V>();
        }

        xInQueue_.FreeTensor(xInLocal_);
        yOutQueue_.EnQue(yOutLocal_);
    }

    __aicore__ inline void ProcessRow(int32_t rowIdx)
    {
        CopyInX(rowIdx);
        ComputeRow();
        CopyOutY(rowIdx);
    }

private:
    SumKernelTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};
    int subBlockRows_{0};

    AscendC::GlobalTensor<dataType> xGM_;
    AscendC::GlobalTensor<dataType> yGM_;

    AscendC::TQue<AscendC::TPosition::VECIN, 0> xInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> yOutQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yCastBuf_;

    AscendC::LocalTensor<dataType> xInLocal_;
    AscendC::LocalTensor<dataType> yOutLocal_;
    AscendC::LocalTensor<float> reduceLocal_;
    AscendC::LocalTensor<float> sumLocal_;
};
