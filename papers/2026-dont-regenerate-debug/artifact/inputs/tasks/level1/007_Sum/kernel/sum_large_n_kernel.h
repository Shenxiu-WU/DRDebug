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
class SumLargeNKernel {
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
            pipe_->InitBuffer(xInQueue_, 1, tiling_.tileN * sizeof(dataType));
            pipe_->InitBuffer(yOutQueue_, 1, sizeof(dataType));
            pipe_->InitBuffer(reduceBuf_, tiling_.tileN * sizeof(float));
            pipe_->InitBuffer(sumBuf_, tiling_.N * sizeof(float));
            pipe_->InitBuffer(accumBuf_, sizeof(float));
            if constexpr (!std::is_same_v<dataType, float>) {
                pipe_->InitBuffer(xCastBuf_, tiling_.tileN * sizeof(float));
                pipe_->InitBuffer(yCastBuf_, sizeof(float));
            }
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

    __aicore__ inline void ProcessRow(int32_t rowIdx)
    {
        // Zero-initialize accumulator
        AscendC::LocalTensor<float> accumLocal = accumBuf_.Get<float>();
        AscendC::Duplicate(accumLocal, 0.0f, 1);
        AscendC::PipeBarrier<PIPE_V>();

        reduceLocal_ = reduceBuf_.Get<float>();
        sumLocal_ = sumBuf_.Get<float>();

        for (int tileIdx = 0; tileIdx < tiling_.nTiles; ++tileIdx) {
            int32_t colStart = tileIdx * tiling_.tileN;
            int32_t validN = tiling_.tileN;
            if (colStart + tiling_.tileN > tiling_.N) {
                validN = tiling_.lastTileValid;
            }

            // CopyIn
            xInQueue_.AllocTensor<dataType>(xInLocal_);
            LoadGmToUb(xInLocal_, xGM_[rowIdx * tiling_.N + colStart],
                        static_cast<uint32_t>(validN));
            xInQueue_.EnQue(xInLocal_);

            // Compute partial sum
            xInQueue_.DeQue<dataType>(xInLocal_);

            AscendC::LocalTensor<float> xFp32;
            if constexpr (std::is_same_v<dataType, float>) {
                xFp32 = xInLocal_.template ReinterpretCast<float>();
            } else {
                xFp32 = xCastBuf_.Get<float>();
                AscendC::Cast(xFp32, xInLocal_, AscendC::RoundMode::CAST_NONE, validN);
                AscendC::PipeBarrier<PIPE_V>();
            }

            AscendC::ReduceSum<float>(sumLocal_, xFp32, reduceLocal_, validN);
            AscendC::PipeBarrier<PIPE_ALL>();

            // Accumulate: accum += partial
            AscendC::Add(accumLocal, accumLocal, sumLocal_, 1);
            AscendC::PipeBarrier<PIPE_V>();

            xInQueue_.FreeTensor(xInLocal_);
        }

        // CopyOut: store accumulated sum
        yOutQueue_.AllocTensor<dataType>(yOutLocal_);
        if constexpr (std::is_same_v<dataType, float>) {
            AscendC::LocalTensor<float> yFp32 = yOutLocal_.template ReinterpretCast<float>();
            AscendC::DataCopy(yFp32, accumLocal, 1);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::Cast(yOutLocal_, accumLocal, OutputRoundMode(), 1);
            AscendC::PipeBarrier<PIPE_V>();
        }
        yOutQueue_.EnQue(yOutLocal_);

        yOutQueue_.DeQue<dataType>(yOutLocal_);
        StoreUbToGm(yGM_[rowIdx], yOutLocal_, 1);
        yOutQueue_.FreeTensor(yOutLocal_);
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
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yCastBuf_;

    AscendC::LocalTensor<dataType> xInLocal_;
    AscendC::LocalTensor<dataType> yOutLocal_;
    AscendC::LocalTensor<float> reduceLocal_;
    AscendC::LocalTensor<float> sumLocal_;
};
