#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <type_traits>

#include "kernel_operator.h"

#include "kernel_common.h"
#include "add_tiling.h"
#include "vector_tile.h"

template <typename dataType>
class AddAlphaKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR out, GM_ADDR tilingGM, AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);
        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(x), tiling_.N);
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(y), tiling_.N);
        outGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(out), tiling_.N);

        if ASCEND_IS_AIV {
            pipe_ = pipe;
            pipe_->InitBuffer(xInQueue_, 1, tiling_.tileSize * sizeof(dataType));
            pipe_->InitBuffer(yInQueue_, 1, tiling_.tileSize * sizeof(dataType));
            pipe_->InitBuffer(outQueue_, 1, tiling_.tileSize * sizeof(dataType));

            if constexpr (!std::is_same_v<dataType, float>) {
                pipe_->InitBuffer(xF32Buf_, tiling_.tileSize * sizeof(float));
                pipe_->InitBuffer(yF32Buf_, tiling_.tileSize * sizeof(float));
                pipe_->InitBuffer(alphaYF32Buf_, tiling_.tileSize * sizeof(float));
                pipe_->InitBuffer(outF32Buf_, tiling_.tileSize * sizeof(float));
            }
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();

            for (int localIdx = 0; localIdx < tiling_.tasksPerCore; ++localIdx) {
                int tileIdx = coreIdx * tiling_.tasksPerCore + localIdx;
                if (tileIdx >= tiling_.totalTiles) {
                    continue;
                }
                int offset = tileIdx * tiling_.tileSize;
                int actualLen = tiling_.tileSize;
                if (offset + actualLen > tiling_.N) {
                    actualLen = tiling_.N - offset;
                }
                ProcessTile(offset, actualLen);
            }
        }
    }

private:
    __aicore__ inline void ProcessTile(int offset, int count)
    {
        // CopyIn: load x and y tiles from GM to UB
        xInQueue_.AllocTensor<dataType>(xInLocal_);
        yInQueue_.AllocTensor<dataType>(yInLocal_);
        LoadGmToUb(xInLocal_, xGM_[offset], static_cast<uint32_t>(count));
        LoadGmToUb(yInLocal_, yGM_[offset], static_cast<uint32_t>(count));
        xInQueue_.EnQue(xInLocal_);
        yInQueue_.EnQue(yInLocal_);

        // Compute
        ComputeAlpha(offset, count);
    }

    __aicore__ inline void ComputeAlpha(int offset, int count)
    {
        xInQueue_.DeQue<dataType>(xInLocal_);
        yInQueue_.DeQue<dataType>(yInLocal_);

        outQueue_.AllocTensor<dataType>(outLocal_);

        if constexpr (std::is_same_v<dataType, float>) {
            // float32: direct computation
            // alpha * y
            AscendC::Muls(outLocal_, yInLocal_, tiling_.alpha, static_cast<uint32_t>(count));
            // x + alpha*y
            AscendC::Add(outLocal_, xInLocal_, outLocal_, static_cast<uint32_t>(count));
        } else {
            // fp16/bf16: cast to float32, compute, cast back
            AscendC::LocalTensor<float> xF32 = xF32Buf_.Get<float>();
            AscendC::LocalTensor<float> yF32 = yF32Buf_.Get<float>();
            AscendC::LocalTensor<float> alphaYF32 = alphaYF32Buf_.Get<float>();
            AscendC::LocalTensor<float> outF32 = outF32Buf_.Get<float>();

            // Cast inputs to float32
            AscendC::Cast(xF32, xInLocal_, AscendC::RoundMode::CAST_NONE,
                         static_cast<uint32_t>(count));
            AscendC::Cast(yF32, yInLocal_, AscendC::RoundMode::CAST_NONE,
                         static_cast<uint32_t>(count));
            AscendC::PipeBarrier<PIPE_V>();

            // alpha * y in float32
            AscendC::Muls(alphaYF32, yF32, tiling_.alpha, static_cast<uint32_t>(count));
            // x + alpha*y in float32
            AscendC::Add(outF32, xF32, alphaYF32, static_cast<uint32_t>(count));
            AscendC::PipeBarrier<PIPE_V>();

            // Cast result back to original dtype
            AscendC::RoundMode castMode = AscendC::RoundMode::CAST_NONE;
            if constexpr (std::is_same_v<dataType, bfloat16_t>) {
                castMode = AscendC::RoundMode::CAST_ROUND;
            }
            AscendC::Cast(outLocal_, outF32, castMode, static_cast<uint32_t>(count));
        }

        // CopyOut: store result back to GM
        outQueue_.EnQue(outLocal_);
        outQueue_.DeQue<dataType>(outLocal_);
        StoreUbToGm(outGM_[offset], outLocal_, static_cast<uint32_t>(count));
        outQueue_.FreeTensor(outLocal_);

        xInQueue_.FreeTensor(xInLocal_);
        yInQueue_.FreeTensor(yInLocal_);
    }

    AddKernelTiling tiling_;
    AscendC::TPipe *pipe_{nullptr};
    AscendC::GlobalTensor<dataType> xGM_, yGM_, outGM_;

    // Queues for GM <-> UB data movement
    AscendC::TQue<AscendC::TPosition::VECIN, 0> xInQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> yInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> outQueue_;

    // Local tensors for queue operations
    AscendC::LocalTensor<dataType> xInLocal_;
    AscendC::LocalTensor<dataType> yInLocal_;
    AscendC::LocalTensor<dataType> outLocal_;

    // Temporary float32 buffers (only used for fp16/bf16)
    AscendC::TBuf<AscendC::TPosition::VECCALC> xF32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yF32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> alphaYF32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outF32Buf_;
};
