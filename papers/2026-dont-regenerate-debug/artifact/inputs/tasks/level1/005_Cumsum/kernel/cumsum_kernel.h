#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"

#include "kernel_common.h"
#include "cumsum_tiling.h"
#include "vector_tile.h"

template <typename dataType>
class CumsumKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR tilingGM, AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);
        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(x),
                             tiling_.S * tiling_.P);
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(y),
                             tiling_.S * tiling_.P);

        if ASCEND_IS_AIV {
            pipe_ = pipe;
            pipe_->InitBuffer(inQueue_, 1, tiling_.S * sizeof(dataType));
            pipe_->InitBuffer(outQueue_, 1, tiling_.S * sizeof(dataType));
            // Cast buffer: always fp32, used for accumulation
            // For fp32 input, castLocal aliases xInLocal_ via ReinterpretCast
            // For fp16/bf16, castBuf_ provides separate fp32 storage
            if constexpr (!std::is_same_v<dataType, float>) {
                pipe_->InitBuffer(castBuf_, tiling_.S * sizeof(float));
            }
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();

            for (int localIdx = 0; localIdx < tiling_.tasksPerCore; ++localIdx) {
                const int col = coreIdx * tiling_.tasksPerCore + localIdx;
                if (col < tiling_.P) {
                    ProcessColumn(col);
                }
            }
        }
    }

private:
    __aicore__ inline void ProcessColumn(int col)
    {
        // CopyIn: GM -> UB via VECIN queue
        inQueue_.AllocTensor<dataType>(xInLocal_);
        LoadGmToUb(xInLocal_, xGM_[col * tiling_.S],
                   static_cast<uint32_t>(tiling_.S));
        inQueue_.EnQue(xInLocal_);

        // Allocate output
        outQueue_.AllocTensor<dataType>(yOutLocal_);
        inQueue_.DeQue<dataType>(xInLocal_);

        // Get fp32 accumulation buffer
        AscendC::LocalTensor<float> accumLocal;
        PrepareAccumBuffer(accumLocal, xInLocal_);

        // Sequential prefix sum in fp32 (in-place on accum buffer)
        for (int i = 1; i < tiling_.S; ++i) {
            accumLocal.SetValue(i,
                accumLocal.GetValue(i) + accumLocal.GetValue(i - 1));
        }

        // Cast result back to original dtype
        FinalizeOutput(yOutLocal_, accumLocal, tiling_.S);

        inQueue_.FreeTensor(xInLocal_);
        outQueue_.EnQue(yOutLocal_);

        // CopyOut: UB -> GM via VECOUT queue
        outQueue_.DeQue<dataType>(yOutLocal_);
        StoreUbToGm(yGM_[col * tiling_.S], yOutLocal_,
                     static_cast<uint32_t>(tiling_.S));
        outQueue_.FreeTensor(yOutLocal_);
    }

    __aicore__ inline void PrepareAccumBuffer(
        AscendC::LocalTensor<float> &dst,
        AscendC::LocalTensor<dataType> &src)
    {
        if constexpr (std::is_same_v<dataType, float>) {
            // For float32: reuse input buffer as fp32 accum buffer
            dst = src.template ReinterpretCast<float>();
        } else {
            // For fp16/bf16: cast input to fp32 into castBuf_
            dst = castBuf_.Get<float>();
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, tiling_.S);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void FinalizeOutput(
        AscendC::LocalTensor<dataType> &out,
        AscendC::LocalTensor<float> &src,
        int32_t count)
    {
        if constexpr (std::is_same_v<dataType, float>) {
            // For float32: accum was in-place in xInLocal_, copy to output
            // Note: accumLocal and xInLocal_ share same memory.
            // We need to copy from xInLocal_ to yOutLocal_.
            AscendC::DataCopy(out, src.ReinterpretCast<dataType>(), count);
        } else {
            // For fp16/bf16: cast fp32 result back to original dtype
            AscendC::RoundMode roundMode = AscendC::RoundMode::CAST_NONE;
            if constexpr (std::is_same_v<dataType, bfloat16_t>) {
                roundMode = AscendC::RoundMode::CAST_ROUND;
            }
            AscendC::Cast(out, src, roundMode, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

private:
    CumsumTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};

    AscendC::GlobalTensor<dataType> xGM_;
    AscendC::GlobalTensor<dataType> yGM_;

    AscendC::TQue<AscendC::TPosition::VECIN, 0> inQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> outQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castBuf_;

    AscendC::LocalTensor<dataType> xInLocal_;
    AscendC::LocalTensor<dataType> yOutLocal_;
};
