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
            if constexpr (std::is_same_v<dataType, half>) {
                pipe_->InitBuffer(scanBuf_, tiling_.S * sizeof(float));
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

        if constexpr (std::is_same_v<dataType, bfloat16_t>) {
            SequentialScanAndFinalize();
        } else {
            if (tiling_.scanMode == 1) {
                if constexpr (std::is_same_v<dataType, half>) {
                    HillisFp16Scan();
                } else {
                    HillisSteeleScan(xInLocal_, yOutLocal_, tiling_.S);
                }
            } else {
                if (tiling_.scanMode == 0) {
                    PartitionedScan();
                } else {
                    SequentialScanAndFinalize();
                }
            }
        }

        inQueue_.FreeTensor(xInLocal_);
        outQueue_.EnQue(yOutLocal_);

        // CopyOut: UB -> GM via VECOUT queue
        outQueue_.DeQue<dataType>(yOutLocal_);
        StoreUbToGm(yGM_[col * tiling_.S], yOutLocal_,
                     static_cast<uint32_t>(tiling_.S));
        outQueue_.FreeTensor(yOutLocal_);
    }

    __aicore__ inline void PartitionedScan()
    {
        constexpr int32_t kPartitions = 40;
        const int32_t base = tiling_.S / kPartitions;
        const int32_t remainder = tiling_.S % kPartitions;
        AscendC::LocalTensor<float> accumLocal;
        if constexpr (std::is_same_v<dataType, float>) {
            accumLocal = xInLocal_.template ReinterpretCast<float>();
        } else {
            accumLocal = castBuf_.Get<float>();
        }

        int32_t part = 0;
        int32_t start = 0;
        int32_t count = base + (remainder > 0 ? 1 : 0);
        for (int32_t i = 0; i < tiling_.S; ++i) {
            const float value = static_cast<float>(xInLocal_.GetValue(i));
            if (i == start) {
                accumLocal.SetValue(i, value);
            } else {
                float local = accumLocal.GetValue(i - 1) + value;
                if constexpr (std::is_same_v<dataType, half>) {
                    local = static_cast<float>(static_cast<half>(local));
                }
                accumLocal.SetValue(i, local);
            }
            if (i + 1 < tiling_.S && i + 1 == start + count) {
                ++part;
                start += count;
                count = base + (part < remainder ? 1 : 0);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        part = 0;
        start = 0;
        count = base + (remainder > 0 ? 1 : 0);
        for (int32_t i = 0; i < tiling_.S; ++i) {
            const float carry = start == 0 ? 0.0f : accumLocal.GetValue(start - 1);
            float result = carry + accumLocal.GetValue(i);
            if constexpr (std::is_same_v<dataType, half>) {
                result = static_cast<float>(static_cast<half>(result));
            }
            accumLocal.SetValue(i, result);
            if (i + 1 < tiling_.S && i + 1 == start + count) {
                ++part;
                start += count;
                count = base + (part < remainder ? 1 : 0);
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        FinalizeOutput(yOutLocal_, accumLocal, tiling_.S);
    }

    __aicore__ inline void SequentialScanAndFinalize()
    {
        AscendC::LocalTensor<float> accumLocal;
        PrepareAccumBuffer(accumLocal, xInLocal_);

        for (int i = 1; i < tiling_.S; ++i) {
            float sum = accumLocal.GetValue(i) + accumLocal.GetValue(i - 1);
            if constexpr (std::is_same_v<dataType, half>) {
                if (tiling_.scanMode == 2) {
                    sum = static_cast<float>(static_cast<half>(sum));
                }
            }
            accumLocal.SetValue(i, sum);
        }
        FinalizeOutput(yOutLocal_, accumLocal, tiling_.S);
    }

    __aicore__ inline void HillisFp16Step(
        AscendC::LocalTensor<float> &dst,
        AscendC::LocalTensor<float> &src,
        int32_t stride)
    {
        for (int32_t i = 0; i < tiling_.S; ++i) {
            float value = src.GetValue(i);
            if (i >= stride) {
                value = static_cast<float>(static_cast<half>(
                    src.GetValue(i) + src.GetValue(i - stride)));
            }
            dst.SetValue(i, value);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void HillisFp16Scan()
    {
        AscendC::LocalTensor<float> first = castBuf_.Get<float>();
        AscendC::LocalTensor<float> second = scanBuf_.Get<float>();
        AscendC::Cast(first, xInLocal_, AscendC::RoundMode::CAST_NONE, tiling_.S);
        AscendC::PipeBarrier<PIPE_V>();

        bool sourceIsFirst = true;
        for (int32_t stride = 1; stride < tiling_.S; stride *= 2) {
            if (sourceIsFirst) {
                HillisFp16Step(second, first, stride);
            } else {
                HillisFp16Step(first, second, stride);
            }
            sourceIsFirst = !sourceIsFirst;
        }
        if (sourceIsFirst) {
            FinalizeOutput(yOutLocal_, first, tiling_.S);
        } else {
            FinalizeOutput(yOutLocal_, second, tiling_.S);
        }
    }

    __aicore__ inline void HillisSteeleStep(
        AscendC::LocalTensor<dataType> &dst,
        AscendC::LocalTensor<dataType> &src,
        int32_t stride,
        int32_t count)
    {
        if constexpr (std::is_same_v<dataType, float>) {
            if ((stride * static_cast<int32_t>(sizeof(dataType))) % 32 == 0) {
                AscendC::Adds(dst, src, static_cast<dataType>(0), stride);
                AscendC::Add(dst[stride], src[stride], src, count - stride);
                return;
            }
        }
        for (int32_t i = 0; i < count; ++i) {
            dataType value = src.GetValue(i);
            if (i >= stride) {
                value = static_cast<dataType>(
                    static_cast<float>(value) +
                    static_cast<float>(src.GetValue(i - stride)));
            }
            dst.SetValue(i, value);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void HillisSteeleScan(
        AscendC::LocalTensor<dataType> &input,
        AscendC::LocalTensor<dataType> &output,
        int32_t count)
    {
        bool sourceIsInput = true;
        for (int32_t stride = 1; stride < count; stride *= 2) {
            if (sourceIsInput) {
                HillisSteeleStep(output, input, stride, count);
            } else {
                HillisSteeleStep(input, output, stride, count);
            }
            sourceIsInput = !sourceIsInput;
        }
        if (sourceIsInput) {
            for (int32_t i = 0; i < count; ++i) {
                output.SetValue(i, input.GetValue(i));
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::PipeBarrier<PIPE_V>();
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
            AscendC::Adds(out, src.ReinterpretCast<dataType>(),
                          static_cast<dataType>(0), count);
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
    AscendC::TBuf<AscendC::TPosition::VECCALC> scanBuf_;

    AscendC::LocalTensor<dataType> xInLocal_;
    AscendC::LocalTensor<dataType> yOutLocal_;
};
