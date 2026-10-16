#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <type_traits>

#include "kernel_operator.h"

#include "kernel_common.h"
#include "rms_norm_tiling.h"
#include "vector_tile.h"

template <typename dataType>
class RmsNormSplitDKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR invRms, GM_ADDR tilingGM, AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);
        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(x), tiling_.M * tiling_.N);
        gammaGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(gamma), tiling_.N);
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(y), tiling_.M * tiling_.N);
        invRmsGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(invRms), tiling_.M);

        if ASCEND_IS_AIV {
            pipe_ = pipe;
            subBlockRows_ = tiling_.blockM / AscendC::GetSubBlockNum();
            pipe_->InitBuffer(xInQueue_, 1, kBlockN * sizeof(dataType));
            pipe_->InitBuffer(gammaInQueue_, 1, kBlockN * sizeof(dataType));
            pipe_->InitBuffer(yOutQueue_, 1, kBlockN * sizeof(dataType));
            pipe_->InitBuffer(invRmsOutQueue_, 1, sizeof(dataType));
            pipe_->InitBuffer(reduceBuf_, kBlockN * sizeof(float));
            pipe_->InitBuffer(sumBuf_, 16 * sizeof(float));
            pipe_->InitBuffer(tempBuf_, kTileFloatBytes);
            pipe_->InitBuffer(invRmsBuf_, sizeof(float));
            if constexpr (!std::is_same_v<dataType, float>) {
                pipe_->InitBuffer(xCastBuf_, kTileFloatBytes);
                pipe_->InitBuffer(gammaCastBuf_, kTileFloatBytes);
                pipe_->InitBuffer(yCastBuf_, kTileFloatBytes);
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
                if (bx >= BlockCount()) {
                    continue;
                }

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
    static constexpr int kBlockN = 1024;
    static constexpr uint32_t kTileFloatBytes = kBlockN * sizeof(float);

    __aicore__ inline int32_t BlockCount() const
    {
        return (tiling_.M + tiling_.blockM - 1) / tiling_.blockM;
    }

    __aicore__ inline int32_t NumTiles() const
    {
        return (tiling_.N + kBlockN - 1) / kBlockN;
    }

    __aicore__ inline int32_t GetValidN(int32_t colBase) const
    {
        return (colBase + kBlockN <= tiling_.N) ? kBlockN : (tiling_.N - colBase);
    }

    __aicore__ inline AscendC::RoundMode OutputRoundMode() const
    {
        if constexpr (std::is_same_v<dataType, bfloat16_t>) {
            return AscendC::RoundMode::CAST_ROUND;
        }
        return AscendC::RoundMode::CAST_NONE;
    }

    __aicore__ inline void PrepareInputTensor(
        AscendC::LocalTensor<float> &dst,
        AscendC::LocalTensor<dataType> &src,
        AscendC::TBuf<AscendC::TPosition::VECCALC> &castBuf,
        int32_t count)
    {
        if constexpr (std::is_same_v<dataType, float>) {
            dst = src.template ReinterpretCast<float>();
        } else {
            dst = castBuf.Get<float>();
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void PrepareOutputTensor(
        AscendC::LocalTensor<float> &dst,
        AscendC::LocalTensor<dataType> &out,
        AscendC::TBuf<AscendC::TPosition::VECCALC> &castBuf)
    {
        if constexpr (std::is_same_v<dataType, float>) {
            dst = out.template ReinterpretCast<float>();
        } else {
            dst = castBuf.Get<float>();
        }
    }

    __aicore__ inline void FinalizeOutputTensor(
        AscendC::LocalTensor<dataType> &out,
        AscendC::LocalTensor<float> &src,
        int32_t count)
    {
        if constexpr (!std::is_same_v<dataType, float>) {
            AscendC::Cast(out, src, OutputRoundMode(), count);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void PrepareInvRmsTensor(
        AscendC::LocalTensor<float> &dst,
        AscendC::LocalTensor<dataType> &out)
    {
        if constexpr (std::is_same_v<dataType, float>) {
            dst = out.template ReinterpretCast<float>();
        } else {
            dst = invRmsBuf_.Get<float>();
        }
    }

    __aicore__ inline void CopyInX(int32_t rowIdx, int32_t colBase, int32_t validN)
    {
        xInQueue_.AllocTensor<dataType>(xInLocal_);
        LoadGmToUb(xInLocal_, xGM_[rowIdx * tiling_.N + colBase], static_cast<uint32_t>(validN));
        xInQueue_.EnQue(xInLocal_);
    }

    __aicore__ inline void CopyInGamma(int32_t colBase, int32_t validN)
    {
        gammaInQueue_.AllocTensor<dataType>(gammaInLocal_);
        LoadGmToUb(gammaInLocal_, gammaGM_[colBase], static_cast<uint32_t>(validN));
        gammaInQueue_.EnQue(gammaInLocal_);
    }

    __aicore__ inline void CopyOutY(int32_t rowIdx, int32_t colBase, int32_t validN)
    {
        yOutQueue_.DeQue<dataType>(yOutLocal_);
        StoreUbToGm(yGM_[rowIdx * tiling_.N + colBase], yOutLocal_, static_cast<uint32_t>(validN));
        yOutQueue_.FreeTensor(yOutLocal_);
    }

    __aicore__ inline void CopyOutInvRms(int32_t rowIdx)
    {
        invRmsOutQueue_.DeQue<dataType>(invRmsOutLocal_);
        StoreUbToGm(invRmsGM_[rowIdx], invRmsOutLocal_, 1);
        invRmsOutQueue_.FreeTensor(invRmsOutLocal_);
    }

    __aicore__ inline float ComputeInvRms(int32_t rowIdx)
    {
        reduceLocal_ = reduceBuf_.Get<float>();
        sumLocal_ = sumBuf_.Get<float>();
        tempLocal_ = tempBuf_.Get<float>();
        invRmsOutQueue_.AllocTensor<dataType>(invRmsOutLocal_);
        PrepareInvRmsTensor(invRmsLocal_, invRmsOutLocal_);
        float sumSq = 0.0f;

        for (int by = 0; by < NumTiles(); ++by) {
            const int colBase = by * kBlockN;
            const int validN = GetValidN(colBase);

            CopyInX(rowIdx, colBase, validN);

            xInQueue_.DeQue<dataType>(xInLocal_);
            PrepareInputTensor(xLocal_, xInLocal_, xCastBuf_, validN);
            AscendC::Mul(tempLocal_, xLocal_, xLocal_, validN);
            AscendC::ReduceSum<float>(sumLocal_, tempLocal_, reduceLocal_, validN);
            sumSq += sumLocal_.GetValue(0);
            xInQueue_.FreeTensor(xInLocal_);
        }

        AscendC::Duplicate(sumLocal_, sumSq * tiling_.invN + tiling_.eps, 1);
        AscendC::Rsqrt(invRmsLocal_, sumLocal_, 1);
        AscendC::PipeBarrier<PIPE_ALL>();
        float invRms = invRmsLocal_.GetValue(0);
        AscendC::PipeBarrier<PIPE_ALL>();
        FinalizeOutputTensor(invRmsOutLocal_, invRmsLocal_, 1);

        invRmsOutQueue_.EnQue(invRmsOutLocal_);
        return invRms;
    }

    __aicore__ inline void ComputeTile(float invRms, int32_t validN)
    {
        yOutQueue_.AllocTensor<dataType>(yOutLocal_);
        xInQueue_.DeQue<dataType>(xInLocal_);
        gammaInQueue_.DeQue<dataType>(gammaInLocal_);
        PrepareInputTensor(xLocal_, xInLocal_, xCastBuf_, validN);
        PrepareInputTensor(gammaLocal_, gammaInLocal_, gammaCastBuf_, validN);
        PrepareOutputTensor(yLocal_, yOutLocal_, yCastBuf_);
        AscendC::Muls(yLocal_, xLocal_, invRms, validN);
        AscendC::Mul(yLocal_, yLocal_, gammaLocal_, validN);
        FinalizeOutputTensor(yOutLocal_, yLocal_, validN);
        xInQueue_.FreeTensor(xInLocal_);
        gammaInQueue_.FreeTensor(gammaInLocal_);
        yOutQueue_.EnQue(yOutLocal_);
    }

    __aicore__ inline void ProcessRow(int rowIdx)
    {
        float invRms = ComputeInvRms(rowIdx);
        CopyOutInvRms(rowIdx);

        for (int by = 0; by < NumTiles(); ++by) {
            const int colBase = by * kBlockN;
            const int validN = GetValidN(colBase);

            CopyInX(rowIdx, colBase, validN);
            CopyInGamma(colBase, validN);
            ComputeTile(invRms, validN);
            CopyOutY(rowIdx, colBase, validN);
        }
    }

private:
    RmsNormKernelTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};
    int subBlockRows_{0};

    AscendC::GlobalTensor<dataType> xGM_;
    AscendC::GlobalTensor<dataType> gammaGM_;
    AscendC::GlobalTensor<dataType> yGM_;
    AscendC::GlobalTensor<dataType> invRmsGM_;

    AscendC::TQue<AscendC::TPosition::VECIN, 0> xInQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> gammaInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> yOutQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> invRmsOutQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tempBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> invRmsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yCastBuf_;

    AscendC::LocalTensor<dataType> xInLocal_;
    AscendC::LocalTensor<dataType> gammaInLocal_;
    AscendC::LocalTensor<dataType> yOutLocal_;
    AscendC::LocalTensor<dataType> invRmsOutLocal_;
    AscendC::LocalTensor<float> invRmsLocal_;
    AscendC::LocalTensor<float> xLocal_;
    AscendC::LocalTensor<float> gammaLocal_;
    AscendC::LocalTensor<float> yLocal_;
    AscendC::LocalTensor<float> reduceLocal_;
    AscendC::LocalTensor<float> sumLocal_;
    AscendC::LocalTensor<float> tempLocal_;
};
