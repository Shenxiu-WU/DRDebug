#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <type_traits>
#include "kernel_operator.h"
#include "kernel_common.h"
#include "layer_norm_tiling.h"
#include "vector_tile.h"

template <typename dataType>
class LayerNormTwopassKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR tilingGM, AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);
        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(x), tiling_.M * tiling_.N);
        gammaGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(gamma), tiling_.N);
        betaGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(beta), tiling_.N);
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(y), tiling_.M * tiling_.N);

        if ASCEND_IS_AIV {
            pipe_ = pipe;
            subBlockRows_ = tiling_.subBlockM;
            blockNChunk_ = tiling_.blockNChunk;
            nChunks_ = tiling_.nChunks;

            pipe_->InitBuffer(gammaBuf_, blockNChunk_ * sizeof(dataType));
            pipe_->InitBuffer(betaBuf_, blockNChunk_ * sizeof(dataType));
            pipe_->InitBuffer(xInQueue_, 1, blockNChunk_ * sizeof(dataType));
            pipe_->InitBuffer(yOutQueue_, 1, blockNChunk_ * sizeof(dataType));
            pipe_->InitBuffer(reduceBuf_, blockNChunk_ * sizeof(float));
            pipe_->InitBuffer(sumBuf_, 16 * sizeof(float));
            pipe_->InitBuffer(invStdBuf_, sizeof(float));
            if constexpr (!std::is_same_v<dataType, float>) {
                pipe_->InitBuffer(xCastBuf_, blockNChunk_ * sizeof(float));
                pipe_->InitBuffer(gammaCastBuf_, blockNChunk_ * sizeof(float));
                pipe_->InitBuffer(betaCastBuf_, blockNChunk_ * sizeof(float));
                pipe_->InitBuffer(yCastBuf_, blockNChunk_ * sizeof(float));
            }
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
            const int subBlockIdx = AscendC::GetSubBlockIdx();
            const int mNum = (tiling_.M + tiling_.blockM - 1) / tiling_.blockM;

            for (int localIdx = 0; localIdx < tiling_.tasksPerCore; ++localIdx) {
                const int bx = coreIdx * tiling_.tasksPerCore + localIdx;
                if (bx >= mNum) continue;

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
    __aicore__ inline AscendC::RoundMode OutputRoundMode() const
    {
        if constexpr (std::is_same_v<dataType, bfloat16_t>) {
            return AscendC::RoundMode::CAST_ROUND;
        }
        return AscendC::RoundMode::CAST_NONE;
    }

    __aicore__ inline void CopyInChunk(int32_t rowIdx, int32_t colBase, int32_t count)
    {
        xInQueue_.AllocTensor<dataType>(xInLocal_);
        LoadGmToUb(xInLocal_, xGM_[rowIdx * tiling_.N + colBase], static_cast<uint32_t>(count));
        xInQueue_.EnQue(xInLocal_);
    }

    __aicore__ inline void CopyOutChunk(int32_t rowIdx, int32_t colBase, int32_t count)
    {
        yOutQueue_.DeQue<dataType>(yOutLocal_);
        StoreUbToGm(yGM_[rowIdx * tiling_.N + colBase], yOutLocal_, static_cast<uint32_t>(count));
        yOutQueue_.FreeTensor(yOutLocal_);
    }

    __aicore__ inline void ProcessRow(int32_t rowIdx)
    {
        reduceLocal_ = reduceBuf_.Get<float>();
        sumLocal_ = sumBuf_.Get<float>();
        int32_t N = tiling_.N;
        int32_t blockN = blockNChunk_;

        // === Pass 1: compute mean ===
        float sumX = 0.0f;

        for (int32_t chunk = 0; chunk < nChunks_; ++chunk) {
            int32_t colBase = chunk * blockN;
            int32_t validN = (colBase + blockN <= N) ? blockN : (N - colBase);

            CopyInChunk(rowIdx, colBase, validN);
            xInQueue_.DeQue<dataType>(xInLocal_);

            AscendC::LocalTensor<float> xF32;
            if constexpr (std::is_same_v<dataType, float>) {
                xF32 = xInLocal_.template ReinterpretCast<float>();
            } else {
                xF32 = xCastBuf_.Get<float>();
                AscendC::Cast(xF32, xInLocal_, AscendC::RoundMode::CAST_NONE, validN);
                AscendC::PipeBarrier<PIPE_V>();
            }

            // sum(x)
            AscendC::ReduceSum<float>(sumLocal_, xF32, reduceLocal_, validN);
            sumX += sumLocal_.GetValue(0);

            xInQueue_.FreeTensor(xInLocal_);
        }

        float mean = sumX * tiling_.invN;

        // === Pass 1.5: compute variance = sum((x-mean)^2) / N ===
        float sumXCentered2 = 0.0f;

        for (int32_t chunk = 0; chunk < nChunks_; ++chunk) {
            int32_t colBase = chunk * blockN;
            int32_t validN = (colBase + blockN <= N) ? blockN : (N - colBase);

            CopyInChunk(rowIdx, colBase, validN);
            xInQueue_.DeQue<dataType>(xInLocal_);

            AscendC::LocalTensor<float> xF32;
            if constexpr (std::is_same_v<dataType, float>) {
                xF32 = xInLocal_.template ReinterpretCast<float>();
            } else {
                xF32 = xCastBuf_.Get<float>();
                AscendC::Cast(xF32, xInLocal_, AscendC::RoundMode::CAST_NONE, validN);
                AscendC::PipeBarrier<PIPE_V>();
            }

            // x - mean
            AscendC::Adds(reduceLocal_, xF32, -mean, validN);
            // (x-mean)^2
            AscendC::Mul(reduceLocal_, reduceLocal_, reduceLocal_, validN);
            // sum((x-mean)^2) for this chunk
            AscendC::ReduceSum<float>(sumLocal_, reduceLocal_, reduceLocal_, validN);
            sumXCentered2 += sumLocal_.GetValue(0);

            xInQueue_.FreeTensor(xInLocal_);
        }

        // var = sum((x-mean)^2) / N + eps; inv_std = rsqrt(var)
        float var = sumXCentered2 * tiling_.invN + tiling_.eps;
        AscendC::Duplicate(sumLocal_, var, 1);
        AscendC::Rsqrt(sumLocal_, sumLocal_, 1);
        AscendC::PipeBarrier<PIPE_ALL>();
        float invStd = sumLocal_.GetValue(0);

        // === Pass 2: normalize and write output ===
        for (int32_t chunk = 0; chunk < nChunks_; ++chunk) {
            int32_t colBase = chunk * blockN;
            int32_t validN = (colBase + blockN <= N) ? blockN : (N - colBase);

            CopyInChunk(rowIdx, colBase, validN);
            xInQueue_.DeQue<dataType>(xInLocal_);

            yOutQueue_.AllocTensor<dataType>(yOutLocal_);

            AscendC::LocalTensor<float> xF32;
            if constexpr (std::is_same_v<dataType, float>) {
                xF32 = xInLocal_.template ReinterpretCast<float>();
            } else {
                xF32 = xCastBuf_.Get<float>();
                AscendC::Cast(xF32, xInLocal_, AscendC::RoundMode::CAST_NONE, validN);
                AscendC::PipeBarrier<PIPE_V>();
            }

            AscendC::LocalTensor<float> yF32;
            if constexpr (std::is_same_v<dataType, float>) {
                yF32 = yOutLocal_.template ReinterpretCast<float>();
            } else {
                yF32 = yCastBuf_.Get<float>();
            }

            // x - mean
            AscendC::Adds(yF32, xF32, -mean, validN);
            // * inv_std
            AscendC::Muls(yF32, yF32, invStd, validN);

            // Load gamma chunk
            gammaInLocal_ = gammaBuf_.Get<dataType>();
            LoadGmToUb(gammaInLocal_, gammaGM_[colBase], static_cast<uint32_t>(validN));
            AscendC::PipeBarrier<PIPE_MTE2>();
            AscendC::LocalTensor<float> gammaF32;
            if constexpr (std::is_same_v<dataType, float>) {
                gammaF32 = gammaInLocal_.template ReinterpretCast<float>();
            } else {
                gammaF32 = gammaCastBuf_.Get<float>();
                AscendC::Cast(gammaF32, gammaInLocal_, AscendC::RoundMode::CAST_NONE, validN);
                AscendC::PipeBarrier<PIPE_V>();
            }
            // * gamma
            AscendC::Mul(yF32, yF32, gammaF32, validN);

            // + beta
            if (tiling_.hasBias) {
                betaInLocal_ = betaBuf_.Get<dataType>();
                LoadGmToUb(betaInLocal_, betaGM_[colBase], static_cast<uint32_t>(validN));
                AscendC::PipeBarrier<PIPE_MTE2>();
                AscendC::LocalTensor<float> betaF32;
                if constexpr (std::is_same_v<dataType, float>) {
                    betaF32 = betaInLocal_.template ReinterpretCast<float>();
                } else {
                    betaF32 = betaCastBuf_.Get<float>();
                    AscendC::Cast(betaF32, betaInLocal_, AscendC::RoundMode::CAST_NONE, validN);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::Add(yF32, yF32, betaF32, validN);
            }

            if constexpr (!std::is_same_v<dataType, float>) {
                AscendC::Cast(yOutLocal_, yF32, OutputRoundMode(), validN);
                AscendC::PipeBarrier<PIPE_V>();
            }

            xInQueue_.FreeTensor(xInLocal_);
            yOutQueue_.EnQue(yOutLocal_);
            CopyOutChunk(rowIdx, colBase, validN);
        }
    }

private:
    LayerNormKernelTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};
    int subBlockRows_{0};
    int32_t blockNChunk_{0};
    int32_t nChunks_{0};

    AscendC::GlobalTensor<dataType> xGM_;
    AscendC::GlobalTensor<dataType> gammaGM_;
    AscendC::GlobalTensor<dataType> betaGM_;
    AscendC::GlobalTensor<dataType> yGM_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> betaBuf_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> xInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> yOutQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> invStdBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> betaCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yCastBuf_;

    AscendC::LocalTensor<dataType> gammaInLocal_;
    AscendC::LocalTensor<dataType> betaInLocal_;
    AscendC::LocalTensor<dataType> xInLocal_;
    AscendC::LocalTensor<dataType> yOutLocal_;
    AscendC::LocalTensor<float> reduceLocal_;
    AscendC::LocalTensor<float> sumLocal_;
};
