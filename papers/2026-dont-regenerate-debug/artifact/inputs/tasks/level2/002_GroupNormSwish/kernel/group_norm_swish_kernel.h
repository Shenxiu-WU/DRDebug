#pragma once

#include <type_traits>
#include "kernel_operator.h"
#include "kernel_common.h"
#include "group_norm_swish_tiling.h"
#include "vector_tile.h"

template <typename dataType>
class GroupNormSwishKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
        GM_ADDR output, GM_ADDR mean, GM_ADDR rstd,
        GM_ADDR tilingGM, AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);
        int64_t xSize = static_cast<int64_t>(tiling_.N) * tiling_.C * tiling_.spatialSize;
        int64_t outSize = xSize;
        int64_t meanSize = static_cast<int64_t>(tiling_.N) * tiling_.numGroups;
        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(x), xSize);
        weightGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(weight), tiling_.C);
        biasGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(bias), tiling_.C);
        outputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(output), outSize);
        meanGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(mean), meanSize);
        rstdGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(rstd), meanSize);

        pipe_ = pipe;
        int32_t maxTileSize = tiling_.groupSize * kTileSpatial;

        pipe_->InitBuffer(xInQueue_, 1, maxTileSize * sizeof(dataType));
        pipe_->InitBuffer(outQueue_, 1, maxTileSize * sizeof(dataType));
        pipe_->InitBuffer(meanOutQueue_, 1, sizeof(dataType));
        pipe_->InitBuffer(rstdOutQueue_, 1, sizeof(dataType));

        pipe_->InitBuffer(tmpBuf1_, maxTileSize * sizeof(float));
        pipe_->InitBuffer(tmpBuf2_, maxTileSize * sizeof(float));
        pipe_->InitBuffer(sigmoidTmpBuf_, maxTileSize * sizeof(float));
        pipe_->InitBuffer(scaleBroadBuf_, maxTileSize * sizeof(float));
        pipe_->InitBuffer(broadcastTmpBuf_, 2 * maxTileSize * sizeof(uint8_t));
        pipe_->InitBuffer(reduceBuf_, 8192);
        pipe_->InitBuffer(sumBuf_, tiling_.groupSize * sizeof(float));
        pipe_->InitBuffer(sumSqBuf_, tiling_.groupSize * sizeof(float));
        pipe_->InitBuffer(wBuf_, tiling_.groupSize * sizeof(dataType));
        pipe_->InitBuffer(bBuf_, tiling_.groupSize * sizeof(dataType));
        pipe_->InitBuffer(meanBuf_, sizeof(float));
        pipe_->InitBuffer(rstdBuf_, sizeof(float));

        if constexpr (!std::is_same_v<dataType, float>) {
            pipe_->InitBuffer(xCastBuf_, maxTileSize * sizeof(float));
            pipe_->InitBuffer(outCastBuf_, maxTileSize * sizeof(float));
            pipe_->InitBuffer(wCastBuf_, tiling_.groupSize * sizeof(float));
            pipe_->InitBuffer(bCastBuf_, tiling_.groupSize * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
            for (int localIdx = 0; localIdx < tiling_.tasksPerCore; ++localIdx) {
                const int mg = coreIdx * tiling_.tasksPerCore + localIdx;
                if (mg >= tiling_.M) {
                    continue;
                }
                const int n = mg / tiling_.numGroups;
                const int g = mg % tiling_.numGroups;
                const int cStart = g * tiling_.groupSize;

                // ---------- Reduce phase ----------
                float sumAcc = 0.0f;
                float sumSqAcc = 0.0f;
                int fullTiles = tiling_.spatialSize / kTileSpatial;
                int partialLen = tiling_.spatialSize % kTileSpatial;

                for (int st = 0; st < fullTiles; ++st) {
                    int sStart = st * kTileSpatial;
                    ReduceTile(n, cStart, sStart, kTileSpatial, sumAcc, sumSqAcc);
                }
                if (partialLen > 0) {
                    int sStart = fullTiles * kTileSpatial;
                    ReduceTile(n, cStart, sStart, partialLen, sumAcc, sumSqAcc);
                }

                float meanVal = sumAcc * tiling_.invK;
                float varVal = sumSqAcc * tiling_.invK - meanVal * meanVal;
                float rstdVal = 1.0f / sqrt(varVal + tiling_.eps);

                // Load weight and bias
                LoadWeightBias(cStart);

                // ---------- Normalize + Swish phase ----------
                for (int st = 0; st < fullTiles; ++st) {
                    int sStart = st * kTileSpatial;
                    NormalizeSwishTile(n, cStart, sStart, kTileSpatial, meanVal, rstdVal);
                }
                if (partialLen > 0) {
                    int sStart = fullTiles * kTileSpatial;
                    NormalizeSwishTile(n, cStart, sStart, partialLen, meanVal, rstdVal);
                }

                // Write mean and rstd
                WriteMeanRstd(n, g, meanVal, rstdVal);
            }
        }
    }

private:
    template <typename T>
    __aicore__ inline AscendC::RoundMode OutputRoundMode() const
    {
        if constexpr (std::is_same_v<T, bfloat16_t>) {
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
            AscendC::Cast(out, src, OutputRoundMode<dataType>(), count);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ReduceTile(int n, int cStart, int sStart, int tileSpatial,
                                       float &sumAcc, float &sumSqAcc)
    {
        int tileSize = tileSpatial * tiling_.groupSize;

        xInQueue_.AllocTensor<dataType>(xInLocal_);
        for (int ci = 0; ci < tiling_.groupSize; ++ci) {
            int64_t chOffset = (static_cast<int64_t>(n) * tiling_.C + cStart + ci) * tiling_.spatialSize + sStart;
            auto xSub = xInLocal_[ci * tileSpatial];
            LoadGmToUb(xSub, xGM_[chOffset], static_cast<uint32_t>(tileSpatial));
        }
        AscendC::PipeBarrier<PIPE_MTE2>();
        xInQueue_.EnQue(xInLocal_);

        xInQueue_.DeQue<dataType>(xInLocal_);
        if constexpr (std::is_same_v<dataType, float>) {
            xLocal_ = xInLocal_.template ReinterpretCast<float>();
        } else {
            xLocal_ = xCastBuf_.Get<float>();
            AscendC::Cast(xLocal_, xInLocal_, AscendC::RoundMode::CAST_NONE, tileSize);
            AscendC::PipeBarrier<PIPE_V>();
        }

        auto xSqLocal = tmpBuf1_.Get<float>();
        AscendC::Mul(xSqLocal, xLocal_, xLocal_, tileSize);

        auto sumLocal = sumBuf_.Get<float>();
        auto sumSqLocal = sumSqBuf_.Get<float>();
        auto reduceTmp = reduceBuf_.Get<uint8_t>();

        const uint32_t reduceShape[2] = {
            static_cast<uint32_t>(tiling_.groupSize),
            static_cast<uint32_t>(tileSpatial),
        };
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(
            sumLocal, xLocal_, reduceTmp, reduceShape, true);
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(
            sumSqLocal, xSqLocal, reduceTmp, reduceShape, true);

        for (int gi = 0; gi < tiling_.groupSize; ++gi) {
            sumAcc += sumLocal.GetValue(gi);
            sumSqAcc += sumSqLocal.GetValue(gi);
        }

        xInQueue_.FreeTensor(xInLocal_);
    }

    __aicore__ inline void LoadWeightBias(int cStart)
    {
        auto wInLocal = wBuf_.Get<dataType>();
        auto bInLocal = bBuf_.Get<dataType>();
        LoadGmToUb(wInLocal, weightGM_[cStart], static_cast<uint32_t>(tiling_.groupSize));
        LoadGmToUb(bInLocal, biasGM_[cStart], static_cast<uint32_t>(tiling_.groupSize));
        AscendC::PipeBarrier<PIPE_MTE2>();

        if constexpr (std::is_same_v<dataType, float>) {
            wLocal_ = wInLocal.template ReinterpretCast<float>();
            bLocal_ = bInLocal.template ReinterpretCast<float>();
        } else {
            wLocal_ = wCastBuf_.Get<float>();
            bLocal_ = bCastBuf_.Get<float>();
            AscendC::Cast(wLocal_, wInLocal, AscendC::RoundMode::CAST_NONE, tiling_.groupSize);
            AscendC::Cast(bLocal_, bInLocal, AscendC::RoundMode::CAST_NONE, tiling_.groupSize);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void NormalizeSwishTile(int n, int cStart, int sStart, int tileSpatial,
                                               float meanVal, float rstdVal)
    {
        int tileSize = tileSpatial * tiling_.groupSize;

        xInQueue_.AllocTensor<dataType>(xInLocal_);
        for (int ci = 0; ci < tiling_.groupSize; ++ci) {
            int64_t chOffset = (static_cast<int64_t>(n) * tiling_.C + cStart + ci) * tiling_.spatialSize + sStart;
            auto xSub = xInLocal_[ci * tileSpatial];
            LoadGmToUb(xSub, xGM_[chOffset], static_cast<uint32_t>(tileSpatial));
        }
        AscendC::PipeBarrier<PIPE_MTE2>();
        xInQueue_.EnQue(xInLocal_);

        outQueue_.AllocTensor<dataType>(outLocal_);

        xInQueue_.DeQue<dataType>(xInLocal_);
        if constexpr (std::is_same_v<dataType, float>) {
            xLocal_ = xInLocal_.template ReinterpretCast<float>();
            yLocal_ = outLocal_.template ReinterpretCast<float>();
        } else {
            xLocal_ = xCastBuf_.Get<float>();
            yLocal_ = outCastBuf_.Get<float>();
            AscendC::Cast(xLocal_, xInLocal_, AscendC::RoundMode::CAST_NONE, tileSize);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // (x - mean) * rstd
        AscendC::Adds(yLocal_, xLocal_, -meanVal, tileSize);
        AscendC::Muls(yLocal_, yLocal_, rstdVal, tileSize);

        // scale and shift: x * w + b
        auto scaleBroad = scaleBroadBuf_.Get<float>();
        auto broadcastTmp = broadcastTmpBuf_.Get<uint8_t>();
        const uint32_t wSrcShape[2] = {static_cast<uint32_t>(tiling_.groupSize), 1U};
        const uint32_t wDstShape[2] = {
            static_cast<uint32_t>(tiling_.groupSize),
            static_cast<uint32_t>(tileSpatial),
        };
        AscendC::Broadcast<float, 2, 1>(scaleBroad, wLocal_, wDstShape, wSrcShape, broadcastTmp);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(yLocal_, yLocal_, scaleBroad, tileSize);

        AscendC::Broadcast<float, 2, 1>(scaleBroad, bLocal_, wDstShape, wSrcShape, broadcastTmp);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(yLocal_, yLocal_, scaleBroad, tileSize);

        // Swish: x * sigmoid(swish_scale * x)
        auto sigmoidIn = tmpBuf1_.Get<float>();
        auto sigmoidOut = tmpBuf2_.Get<float>();
        auto sigmoidTmp = sigmoidTmpBuf_.Get<uint8_t>();
        AscendC::Muls(sigmoidIn, yLocal_, tiling_.swishScale, tileSize);
        AscendC::Sigmoid(sigmoidOut, sigmoidIn, sigmoidTmp, static_cast<uint32_t>(tileSize));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(yLocal_, yLocal_, sigmoidOut, tileSize);

        FinalizeOutputTensor(outLocal_, yLocal_, tileSize);
        outQueue_.EnQue(outLocal_);

        xInQueue_.FreeTensor(xInLocal_);

        outQueue_.DeQue<dataType>(outLocal_);
        for (int ci = 0; ci < tiling_.groupSize; ++ci) {
            int64_t chOffset = (static_cast<int64_t>(n) * tiling_.C + cStart + ci) * tiling_.spatialSize + sStart;
            auto outSub = outLocal_[ci * tileSpatial];
            StoreUbToGm(outputGM_[chOffset], outSub, static_cast<uint32_t>(tileSpatial));
        }
        AscendC::PipeBarrier<PIPE_MTE2>();
        outQueue_.FreeTensor(outLocal_);
    }

    __aicore__ inline void WriteMeanRstd(int n, int g, float meanVal, float rstdVal)
    {
        auto meanLocal = meanBuf_.Get<float>();
        auto rstdLocal = rstdBuf_.Get<float>();
        meanLocal.SetValue(0, meanVal);
        rstdLocal.SetValue(0, rstdVal);
        AscendC::PipeBarrier<PIPE_V>();

        meanOutQueue_.AllocTensor<dataType>(meanOutLocal_);
        rstdOutQueue_.AllocTensor<dataType>(rstdOutLocal_);
        if constexpr (std::is_same_v<dataType, float>) {
            auto meanOut = meanOutLocal_.template ReinterpretCast<float>();
            auto rstdOut = rstdOutLocal_.template ReinterpretCast<float>();
            meanOut.SetValue(0, meanVal);
            rstdOut.SetValue(0, rstdVal);
        } else {
            AscendC::Cast(meanOutLocal_, meanLocal, OutputRoundMode<dataType>(), 1);
            AscendC::Cast(rstdOutLocal_, rstdLocal, OutputRoundMode<dataType>(), 1);
            AscendC::PipeBarrier<PIPE_V>();
        }
        StoreUbToGm(meanGM_[n * tiling_.numGroups + g], meanOutLocal_, 1);
        StoreUbToGm(rstdGM_[n * tiling_.numGroups + g], rstdOutLocal_, 1);
        meanOutQueue_.FreeTensor(meanOutLocal_);
        rstdOutQueue_.FreeTensor(rstdOutLocal_);
    }

private:
    GroupNormSwishTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};

    AscendC::GlobalTensor<dataType> xGM_;
    AscendC::GlobalTensor<dataType> weightGM_;
    AscendC::GlobalTensor<dataType> biasGM_;
    AscendC::GlobalTensor<dataType> outputGM_;
    AscendC::GlobalTensor<dataType> meanGM_;
    AscendC::GlobalTensor<dataType> rstdGM_;

    AscendC::TQue<AscendC::TPosition::VECIN, 0> xInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> outQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> meanOutQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> rstdOutQueue_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf2_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sigmoidTmpBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scaleBroadBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> broadcastTmpBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumSqBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> wBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> meanBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rstdBuf_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> xCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> wCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bCastBuf_;

    AscendC::LocalTensor<dataType> xInLocal_;
    AscendC::LocalTensor<dataType> outLocal_;
    AscendC::LocalTensor<dataType> meanOutLocal_;
    AscendC::LocalTensor<dataType> rstdOutLocal_;
    AscendC::LocalTensor<float> xLocal_;
    AscendC::LocalTensor<float> yLocal_;
    AscendC::LocalTensor<float> wLocal_;
    AscendC::LocalTensor<float> bLocal_;
};
