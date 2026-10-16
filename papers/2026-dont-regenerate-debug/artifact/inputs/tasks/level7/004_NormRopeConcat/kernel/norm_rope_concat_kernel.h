#pragma once

#include <type_traits>

#include "kernel_operator.h"

#include "kernel_common.h"
#include "norm_rope_concat_tiling.h"
#include "vector_tile.h"

// norm type constants
constexpr int NORM_NONE = 0;
constexpr int NORM_LAYERNORM = 1;
constexpr int NORM_LAYERNORM_AFFINE = 2;
constexpr int NORM_RMSNORM = 3;
constexpr int NORM_RMSNORM_AFFINE = 4;

// rope type constants
constexpr int ROPE_NONE = 0;
constexpr int ROPE_INTERLEAVE = 1;
constexpr int ROPE_HALF = 2;

template <typename dataType>
class NormRopeConcatKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR query,
        GM_ADDR key,
        GM_ADDR value,
        GM_ADDR queryWeight,
        GM_ADDR queryBias,
        GM_ADDR keyWeight,
        GM_ADDR keyBias,
        GM_ADDR sin,
        GM_ADDR cos,
        GM_ADDR queryOut,
        GM_ADDR keyOut,
        GM_ADDR valueOut,
        GM_ADDR tilingGM,
        AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);
        queryGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(query), tiling_.B * tiling_.S * tiling_.N * tiling_.D);
        keyGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(key), tiling_.B * tiling_.S * tiling_.N * tiling_.D);
        valueGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(value), tiling_.B * tiling_.S * tiling_.N * tiling_.D);
        queryWeightGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(queryWeight), tiling_.D);
        queryBiasGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(queryBias), tiling_.D);
        keyWeightGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(keyWeight), tiling_.D);
        keyBiasGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(keyBias), tiling_.D);
        sinGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(sin), tiling_.S * tiling_.D);
        cosGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(cos), tiling_.S * tiling_.D);
        queryOutGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(queryOut), tiling_.B * tiling_.N * tiling_.S * tiling_.D);
        keyOutGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(keyOut), tiling_.B * tiling_.N * tiling_.S * tiling_.D);
        valueOutGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(valueOut), tiling_.B * tiling_.N * tiling_.S * tiling_.D);

        if ASCEND_IS_AIV {
            pipe_ = pipe;
            const int32_t D = tiling_.D;

            pipe_->InitBuffer(rowInQueue_, 1, D * sizeof(dataType));
            pipe_->InitBuffer(rowOutQueue_, 1, D * sizeof(dataType));
            pipe_->InitBuffer(weightBuf_, D * sizeof(dataType));
            pipe_->InitBuffer(biasBuf_, D * sizeof(dataType));
            pipe_->InitBuffer(sinBuf_, D * sizeof(dataType));
            pipe_->InitBuffer(cosBuf_, D * sizeof(dataType));
            pipe_->InitBuffer(fp32RowBuf_, D * sizeof(float));
            pipe_->InitBuffer(fp32Tmp1_, D * sizeof(float));
            pipe_->InitBuffer(fp32Tmp2_, D * sizeof(float));
            pipe_->InitBuffer(sumBuf_, 16 * sizeof(float));
            pipe_->InitBuffer(rotatedBuf_, D * sizeof(float));

            if constexpr (!std::is_same_v<dataType, float>) {
                pipe_->InitBuffer(castInBuf_, D * sizeof(float));
                pipe_->InitBuffer(castOutBuf_, D * sizeof(float));
                pipe_->InitBuffer(castWeightBuf_, D * sizeof(float));
                pipe_->InitBuffer(castBiasBuf_, D * sizeof(float));
                pipe_->InitBuffer(castSinBuf_, D * sizeof(float));
                pipe_->InitBuffer(castCosBuf_, D * sizeof(float));
            }

            auto weightDummy = weightBuf_.Get<dataType>();
            auto biasDummy = biasBuf_.Get<dataType>();
            if constexpr (std::is_same_v<dataType, float>) {
                queryWeight_ = weightDummy.template ReinterpretCast<float>();
                queryBias_ = biasDummy.template ReinterpretCast<float>();
                keyWeight_ = weightDummy.template ReinterpretCast<float>();
                keyBias_ = biasDummy.template ReinterpretCast<float>();
            } else {
                queryWeight_ = castWeightBuf_.Get<float>();
                queryBias_ = castBiasBuf_.Get<float>();
                keyWeight_ = castWeightBuf_.Get<float>();
                keyBias_ = castBiasBuf_.Get<float>();
            }

            if (tiling_.normType == NORM_LAYERNORM_AFFINE || tiling_.normType == NORM_RMSNORM_AFFINE) {
                auto qw = weightBuf_.Get<dataType>();
                LoadGmToUb(qw, queryWeightGM_, static_cast<uint32_t>(D));
                AscendC::PipeBarrier<PIPE_MTE2>();
                AscendC::PipeBarrier<PIPE_ALL>();
                PrepareInputTensor(queryWeight_, qw, castWeightBuf_, D);

                auto qb = biasBuf_.Get<dataType>();
                LoadGmToUb(qb, queryBiasGM_, static_cast<uint32_t>(D));
                AscendC::PipeBarrier<PIPE_MTE2>();
                AscendC::PipeBarrier<PIPE_ALL>();
                PrepareInputTensor(queryBias_, qb, castBiasBuf_, D);

                auto kw = weightBuf_.Get<dataType>();
                LoadGmToUb(kw, keyWeightGM_, static_cast<uint32_t>(D));
                AscendC::PipeBarrier<PIPE_MTE2>();
                AscendC::PipeBarrier<PIPE_ALL>();
                PrepareInputTensor(keyWeight_, kw, castWeightBuf_, D);

                auto kb = biasBuf_.Get<dataType>();
                LoadGmToUb(kb, keyBiasGM_, static_cast<uint32_t>(D));
                AscendC::PipeBarrier<PIPE_MTE2>();
                AscendC::PipeBarrier<PIPE_ALL>();
                PrepareInputTensor(keyBias_, kb, castBiasBuf_, D);
            }
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
            const int32_t M = tiling_.B * tiling_.S * tiling_.N;
            for (int localIdx = 0; localIdx < tiling_.tasksPerCore; ++localIdx) {
                const int bx = coreIdx * tiling_.tasksPerCore + localIdx;
                if (bx >= BlockCount()) {
                    continue;
                }
                for (int row = 0; row < tiling_.blockM; ++row) {
                    const int rowIdx = bx * tiling_.blockM + row;
                    if (rowIdx >= M) {
                        continue;
                    }
                    ProcessRow(rowIdx);
                }
            }
        }
    }

private:
    __aicore__ inline int32_t BlockCount() const
    {
        int32_t M = tiling_.B * tiling_.S * tiling_.N;
        return (M + tiling_.blockM - 1) / tiling_.blockM;
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

    __aicore__ inline void DecomposeRow(int32_t rowIdx, int32_t &b, int32_t &s, int32_t &n)
    {
        const int32_t SN = tiling_.S * tiling_.N;
        b = rowIdx / SN;
        int rest = rowIdx % SN;
        s = rest / tiling_.N;
        n = rest % tiling_.N;
    }

    __aicore__ inline int64_t InputOffset(int32_t b, int32_t s, int32_t n, int32_t d)
    {
        return static_cast<int64_t>(b) * tiling_.S * tiling_.N * tiling_.D +
               static_cast<int64_t>(s) * tiling_.N * tiling_.D +
               static_cast<int64_t>(n) * tiling_.D + d;
    }

    __aicore__ inline int64_t OutputOffset(int32_t b, int32_t n, int32_t s, int32_t d)
    {
        return static_cast<int64_t>(b) * tiling_.N * tiling_.S * tiling_.D +
               static_cast<int64_t>(n) * tiling_.S * tiling_.D +
               static_cast<int64_t>(s) * tiling_.D + d;
    }

    __aicore__ inline void CopyInRow(AscendC::GlobalTensor<dataType> &gm, int32_t b, int32_t s, int32_t n)
    {
        rowInQueue_.AllocTensor<dataType>(rowInLocal_);
        LoadGmToUb(rowInLocal_, gm[InputOffset(b, s, n, 0)], static_cast<uint32_t>(tiling_.D));
        rowInQueue_.EnQue(rowInLocal_);
    }

    __aicore__ inline void CopyOutRow(AscendC::GlobalTensor<dataType> &gm, int32_t b, int32_t n, int32_t s)
    {
        rowOutQueue_.DeQue<dataType>(rowOutLocal_);
        StoreUbToGm(gm[OutputOffset(b, n, s, 0)], rowOutLocal_, static_cast<uint32_t>(tiling_.D));
        rowOutQueue_.FreeTensor(rowOutLocal_);
    }

    __aicore__ inline void ComputeNorm(
        AscendC::LocalTensor<float> row,
        AscendC::LocalTensor<float> weight,
        AscendC::LocalTensor<float> bias,
        int32_t normType)
    {
        const int32_t D = tiling_.D;
        auto tmp1 = fp32Tmp1_.Get<float>();
        auto tmp2 = fp32Tmp2_.Get<float>();
        auto sumLocal = sumBuf_.Get<float>();

        if (normType == NORM_LAYERNORM || normType == NORM_LAYERNORM_AFFINE) {
            AscendC::ReduceSum<float>(sumLocal, row, tmp1, D);
            float mean = sumLocal.GetValue(0) * tiling_.invD;
            AscendC::Duplicate(sumLocal, mean, 1);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::Sub(tmp1, row, sumLocal, D);
            AscendC::Mul(tmp2, tmp1, tmp1, D);
            AscendC::ReduceSum<float>(sumLocal, tmp2, row, D);
            float var = sumLocal.GetValue(0) * tiling_.invD + tiling_.eps;
            AscendC::Duplicate(sumLocal, var, 1);
            AscendC::Rsqrt(sumLocal, sumLocal, 1);
            AscendC::PipeBarrier<PIPE_ALL>();

            float rstd = sumLocal.GetValue(0);
            AscendC::Duplicate(sumLocal, rstd, 1);
            AscendC::Mul(row, tmp1, sumLocal, D);

            if (normType == NORM_LAYERNORM_AFFINE) {
                AscendC::Mul(row, row, weight, D);
                AscendC::Add(row, row, bias, D);
            }
        } else if (normType == NORM_RMSNORM || normType == NORM_RMSNORM_AFFINE) {
            AscendC::Mul(tmp1, row, row, D);
            AscendC::ReduceSum<float>(sumLocal, tmp1, tmp2, D);
            float ms = sumLocal.GetValue(0) * tiling_.invD + tiling_.eps;
            AscendC::Duplicate(sumLocal, ms, 1);
            AscendC::Rsqrt(sumLocal, sumLocal, 1);
            AscendC::PipeBarrier<PIPE_ALL>();

            float rms = sumLocal.GetValue(0);
            AscendC::Duplicate(sumLocal, rms, 1);
            AscendC::Mul(row, row, sumLocal, D);

            if (normType == NORM_RMSNORM_AFFINE) {
                AscendC::Mul(row, row, weight, D);
            }
        }
    }

    __aicore__ inline void ComputeRope(
        AscendC::LocalTensor<float> row,
        int32_t s)
    {
        const int32_t D = tiling_.D;
        int32_t ropeType = tiling_.ropeType;
        if (ropeType == ROPE_NONE) {
            return;
        }

        auto sinLocal = sinBuf_.Get<dataType>();
        auto cosLocal = cosBuf_.Get<dataType>();
        auto sinCast = fp32Tmp1_.Get<float>();
        auto cosCast = fp32Tmp2_.Get<float>();

        int64_t sinOff = static_cast<int64_t>(s) * D;
        LoadGmToUb(sinLocal, sinGM_[sinOff], static_cast<uint32_t>(D));
        AscendC::PipeBarrier<PIPE_MTE2>();
        LoadGmToUb(cosLocal, cosGM_[sinOff], static_cast<uint32_t>(D));
        AscendC::PipeBarrier<PIPE_MTE2>();

        if constexpr (!std::is_same_v<dataType, float>) {
            AscendC::Cast(sinCast, sinLocal, AscendC::RoundMode::CAST_NONE, D);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(cosCast, cosLocal, AscendC::RoundMode::CAST_NONE, D);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            sinCast = sinLocal.template ReinterpretCast<float>();
            cosCast = cosLocal.template ReinterpretCast<float>();
        }

        auto rotated = rotatedBuf_.Get<float>();

        if (ropeType == ROPE_INTERLEAVE) {
            for (int i = 0; i < D; i += 2) {
                float v0 = row.GetValue(i);
                float v1 = row.GetValue(i + 1);
                rotated.SetValue(i, -v1);
                rotated.SetValue(i + 1, v0);
            }
            AscendC::PipeBarrier<PIPE_V>();
        } else if (ropeType == ROPE_HALF) {
            const int half = D / 2;
            for (int i = 0; i < half; ++i) {
                float vNeg = -row.GetValue(half + i);
                rotated.SetValue(i, vNeg);
            }
            for (int i = 0; i < half; ++i) {
                float v = row.GetValue(i);
                rotated.SetValue(half + i, v);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::Mul(row, row, cosCast, D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(rotated, rotated, sinCast, D);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(row, row, rotated, D);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessQueryOrKey(
        AscendC::GlobalTensor<dataType> &inGM,
        AscendC::GlobalTensor<dataType> &outGM,
        AscendC::LocalTensor<float> &weight,
        AscendC::LocalTensor<float> &bias,
        int32_t b, int32_t s, int32_t n)
    {
        const int32_t D = tiling_.D;
        CopyInRow(inGM, b, s, n);

        rowInQueue_.DeQue<dataType>(rowInLocal_);
        auto fp32Row = fp32RowBuf_.Get<float>();

        if constexpr (std::is_same_v<dataType, float>) {
            auto rowInFp32 = rowInLocal_.template ReinterpretCast<float>();
            AscendC::DataCopy(fp32Row, rowInFp32, D);
            AscendC::PipeBarrier<PIPE_MTE2>();
        } else {
            auto castBuf = castInBuf_.Get<float>();
            AscendC::Cast(castBuf, rowInLocal_, AscendC::RoundMode::CAST_NONE, D);
            AscendC::PipeBarrier<PIPE_V>();
            fp32Row = castBuf;
        }

        rowInQueue_.FreeTensor(rowInLocal_);

        if (tiling_.normType != NORM_NONE) {
            ComputeNorm(fp32Row, weight, bias, tiling_.normType);
        }

        if (tiling_.ropeType != ROPE_NONE) {
            ComputeRope(fp32Row, s);
        }

        rowOutQueue_.AllocTensor<dataType>(rowOutLocal_);
        if constexpr (std::is_same_v<dataType, float>) {
            auto fp32Out = rowOutLocal_.template ReinterpretCast<float>();
            AscendC::DataCopy(fp32Out, fp32Row, D);
            AscendC::PipeBarrier<PIPE_MTE2>();
        } else {
            AscendC::Cast(rowOutLocal_, fp32Row, OutputRoundMode(), D);
            AscendC::PipeBarrier<PIPE_V>();
        }
        rowOutQueue_.EnQue(rowOutLocal_);
        CopyOutRow(outGM, b, n, s);
    }

    __aicore__ inline void ProcessValue(
        AscendC::GlobalTensor<dataType> &inGM,
        AscendC::GlobalTensor<dataType> &outGM,
        int32_t b, int32_t s, int32_t n)
    {
        const int32_t D = tiling_.D;
        CopyInRow(inGM, b, s, n);
        rowInQueue_.DeQue<dataType>(rowInLocal_);

        rowOutQueue_.AllocTensor<dataType>(rowOutLocal_);
        AscendC::DataCopy(rowOutLocal_, rowInLocal_, D);
        AscendC::PipeBarrier<PIPE_MTE2>();

        rowInQueue_.FreeTensor(rowInLocal_);
        rowOutQueue_.EnQue(rowOutLocal_);
        CopyOutRow(outGM, b, n, s);
    }

    __aicore__ inline void ProcessRow(int32_t rowIdx)
    {
        int32_t b, s, n;
        DecomposeRow(rowIdx, b, s, n);
        ProcessQueryOrKey(queryGM_, queryOutGM_, queryWeight_, queryBias_, b, s, n);
        ProcessQueryOrKey(keyGM_, keyOutGM_, keyWeight_, keyBias_, b, s, n);
        ProcessValue(valueGM_, valueOutGM_, b, s, n);
    }

private:
    NormRopeConcatKernelTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};

    AscendC::GlobalTensor<dataType> queryGM_;
    AscendC::GlobalTensor<dataType> keyGM_;
    AscendC::GlobalTensor<dataType> valueGM_;
    AscendC::GlobalTensor<dataType> queryWeightGM_;
    AscendC::GlobalTensor<dataType> queryBiasGM_;
    AscendC::GlobalTensor<dataType> keyWeightGM_;
    AscendC::GlobalTensor<dataType> keyBiasGM_;
    AscendC::GlobalTensor<dataType> sinGM_;
    AscendC::GlobalTensor<dataType> cosGM_;
    AscendC::GlobalTensor<dataType> queryOutGM_;
    AscendC::GlobalTensor<dataType> keyOutGM_;
    AscendC::GlobalTensor<dataType> valueOutGM_;

    AscendC::TQue<AscendC::TPosition::VECIN, 0> rowInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> rowOutQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> weightBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> biasBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sinBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cosBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32RowBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32Tmp1_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32Tmp2_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rotatedBuf_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> castInBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castOutBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castWeightBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castBiasBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castSinBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castCosBuf_;

    AscendC::LocalTensor<dataType> rowInLocal_;
    AscendC::LocalTensor<dataType> rowOutLocal_;
    AscendC::LocalTensor<float> queryWeight_;
    AscendC::LocalTensor<float> queryBias_;
    AscendC::LocalTensor<float> keyWeight_;
    AscendC::LocalTensor<float> keyBias_;
};
