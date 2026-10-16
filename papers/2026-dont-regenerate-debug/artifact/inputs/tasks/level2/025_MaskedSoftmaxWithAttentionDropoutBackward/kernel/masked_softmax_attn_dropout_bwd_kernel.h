#pragma once

#include <type_traits>
#include "kernel_operator.h"
#include "kernel_common.h"
#include "masked_softmax_attn_dropout_bwd_tiling.h"
#include "vector_tile.h"

template <typename dataType>
class MaskedSoftmaxAttnDropoutBwdKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR gradOutput,
        GM_ADDR pAttn,
        GM_ADDR mask,
        GM_ADDR dropoutMask,
        GM_ADDR out,
        GM_ADDR tilingGM,
        AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);
        int32_t totalElements = tiling_.M * tiling_.T;
        gradOutputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(gradOutput), totalElements);
        pAttnGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(pAttn), totalElements);
        maskGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(mask), totalElements);
        dropoutMaskGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(dropoutMask), totalElements);
        outGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(out), totalElements);

        if ASCEND_IS_AIV {
            pipe_ = pipe;
            pipe_->InitBuffer(gradBuf_, tiling_.T * sizeof(dataType));
            pipe_->InitBuffer(pAttnBuf_, tiling_.T * sizeof(dataType));
            pipe_->InitBuffer(maskBuf_, tiling_.T * sizeof(dataType));
            pipe_->InitBuffer(dropoutBuf_, tiling_.T * sizeof(dataType));
            pipe_->InitBuffer(tmpBuf1_, tiling_.T * sizeof(float));
            pipe_->InitBuffer(tmpBuf2_, tiling_.T * sizeof(float));
            pipe_->InitBuffer(outBuf_, tiling_.T * sizeof(float));
            pipe_->InitBuffer(reduceBuf_, tiling_.T * sizeof(float));
            pipe_->InitBuffer(sumBuf_, 16 * sizeof(float));

            if constexpr (!std::is_same_v<dataType, float>) {
                pipe_->InitBuffer(castGradBuf_, tiling_.T * sizeof(float));
                pipe_->InitBuffer(castPAttnBuf_, tiling_.T * sizeof(float));
                pipe_->InitBuffer(castMaskBuf_, tiling_.T * sizeof(float));
                pipe_->InitBuffer(castDropoutBuf_, tiling_.T * sizeof(float));
                pipe_->InitBuffer(castOutBuf_, tiling_.T * sizeof(dataType));
            }
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int aivIdx = AscendC::GetBlockIdx() * AscendC::GetSubBlockNum() + AscendC::GetSubBlockIdx();
            const int totalAivs = tiling_.usedCoreNum * AscendC::GetSubBlockNum();
            for (int row = aivIdx; row < tiling_.M; row += totalAivs) {
                ProcessRow(row);
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

    __aicore__ inline AscendC::LocalTensor<float> GetFloatTensor(
        AscendC::LocalTensor<dataType> &src,
        AscendC::TBuf<AscendC::TPosition::VECCALC> &castBuf,
        int32_t count)
    {
        if constexpr (std::is_same_v<dataType, float>) {
            return src.template ReinterpretCast<float>();
        } else {
            auto dst = castBuf.Get<float>();
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
            return dst;
        }
    }

    __aicore__ inline void ProcessRow(int32_t row)
    {
        auto gradLocal = gradBuf_.Get<dataType>();
        auto pAttnLocal = pAttnBuf_.Get<dataType>();
        auto maskLocal = maskBuf_.Get<dataType>();

        LoadGmToUb(gradLocal, gradOutputGM_[row * tiling_.T], tiling_.T);
        LoadGmToUb(pAttnLocal, pAttnGM_[row * tiling_.T], tiling_.T);
        LoadGmToUb(maskLocal, maskGM_[row * tiling_.T], tiling_.T);

        auto gradF = GetFloatTensor(gradLocal, castGradBuf_, tiling_.T);
        auto pAttnF = GetFloatTensor(pAttnLocal, castPAttnBuf_, tiling_.T);
        auto maskF = GetFloatTensor(maskLocal, castMaskBuf_, tiling_.T);

        auto gradSoftmaxF = tmpBuf1_.Get<float>();

        if (tiling_.p_dropout > 0.0f) {
            auto dropoutLocal = dropoutBuf_.Get<dataType>();
            LoadGmToUb(dropoutLocal, dropoutMaskGM_[row * tiling_.T], tiling_.T);
            auto dropoutF = GetFloatTensor(dropoutLocal, castDropoutBuf_, tiling_.T);
            AscendC::Mul(gradSoftmaxF, gradF, dropoutF, tiling_.T);
            AscendC::Muls(gradSoftmaxF, gradSoftmaxF, tiling_.scale, tiling_.T);
        } else {
            gradSoftmaxF = gradF;
        }

        auto prodF = tmpBuf2_.Get<float>();
        AscendC::Mul(prodF, pAttnF, gradSoftmaxF, tiling_.T);

        auto sumLocal = sumBuf_.Get<float>();
        auto reduceLocal = reduceBuf_.Get<float>();
        AscendC::ReduceSum<float>(sumLocal, prodF, reduceLocal, tiling_.T);
        float sum_val = sumLocal.GetValue(0);

        auto diffF = prodF;
        AscendC::Adds(diffF, gradSoftmaxF, -sum_val, tiling_.T);

        auto outF = outBuf_.Get<float>();
        AscendC::Mul(outF, pAttnF, diffF, tiling_.T);
        AscendC::Mul(outF, outF, maskF, tiling_.T);

        if constexpr (std::is_same_v<dataType, float>) {
            StoreUbToGm(outGM_[row * tiling_.T], outF, tiling_.T);
        } else {
            auto outLocal = castOutBuf_.Get<dataType>();
            AscendC::Cast(outLocal, outF, OutputRoundMode(), tiling_.T);
            AscendC::PipeBarrier<PIPE_V>();
            StoreUbToGm(outGM_[row * tiling_.T], outLocal, tiling_.T);
        }
    }

    MaskedSoftmaxAttnDropoutBwdTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};

    AscendC::GlobalTensor<dataType> gradOutputGM_;
    AscendC::GlobalTensor<dataType> pAttnGM_;
    AscendC::GlobalTensor<dataType> maskGM_;
    AscendC::GlobalTensor<dataType> dropoutMaskGM_;
    AscendC::GlobalTensor<dataType> outGM_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> gradBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> pAttnBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dropoutBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf2_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> castGradBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castPAttnBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castMaskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castDropoutBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castOutBuf_;
};
