#pragma once

#include <type_traits>
#include "kernel_operator.h"
#include "kernel_common.h"
#include "fused_rope_qk_norm_kv_cache_tiling.h"
#include "vector_tile.h"

template <typename dataType>
class FusedRopeQkNormKvCacheKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR query, GM_ADDR key, GM_ADDR value,
        GM_ADDR positionIds, GM_ADDR cachePosition,
        GM_ADDR qNormWeight, GM_ADDR kNormWeight, GM_ADDR invFreq,
        GM_ADDR queryRotated, GM_ADDR keyRotated,
        GM_ADDR keyCacheOut, GM_ADDR valueCacheOut,
        GM_ADDR tilingGM, AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);
        int64_t qSize = static_cast<int64_t>(tiling_.B) * tiling_.Hq * tiling_.S * tiling_.D;
        int64_t kSize = static_cast<int64_t>(tiling_.B) * tiling_.Hkv * tiling_.S * tiling_.D;
        int64_t vSize = kSize;
        int64_t cacheSize = static_cast<int64_t>(tiling_.B) * tiling_.Hkv * tiling_.maxS * tiling_.D;
        int64_t posIdsSize = static_cast<int64_t>(tiling_.B) * tiling_.S;
        int64_t cachePosSize = tiling_.S;
        int64_t halfD = tiling_.D / 2;

        queryGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(query), qSize);
        keyGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(key), kSize);
        valueGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(value), vSize);
        positionIdsGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(positionIds), posIdsSize);
        cachePositionGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(cachePosition), cachePosSize);
        qNormWeightGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(qNormWeight), tiling_.D);
        kNormWeightGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(kNormWeight), tiling_.D);
        invFreqGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(invFreq), halfD);
        queryRotatedGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(queryRotated), qSize);
        keyRotatedGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(keyRotated), kSize);
        keyCacheOutGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(keyCacheOut), cacheSize);
        valueCacheOutGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(valueCacheOut), cacheSize);

        pipe_ = pipe;
        subBlockRows_ = tiling_.blockM / AscendC::GetSubBlockNum();
        halfD_ = halfD;

        if ASCEND_IS_AIV {
            pipe_->InitBuffer(rowInQueue_, 1, tiling_.D * sizeof(dataType));
            pipe_->InitBuffer(rowOutQueue_, 1, tiling_.D * sizeof(dataType));
            pipe_->InitBuffer(invFreqBuf_, halfD * sizeof(float));
            pipe_->InitBuffer(qNormWeightBuf_, tiling_.D * sizeof(dataType));
            pipe_->InitBuffer(kNormWeightBuf_, tiling_.D * sizeof(dataType));
            pipe_->InitBuffer(reduceBuf_, tiling_.D * sizeof(float));
            pipe_->InitBuffer(sumBuf_, 16 * sizeof(float));
            pipe_->InitBuffer(xFp32Buf_, tiling_.D * sizeof(float));
            pipe_->InitBuffer(tempBuf_, halfD * sizeof(float));
            pipe_->InitBuffer(x2Buf_, halfD * sizeof(float));
            pipe_->InitBuffer(outFp32Buf_, tiling_.D * sizeof(float));
            pipe_->InitBuffer(anglesBuf_, halfD * sizeof(float));
            pipe_->InitBuffer(cosBuf_, halfD * sizeof(float));
            pipe_->InitBuffer(sinBuf_, halfD * sizeof(float));
            pipe_->InitBuffer(posIdBuf_, sizeof(int64_t));
            pipe_->InitBuffer(cachePosBuf_, sizeof(int64_t));
            if constexpr (!std::is_same_v<dataType, float>) {
                pipe_->InitBuffer(castBuf_, tiling_.D * sizeof(float));
            }

            invFreqLocal_ = invFreqBuf_.Get<float>();
            LoadGmToUb(invFreqLocal_, invFreqGM_, static_cast<uint32_t>(halfD));
            AscendC::PipeBarrier<PIPE_MTE2>();

            auto qNormWeightInLocal = qNormWeightBuf_.Get<dataType>();
            LoadGmToUb(qNormWeightInLocal, qNormWeightGM_, static_cast<uint32_t>(tiling_.D));
            AscendC::PipeBarrier<PIPE_MTE2>();
            PrepareInputTensor(qNormWeightLocal_, qNormWeightInLocal, tiling_.D);

            auto kNormWeightInLocal = kNormWeightBuf_.Get<dataType>();
            LoadGmToUb(kNormWeightInLocal, kNormWeightGM_, static_cast<uint32_t>(tiling_.D));
            AscendC::PipeBarrier<PIPE_MTE2>();
            PrepareInputTensor(kNormWeightLocal_, kNormWeightInLocal, tiling_.D);

            posIdLocal_ = posIdBuf_.Get<int64_t>();
            cachePosLocal_ = cachePosBuf_.Get<int64_t>();
            reduceLocal_ = reduceBuf_.Get<float>();
            sumLocal_ = sumBuf_.Get<float>();
            xFp32Local_ = xFp32Buf_.Get<float>();
            tempLocal_ = tempBuf_.Get<float>();
            x2Local_ = x2Buf_.Get<float>();
            outFp32Local_ = outFp32Buf_.Get<float>();
            anglesLocal_ = anglesBuf_.Get<float>();
            cosLocal_ = cosBuf_.Get<float>();
            sinLocal_ = sinBuf_.Get<float>();
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
                    if (rowIdx >= tiling_.totalRows) {
                        continue;
                    }
                    if (rowIdx < tiling_.Mq) {
                        ProcessQRow(rowIdx);
                    } else if (rowIdx < tiling_.Mq + tiling_.Mk) {
                        ProcessKRow(rowIdx);
                    } else {
                        ProcessVRow(rowIdx);
                    }
                }
            }
        }
    }

private:
    __aicore__ inline int32_t BlockCount() const
    {
        return (tiling_.totalRows + tiling_.blockM - 1) / tiling_.blockM;
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
        int32_t count)
    {
        if constexpr (std::is_same_v<dataType, float>) {
            dst = src.template ReinterpretCast<float>();
        } else {
            dst = castBuf_.Get<float>();
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
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

    __aicore__ inline void CopyInRow(AscendC::GlobalTensor<dataType> &gm, int64_t offset)
    {
        rowInQueue_.AllocTensor<dataType>(rowInLocal_);
        LoadGmToUb(rowInLocal_, gm[offset], static_cast<uint32_t>(tiling_.D));
        rowInQueue_.EnQue(rowInLocal_);
    }

    __aicore__ inline void CopyOutRow(AscendC::GlobalTensor<dataType> &gm, int64_t offset)
    {
        rowOutQueue_.DeQue<dataType>(rowOutLocal_);
        StoreUbToGm(gm[offset], rowOutLocal_, static_cast<uint32_t>(tiling_.D));
        rowOutQueue_.FreeTensor(rowOutLocal_);
    }

    __aicore__ inline void CopyOutKRow(int64_t kOffset, int64_t cacheOffset)
    {
        rowOutQueue_.DeQue<dataType>(rowOutLocal_);
        StoreUbToGm(keyRotatedGM_[kOffset], rowOutLocal_, static_cast<uint32_t>(tiling_.D));
        StoreUbToGm(keyCacheOutGM_[cacheOffset], rowOutLocal_, static_cast<uint32_t>(tiling_.D));
        rowOutQueue_.FreeTensor(rowOutLocal_);
    }

    __aicore__ inline void ComputeRmsNorm(AscendC::LocalTensor<float> &weightLocal)
    {
        rowInQueue_.DeQue<dataType>(rowInLocal_);
        PrepareInputTensor(xFp32Local_, rowInLocal_, tiling_.D);
        rowOutQueue_.AllocTensor<dataType>(rowOutLocal_);

        AscendC::Mul(outFp32Local_, xFp32Local_, xFp32Local_, tiling_.D);
        AscendC::ReduceSum<float>(sumLocal_, outFp32Local_, reduceLocal_, tiling_.D);

        float meanSq = sumLocal_.GetValue(0) * tiling_.invD + tiling_.eps;
        AscendC::Duplicate(sumLocal_, meanSq, 1);
        AscendC::Rsqrt(sumLocal_, sumLocal_, 1);
        AscendC::PipeBarrier<PIPE_ALL>();
        float invRms = sumLocal_.GetValue(0);
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::Muls(outFp32Local_, xFp32Local_, invRms, tiling_.D);
        AscendC::Mul(outFp32Local_, outFp32Local_, weightLocal, tiling_.D);
    }

    __aicore__ inline void ComputeRope(int64_t b, int64_t s)
    {
        // Save normalized x1 and x2 before computing angles
        AscendC::DataCopy(tempLocal_, outFp32Local_, static_cast<uint32_t>(halfD_));
        AscendC::DataCopy(x2Local_, outFp32Local_[halfD_], static_cast<uint32_t>(halfD_));

        int64_t posIdOffset = b * tiling_.S + s;
        LoadGmToUb(posIdLocal_, positionIdsGM_[posIdOffset], 1);
        int64_t posId = posIdLocal_.GetValue(0);
        float posIdFloat = static_cast<float>(posId);

        AscendC::Muls(anglesLocal_, invFreqLocal_, posIdFloat, static_cast<uint32_t>(halfD_));
        AscendC::Cos(cosLocal_, anglesLocal_, static_cast<uint32_t>(halfD_));
        AscendC::Sin(sinLocal_, anglesLocal_, static_cast<uint32_t>(halfD_));

        // First half output = x1 * cos - x2 * sin
        AscendC::Mul(outFp32Local_, tempLocal_, cosLocal_, static_cast<uint32_t>(halfD_));
        AscendC::Mul(xFp32Local_, x2Local_, sinLocal_, static_cast<uint32_t>(halfD_));
        AscendC::Sub(outFp32Local_, outFp32Local_, xFp32Local_, static_cast<uint32_t>(halfD_));

        // Second half output = x2 * cos + x1 * sin
        AscendC::Mul(outFp32Local_[halfD_], x2Local_, cosLocal_, static_cast<uint32_t>(halfD_));
        AscendC::Mul(xFp32Local_, tempLocal_, sinLocal_, static_cast<uint32_t>(halfD_));
        AscendC::Add(outFp32Local_[halfD_], outFp32Local_[halfD_], xFp32Local_, static_cast<uint32_t>(halfD_));
    }

    __aicore__ inline void ProcessQRow(int64_t rowIdx)
    {
        int64_t b = rowIdx / (tiling_.Hq * tiling_.S);
        int64_t rem = rowIdx % (tiling_.Hq * tiling_.S);
        int64_t h = rem / tiling_.S;
        int64_t s = rem % tiling_.S;
        int64_t qOffset = ((b * tiling_.Hq + h) * tiling_.S + s) * tiling_.D;

        CopyInRow(queryGM_, qOffset);
        ComputeRmsNorm(qNormWeightLocal_);
        ComputeRope(b, s);
        FinalizeOutputTensor(rowOutLocal_, outFp32Local_, tiling_.D);
        rowInQueue_.FreeTensor(rowInLocal_);
        rowOutQueue_.EnQue(rowOutLocal_);
        CopyOutRow(queryRotatedGM_, qOffset);
    }

    __aicore__ inline void ProcessKRow(int64_t rowIdx)
    {
        int64_t kRow = rowIdx - tiling_.Mq;
        int64_t b = kRow / (tiling_.Hkv * tiling_.S);
        int64_t rem = kRow % (tiling_.Hkv * tiling_.S);
        int64_t h = rem / tiling_.S;
        int64_t s = rem % tiling_.S;
        int64_t kOffset = ((b * tiling_.Hkv + h) * tiling_.S + s) * tiling_.D;

        CopyInRow(keyGM_, kOffset);
        ComputeRmsNorm(kNormWeightLocal_);
        ComputeRope(b, s);
        FinalizeOutputTensor(rowOutLocal_, outFp32Local_, tiling_.D);
        rowInQueue_.FreeTensor(rowInLocal_);
        rowOutQueue_.EnQue(rowOutLocal_);

        LoadGmToUb(cachePosLocal_, cachePositionGM_[s], 1);
        int64_t cachePos = cachePosLocal_.GetValue(0);
        int64_t cacheOffset = ((b * tiling_.Hkv + h) * tiling_.maxS + cachePos) * tiling_.D;
        CopyOutKRow(kOffset, cacheOffset);
    }

    __aicore__ inline void ProcessVRow(int64_t rowIdx)
    {
        int64_t vRow = rowIdx - tiling_.Mq - tiling_.Mk;
        int64_t b = vRow / (tiling_.Hkv * tiling_.S);
        int64_t rem = vRow % (tiling_.Hkv * tiling_.S);
        int64_t h = rem / tiling_.S;
        int64_t s = rem % tiling_.S;
        int64_t vOffset = ((b * tiling_.Hkv + h) * tiling_.S + s) * tiling_.D;

        CopyInRow(valueGM_, vOffset);
        rowInQueue_.DeQue<dataType>(rowInLocal_);
        rowOutQueue_.AllocTensor<dataType>(rowOutLocal_);
        AscendC::DataCopy(rowOutLocal_, rowInLocal_, static_cast<uint32_t>(tiling_.D));
        rowInQueue_.FreeTensor(rowInLocal_);
        rowOutQueue_.EnQue(rowOutLocal_);

        LoadGmToUb(cachePosLocal_, cachePositionGM_[s], 1);
        int64_t cachePos = cachePosLocal_.GetValue(0);
        int64_t cacheOffset = ((b * tiling_.Hkv + h) * tiling_.maxS + cachePos) * tiling_.D;
        CopyOutRow(valueCacheOutGM_, cacheOffset);
    }

private:
    FusedRopeQkNormKvCacheTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};
    int subBlockRows_{0};
    int64_t halfD_{0};

    AscendC::GlobalTensor<dataType> queryGM_;
    AscendC::GlobalTensor<dataType> keyGM_;
    AscendC::GlobalTensor<dataType> valueGM_;
    AscendC::GlobalTensor<int64_t> positionIdsGM_;
    AscendC::GlobalTensor<int64_t> cachePositionGM_;
    AscendC::GlobalTensor<dataType> qNormWeightGM_;
    AscendC::GlobalTensor<dataType> kNormWeightGM_;
    AscendC::GlobalTensor<float> invFreqGM_;
    AscendC::GlobalTensor<dataType> queryRotatedGM_;
    AscendC::GlobalTensor<dataType> keyRotatedGM_;
    AscendC::GlobalTensor<dataType> keyCacheOutGM_;
    AscendC::GlobalTensor<dataType> valueCacheOutGM_;

    AscendC::TQue<AscendC::TPosition::VECIN, 0> rowInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> rowOutQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> invFreqBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> qNormWeightBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> kNormWeightBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xFp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tempBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x2Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outFp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> anglesBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cosBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sinBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> posIdBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cachePosBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castBuf_;

    AscendC::LocalTensor<dataType> rowInLocal_;
    AscendC::LocalTensor<dataType> rowOutLocal_;
    AscendC::LocalTensor<float> invFreqLocal_;
    AscendC::LocalTensor<float> qNormWeightLocal_;
    AscendC::LocalTensor<float> kNormWeightLocal_;
    AscendC::LocalTensor<float> reduceLocal_;
    AscendC::LocalTensor<float> sumLocal_;
    AscendC::LocalTensor<float> xFp32Local_;
    AscendC::LocalTensor<float> tempLocal_;
    AscendC::LocalTensor<float> x2Local_;
    AscendC::LocalTensor<float> outFp32Local_;
    AscendC::LocalTensor<float> anglesLocal_;
    AscendC::LocalTensor<float> cosLocal_;
    AscendC::LocalTensor<float> sinLocal_;
    AscendC::LocalTensor<int64_t> posIdLocal_;
    AscendC::LocalTensor<int64_t> cachePosLocal_;
};
