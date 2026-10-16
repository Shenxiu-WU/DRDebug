#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_common.h"
#include "embedding_dense_backward_tiling.h"

template <typename dataType>
class EmbeddingDenseBackwardKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR gradOutput,
        GM_ADDR indices,
        GM_ADDR gradWeight,
        GM_ADDR tilingGM,
        AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);
        gradOutputGM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ dataType *>(gradOutput),
            tiling_.totalIndices * tiling_.D);
        indicesGM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(indices),
            tiling_.totalIndices);
        gradWeightGM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ dataType *>(gradWeight),
            tiling_.numWeights * tiling_.D);
        pipe_ = pipe;

        pipe_->InitBuffer(gwBuf_, tiling_.blockWeights * tiling_.blockD * sizeof(float));
        pipe_->InitBuffer(gradRowBuf_, tiling_.blockD * sizeof(dataType));
        pipe_->InitBuffer(idxBuf_, tiling_.blockI * sizeof(int32_t));
        pipe_->InitBuffer(freqBuf_, tiling_.blockWeights * sizeof(int32_t));
        if constexpr (!std::is_same_v<dataType, float>) {
            pipe_->InitBuffer(gradRowFloatBuf_, tiling_.blockD * sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int coreIdx = AscendC::GetBlockIdx();
            for (int localIdx = 0; localIdx < tiling_.tasksPerCore; ++localIdx) {
                int blockId = coreIdx * tiling_.tasksPerCore + localIdx;
                if (blockId >= tiling_.totalBlocks) {
                    continue;
                }

                int bm = blockId / tiling_.dNum;
                int bd = blockId % tiling_.dNum;

                int wStart = bm * tiling_.blockWeights;
                int wEnd = wStart + static_cast<int>(tiling_.blockWeights);
                if (wEnd > tiling_.numWeights) {
                    wEnd = tiling_.numWeights;
                }
                int dStart = bd * tiling_.blockD;
                int dEnd = dStart + static_cast<int>(tiling_.blockD);
                if (dEnd > tiling_.D) {
                    dEnd = tiling_.D;
                }

                int wSize = wEnd - wStart;
                int dSize = dEnd - dStart;

                auto gwLocal = gwBuf_.Get<float>();
                AscendC::Duplicate(gwLocal, 0.0f, wSize * tiling_.blockD);

                auto freqLocal = freqBuf_.Get<int32_t>();
                if (tiling_.scaleGradByFreq) {
                    AscendC::Duplicate(freqLocal, 0, wSize);
                    CountFrequency(wStart, wEnd, freqLocal);
                }

                AccumulateGradOutput(wStart, wEnd, dStart, dSize, gwLocal, freqLocal);
                WriteBack(wStart, wSize, dStart, dSize, gwLocal);
            }
        }
    }

private:
    __aicore__ inline void CountFrequency(
        int wStart, int wEnd,
        AscendC::LocalTensor<int32_t> freqLocal)
    {
        int indicesTileNum = (tiling_.totalIndices + tiling_.blockI - 1) / tiling_.blockI;
        for (int it = 0; it < indicesTileNum; ++it) {
            int iStart = it * tiling_.blockI;
            int iEnd = iStart + static_cast<int>(tiling_.blockI);
            if (iEnd > tiling_.totalIndices) {
                iEnd = tiling_.totalIndices;
            }
            int iSize = iEnd - iStart;

            auto idxLocal = idxBuf_.Get<int32_t>();
            LoadIndices(idxLocal, indicesGM_[iStart], iSize);
            AscendC::PipeBarrier<PIPE_MTE2>();

            for (int ti = 0; ti < iSize; ++ti) {
                int32_t idx = idxLocal.GetValue(ti);
                if (tiling_.paddingIdx >= 0 && idx == tiling_.paddingIdx) {
                    continue;
                }
                if (idx >= wStart && idx < wEnd) {
                    int f = freqLocal.GetValue(idx - wStart) + 1;
                    freqLocal.SetValue(idx - wStart, f);
                }
            }
        }
    }

    __aicore__ inline void AccumulateGradOutput(
        int wStart, int wEnd, int dStart, int dSize,
        AscendC::LocalTensor<float> gwLocal,
        AscendC::LocalTensor<int32_t> freqLocal)
    {
        int indicesTileNum = (tiling_.totalIndices + tiling_.blockI - 1) / tiling_.blockI;
        for (int it = 0; it < indicesTileNum; ++it) {
            int iStart = it * tiling_.blockI;
            int iEnd = iStart + static_cast<int>(tiling_.blockI);
            if (iEnd > tiling_.totalIndices) {
                iEnd = tiling_.totalIndices;
            }
            int iSize = iEnd - iStart;

            auto idxLocal = idxBuf_.Get<int32_t>();
            LoadIndices(idxLocal, indicesGM_[iStart], iSize);
            AscendC::PipeBarrier<PIPE_MTE2>();

            for (int ti = 0; ti < iSize; ++ti) {
                int32_t idx = idxLocal.GetValue(ti);
                if (tiling_.paddingIdx >= 0 && idx == tiling_.paddingIdx) {
                    continue;
                }
                if (idx >= wStart && idx < wEnd) {
                    int rowOffset = (idx - wStart) * tiling_.blockD;
                    auto gradRowLocal = gradRowBuf_.Get<dataType>();
                    LoadGradOutput(gradRowLocal, iStart + ti, dStart, dSize);
                    AscendC::PipeBarrier<PIPE_MTE2>();

                    if constexpr (std::is_same_v<dataType, float>) {
                        if (tiling_.scaleGradByFreq) {
                            float freq = static_cast<float>(freqLocal.GetValue(idx - wStart));
                            if (freq > 0.0f) {
                                float scale = 1.0f / freq;
                                AscendC::Muls(gradRowLocal, gradRowLocal, scale, dSize);
                            }
                        }
                        for (int d = 0; d < dSize; ++d) {
                            float val = gwLocal.GetValue(rowOffset + d) + gradRowLocal.GetValue(d);
                            gwLocal.SetValue(rowOffset + d, val);
                        }
                    } else {
                        auto gradRowFloat = gradRowFloatBuf_.Get<float>();
                        AscendC::Cast(gradRowFloat, gradRowLocal, AscendC::RoundMode::CAST_NONE, dSize);
                        AscendC::PipeBarrier<PIPE_V>();

                        if (tiling_.scaleGradByFreq) {
                            float freq = static_cast<float>(freqLocal.GetValue(idx - wStart));
                            if (freq > 0.0f) {
                                float scale = 1.0f / freq;
                                AscendC::Muls(gradRowFloat, gradRowFloat, scale, dSize);
                            }
                        }
                        for (int d = 0; d < dSize; ++d) {
                            float val = gwLocal.GetValue(rowOffset + d) + gradRowFloat.GetValue(d);
                            gwLocal.SetValue(rowOffset + d, val);
                        }
                    }
                }
            }
        }
    }

    __aicore__ inline void WriteBack(
        int wStart, int wSize, int dStart, int dSize,
        AscendC::LocalTensor<float> gwLocal)
    {
        if constexpr (std::is_same_v<dataType, float>) {
            for (int w = 0; w < wSize; ++w) {
                int rowOffset = w * tiling_.blockD;
                StoreGradOutput(wStart + w, dStart, gwLocal[rowOffset], dSize);
            }
        } else {
            auto outLocal = gradRowBuf_.Get<dataType>();
            AscendC::RoundMode roundMode = AscendC::RoundMode::CAST_NONE;
            if constexpr (std::is_same_v<dataType, bfloat16_t>) {
                roundMode = AscendC::RoundMode::CAST_ROUND;
            }
            for (int w = 0; w < wSize; ++w) {
                int rowOffset = w * tiling_.blockD;
                auto gwSlice = gwLocal[rowOffset];
                AscendC::Cast(outLocal, gwSlice, roundMode, dSize);
                AscendC::PipeBarrier<PIPE_V>();
                StoreGradOutput(wStart + w, dStart, outLocal, dSize);
            }
        }
    }

    __aicore__ inline void LoadIndices(
        AscendC::LocalTensor<int32_t> &dst,
        AscendC::GlobalTensor<int32_t> src,
        int count)
    {
        AscendC::DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
        AscendC::DataCopyPad(dst, src, copyParams, padParams);
    }

    __aicore__ inline void LoadGradOutput(
        AscendC::LocalTensor<dataType> &dst,
        int row, int colStart, int count)
    {
        AscendC::DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(count * sizeof(dataType)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<dataType> padParams{true, 0, 0, static_cast<dataType>(0)};
        AscendC::DataCopyPad(dst, gradOutputGM_[row * tiling_.D + colStart], copyParams, padParams);
    }

    __aicore__ inline void StoreGradOutput(
        int row, int colStart,
        AscendC::LocalTensor<dataType> src,
        int count)
    {
        AscendC::DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(count * sizeof(dataType)), 0, 0, 0};
        AscendC::DataCopyPad(gradWeightGM_[row * tiling_.D + colStart], src, copyParams);
    }

private:
    EmbeddingDenseBackwardTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};

    AscendC::GlobalTensor<dataType> gradOutputGM_;
    AscendC::GlobalTensor<int32_t> indicesGM_;
    AscendC::GlobalTensor<dataType> gradWeightGM_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> gwBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gradRowBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> idxBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> freqBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gradRowFloatBuf_;
};
