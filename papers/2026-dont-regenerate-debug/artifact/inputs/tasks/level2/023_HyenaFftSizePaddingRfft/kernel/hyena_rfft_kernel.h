#pragma once

#include <kernel_operator.h>
#include <math.h>
#include <stdint.h>
#include <type_traits>

#include "hyena_rfft_tiling.h"

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

__aicore__ inline void CopyTiling(HyenaRfftTiling* tiling, GM_ADDR tilingGM) {
    int32_t* dst = reinterpret_cast<int32_t*>(tiling);
    auto* src = reinterpret_cast<__gm__ int32_t*>(tilingGM);
    for (size_t i = 0; i < sizeof(HyenaRfftTiling) / sizeof(int32_t); ++i) {
        dst[i] = src[i];
    }
}

template <typename T>
class HyenaRfftKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR real, GM_ADDR imag, GM_ADDR tilingGM) {
        CopyTiling(&tiling_, tilingGM);
        int64_t xSize = (int64_t)tiling_.B * tiling_.C * tiling_.L;
        int64_t outSize = (int64_t)tiling_.B * tiling_.C * (tiling_.L + 1);
        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), xSize);
        realGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(real), outSize);
        imagGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(imag), outSize);

        int32_t maxL = 4096;

        pipe_.InitBuffer(xBuf_, maxL * sizeof(float));
        if constexpr (!std::is_same_v<T, float>) {
            pipe_.InitBuffer(rawBuf_, maxL * sizeof(T));
        }
        pipe_.InitBuffer(indexBuf_, (maxL + 1) * sizeof(float));
        pipe_.InitBuffer(anglesBuf_, (maxL + 1) * sizeof(float));
        pipe_.InitBuffer(sinBuf_, (maxL + 1) * sizeof(float));
        pipe_.InitBuffer(cosBuf_, (maxL + 1) * sizeof(float));
        pipe_.InitBuffer(tmpBuf_, (maxL + 1) * sizeof(float));
        pipe_.InitBuffer(realOutBuf_, (maxL + 1) * sizeof(float));
        pipe_.InitBuffer(imagOutBuf_, (maxL + 1) * sizeof(float));
    }

    __aicore__ inline void Process() {
        int32_t m = AscendC::GetBlockIdx();
        if (m >= tiling_.totalSeq) {
            return;
        }
        int32_t b = m / tiling_.C;
        int32_t c = m % tiling_.C;

        CopyIn(b, c);
        ComputeDft();
        CopyOut(b, c);
    }

private:
    __aicore__ inline void CopyIn(int32_t b, int32_t c) {
        int32_t L = tiling_.L;
        int64_t offset = ((int64_t)b * tiling_.C + c) * L;

        if constexpr (std::is_same_v<T, float>) {
            auto xLocal = xBuf_.Get<float>();
            AscendC::DataCopy(xLocal, xGM_[offset], L);
            AscendC::PipeBarrier<PIPE_MTE2>();
        } else {
            auto rawLocal = rawBuf_.Get<T>();
            AscendC::DataCopy(rawLocal, xGM_[offset], L);
            AscendC::PipeBarrier<PIPE_MTE2>();
            auto xLocal = xBuf_.Get<float>();
            AscendC::Cast(xLocal, rawLocal, AscendC::RoundMode::CAST_NONE, L);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ComputeDft() {
        int32_t L = tiling_.L;
        int32_t N = tiling_.N;
        float norm = 1.0f / (float)N;
        float two_pi_over_N = 2.0f * M_PI_F / (float)N;

        auto xLocal = xBuf_.Get<float>();
        auto indexLocal = indexBuf_.Get<float>();
        auto anglesLocal = anglesBuf_.Get<float>();
        auto sinLocal = sinBuf_.Get<float>();
        auto cosLocal = cosBuf_.Get<float>();
        auto tmpLocal = tmpBuf_.Get<float>();
        auto realOutLocal = realOutBuf_.Get<float>();
        auto imagOutLocal = imagOutBuf_.Get<float>();

        AscendC::Duplicate(realOutLocal, 0.0f, L + 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(imagOutLocal, 0.0f, L + 1);
        AscendC::PipeBarrier<PIPE_V>();

        for (int32_t n = 0; n < L; ++n) {
            float xn = xLocal.GetValue(n);
            float baseAngle = two_pi_over_N * (float)n;

            AscendC::CreateVecIndex(indexLocal, 0.0f, L + 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(anglesLocal, indexLocal, baseAngle, L + 1);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Sin(sinLocal, anglesLocal, L + 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cos(cosLocal, anglesLocal, L + 1);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Muls(tmpLocal, cosLocal, xn, L + 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(realOutLocal, realOutLocal, tmpLocal, L + 1);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Muls(tmpLocal, sinLocal, xn, L + 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(imagOutLocal, imagOutLocal, tmpLocal, L + 1);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::Muls(realOutLocal, realOutLocal, norm, L + 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(imagOutLocal, imagOutLocal, norm, L + 1);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyOut(int32_t b, int32_t c) {
        int32_t L = tiling_.L;
        int64_t offset = ((int64_t)b * tiling_.C + c) * (L + 1);

        auto realOutLocal = realOutBuf_.Get<float>();
        auto imagOutLocal = imagOutBuf_.Get<float>();

        AscendC::DataCopy(realGM_[offset], realOutLocal, L + 1);
        AscendC::PipeBarrier<PIPE_MTE2>();
        AscendC::DataCopy(imagGM_[offset], imagOutLocal, L + 1);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    HyenaRfftTiling tiling_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rawBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> anglesBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sinBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cosBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> realOutBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> imagOutBuf_;
    AscendC::GlobalTensor<T> xGM_;
    AscendC::GlobalTensor<float> realGM_;
    AscendC::GlobalTensor<float> imagGM_;
};
