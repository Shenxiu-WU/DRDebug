#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <type_traits>

#include "kernel_operator.h"

#include "kernel_common.h"
#include "matmul_transb_tiling.h"
#include "vector_tile.h"

// Round up `value` to the nearest multiple of `align`.
__aicore__ inline uint32_t RoundUpU32(uint32_t value, uint32_t align)
{
    return ((value + align - 1U) / align) * align;
}

template <typename dataType>
class MatmulTransBKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tilingGM, AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);

        const uint32_t M = static_cast<uint32_t>(tiling_.M);
        const uint32_t N = static_cast<uint32_t>(tiling_.N);
        const uint32_t K = static_cast<uint32_t>(tiling_.K);

        aGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(a), M * K);
        bGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(b), N * K);
        cGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(c), M * N);

        if ASCEND_IS_AIV {
            pipe_ = pipe;

            // Align internal buffers to 32 bytes worth of elements to keep
            // ReduceSum source addresses 32-byte aligned regardless of K/N.
            constexpr uint32_t kBytesPerBlock = 32U;
            kAlignedFp32_ = RoundUpU32(K, kBytesPerBlock / sizeof(float));
            nAlignedFp32_ = RoundUpU32(N, kBytesPerBlock / sizeof(float));
            kAlignedRaw_ = RoundUpU32(K, kBytesPerBlock / sizeof(dataType));
            nAlignedRaw_ = RoundUpU32(N, kBytesPerBlock / sizeof(dataType));

            // Reduce scratch buffer per ReduceSum API:
            //   elementsPerBlock = 32 / sizeof(float) = 8
            //   elementsPerRepeat = 256 / sizeof(float) = 64
            //   firstMaxRepeat = ceil(K / 64)
            //   iter1OutputCount = firstMaxRepeat
            //   iter1AlignEnd = roundUp(iter1OutputCount, 8) * 1
            // Allocate generously: max(K, 64) * sizeof(float) bytes.
            uint32_t reduceTmpElems = (K > 64U) ? K : 64U;
            reduceTmpElems_ = reduceTmpElems;

            // Input dtype buffers (use TBuf so we can keep A loaded across many B
            // loads; B is reloaded each iteration but stays in the same buffer).
            pipe_->InitBuffer(aRawBuf_, kAlignedRaw_ * sizeof(dataType));
            pipe_->InitBuffer(bRawBuf_, kAlignedRaw_ * sizeof(dataType));
            pipe_->InitBuffer(cRawOutBuf_, nAlignedRaw_ * sizeof(dataType));

            if constexpr (!std::is_same_v<dataType, float>) {
                pipe_->InitBuffer(aFp32Buf_, kAlignedFp32_ * sizeof(float));
                pipe_->InitBuffer(bFp32Buf_, kAlignedFp32_ * sizeof(float));
            }
            pipe_->InitBuffer(mulFp32Buf_, kAlignedFp32_ * sizeof(float));
            pipe_->InitBuffer(cAccFp32Buf_, nAlignedFp32_ * sizeof(float));
            pipe_->InitBuffer(reduceTmpBuf_, reduceTmpElems_ * sizeof(float));
            pipe_->InitBuffer(scalarBuf_, 32U);   // 32 bytes scratch for one fp32 scalar
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const uint32_t M = static_cast<uint32_t>(tiling_.M);
            const uint32_t N = static_cast<uint32_t>(tiling_.N);
            const uint32_t K = static_cast<uint32_t>(tiling_.K);

            const int subBlockNum = AscendC::GetSubBlockNum();
            const int subBlockIdx = AscendC::GetSubBlockIdx();
            const int coreIdx = AscendC::GetBlockIdx() / subBlockNum;
            const uint32_t totalRows = M;

            // Distribute rows: each AIC has 2 AIVs (KERNEL_TYPE_MIX_AIC_1_2). We
            // interleave rows across cores AND sub-blocks so that every AIV
            // gets a deterministic stride-(usedCoreNum * subBlockNum) slice.
            const uint32_t rowStride = static_cast<uint32_t>(tiling_.usedCoreNum) * static_cast<uint32_t>(subBlockNum);
            const uint32_t rowStart = static_cast<uint32_t>(coreIdx) * static_cast<uint32_t>(subBlockNum)
                                      + static_cast<uint32_t>(subBlockIdx);

            for (uint32_t mIdx = rowStart; mIdx < totalRows; mIdx += rowStride) {
                ProcessRow(mIdx, M, N, K);
            }
        }
    }

private:
    __aicore__ inline void ProcessRow(uint32_t mIdx, uint32_t M, uint32_t N, uint32_t K)
    {
        // 1. Load A[mIdx, :K] once.
        AscendC::LocalTensor<dataType> aRaw = aRawBuf_.template Get<dataType>();
        LoadGmToUb<dataType>(aRaw, aGM_[mIdx * K], K);
        AscendC::PipeBarrier<PIPE_MTE2>();

        // 2. Promote A row to fp32 once.
        AscendC::LocalTensor<float> aFp32;
        if constexpr (std::is_same_v<dataType, float>) {
            aFp32 = aRaw.template ReinterpretCast<float>();
        } else {
            aFp32 = aFp32Buf_.template Get<float>();
            AscendC::Cast(aFp32, aRaw, AscendC::RoundMode::CAST_NONE, K);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::LocalTensor<float> mulFp32 = mulFp32Buf_.template Get<float>();
        AscendC::LocalTensor<float> cAccFp32 = cAccFp32Buf_.template Get<float>();
        AscendC::LocalTensor<float> reduceTmp = reduceTmpBuf_.template Get<float>();
        AscendC::LocalTensor<float> scalar = scalarBuf_.template Get<float>();

        // 3. For every n, recompute B row and dot product into cAccFp32[n].
        AscendC::LocalTensor<dataType> bRaw = bRawBuf_.template Get<dataType>();
        AscendC::LocalTensor<float> bFp32;
        if constexpr (std::is_same_v<dataType, float>) {
            bFp32 = bRaw.template ReinterpretCast<float>();
        } else {
            bFp32 = bFp32Buf_.template Get<float>();
        }

        for (uint32_t n = 0; n < N; ++n) {
            LoadGmToUb<dataType>(bRaw, bGM_[n * K], K);
            AscendC::PipeBarrier<PIPE_MTE2>();

            if constexpr (!std::is_same_v<dataType, float>) {
                AscendC::Cast(bFp32, bRaw, AscendC::RoundMode::CAST_NONE, K);
                AscendC::PipeBarrier<PIPE_V>();
            }

            AscendC::Mul(mulFp32, aFp32, bFp32, K);
            AscendC::PipeBarrier<PIPE_V>();

            // Reduce K elements -> 1 fp32 scalar in scalar[0].
            AscendC::ReduceSum<float>(scalar, mulFp32, reduceTmp, static_cast<int32_t>(K));
            AscendC::PipeBarrier<PIPE_V>();

            float dotVal = scalar.GetValue(0);
            AscendC::PipeBarrier<PIPE_ALL>();
            cAccFp32.SetValue(n, dotVal);
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        // 4. Cast C row back to dtype and store to GM.
        AscendC::LocalTensor<dataType> cOut = cRawOutBuf_.template Get<dataType>();
        if constexpr (std::is_same_v<dataType, float>) {
            AscendC::DataCopy(cOut, cAccFp32.template ReinterpretCast<dataType>(), nAlignedFp32_);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::RoundMode rm = std::is_same_v<dataType, bfloat16_t>
                                        ? AscendC::RoundMode::CAST_RINT
                                        : AscendC::RoundMode::CAST_NONE;
            AscendC::Cast(cOut, cAccFp32, rm, N);
            AscendC::PipeBarrier<PIPE_V>();
        }

        StoreUbToGm<dataType>(cGM_[mIdx * N], cOut, N);
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

private:
    MatmulTransBKernelTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};

    AscendC::GlobalTensor<dataType> aGM_;
    AscendC::GlobalTensor<dataType> bGM_;
    AscendC::GlobalTensor<dataType> cGM_;

    // Reusable UB scratch for one row of A / one row of B / accumulator etc.
    AscendC::TBuf<AscendC::TPosition::VECCALC> aRawBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bRawBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aFp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bFp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mulFp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cAccFp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cRawOutBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceTmpBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scalarBuf_;

    uint32_t kAlignedFp32_{0};
    uint32_t nAlignedFp32_{0};
    uint32_t kAlignedRaw_{0};
    uint32_t nAlignedRaw_{0};
    uint32_t reduceTmpElems_{0};
};
