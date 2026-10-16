#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"

#include "kernel_common.h"
#include "conv1d_tiling.h"

constexpr int32_t CONV1D_K_VEC_LEN_ALIGN = 8;       // float -> 8 elements per 32B DataBlock
constexpr int32_t CONV1D_MAX_LOCAL_RED_LEN = 256;   // upper bound used for prod/work tensors

class Conv1DKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
                                GM_ADDR tilingGM, AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);

        const uint64_t xElems = static_cast<uint64_t>(tiling_.N) *
                                static_cast<uint64_t>(tiling_.lIn) *
                                static_cast<uint64_t>(tiling_.cIn);
        const uint64_t wElems = static_cast<uint64_t>(tiling_.cOut) *
                                static_cast<uint64_t>(tiling_.kSize) *
                                static_cast<uint64_t>(tiling_.cInPerGroup);
        const uint64_t bElems = static_cast<uint64_t>(tiling_.cOut);
        const uint64_t yElems = static_cast<uint64_t>(tiling_.N) *
                                static_cast<uint64_t>(tiling_.lOut) *
                                static_cast<uint64_t>(tiling_.cOut);

        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x), xElems);
        wGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weight), wElems);
        biasGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(bias), bElems);
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y), yElems);

        if ASCEND_IS_AIV {
            pipe_ = pipe;

            // Allocate UB buffers. xVecsLen counts K * cIn floats packed.
            xVecsLen_ = tiling_.kSize * tiling_.cIn;
            accLen_ = tiling_.cOut;
            wTileLen_ = tiling_.cInPerGroup;

            // Round buffer sizes up to a 32B (8-float) boundary because
            // AscendC vector ops expect aligned UB tensors.
            const uint32_t xVecsBytes = AlignUp32B(static_cast<uint32_t>(xVecsLen_));
            const uint32_t accBytes = AlignUp32B(static_cast<uint32_t>(accLen_));
            const uint32_t wTileBytes = AlignUp32B(static_cast<uint32_t>(wTileLen_));
            const uint32_t prodBytes = wTileBytes;
            const uint32_t redBytes = AlignUp32B(static_cast<uint32_t>(CONV1D_K_VEC_LEN_ALIGN));

            pipe_->InitBuffer(xVecsBuf_, xVecsBytes);
            pipe_->InitBuffer(accBuf_, accBytes);
            pipe_->InitBuffer(wTileBuf_, wTileBytes);
            pipe_->InitBuffer(prodBuf_, prodBytes);
            pipe_->InitBuffer(redBuf_, redBytes);
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int workerId = static_cast<int>(AscendC::GetBlockIdx());
            const int totalBlocks = tiling_.totalBlocks;
            int workers = tiling_.usedCoreNum;
            if (workers <= 0) {
                workers = 1;
            }

            for (int blkIdx = workerId; blkIdx < totalBlocks; blkIdx += workers) {
                ProcessOneBlock(blkIdx);
            }
        }
    }

private:
    __aicore__ inline static uint32_t AlignUp32B(uint32_t elems)
    {
        // 32B = 8 floats; round up to multiple of 8 floats and convert to bytes.
        const uint32_t aligned = ((elems + 7U) / 8U) * 8U;
        return aligned * sizeof(float);
    }

    __aicore__ inline void LoadXVecs(int nIdx, int lo)
    {
        AscendC::LocalTensor<float> xVecs = xVecsBuf_.Get<float>();
        // Zero-fill once, then overwrite valid kw rows.
        AscendC::Duplicate(xVecs, 0.0f, xVecsLen_);
        AscendC::PipeBarrier<PIPE_V>();

        for (int kw = 0; kw < tiling_.kSize; ++kw) {
            const int lIn = lo * tiling_.stride - tiling_.padding + kw * tiling_.dilation;
            if (lIn < 0 || lIn >= tiling_.lIn) {
                continue;
            }
            const uint64_t srcOffset = (static_cast<uint64_t>(nIdx) * tiling_.lIn +
                                        static_cast<uint64_t>(lIn)) *
                                       static_cast<uint64_t>(tiling_.cIn);
            AscendC::DataCopy(
                xVecs[kw * tiling_.cIn],
                xGM_[srcOffset],
                static_cast<uint32_t>(AlignedCopyLen(tiling_.cIn))
            );
        }
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void InitAcc()
    {
        AscendC::LocalTensor<float> acc = accBuf_.Get<float>();
        if (tiling_.hasBias != 0) {
            AscendC::DataCopy(
                acc,
                biasGM_,
                static_cast<uint32_t>(AlignedCopyLen(tiling_.cOut))
            );
            AscendC::PipeBarrier<PIPE_MTE2>();
        } else {
            AscendC::Duplicate(acc, 0.0f, accLen_);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void StoreAcc(int nIdx, int lo)
    {
        AscendC::LocalTensor<float> acc = accBuf_.Get<float>();
        const uint64_t dstOffset = (static_cast<uint64_t>(nIdx) * tiling_.lOut +
                                    static_cast<uint64_t>(lo)) *
                                   static_cast<uint64_t>(tiling_.cOut);
        AscendC::DataCopy(
            yGM_[dstOffset],
            acc,
            static_cast<uint32_t>(AlignedCopyLen(tiling_.cOut))
        );
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    __aicore__ inline static uint32_t AlignedCopyLen(int elems)
    {
        // DataCopy uint32_t length is in 32B units for some signatures, but
        // AscendC overload taking a single uint32_t treats it as element count.
        // Round up to 8-float alignment for safety.
        const uint32_t e = static_cast<uint32_t>(elems);
        return ((e + 7U) / 8U) * 8U;
    }

    __aicore__ inline void ProcessOneBlock(int blkIdx)
    {
        const int nIdx = blkIdx / tiling_.lOut;
        const int lo = blkIdx - nIdx * tiling_.lOut;

        LoadXVecs(nIdx, lo);
        InitAcc();

        AscendC::LocalTensor<float> xVecs = xVecsBuf_.Get<float>();
        AscendC::LocalTensor<float> acc = accBuf_.Get<float>();
        AscendC::LocalTensor<float> wTile = wTileBuf_.Get<float>();
        AscendC::LocalTensor<float> prod = prodBuf_.Get<float>();
        AscendC::LocalTensor<float> red = redBuf_.Get<float>();

        const int K = tiling_.kSize;
        const int Cin = tiling_.cIn;
        const int CinPerGroup = tiling_.cInPerGroup;
        const int CoutPerGroup = tiling_.cOutPerGroup;
        const int Cout = tiling_.cOut;
        const uint32_t cipgU = static_cast<uint32_t>(CinPerGroup);

        for (int oc = 0; oc < Cout; ++oc) {
            const int groupIdx = oc / CoutPerGroup;
            const int icStart = groupIdx * CinPerGroup;

            // Read current acc[oc] into a scalar.
            float accScalar = acc.GetValue(oc);

            for (int kw = 0; kw < K; ++kw) {
                const uint64_t wOffset = (static_cast<uint64_t>(oc) * K +
                                          static_cast<uint64_t>(kw)) *
                                         static_cast<uint64_t>(CinPerGroup);
                AscendC::DataCopy(
                    wTile,
                    wGM_[wOffset],
                    static_cast<uint32_t>(AlignedCopyLen(CinPerGroup))
                );
                AscendC::PipeBarrier<PIPE_MTE2>();

                // prod[i] = wTile[i] * xVecs[kw * Cin + icStart + i]
                AscendC::Mul(
                    prod,
                    wTile,
                    xVecs[kw * Cin + icStart],
                    static_cast<int32_t>(cipgU)
                );
                AscendC::PipeBarrier<PIPE_V>();

                // delta = sum(prod[0..CinPerGroup))
                AscendC::WholeReduceSum<float>(
                    red,
                    prod,
                    static_cast<int32_t>(cipgU),    // mask = element count
                    1,                              // repeat = 1
                    1,                              // dstRepStride = 1 datablock
                    1,                              // srcBlkStride = 1
                    8                               // srcRepStride = 8 datablocks (default for 1 repeat)
                );
                AscendC::PipeBarrier<PIPE_V>();

                accScalar += red.GetValue(0);
            }

            acc.SetValue(oc, accScalar);
        }

        AscendC::PipeBarrier<PIPE_V>();

        StoreAcc(nIdx, lo);
    }

private:
    Conv1DKernelTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};
    int xVecsLen_{0};
    int accLen_{0};
    int wTileLen_{0};

    AscendC::GlobalTensor<float> xGM_;
    AscendC::GlobalTensor<float> wGM_;
    AscendC::GlobalTensor<float> biasGM_;
    AscendC::GlobalTensor<float> yGM_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> xVecsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> wTileBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> prodBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> redBuf_;
};
