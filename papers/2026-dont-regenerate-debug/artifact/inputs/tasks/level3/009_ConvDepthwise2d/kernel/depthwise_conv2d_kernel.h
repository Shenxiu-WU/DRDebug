#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"

#include "kernel_common.h"
#include "depthwise_conv2d_tiling.h"

class DepthwiseConv2DKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR w,
        GM_ADDR bias,
        GM_ADDR y,
        GM_ADDR tilingGM,
        AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);

        // 32B alignment requirement for UB-side vector ops on fp32 means each
        // logical channel slice must start at a 32B (= 8 fp32 elements)
        // boundary. We therefore pad the UB stride to a multiple of 8.
        paddedC_ = ((tiling_.C + 7) / 8) * 8;
        if (paddedC_ <= 0) {
            paddedC_ = 8;
        }
        weightStride_ = paddedC_;
        weightSlots_ = tiling_.kH * tiling_.kW;

        xGM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(x),
            static_cast<uint64_t>(tiling_.N) * tiling_.inSpatial * tiling_.C);
        wGM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(w),
            static_cast<uint64_t>(tiling_.weightCount));
        biasGM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(bias),
            static_cast<uint64_t>(tiling_.C));
        yGM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float *>(y),
            static_cast<uint64_t>(tiling_.N) * tiling_.outSpatial * tiling_.C);

        if ASCEND_IS_AIV {
            pipe_ = pipe;

            const uint32_t weightBufBytes =
                static_cast<uint32_t>(weightSlots_ * weightStride_ * sizeof(float));
            const uint32_t paddedBytes =
                static_cast<uint32_t>(paddedC_ * sizeof(float));

            pipe_->InitBuffer(weightBuf_, weightBufBytes);
            pipe_->InitBuffer(biasBuf_, paddedBytes);
            pipe_->InitBuffer(accBuf_, paddedBytes);
            pipe_->InitBuffer(tmpBuf_, paddedBytes);
            pipe_->InitBuffer(xInQueue_, 1, paddedBytes);
            pipe_->InitBuffer(yOutQueue_, 1, paddedBytes);
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            // Phase 0: one-shot pre-load of weights and bias to UB.
            weightLocal_ = weightBuf_.Get<float>();
            // Zero the entire weight buffer first so any padded tail bytes are
            // deterministic 0 — ensures Mul over paddedC tail produces 0
            // contribution to acc_ub even if we only intended C valid floats.
            AscendC::Duplicate(weightLocal_, 0.0f,
                               static_cast<int32_t>(weightSlots_ * weightStride_));
            AscendC::PipeBarrier<PIPE_V>();

            // Load each (kh, kw) weight slice into its 32B-aligned UB row.
            for (int wRow = 0; wRow < weightSlots_; ++wRow) {
                AscendC::DataCopyExtParams copyParams{
                    1,
                    static_cast<uint32_t>(tiling_.C * sizeof(float)),
                    0,
                    0,
                    0};
                AscendC::DataCopyPadExtParams<float> padParams{true, 0, 0, 0.0f};
                AscendC::DataCopyPad(
                    weightLocal_[wRow * weightStride_],
                    wGM_[static_cast<uint64_t>(wRow) * tiling_.C],
                    copyParams,
                    padParams);
            }

            biasLocal_ = biasBuf_.Get<float>();
            // Zero pad bias UB tail too so Adds(acc, bias, 0, paddedC) is safe.
            AscendC::Duplicate(biasLocal_, 0.0f, static_cast<int32_t>(paddedC_));
            AscendC::PipeBarrier<PIPE_V>();
            LoadGmToUb(biasLocal_, biasGM_, static_cast<uint32_t>(tiling_.C));

            // Ensure MTE2 → V dependency is honoured before reading from UB.
            AscendC::PipeBarrier<PIPE_ALL>();

            // Phase 1: dispatch output rows to this AIV worker.
            const int workerId = static_cast<int>(AscendC::GetBlockIdx());
            int workerCount = tiling_.launchBlocks;
            if (workerCount <= 0) {
                workerCount = 1;
            }

            const int totalRows = tiling_.mOut;
            for (int outRow = workerId; outRow < totalRows; outRow += workerCount) {
                ProcessOneRow(outRow);
            }
        }
    }

private:
    __aicore__ inline void ProcessOneRow(int outRow)
    {
        const int n = outRow / tiling_.outSpatial;
        const int rem = outRow - n * tiling_.outSpatial;
        const int oh = rem / tiling_.OW;
        const int ow = rem - oh * tiling_.OW;

        AscendC::LocalTensor<float> accLocal = accBuf_.Get<float>();

        // acc <- bias (or zero if no bias). All of acc, bias, weight slices
        // are paddedC-sized, so we run vec ops over the full paddedC count;
        // the tail beyond C contains only zeros and never contaminates the
        // first C valid lanes that we eventually copy back to GM.
        if (tiling_.hasBias > 0) {
            AscendC::Adds(accLocal, biasLocal_, 0.0f, paddedC_);
        } else {
            AscendC::Duplicate(accLocal, 0.0f, paddedC_);
        }
        AscendC::PipeBarrier<PIPE_V>();

        for (int kh = 0; kh < tiling_.kH; ++kh) {
            const int ih = oh * tiling_.sH - tiling_.pH + kh;
            if (ih < 0 || ih >= tiling_.H) {
                continue;
            }

            for (int kw = 0; kw < tiling_.kW; ++kw) {
                const int iw = ow * tiling_.sW - tiling_.pW + kw;
                if (iw < 0 || iw >= tiling_.W) {
                    continue;
                }

                const int xRow = n * tiling_.inSpatial + ih * tiling_.W + iw;
                const int wOff = (kh * tiling_.kW + kw) * weightStride_;

                // x[xRow, :] -> xLocal (paddedC slot, with right-pad zeros)
                AscendC::LocalTensor<float> xLocal;
                xInQueue_.AllocTensor<float>(xLocal);
                AscendC::Duplicate(xLocal, 0.0f, paddedC_);
                AscendC::PipeBarrier<PIPE_V>();
                LoadGmToUb(
                    xLocal,
                    xGM_[static_cast<uint64_t>(xRow) * tiling_.C],
                    static_cast<uint32_t>(tiling_.C));
                xInQueue_.EnQue(xLocal);
                xInQueue_.DeQue<float>(xLocal);

                // tmp = x * weight[kh, kw]
                AscendC::LocalTensor<float> tmpLocal = tmpBuf_.Get<float>();
                AscendC::Mul(tmpLocal, xLocal, weightLocal_[wOff], paddedC_);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Add(accLocal, accLocal, tmpLocal, paddedC_);
                AscendC::PipeBarrier<PIPE_V>();

                xInQueue_.FreeTensor(xLocal);
            }
        }

        // acc -> y[outRow, :] (only C valid floats are flushed to GM).
        AscendC::LocalTensor<float> yLocal;
        yOutQueue_.AllocTensor<float>(yLocal);
        AscendC::Adds(yLocal, accLocal, 0.0f, paddedC_);
        yOutQueue_.EnQue(yLocal);
        yOutQueue_.DeQue<float>(yLocal);
        StoreUbToGm(
            yGM_[static_cast<uint64_t>(outRow) * tiling_.C],
            yLocal,
            static_cast<uint32_t>(tiling_.C));
        yOutQueue_.FreeTensor(yLocal);
    }

private:
    DepthwiseConv2DKernelTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};

    int32_t paddedC_{0};
    int32_t weightStride_{0};
    int32_t weightSlots_{0};

    AscendC::GlobalTensor<float> xGM_;
    AscendC::GlobalTensor<float> wGM_;
    AscendC::GlobalTensor<float> biasGM_;
    AscendC::GlobalTensor<float> yGM_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> weightBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> biasBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> xInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> yOutQueue_;

    AscendC::LocalTensor<float> weightLocal_;
    AscendC::LocalTensor<float> biasLocal_;
};
