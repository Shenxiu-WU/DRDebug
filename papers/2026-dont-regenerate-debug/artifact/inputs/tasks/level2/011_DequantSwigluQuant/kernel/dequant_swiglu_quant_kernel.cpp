#include "dequant_swiglu_quant_tiling.h"
#include "kernel_common.h"

class DequantSwigluQuantKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR weight_scale, GM_ADDR activation_scale,
        GM_ADDR quant_scale, GM_ADDR y, GM_ADDR scales,
        GM_ADDR tilingGM, AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);
        pipe_ = pipe;

        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(x), tiling_.T * tiling_.twoH);
        weightScaleGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(weight_scale), tiling_.twoH);
        activationScaleGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(activation_scale), tiling_.T);
        quantScaleGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(quant_scale), tiling_.H);
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(y), tiling_.T * tiling_.H);
        scalesGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scales), tiling_.T);

        pipe_->InitBuffer(weightScaleBuf_, tiling_.twoH * sizeof(float));
        pipe_->InitBuffer(quantScaleBuf_, tiling_.H * sizeof(float));
        pipe_->InitBuffer(actScaleBuf_, tiling_.T * sizeof(float));
        pipe_->InitBuffer(xInQueue_, 1, tiling_.twoH * sizeof(int32_t));
        pipe_->InitBuffer(gateBuf_, tiling_.H * sizeof(float));
        pipe_->InitBuffer(upBuf_, tiling_.H * sizeof(float));
        pipe_->InitBuffer(outQueue_, 1, tiling_.H * sizeof(int8_t));
        pipe_->InitBuffer(maxBuf_, 16 * sizeof(float));
        pipe_->InitBuffer(fp16Buf_, tiling_.H * sizeof(half));

        auto weightScaleLocal = weightScaleBuf_.Get<float>();
        auto quantScaleLocal = quantScaleBuf_.Get<float>();
        auto actScaleLocal = actScaleBuf_.Get<float>();

        AscendC::DataCopy(weightScaleLocal, weightScaleGM_, tiling_.twoH);
        AscendC::DataCopy(quantScaleLocal, quantScaleGM_, tiling_.H);
        AscendC::DataCopy(actScaleLocal, activationScaleGM_, tiling_.T);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void Process()
    {
        uint32_t coreNum = AscendC::GetBlockNum();
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t tokensPerCore = CeilDivU32(static_cast<uint32_t>(tiling_.T), coreNum);
        uint32_t startToken = coreIdx * tokensPerCore;
        uint32_t endToken = startToken + tokensPerCore;
        if (endToken > static_cast<uint32_t>(tiling_.T)) {
            endToken = tiling_.T;
        }

        for (uint32_t tokenIdx = startToken; tokenIdx < endToken; ++tokenIdx) {
            ProcessToken(static_cast<int32_t>(tokenIdx));
        }
    }

private:
    __aicore__ inline void ProcessToken(int32_t tokenIdx)
    {
        xInQueue_.AllocTensor<int32_t>(xInLocal_);
        AscendC::DataCopy(xInLocal_, xGM_[tokenIdx * tiling_.twoH], tiling_.twoH);
        xInQueue_.EnQue(xInLocal_);

        xInQueue_.DeQue<int32_t>(xInLocal_);
        auto xFpLocal = xInLocal_.template ReinterpretCast<float>();
        AscendC::Cast(xFpLocal, xInLocal_, AscendC::RoundMode::CAST_NONE, tiling_.twoH);
        AscendC::PipeBarrier<PIPE_V>();

        auto weightScaleLocal = weightScaleBuf_.Get<float>();
        AscendC::Mul(xFpLocal, xFpLocal, weightScaleLocal, tiling_.twoH);
        AscendC::PipeBarrier<PIPE_V>();

        auto actScaleLocal = actScaleBuf_.Get<float>();
        float actScale = actScaleLocal.GetValue(tokenIdx);
        AscendC::Muls(xFpLocal, xFpLocal, actScale, tiling_.twoH);
        AscendC::PipeBarrier<PIPE_V>();

        auto gateLocal = gateBuf_.Get<float>();
        auto upLocal = upBuf_.Get<float>();

        if (tiling_.swiglu_mode == 0) {
            int gateOffset = tiling_.activate_left ? 0 : tiling_.H;
            int upOffset = tiling_.activate_left ? tiling_.H : 0;
            for (int i = 0; i < tiling_.H; ++i) {
                gateLocal[i] = xFpLocal[gateOffset + i];
                upLocal[i] = xFpLocal[upOffset + i];
            }
        } else {
            for (int i = 0; i < tiling_.H; ++i) {
                gateLocal[i] = xFpLocal[i * 2];
                upLocal[i] = xFpLocal[i * 2 + 1];
            }
        }
        AscendC::PipeBarrier<PIPE_V>();

        auto tmpBuf = xFpLocal;

        if (tiling_.swiglu_mode == 0) {
            AscendC::Muls(tmpBuf, gateLocal, -1.0f, tiling_.H);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(tmpBuf, tmpBuf, tiling_.H);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(tmpBuf, tmpBuf, 1.0f, tiling_.H);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Reciprocal(tmpBuf, tmpBuf, tiling_.H);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Mul(gateLocal, gateLocal, tmpBuf, tiling_.H);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(gateLocal, gateLocal, upLocal, tiling_.H);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            float limit = tiling_.clamp_limit;
            AscendC::Mins(gateLocal, gateLocal, limit, tiling_.H);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Maxs(gateLocal, gateLocal, -limit, tiling_.H);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Muls(tmpBuf, gateLocal, -tiling_.glu_alpha, tiling_.H);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(tmpBuf, tmpBuf, tiling_.H);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(tmpBuf, tmpBuf, 1.0f, tiling_.H);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Reciprocal(tmpBuf, tmpBuf, tiling_.H);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Mul(gateLocal, gateLocal, tmpBuf, tiling_.H);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Mins(upLocal, upLocal, limit, tiling_.H);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Maxs(upLocal, upLocal, -limit, tiling_.H);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(upLocal, upLocal, tiling_.glu_bias, tiling_.H);
            AscendC::PipeBarrier<PIPE_V>();

            AscendC::Mul(gateLocal, gateLocal, upLocal, tiling_.H);
            AscendC::PipeBarrier<PIPE_V>();
        }

        auto quantScaleLocal = quantScaleBuf_.Get<float>();
        AscendC::Mul(gateLocal, gateLocal, quantScaleLocal, tiling_.H);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Abs(tmpBuf, gateLocal, tiling_.H);
        AscendC::PipeBarrier<PIPE_V>();

        auto maxLocal = maxBuf_.Get<float>();
        AscendC::ReduceMax(maxLocal, tmpBuf, upLocal, tiling_.H);
        AscendC::PipeBarrier<PIPE_V>();
        float maxAbs = maxLocal.GetValue(0);

        float rawScale = maxAbs / 127.0f;
        float clampedScale = rawScale < 1e-10f ? 1e-10f : rawScale;
        float recip = 1.0f / clampedScale;

        maxLocal.SetValue(0, rawScale);
        AscendC::DataCopy(scalesGM_[tokenIdx], maxLocal, 1);
        AscendC::PipeBarrier<PIPE_MTE2>();

        AscendC::Muls(gateLocal, gateLocal, recip, tiling_.H);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mins(gateLocal, gateLocal, 127.0f, tiling_.H);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Maxs(gateLocal, gateLocal, -128.0f, tiling_.H);
        AscendC::PipeBarrier<PIPE_V>();

        auto fp16Local = fp16Buf_.Get<half>();
        AscendC::Cast(fp16Local, gateLocal, AscendC::RoundMode::CAST_NONE, tiling_.H);
        AscendC::PipeBarrier<PIPE_V>();

        outQueue_.AllocTensor<int8_t>(outLocal_);
        AscendC::Cast(outLocal_, fp16Local, AscendC::RoundMode::CAST_NONE, tiling_.H);
        AscendC::PipeBarrier<PIPE_V>();
        outQueue_.EnQue(outLocal_);

        outQueue_.DeQue<int8_t>(outLocal_);
        AscendC::DataCopy(yGM_[tokenIdx * tiling_.H], outLocal_, tiling_.H);
        AscendC::PipeBarrier<PIPE_MTE2>();

        outQueue_.FreeTensor(outLocal_);
        xInQueue_.FreeTensor(xInLocal_);
    }

private:
    DequantSwigluQuantTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};

    AscendC::GlobalTensor<int32_t> xGM_;
    AscendC::GlobalTensor<float> weightScaleGM_;
    AscendC::GlobalTensor<float> activationScaleGM_;
    AscendC::GlobalTensor<float> quantScaleGM_;
    AscendC::GlobalTensor<int8_t> yGM_;
    AscendC::GlobalTensor<float> scalesGM_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> weightScaleBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> quantScaleBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> actScaleBuf_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> xInQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gateBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> upBuf_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> outQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maxBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16Buf_;

    AscendC::LocalTensor<int32_t> xInLocal_;
    AscendC::LocalTensor<int8_t> outLocal_;
};

extern "C" __global__ __aicore__ void dequant_swiglu_quant_custom_kernel(
    GM_ADDR x, GM_ADDR weight_scale, GM_ADDR activation_scale,
    GM_ADDR quant_scale, GM_ADDR y, GM_ADDR scales, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    DequantSwigluQuantKernel kernel;
    kernel.Init(x, weight_scale, activation_scale, quant_scale, y, scales, tiling, &pipe);
    kernel.Process();
}

extern "C" void dequant_swiglu_quant_do(
    uint32_t blockDim, void *stream,
    uint8_t *x, uint8_t *weight_scale, uint8_t *activation_scale,
    uint8_t *quant_scale, uint8_t *y, uint8_t *scales, uint8_t *tiling)
{
    dequant_swiglu_quant_custom_kernel<<<blockDim, nullptr, stream>>>(
        x, weight_scale, activation_scale, quant_scale, y, scales, tiling);
}
