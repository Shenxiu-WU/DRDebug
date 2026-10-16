#pragma once

#include <type_traits>
#include "kernel_operator.h"
#include "kernel_common.h"
#include "tanh_gated_residual_add_backward_tiling.h"

template <typename dataType>
class TanhGatedResidualAddBackwardKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR gradOutput, GM_ADDR gate, GM_ADDR hiddenStates, GM_ADDR mask,
        GM_ADDR gradResidual, GM_ADDR gradHiddenStates, GM_ADDR gradGate,
        GM_ADDR workspace, GM_ADDR tilingGM, AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);

        goGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(gradOutput), tiling_.numElements);
        gateGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(gate), tiling_.numElements);
        hsGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(hiddenStates), tiling_.numElements);
        maskGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(mask), tiling_.numElements);
        grGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(gradResidual), tiling_.numElements);
        ghsGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(gradHiddenStates), tiling_.numElements);
        ggGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(gradGate), tiling_.numElements);
        workspaceGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace), tiling_.usedCoreNum);

        if ASCEND_IS_AIV {
            pipe_ = pipe;
            const uint32_t bs = static_cast<uint32_t>(tiling_.blockSize);

            // Input / output buffers (bfloat16)
            pipe_->InitBuffer(goBuf_, bs * sizeof(dataType));
            pipe_->InitBuffer(gateBuf_, bs * sizeof(dataType));
            pipe_->InitBuffer(hsBuf_, bs * sizeof(dataType));
            pipe_->InitBuffer(maskBuf_, bs * sizeof(dataType));
            pipe_->InitBuffer(grBuf_, bs * sizeof(dataType));
            pipe_->InitBuffer(ghsBuf_, bs * sizeof(dataType));
            pipe_->InitBuffer(ggBuf_, bs * sizeof(dataType));

            // Compute buffers (float32)
            pipe_->InitBuffer(goF32Buf_, bs * sizeof(float));
            pipe_->InitBuffer(gateF32Buf_, bs * sizeof(float));
            pipe_->InitBuffer(hsF32Buf_, bs * sizeof(float));
            pipe_->InitBuffer(maskF32Buf_, bs * sizeof(float));
            pipe_->InitBuffer(tanhGateBuf_, bs * sizeof(float));
            pipe_->InitBuffer(sechSqBuf_, bs * sizeof(float));
            pipe_->InitBuffer(ghsF32Buf_, bs * sizeof(float));
            pipe_->InitBuffer(partialBuf_, bs * sizeof(float));
            pipe_->InitBuffer(tmpMulBuf_, bs * sizeof(float));
            pipe_->InitBuffer(sigmoidTmpBuf_, bs * sizeof(float));

            // Reduce tmp buffer (float32)
            pipe_->InitBuffer(reduceTmpBuf_, 2 * bs * sizeof(float));

            // Scalar buffer (float32)
            pipe_->InitBuffer(sumRwBuf_, sizeof(float));
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int cid = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();

            // ========== Pass 1: compute grad_residual, grad_hidden_states, local_sum ==========
            float localSum = 0.0f;
            for (int localIdx = 0; localIdx < tiling_.tasksPerCore; ++localIdx) {
                const int bx = cid * tiling_.tasksPerCore + localIdx;
                if (bx >= tiling_.numBlocks) {
                    continue;
                }
                const int elemBase = bx * tiling_.blockSize;
                const int validElems = (elemBase + tiling_.blockSize <= tiling_.numElements)
                    ? tiling_.blockSize
                    : (tiling_.numElements - elemBase);
                localSum += ProcessPass1(elemBase, validElems);
            }

            // Write local sum to workspace
            WriteWorkspace(cid, localSum);

            // Barrier: ensure all cores finish pass 1
            AscendC::PipeBarrier<PIPE_ALL>();

            // Core 0: global reduction
            if (cid == 0) {
                float globalSum = 0.0f;
                for (int i = 0; i < tiling_.usedCoreNum; ++i) {
                    globalSum += ReadWorkspace(i);
                }
                WriteWorkspace(0, globalSum);
            }

            // Barrier: ensure core 0 finishes global reduce
            AscendC::PipeBarrier<PIPE_ALL>();

            // Read global sum
            float globalSum = ReadWorkspace(0);

            // ========== Pass 2: compute grad_gate = global_sum * sech_squared ==========
            for (int localIdx = 0; localIdx < tiling_.tasksPerCore; ++localIdx) {
                const int bx = cid * tiling_.tasksPerCore + localIdx;
                if (bx >= tiling_.numBlocks) {
                    continue;
                }
                const int elemBase = bx * tiling_.blockSize;
                const int validElems = (elemBase + tiling_.blockSize <= tiling_.numElements)
                    ? tiling_.blockSize
                    : (tiling_.numElements - elemBase);
                ProcessPass2(elemBase, validElems, globalSum);
            }
        }
    }

private:
    __aicore__ inline float ProcessPass1(int32_t elemBase, int32_t validElems)
    {
        const uint32_t count = static_cast<uint32_t>(validElems);

        // Load inputs from GM to UB
        auto goLocal = goBuf_.Get<dataType>();
        auto gateLocal = gateBuf_.Get<dataType>();
        auto hsLocal = hsBuf_.Get<dataType>();
        auto maskLocal = maskBuf_.Get<dataType>();

        LoadData(goLocal, goGM_[elemBase], count);
        LoadData(gateLocal, gateGM_[elemBase], count);
        LoadData(hsLocal, hsGM_[elemBase], count);
        LoadData(maskLocal, maskGM_[elemBase], count);
        AscendC::PipeBarrier<PIPE_MTE2>();

        // Cast to float32 if needed
        auto goF32 = goF32Buf_.Get<float>();
        auto gateF32 = gateF32Buf_.Get<float>();
        auto hsF32 = hsF32Buf_.Get<float>();
        auto maskF32 = maskF32Buf_.Get<float>();

        if constexpr (!std::is_same_v<dataType, float>) {
            AscendC::Cast(goF32, goLocal, AscendC::RoundMode::CAST_NONE, validElems);
            AscendC::Cast(gateF32, gateLocal, AscendC::RoundMode::CAST_NONE, validElems);
            AscendC::Cast(hsF32, hsLocal, AscendC::RoundMode::CAST_NONE, validElems);
            AscendC::Cast(maskF32, maskLocal, AscendC::RoundMode::CAST_NONE, validElems);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            goF32 = goLocal.template ReinterpretCast<float>();
            gateF32 = gateLocal.template ReinterpretCast<float>();
            hsF32 = hsLocal.template ReinterpretCast<float>();
            maskF32 = maskLocal.template ReinterpretCast<float>();
        }

        // tanh(gate)
        auto tanhGate = tanhGateBuf_.Get<float>();
        ComputeTanh(gateF32, tanhGate, validElems);

        // sech_squared = 1 - tanh^2(gate)
        auto sechSq = sechSqBuf_.Get<float>();
        AscendC::Mul(sechSq, tanhGate, tanhGate, validElems);
        AscendC::Adds(sechSq, sechSq, -1.0f, validElems);
        AscendC::Muls(sechSq, sechSq, -1.0f, validElems);  // 1 - x = -(x - 1) = Adds then Muls(-1) ... wait, Adds(sechSq, sechSq, -1) gives tanh^2 - 1, then Muls by -1 gives 1 - tanh^2
        AscendC::PipeBarrier<PIPE_V>();

        // grad_residual = grad_output (direct copy through UB)
        auto grLocal = grBuf_.Get<dataType>();
        AscendC::DataCopy(grLocal, goLocal, validElems);
        AscendC::PipeBarrier<PIPE_V>();
        StoreData(grGM_[elemBase], grLocal, count);
        AscendC::PipeBarrier<PIPE_MTE2>();

        // grad_hidden_states = grad_output * tanh(gate) * mask
        auto ghsF32 = ghsF32Buf_.Get<float>();
        AscendC::Mul(ghsF32, goF32, tanhGate, validElems);
        AscendC::Mul(ghsF32, ghsF32, maskF32, validElems);
        AscendC::PipeBarrier<PIPE_V>();

        auto ghsLocal = ghsBuf_.Get<dataType>();
        if constexpr (!std::is_same_v<dataType, float>) {
            AscendC::Cast(ghsLocal, ghsF32, OutputRoundMode(), validElems);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            ghsLocal = ghsF32.template ReinterpretCast<dataType>();
        }
        StoreData(ghsGM_[elemBase], ghsLocal, count);
        AscendC::PipeBarrier<PIPE_MTE2>();

        // partial = grad_output * hidden_states * mask (for grad_gate)
        auto partial = partialBuf_.Get<float>();
        AscendC::Mul(partial, goF32, hsF32, validElems);
        AscendC::Mul(partial, partial, maskF32, validElems);
        AscendC::PipeBarrier<PIPE_V>();

        // Reduce sum over tile
        auto reduceTmp = reduceTmpBuf_.Get<float>();
        auto sumRwLocal = sumRwBuf_.Get<float>();
        AscendC::ReduceSum<float>(sumRwLocal, partial, reduceTmp, validElems);
        AscendC::PipeBarrier<PIPE_V>();

        return sumRwLocal.GetValue(0);
    }

    __aicore__ inline void ProcessPass2(int32_t elemBase, int32_t validElems, float globalSum)
    {
        const uint32_t count = static_cast<uint32_t>(validElems);

        // Reload gate and recompute sech_squared
        auto gateLocal = gateBuf_.Get<dataType>();
        LoadData(gateLocal, gateGM_[elemBase], count);
        AscendC::PipeBarrier<PIPE_MTE2>();

        auto gateF32 = gateF32Buf_.Get<float>();
        if constexpr (!std::is_same_v<dataType, float>) {
            AscendC::Cast(gateF32, gateLocal, AscendC::RoundMode::CAST_NONE, validElems);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            gateF32 = gateLocal.template ReinterpretCast<float>();
        }

        auto tanhGate = tanhGateBuf_.Get<float>();
        ComputeTanh(gateF32, tanhGate, validElems);

        auto sechSq = sechSqBuf_.Get<float>();
        AscendC::Mul(sechSq, tanhGate, tanhGate, validElems);
        AscendC::Adds(sechSq, sechSq, -1.0f, validElems);
        AscendC::Muls(sechSq, sechSq, -1.0f, validElems);
        AscendC::PipeBarrier<PIPE_V>();

        // grad_gate = global_sum * sech_squared
        auto tmpMul = tmpMulBuf_.Get<float>();
        AscendC::Muls(tmpMul, sechSq, globalSum, validElems);
        AscendC::PipeBarrier<PIPE_V>();

        auto ggLocal = ggBuf_.Get<dataType>();
        if constexpr (!std::is_same_v<dataType, float>) {
            AscendC::Cast(ggLocal, tmpMul, OutputRoundMode(), validElems);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            ggLocal = tmpMul.template ReinterpretCast<dataType>();
        }
        StoreData(ggGM_[elemBase], ggLocal, count);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void ComputeTanh(AscendC::LocalTensor<float> &src,
                                       AscendC::LocalTensor<float> &dst,
                                       int32_t count)
    {
        // tanh(x) = 2 * sigmoid(2x) - 1
        auto sigmoidTmp = sigmoidTmpBuf_.Get<uint8_t>();
        auto tmp = tmpMulBuf_.Get<float>();
        AscendC::Muls(tmp, src, 2.0f, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sigmoid<float>(dst, tmp, sigmoidTmp, static_cast<uint32_t>(count));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(dst, dst, 2.0f, count);
        AscendC::Adds(dst, dst, -1.0f, count);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void WriteWorkspace(int32_t idx, float value)
    {
        auto sumRwLocal = sumRwBuf_.Get<float>();
        sumRwLocal.SetValue(0, value);
        AscendC::DataCopy(workspaceGM_[idx], sumRwLocal, 1);
    }

    __aicore__ inline float ReadWorkspace(int32_t idx)
    {
        auto sumRwLocal = sumRwBuf_.Get<float>();
        AscendC::DataCopy(sumRwLocal, workspaceGM_[idx], 1);
        AscendC::PipeBarrier<PIPE_MTE2>();
        return sumRwLocal.GetValue(0);
    }

    template <typename T>
    __aicore__ inline void LoadData(AscendC::LocalTensor<T> &dst,
                                    AscendC::GlobalTensor<T> src,
                                    uint32_t count)
    {
        AscendC::DataCopy(dst, src, count);
    }

    template <typename T>
    __aicore__ inline void StoreData(AscendC::GlobalTensor<T> dst,
                                     AscendC::LocalTensor<T> &src,
                                     uint32_t count)
    {
        AscendC::DataCopy(dst, src, count);
    }

    __aicore__ inline AscendC::RoundMode OutputRoundMode() const
    {
        if constexpr (std::is_same_v<dataType, bfloat16_t>) {
            return AscendC::RoundMode::CAST_ROUND;
        }
        return AscendC::RoundMode::CAST_NONE;
    }

    TanhGatedResidualAddBackwardTiling tiling_;

    // GM tensors
    AscendC::GlobalTensor<dataType> goGM_;
    AscendC::GlobalTensor<dataType> gateGM_;
    AscendC::GlobalTensor<dataType> hsGM_;
    AscendC::GlobalTensor<dataType> maskGM_;
    AscendC::GlobalTensor<dataType> grGM_;
    AscendC::GlobalTensor<dataType> ghsGM_;
    AscendC::GlobalTensor<dataType> ggGM_;
    AscendC::GlobalTensor<float> workspaceGM_;

    AscendC::TPipe *pipe_;

    // Input / output buffers
    AscendC::TBuf<AscendC::TPosition::VECCALC> goBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gateBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> hsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> grBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ghsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ggBuf_;

    // Compute buffers (float32)
    AscendC::TBuf<AscendC::TPosition::VECCALC> goF32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gateF32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> hsF32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskF32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tanhGateBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sechSqBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ghsF32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> partialBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpMulBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sigmoidTmpBuf_;

    // Reduce tmp buffer
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceTmpBuf_;

    // Scalar buffer
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumRwBuf_;
};
