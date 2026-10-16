#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <type_traits>

#include "kernel_operator.h"

#include "adamw_tiling.h"
#include "kernel_common.h"
#include "vector_tile.h"

template <typename T>
class AdamWKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR var,
        GM_ADDR m,
        GM_ADDR v,
        GM_ADDR grad,
        GM_ADDR varOut,
        GM_ADDR mOut,
        GM_ADDR vOut,
        GM_ADDR tilingGM,
        AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);

        const uint32_t total = static_cast<uint32_t>(tiling_.nTotal);
        varGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(var), total);
        gradGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(grad), total);
        mGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(m), total);
        vGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(v), total);
        varOutGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(varOut), total);
        mOutGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(mOut), total);
        vOutGM_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(vOut), total);

        if ASCEND_IS_AIV {
            pipe_ = pipe;
            const uint32_t subBlockN = static_cast<uint32_t>(tiling_.subBlockN);

            // VECIN queues for the four loop-level inputs.
            pipe_->InitBuffer(varInQueue_, 1, subBlockN * sizeof(T));
            pipe_->InitBuffer(gradInQueue_, 1, subBlockN * sizeof(T));
            pipe_->InitBuffer(mInQueue_, 1, subBlockN * sizeof(T));
            pipe_->InitBuffer(vInQueue_, 1, subBlockN * sizeof(T));

            // VECOUT queues for the three outputs.
            pipe_->InitBuffer(varOutQueue_, 1, subBlockN * sizeof(T));
            pipe_->InitBuffer(mOutQueue_, 1, subBlockN * sizeof(T));
            pipe_->InitBuffer(vOutQueue_, 1, subBlockN * sizeof(T));

            // VECCALC scratch buffers (fp32 working space).
            pipe_->InitBuffer(tmpABuf_, subBlockN * sizeof(float));
            pipe_->InitBuffer(tmpBBuf_, subBlockN * sizeof(float));

            // Cast buffers required only when dtype is not float.
            //   varCast/gradCast/mCast/vCast hold the fp32 working copies of
            //   var, grad, m, v respectively; mCast/vCast are also reused to
            //   hold m_new/v_new before the final downcast to T.
            if constexpr (!std::is_same_v<T, float>) {
                pipe_->InitBuffer(varCastBuf_, subBlockN * sizeof(float));
                pipe_->InitBuffer(gradCastBuf_, subBlockN * sizeof(float));
                pipe_->InitBuffer(mCastBuf_, subBlockN * sizeof(float));
                pipe_->InitBuffer(vCastBuf_, subBlockN * sizeof(float));
            }
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
            const int subBlockIdx = AscendC::GetSubBlockIdx();
            const int blockN = tiling_.blockN;
            const int subBlockN = tiling_.subBlockN;
            const int nTotal = tiling_.nTotal;

            for (int localIdx = 0; localIdx < tiling_.tasksPerCore; ++localIdx) {
                const int bx = coreIdx * tiling_.tasksPerCore + localIdx;
                if (bx >= tiling_.nNum) {
                    continue;
                }
                const int offset = bx * blockN + subBlockIdx * subBlockN;
                if (offset >= nTotal) {
                    continue;
                }
                int validN = subBlockN;
                if (offset + validN > nTotal) {
                    validN = nTotal - offset;
                }
                ProcessTile(offset, validN);
            }
        }
    }

private:
    __aicore__ inline AscendC::RoundMode VarOutRoundMode() const
    {
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            return AscendC::RoundMode::CAST_ROUND;
        }
        return AscendC::RoundMode::CAST_NONE;
    }

    __aicore__ inline void PrepareInputFp32(
        AscendC::LocalTensor<float> &dst,
        AscendC::LocalTensor<T> &src,
        AscendC::TBuf<AscendC::TPosition::VECCALC> &castBuf,
        int32_t count)
    {
        if constexpr (std::is_same_v<T, float>) {
            dst = src.template ReinterpretCast<float>();
        } else {
            dst = castBuf.template Get<float>();
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void FinalizeOutTensor(
        AscendC::LocalTensor<T> &dst,
        AscendC::LocalTensor<float> &src,
        int32_t count)
    {
        if constexpr (std::is_same_v<T, float>) {
            // dst is a separate VECOUT slot; copy fp32 result into it.
            AscendC::DataCopy(dst, src, static_cast<uint32_t>(count));
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::Cast(dst, src, VarOutRoundMode(), count);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CopyInVar(int offset, int validN)
    {
        varInQueue_.AllocTensor<T>(varInLocal_);
        LoadGmToUb(varInLocal_, varGM_[offset], static_cast<uint32_t>(validN));
        varInQueue_.EnQue(varInLocal_);
    }

    __aicore__ inline void CopyInGrad(int offset, int validN)
    {
        gradInQueue_.AllocTensor<T>(gradInLocal_);
        LoadGmToUb(gradInLocal_, gradGM_[offset], static_cast<uint32_t>(validN));
        gradInQueue_.EnQue(gradInLocal_);
    }

    __aicore__ inline void CopyInM(int offset, int validN)
    {
        mInQueue_.AllocTensor<T>(mInLocal_);
        LoadGmToUb(mInLocal_, mGM_[offset], static_cast<uint32_t>(validN));
        mInQueue_.EnQue(mInLocal_);
    }

    __aicore__ inline void CopyInV(int offset, int validN)
    {
        vInQueue_.AllocTensor<T>(vInLocal_);
        LoadGmToUb(vInLocal_, vGM_[offset], static_cast<uint32_t>(validN));
        vInQueue_.EnQue(vInLocal_);
    }

    __aicore__ inline void CopyOutVar(int offset, int validN)
    {
        varOutQueue_.DeQue<T>(varOutLocal_);
        StoreUbToGm(varOutGM_[offset], varOutLocal_, static_cast<uint32_t>(validN));
        varOutQueue_.FreeTensor(varOutLocal_);
    }

    __aicore__ inline void CopyOutM(int offset, int validN)
    {
        mOutQueue_.DeQue<T>(mOutLocal_);
        StoreUbToGm(mOutGM_[offset], mOutLocal_, static_cast<uint32_t>(validN));
        mOutQueue_.FreeTensor(mOutLocal_);
    }

    __aicore__ inline void CopyOutV(int offset, int validN)
    {
        vOutQueue_.DeQue<T>(vOutLocal_);
        StoreUbToGm(vOutGM_[offset], vOutLocal_, static_cast<uint32_t>(validN));
        vOutQueue_.FreeTensor(vOutLocal_);
    }

    __aicore__ inline void Compute(int validN)
    {
        const float beta1 = tiling_.beta1;
        const float beta2 = tiling_.beta2;
        const float oneMinusBeta1 = tiling_.oneMinusBeta1;
        const float oneMinusBeta2 = tiling_.oneMinusBeta2;
        const float invBias1 = tiling_.invBias1;
        const float invBias2 = tiling_.invBias2;
        const float negLr = tiling_.negLr;
        const float epsilon = tiling_.epsilon;
        const float coeffGradForNesterov = tiling_.coeffGradForNesterov;
        const bool useNesterov = (tiling_.useNesterov != 0);

        // ----- Allocate output tensors so EnQue can be called once we are done. -----
        varOutQueue_.AllocTensor<T>(varOutLocal_);
        mOutQueue_.AllocTensor<T>(mOutLocal_);
        vOutQueue_.AllocTensor<T>(vOutLocal_);

        // ----- DeQue inputs and prepare fp32 working tensors. -----
        varInQueue_.DeQue<T>(varInLocal_);
        gradInQueue_.DeQue<T>(gradInLocal_);
        mInQueue_.DeQue<T>(mInLocal_);
        vInQueue_.DeQue<T>(vInLocal_);

        // For T == float: f32 alias inputs directly (zero-copy).
        // For T != float: cast input T tensors into VECCALC fp32 buffers.
        AscendC::LocalTensor<float> varF32;
        AscendC::LocalTensor<float> gradF32;
        AscendC::LocalTensor<float> mF32;
        AscendC::LocalTensor<float> vF32;
        PrepareInputFp32(varF32, varInLocal_, varCastBuf_, validN);
        PrepareInputFp32(gradF32, gradInLocal_, gradCastBuf_, validN);
        PrepareInputFp32(mF32, mInLocal_, mCastBuf_, validN);
        PrepareInputFp32(vF32, vInLocal_, vCastBuf_, validN);

        AscendC::LocalTensor<float> tmpA = tmpABuf_.template Get<float>();
        AscendC::LocalTensor<float> tmpB = tmpBBuf_.template Get<float>();

        // ----- m_new = beta1 * m + (1 - beta1) * grad (kept in mF32) -----
        AscendC::Muls(tmpA, gradF32, oneMinusBeta1, validN);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(mF32, mF32, beta1, validN);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(mF32, mF32, tmpA, validN);
        AscendC::PipeBarrier<PIPE_V>();

        // ----- v_new = beta2 * v + (1 - beta2) * grad * grad (kept in vF32) -----
        AscendC::Mul(tmpA, gradF32, gradF32, validN);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(tmpA, tmpA, oneMinusBeta2, validN);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(vF32, vF32, beta2, validN);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(vF32, vF32, tmpA, validN);
        AscendC::PipeBarrier<PIPE_V>();

        // ----- m_hat = m_new * inv_bias1 (kept in tmpA) -----
        AscendC::Muls(tmpA, mF32, invBias1, validN);
        AscendC::PipeBarrier<PIPE_V>();

        // ----- denom = sqrt(v_new * inv_bias2) + eps (kept in tmpB) -----
        AscendC::Muls(tmpB, vF32, invBias2, validN);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sqrt(tmpB, tmpB, validN);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(tmpB, tmpB, epsilon, validN);
        AscendC::PipeBarrier<PIPE_V>();

        if (useNesterov) {
            // m_nesterov = beta1 * m_hat + (1 - beta1) / (1 - beta1_power) * grad
            AscendC::Muls(tmpA, tmpA, beta1, validN);
            AscendC::PipeBarrier<PIPE_V>();
            // Reuse gradF32 as scratch — grad is no longer needed beyond this point.
            AscendC::Muls(gradF32, gradF32, coeffGradForNesterov, validN);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(tmpA, tmpA, gradF32, validN);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // update = (m_hat or m_nesterov) / denom; var_f32 += (-lr) * update
        AscendC::Div(tmpA, tmpA, tmpB, validN);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(tmpA, tmpA, negLr, validN);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(varF32, varF32, tmpA, validN);
        AscendC::PipeBarrier<PIPE_V>();

        // ----- Downcast (or copy) outputs back to T. -----
        FinalizeOutTensor(varOutLocal_, varF32, validN);
        FinalizeOutTensor(mOutLocal_, mF32, validN);
        FinalizeOutTensor(vOutLocal_, vF32, validN);

        // Free input slots before EnQueing outputs.
        varInQueue_.FreeTensor(varInLocal_);
        gradInQueue_.FreeTensor(gradInLocal_);
        mInQueue_.FreeTensor(mInLocal_);
        vInQueue_.FreeTensor(vInLocal_);

        varOutQueue_.EnQue(varOutLocal_);
        mOutQueue_.EnQue(mOutLocal_);
        vOutQueue_.EnQue(vOutLocal_);
    }

    __aicore__ inline void ProcessTile(int offset, int validN)
    {
        CopyInVar(offset, validN);
        CopyInGrad(offset, validN);
        CopyInM(offset, validN);
        CopyInV(offset, validN);
        Compute(validN);
        CopyOutVar(offset, validN);
        CopyOutM(offset, validN);
        CopyOutV(offset, validN);
    }

private:
    AdamWKernelTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};

    AscendC::GlobalTensor<T> varGM_;
    AscendC::GlobalTensor<T> gradGM_;
    AscendC::GlobalTensor<T> mGM_;
    AscendC::GlobalTensor<T> vGM_;
    AscendC::GlobalTensor<T> varOutGM_;
    AscendC::GlobalTensor<T> mOutGM_;
    AscendC::GlobalTensor<T> vOutGM_;

    AscendC::TQue<AscendC::TPosition::VECIN, 0> varInQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> gradInQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> mInQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> vInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> varOutQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> mOutQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> vOutQueue_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpABuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> varCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gradCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> mCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> vCastBuf_;

    AscendC::LocalTensor<T> varInLocal_;
    AscendC::LocalTensor<T> gradInLocal_;
    AscendC::LocalTensor<T> mInLocal_;
    AscendC::LocalTensor<T> vInLocal_;
    AscendC::LocalTensor<T> varOutLocal_;
    AscendC::LocalTensor<T> mOutLocal_;
    AscendC::LocalTensor<T> vOutLocal_;
};
