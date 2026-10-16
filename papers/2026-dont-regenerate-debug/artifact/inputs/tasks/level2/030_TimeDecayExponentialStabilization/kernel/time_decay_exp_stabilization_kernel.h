#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"
#include "kernel_common.h"
#include "time_decay_exp_stabilization_tiling.h"
#include "vector_tile.h"

class TimeDecayExpStabilizationKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR timeDecay, GM_ADDR key, GM_ADDR timeFirst,
        GM_ADDR value, GM_ADDR maxState, GM_ADDR numState,
        GM_ADDR denState, GM_ADDR output,
        GM_ADDR maxStateOut, GM_ADDR numStateOut, GM_ADDR denStateOut,
        GM_ADDR workspace, GM_ADDR tilingGM, AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);

        int32_t hiddenSize = tiling_.hiddenSize;
        int32_t seqLen = tiling_.seqLen;
        int32_t batch = tiling_.batch;
        int32_t tileSize = tiling_.tileSize;

        timeDecayGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(timeDecay), hiddenSize);
        keyGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(key), batch * seqLen * hiddenSize);
        timeFirstGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(timeFirst), hiddenSize);
        valueGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(value), batch * seqLen * hiddenSize);
        maxStateGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(maxState), batch * hiddenSize);
        numStateGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(numState), batch * hiddenSize);
        denStateGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(denState), batch * hiddenSize);
        outputGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output), batch * seqLen * hiddenSize);
        maxStateOutGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(maxStateOut), batch * hiddenSize);
        numStateOutGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(numStateOut), batch * hiddenSize);
        denStateOutGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(denStateOut), batch * hiddenSize);

        pipe_ = pipe;

        pipe_->InitBuffer(timeDecayBuf_, tileSize * sizeof(float));
        pipe_->InitBuffer(timeFirstBuf_, tileSize * sizeof(float));
        pipe_->InitBuffer(maxStateBuf_, tileSize * sizeof(float));
        pipe_->InitBuffer(numStateBuf_, tileSize * sizeof(float));
        pipe_->InitBuffer(denStateBuf_, tileSize * sizeof(float));
        pipe_->InitBuffer(keyBuf_, tileSize * sizeof(float));
        pipe_->InitBuffer(valBuf_, tileSize * sizeof(float));
        pipe_->InitBuffer(e1Buf_, tileSize * sizeof(float));
        pipe_->InitBuffer(e2Buf_, tileSize * sizeof(float));
        pipe_->InitBuffer(tmp1Buf_, tileSize * sizeof(float));
        pipe_->InitBuffer(tmp2Buf_, tileSize * sizeof(float));
        pipe_->InitBuffer(outBuf_, tileSize * sizeof(float));

        if ASCEND_IS_AIV {
            timeDecayLocal_ = timeDecayBuf_.Get<float>();
            timeFirstLocal_ = timeFirstBuf_.Get<float>();
            maxStateLocal_ = maxStateBuf_.Get<float>();
            numStateLocal_ = numStateBuf_.Get<float>();
            denStateLocal_ = denStateBuf_.Get<float>();
            keyLocal_ = keyBuf_.Get<float>();
            valLocal_ = valBuf_.Get<float>();
            e1Local_ = e1Buf_.Get<float>();
            e2Local_ = e2Buf_.Get<float>();
            tmp1Local_ = tmp1Buf_.Get<float>();
            tmp2Local_ = tmp2Buf_.Get<float>();
            outLocal_ = outBuf_.Get<float>();
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            int32_t batch = tiling_.batch;
            int32_t seqLen = tiling_.seqLen;
            int32_t hiddenSize = tiling_.hiddenSize;
            int32_t tileSize = tiling_.tileSize;
            int32_t numTiles = tiling_.numTiles;

            int32_t blockIdx = AscendC::GetBlockIdx();
            int32_t bIdx = blockIdx / numTiles;
            int32_t tIdx = blockIdx % numTiles;
            int32_t hStart = tIdx * tileSize;
            int32_t actualTile = tileSize;
            if (hStart + actualTile > hiddenSize) {
                actualTile = hiddenSize - hStart;
            }

            if (bIdx >= batch) {
                return;
            }

            int32_t stateOffset = bIdx * hiddenSize + hStart;

            // Load time_decay and time_first for this tile
            LoadGmToUb(timeDecayLocal_, timeDecayGM_[hStart], static_cast<uint32_t>(actualTile));
            AscendC::PipeBarrier<PIPE_ALL>();
            LoadGmToUb(timeFirstLocal_, timeFirstGM_[hStart], static_cast<uint32_t>(actualTile));
            AscendC::PipeBarrier<PIPE_ALL>();

            // Load initial states
            LoadGmToUb(maxStateLocal_, maxStateGM_[stateOffset], static_cast<uint32_t>(actualTile));
            AscendC::PipeBarrier<PIPE_ALL>();
            LoadGmToUb(numStateLocal_, numStateGM_[stateOffset], static_cast<uint32_t>(actualTile));
            AscendC::PipeBarrier<PIPE_ALL>();
            LoadGmToUb(denStateLocal_, denStateGM_[stateOffset], static_cast<uint32_t>(actualTile));
            AscendC::PipeBarrier<PIPE_ALL>();

            // Precompute time_decay_exp = -exp(time_decay) into e1Local_
            AscendC::Exp(e1Local_, timeDecayLocal_, static_cast<uint32_t>(actualTile));
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Muls(e1Local_, e1Local_, static_cast<float>(-1.0f), static_cast<uint32_t>(actualTile));
            AscendC::PipeBarrier<PIPE_ALL>();

            for (int t = 0; t < seqLen; ++t) {
                int32_t kvOffset = bIdx * seqLen * hiddenSize + t * hiddenSize + hStart;
                int32_t outOffset = kvOffset;

                // Load key and value for this timestep
                LoadGmToUb(keyLocal_, keyGM_[kvOffset], static_cast<uint32_t>(actualTile));
                AscendC::PipeBarrier<PIPE_ALL>();
                LoadGmToUb(valLocal_, valueGM_[kvOffset], static_cast<uint32_t>(actualTile));
                AscendC::PipeBarrier<PIPE_ALL>();

                // ---- Output computation ----
                // tmp1 = key + time_first
                AscendC::Add(tmp1Local_, keyLocal_, timeFirstLocal_, static_cast<uint32_t>(actualTile));
                AscendC::PipeBarrier<PIPE_ALL>();
                // tmp2 = max(max_state, tmp1)
                AscendC::Max(tmp2Local_, maxStateLocal_, tmp1Local_, static_cast<uint32_t>(actualTile));
                AscendC::PipeBarrier<PIPE_ALL>();

                // e1_out = exp(max_state - tmp2)
                AscendC::Sub(e1Local_, maxStateLocal_, tmp2Local_, static_cast<uint32_t>(actualTile));
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Exp(e1Local_, e1Local_, static_cast<uint32_t>(actualTile));
                AscendC::PipeBarrier<PIPE_ALL>();

                // e2_out = exp(tmp1 - tmp2)
                AscendC::Sub(e2Local_, tmp1Local_, tmp2Local_, static_cast<uint32_t>(actualTile));
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Exp(e2Local_, e2Local_, static_cast<uint32_t>(actualTile));
                AscendC::PipeBarrier<PIPE_ALL>();

                // numerator = e1_out * num_state + e2_out * value
                AscendC::Mul(tmp1Local_, e1Local_, numStateLocal_, static_cast<uint32_t>(actualTile));
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Mul(tmp2Local_, e2Local_, valLocal_, static_cast<uint32_t>(actualTile));
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Add(outLocal_, tmp1Local_, tmp2Local_, static_cast<uint32_t>(actualTile));
                AscendC::PipeBarrier<PIPE_ALL>();

                // denominator = e1_out * den_state + e2_out
                AscendC::Mul(tmp1Local_, e1Local_, denStateLocal_, static_cast<uint32_t>(actualTile));
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Add(tmp2Local_, tmp1Local_, e2Local_, static_cast<uint32_t>(actualTile));
                AscendC::PipeBarrier<PIPE_ALL>();

                // Add small epsilon to prevent division by zero
                AscendC::Adds(tmp2Local_, tmp2Local_, static_cast<float>(1e-7f), static_cast<uint32_t>(actualTile));
                AscendC::PipeBarrier<PIPE_ALL>();

                // output = numerator / denominator
                AscendC::Div(outLocal_, outLocal_, tmp2Local_, static_cast<uint32_t>(actualTile));
                AscendC::PipeBarrier<PIPE_ALL>();

                // Store output
                StoreUbToGm(outputGM_[outOffset], outLocal_, static_cast<uint32_t>(actualTile));
                AscendC::PipeBarrier<PIPE_ALL>();

                // ---- State update ----
                // Recompute time_decay_exp = -exp(time_decay) into e1Local_
                AscendC::Exp(e1Local_, timeDecayLocal_, static_cast<uint32_t>(actualTile));
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Muls(e1Local_, e1Local_, static_cast<float>(-1.0f), static_cast<uint32_t>(actualTile));
                AscendC::PipeBarrier<PIPE_ALL>();

                // tmp1 = max_state + time_decay_exp
                AscendC::Add(tmp1Local_, maxStateLocal_, e1Local_, static_cast<uint32_t>(actualTile));
                AscendC::PipeBarrier<PIPE_ALL>();
                // max_state = max(tmp1, key)  -> update maxStateLocal_ directly
                AscendC::Max(maxStateLocal_, tmp1Local_, keyLocal_, static_cast<uint32_t>(actualTile));
                AscendC::PipeBarrier<PIPE_ALL>();

                // e1_state = exp(tmp1 - max_state)
                AscendC::Sub(e1Local_, tmp1Local_, maxStateLocal_, static_cast<uint32_t>(actualTile));
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Exp(e1Local_, e1Local_, static_cast<uint32_t>(actualTile));
                AscendC::PipeBarrier<PIPE_ALL>();

                // e2_state = exp(key - max_state)
                AscendC::Sub(e2Local_, keyLocal_, maxStateLocal_, static_cast<uint32_t>(actualTile));
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Exp(e2Local_, e2Local_, static_cast<uint32_t>(actualTile));
                AscendC::PipeBarrier<PIPE_ALL>();

                // num_state = e1_state * num_state + e2_state * value
                AscendC::Mul(tmp1Local_, e1Local_, numStateLocal_, static_cast<uint32_t>(actualTile));
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Mul(tmp2Local_, e2Local_, valLocal_, static_cast<uint32_t>(actualTile));
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Add(numStateLocal_, tmp1Local_, tmp2Local_, static_cast<uint32_t>(actualTile));
                AscendC::PipeBarrier<PIPE_ALL>();

                // den_state = e1_state * den_state + e2_state
                AscendC::Mul(tmp1Local_, e1Local_, denStateLocal_, static_cast<uint32_t>(actualTile));
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Add(denStateLocal_, tmp1Local_, e2Local_, static_cast<uint32_t>(actualTile));
                AscendC::PipeBarrier<PIPE_ALL>();
            }

            // Store final states
            StoreUbToGm(maxStateOutGM_[stateOffset], maxStateLocal_, static_cast<uint32_t>(actualTile));
            AscendC::PipeBarrier<PIPE_ALL>();
            StoreUbToGm(numStateOutGM_[stateOffset], numStateLocal_, static_cast<uint32_t>(actualTile));
            AscendC::PipeBarrier<PIPE_ALL>();
            StoreUbToGm(denStateOutGM_[stateOffset], denStateLocal_, static_cast<uint32_t>(actualTile));
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

private:
    TimeDecayExpStabilizationTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};

    AscendC::GlobalTensor<float> timeDecayGM_;
    AscendC::GlobalTensor<float> keyGM_;
    AscendC::GlobalTensor<float> timeFirstGM_;
    AscendC::GlobalTensor<float> valueGM_;
    AscendC::GlobalTensor<float> maxStateGM_;
    AscendC::GlobalTensor<float> numStateGM_;
    AscendC::GlobalTensor<float> denStateGM_;
    AscendC::GlobalTensor<float> outputGM_;
    AscendC::GlobalTensor<float> maxStateOutGM_;
    AscendC::GlobalTensor<float> numStateOutGM_;
    AscendC::GlobalTensor<float> denStateOutGM_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> timeDecayBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> timeFirstBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maxStateBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> numStateBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> denStateBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> keyBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> valBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> e1Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> e2Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp1Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp2Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf_;

    AscendC::LocalTensor<float> timeDecayLocal_;
    AscendC::LocalTensor<float> timeFirstLocal_;
    AscendC::LocalTensor<float> maxStateLocal_;
    AscendC::LocalTensor<float> numStateLocal_;
    AscendC::LocalTensor<float> denStateLocal_;
    AscendC::LocalTensor<float> keyLocal_;
    AscendC::LocalTensor<float> valLocal_;
    AscendC::LocalTensor<float> e1Local_;
    AscendC::LocalTensor<float> e2Local_;
    AscendC::LocalTensor<float> tmp1Local_;
    AscendC::LocalTensor<float> tmp2Local_;
    AscendC::LocalTensor<float> outLocal_;
};
