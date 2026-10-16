#pragma once

#include "kernel_operator.h"
#include "kernel_common.h"
#include "inplace_index_add_tiling.h"

template <typename DataType>
class InplaceIndexAddKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR var,
        GM_ADDR value,
        GM_ADDR sortedIndices,
        GM_ADDR alpha,
        GM_ADDR out,
        GM_ADDR tilingGM,
        AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);
        varGM_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(var), tiling_.M * tiling_.N);
        valueGM_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(value), tiling_.K * tiling_.N);
        indicesGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sortedIndices), tiling_.K);
        alphaGM_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(alpha), 1);
        outGM_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(out), tiling_.M * tiling_.N);

        if ASCEND_IS_AIV {
            pipe_ = pipe;
            const uint32_t rowBytes = static_cast<uint32_t>(tiling_.N * sizeof(DataType));
            const uint32_t alignedRowBytes = AlignBytes(rowBytes);
            pipe_->InitBuffer(varRowQueue_, 1, alignedRowBytes);
            pipe_->InitBuffer(valRowQueue_, 1, alignedRowBytes);
            pipe_->InitBuffer(tmpRowBuf_, alignedRowBytes);
            pipe_->InitBuffer(alphaBuf_, AlignBytes(sizeof(DataType)));
            pipe_->InitBuffer(indicesBuf_, AlignBytes(tiling_.K * sizeof(int32_t)));
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
            const int subBlockIdx = AscendC::GetSubBlockIdx();

            const int mStart = coreIdx * tiling_.blockM;
            const int mEnd = (mStart + tiling_.blockM < tiling_.M) ? (mStart + tiling_.blockM) : tiling_.M;
            const int totalRows = mEnd - mStart;
            const int subBlockM = (totalRows + 1) / 2;

            int localMStart, localMEnd;
            if (subBlockIdx == 0) {
                localMStart = mStart;
                localMEnd = (mStart + subBlockM < mEnd) ? (mStart + subBlockM) : mEnd;
            } else {
                localMStart = (mStart + subBlockM < mEnd) ? (mStart + subBlockM) : mEnd;
                localMEnd = mEnd;
            }

            // Load indices to UB
            indicesLocal_ = indicesBuf_.Get<int32_t>();
            LoadIndices();

            // Load alpha to UB
            alphaLocal_ = alphaBuf_.Get<DataType>();
            LoadAlpha();
            const DataType alphaVal = alphaLocal_.GetValue(0);

            // Step 1: copy var -> out for local rows
            for (int row = localMStart; row < localMEnd; ++row) {
                CopyVarRowToOut(row);
            }

            // Step 2: scatter-add value to out
            for (int i = 0; i < tiling_.K; ++i) {
                const int32_t idx = indicesLocal_.GetValue(i);
                if (idx >= localMStart && idx < localMEnd) {
                    ProcessRow(idx, i, alphaVal);
                }
            }
        }
    }

private:
    __aicore__ inline uint32_t AlignBytes(uint32_t bytes) const
    {
        return ((bytes + 31U) / 32U) * 32U;
    }

    __aicore__ inline uint32_t RowBytes() const
    {
        return static_cast<uint32_t>(tiling_.N * sizeof(DataType));
    }

    __aicore__ inline void LoadIndices()
    {
        const uint32_t indicesBytes = static_cast<uint32_t>(tiling_.K * sizeof(int32_t));
        AscendC::DataCopyExtParams copyParams{1, indicesBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
        AscendC::DataCopyPad(indicesLocal_, indicesGM_, copyParams, padParams);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void LoadAlpha()
    {
        const uint32_t alphaBytes = static_cast<uint32_t>(sizeof(DataType));
        AscendC::DataCopyExtParams copyParams{1, alphaBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<DataType> padParams{true, 0, 0, static_cast<DataType>(0)};
        AscendC::DataCopyPad(alphaLocal_, alphaGM_, copyParams, padParams);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }

    __aicore__ inline void CopyVarRowToOut(int row)
    {
        const uint32_t rowBytes = RowBytes();
        varRowQueue_.AllocTensor<DataType>(varRowLocal_);
        AscendC::DataCopyExtParams copyParams{1, rowBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<DataType> padParams{true, 0, 0, static_cast<DataType>(0)};
        AscendC::DataCopyPad(varRowLocal_, varGM_[row * tiling_.N], copyParams, padParams);
        varRowQueue_.EnQue(varRowLocal_);
        varRowQueue_.DeQue<DataType>(varRowLocal_);
        AscendC::DataCopyPad(outGM_[row * tiling_.N], varRowLocal_, copyParams);
        varRowQueue_.FreeTensor(varRowLocal_);
    }

    __aicore__ inline void ProcessRow(int row, int valueRow, DataType alphaVal)
    {
        const uint32_t rowBytes = RowBytes();
        const uint32_t count = static_cast<uint32_t>(tiling_.N);

        // Load out row to varRowLocal
        varRowQueue_.AllocTensor<DataType>(varRowLocal_);
        AscendC::DataCopyExtParams copyParams{1, rowBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<DataType> padParams{true, 0, 0, static_cast<DataType>(0)};
        AscendC::DataCopyPad(varRowLocal_, outGM_[row * tiling_.N], copyParams, padParams);
        varRowQueue_.EnQue(varRowLocal_);
        varRowQueue_.DeQue<DataType>(varRowLocal_);

        // Load value row to valRowLocal
        valRowQueue_.AllocTensor<DataType>(valRowLocal_);
        AscendC::DataCopyPad(valRowLocal_, valueGM_[valueRow * tiling_.N], copyParams, padParams);
        valRowQueue_.EnQue(valRowLocal_);
        valRowQueue_.DeQue<DataType>(valRowLocal_);

        // Compute: varRow += valRow * alpha
        tmpRowLocal_ = tmpRowBuf_.Get<DataType>();
        AscendC::Muls(tmpRowLocal_, valRowLocal_, alphaVal, count);
        AscendC::Add(varRowLocal_, varRowLocal_, tmpRowLocal_, count);
        AscendC::PipeBarrier<PIPE_ALL>();

        // Write back
        varRowQueue_.EnQue(varRowLocal_);
        varRowQueue_.DeQue<DataType>(varRowLocal_);
        AscendC::DataCopyPad(outGM_[row * tiling_.N], varRowLocal_, copyParams);
        varRowQueue_.FreeTensor(varRowLocal_);
        valRowQueue_.FreeTensor(valRowLocal_);
    }

    InplaceIndexAddTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};

    AscendC::GlobalTensor<DataType> varGM_;
    AscendC::GlobalTensor<DataType> valueGM_;
    AscendC::GlobalTensor<int32_t> indicesGM_;
    AscendC::GlobalTensor<DataType> alphaGM_;
    AscendC::GlobalTensor<DataType> outGM_;

    AscendC::TQue<AscendC::TPosition::VECIN, 0> varRowQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> valRowQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpRowBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> alphaBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indicesBuf_;

    AscendC::LocalTensor<DataType> varRowLocal_;
    AscendC::LocalTensor<DataType> valRowLocal_;
    AscendC::LocalTensor<DataType> tmpRowLocal_;
    AscendC::LocalTensor<DataType> alphaLocal_;
    AscendC::LocalTensor<int32_t> indicesLocal_;
};
