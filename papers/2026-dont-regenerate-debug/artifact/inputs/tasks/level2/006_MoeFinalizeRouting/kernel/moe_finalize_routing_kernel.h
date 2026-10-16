#ifndef MOE_FINALIZE_ROUTING_KERNEL_H
#define MOE_FINALIZE_ROUTING_KERNEL_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <type_traits>
#include "kernel_operator.h"
#include "kernel_common.h"
#include "moe_finalize_routing_tiling.h"
#include "vector_tile.h"

template <typename dataType>
class MoeFinalizeRoutingKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR expanded_permuted_rows,
        GM_ADDR scales,
        GM_ADDR expanded_src_to_dst_row,
        GM_ADDR export_for_source_row,
        GM_ADDR bias,
        GM_ADDR skip1,
        GM_ADDR skip2,
        GM_ADDR output,
        GM_ADDR tilingGM,
        AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);

        if (tiling_.mode == 0 || tiling_.mode == 2) {
            eprNDGM_.SetGlobalBuffer(
                reinterpret_cast<__gm__ dataType *>(expanded_permuted_rows),
                tiling_.NUM_ROWS * tiling_.K * tiling_.H);
        } else {
            eprDGM_.SetGlobalBuffer(
                reinterpret_cast<__gm__ dataType *>(expanded_permuted_rows),
                tiling_.E * tiling_.C * tiling_.H);
        }

        scalesGM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ dataType *>(scales),
            tiling_.NUM_ROWS * tiling_.K);
        idxGM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(expanded_src_to_dst_row),
            tiling_.NUM_ROWS * tiling_.K);
        exportGM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(export_for_source_row),
            tiling_.NUM_ROWS * tiling_.K);

        if (tiling_.has_bias != 0) {
            biasGM_.SetGlobalBuffer(
                reinterpret_cast<__gm__ dataType *>(bias),
                tiling_.E * tiling_.H);
        }
        if (tiling_.has_skip1 != 0) {
            skip1GM_.SetGlobalBuffer(
                reinterpret_cast<__gm__ dataType *>(skip1),
                tiling_.NUM_ROWS * tiling_.H);
        }
        if (tiling_.has_skip2 != 0) {
            skip2GM_.SetGlobalBuffer(
                reinterpret_cast<__gm__ dataType *>(skip2),
                tiling_.NUM_ROWS * tiling_.H);
        }

        outputGM_.SetGlobalBuffer(
            reinterpret_cast<__gm__ dataType *>(output),
            tiling_.NUM_ROWS * tiling_.H);

        if ASCEND_IS_AIV {
            pipe_ = pipe;
            hNum_ = CeilDivU32(static_cast<uint32_t>(tiling_.H),
                               static_cast<uint32_t>(tiling_.H_TILE));

            pipe_->InitBuffer(rowValsBuf_,
                              Align32(tiling_.H_TILE * sizeof(dataType)));
            pipe_->InitBuffer(outBuf_,
                              Align32(tiling_.H_TILE * sizeof(dataType)));
            pipe_->InitBuffer(scaleBuf_,
                              Align32(tiling_.K * sizeof(dataType)));
            pipe_->InitBuffer(idxBuf_,
                              Align32(tiling_.K * sizeof(int32_t)));
            pipe_->InitBuffer(expertBuf_,
                              Align32(tiling_.K * sizeof(int32_t)));

            if constexpr (!std::is_same_v<dataType, float>) {
                pipe_->InitBuffer(rowValsFloatBuf_,
                                  Align32(tiling_.H_TILE * sizeof(float)));
                pipe_->InitBuffer(outFloatBuf_,
                                  Align32(tiling_.H_TILE * sizeof(float)));
                pipe_->InitBuffer(scaleTileFloatBuf_,
                                  Align32(tiling_.H_TILE * sizeof(float)));
            }
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();

            for (int localIdx = 0; localIdx < tiling_.tasksPerCore; ++localIdx) {
                int taskId = coreIdx * tiling_.tasksPerCore + localIdx;
                if (taskId >= tiling_.totalTasks) {
                    continue;
                }

                int rowIdx = taskId / hNum_;
                int hTile = taskId % hNum_;
                int hStart = hTile * tiling_.H_TILE;
                int hEnd = (hStart + tiling_.H_TILE < tiling_.H)
                               ? (hStart + tiling_.H_TILE)
                               : tiling_.H;
                int hLen = hEnd - hStart;

                ProcessTile(rowIdx, hStart, hLen);
            }
        }
    }

private:
    __aicore__ inline void CastToFloat(
        AscendC::LocalTensor<float> &dst,
        AscendC::LocalTensor<dataType> &src,
        int hLen)
    {
        if constexpr (std::is_same_v<dataType, float>) {
            dst = src.template ReinterpretCast<float>();
        } else {
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE,
                          static_cast<uint32_t>(hLen));
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CastFromFloat(
        AscendC::LocalTensor<dataType> &dst,
        AscendC::LocalTensor<float> &src,
        int hLen)
    {
        if constexpr (!std::is_same_v<dataType, float>) {
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE,
                          static_cast<uint32_t>(hLen));
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline float GetScaleFloat(int k)
    {
        dataType scale = scaleLocal_.GetValue(k);
        if constexpr (std::is_same_v<dataType, float>) {
            return scale;
        } else {
            return static_cast<float>(scale);
        }
    }

    __aicore__ inline void ProcessTile(int rowIdx, int hStart, int hLen)
    {
        rowValsLocal_ = rowValsBuf_.Get<dataType>();
        outLocal_ = outBuf_.Get<dataType>();
        scaleLocal_ = scaleBuf_.Get<dataType>();
        idxLocal_ = idxBuf_.Get<int32_t>();
        expertLocal_ = expertBuf_.Get<int32_t>();

        if constexpr (!std::is_same_v<dataType, float>) {
            rowValsFloatLocal_ = rowValsFloatBuf_.Get<float>();
            outFloatLocal_ = outFloatBuf_.Get<float>();
            scaleTileFloatLocal_ = scaleTileFloatBuf_.Get<float>();
        }

        LoadGmToUb(scaleLocal_, scalesGM_[rowIdx * tiling_.K],
                   static_cast<uint32_t>(tiling_.K));
        LoadGmToUb(expertLocal_, exportGM_[rowIdx * tiling_.K],
                   static_cast<uint32_t>(tiling_.K));
        if (tiling_.mode == 0 || tiling_.mode == 2) {
            LoadGmToUb(idxLocal_, idxGM_[rowIdx * tiling_.K],
                       static_cast<uint32_t>(tiling_.K));
        }

        AscendC::PipeBarrier<PIPE_ALL>();

        // Initialize float accumulator
        if constexpr (std::is_same_v<dataType, float>) {
            AscendC::Duplicate(outLocal_, static_cast<dataType>(0),
                               static_cast<uint32_t>(hLen));
        } else {
            AscendC::Duplicate(outFloatLocal_, 0.0f,
                               static_cast<uint32_t>(hLen));
        }

        for (int k = 0; k < tiling_.K; ++k) {
            float scale = GetScaleFloat(k);
            int32_t srcIdx;
            if (tiling_.mode == 0 || tiling_.mode == 2) {
                srcIdx = idxLocal_.GetValue(k);
                LoadGmToUb(rowValsLocal_,
                           eprNDGM_[srcIdx * tiling_.H + hStart],
                           static_cast<uint32_t>(hLen));
            } else {
                srcIdx = expertLocal_.GetValue(k);
                LoadGmToUb(rowValsLocal_,
                           eprDGM_[srcIdx * tiling_.C * tiling_.H +
                                   rowIdx * tiling_.H + hStart],
                           static_cast<uint32_t>(hLen));
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            CastToFloat(rowValsFloatLocal_, rowValsLocal_, hLen);
            AscendC::Duplicate(scaleTileFloatLocal_, scale,
                               static_cast<uint32_t>(hLen));
            AscendC::Mul(rowValsFloatLocal_, rowValsFloatLocal_,
                         scaleTileFloatLocal_,
                         static_cast<uint32_t>(hLen));

            if constexpr (std::is_same_v<dataType, float>) {
                AscendC::Add(outLocal_, outLocal_, rowValsFloatLocal_,
                             static_cast<uint32_t>(hLen));
            } else {
                AscendC::Add(outFloatLocal_, outFloatLocal_,
                             rowValsFloatLocal_,
                             static_cast<uint32_t>(hLen));
            }
        }

        if (tiling_.has_bias != 0) {
            for (int k = 0; k < tiling_.K; ++k) {
                float scale = GetScaleFloat(k);
                int32_t expert = expertLocal_.GetValue(k);
                LoadGmToUb(rowValsLocal_,
                           biasGM_[expert * tiling_.H + hStart],
                           static_cast<uint32_t>(hLen));
                AscendC::PipeBarrier<PIPE_ALL>();

                CastToFloat(rowValsFloatLocal_, rowValsLocal_, hLen);
                AscendC::Duplicate(scaleTileFloatLocal_, scale,
                                   static_cast<uint32_t>(hLen));
                AscendC::Mul(rowValsFloatLocal_, rowValsFloatLocal_,
                             scaleTileFloatLocal_,
                             static_cast<uint32_t>(hLen));

                if constexpr (std::is_same_v<dataType, float>) {
                    AscendC::Add(outLocal_, outLocal_, rowValsFloatLocal_,
                                 static_cast<uint32_t>(hLen));
                } else {
                    AscendC::Add(outFloatLocal_, outFloatLocal_,
                                 rowValsFloatLocal_,
                                 static_cast<uint32_t>(hLen));
                }
            }
        }

        if (tiling_.has_skip1 != 0) {
            LoadGmToUb(rowValsLocal_,
                       skip1GM_[rowIdx * tiling_.H + hStart],
                       static_cast<uint32_t>(hLen));
            AscendC::PipeBarrier<PIPE_ALL>();
            CastToFloat(rowValsFloatLocal_, rowValsLocal_, hLen);

            if constexpr (std::is_same_v<dataType, float>) {
                AscendC::Add(outLocal_, outLocal_, rowValsFloatLocal_,
                             static_cast<uint32_t>(hLen));
            } else {
                AscendC::Add(outFloatLocal_, outFloatLocal_,
                             rowValsFloatLocal_,
                             static_cast<uint32_t>(hLen));
            }
        }

        if (tiling_.has_skip2 != 0) {
            LoadGmToUb(rowValsLocal_,
                       skip2GM_[rowIdx * tiling_.H + hStart],
                       static_cast<uint32_t>(hLen));
            AscendC::PipeBarrier<PIPE_ALL>();
            CastToFloat(rowValsFloatLocal_, rowValsLocal_, hLen);

            if constexpr (std::is_same_v<dataType, float>) {
                AscendC::Add(outLocal_, outLocal_, rowValsFloatLocal_,
                             static_cast<uint32_t>(hLen));
            } else {
                AscendC::Add(outFloatLocal_, outFloatLocal_,
                             rowValsFloatLocal_,
                             static_cast<uint32_t>(hLen));
            }
        }

        if constexpr (!std::is_same_v<dataType, float>) {
            CastFromFloat(outLocal_, outFloatLocal_, hLen);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
        StoreUbToGm(outputGM_[rowIdx * tiling_.H + hStart], outLocal_,
                    static_cast<uint32_t>(hLen));
    }

    MoeFinalizeRoutingTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};
    int hNum_{0};

    AscendC::GlobalTensor<dataType> eprNDGM_;
    AscendC::GlobalTensor<dataType> eprDGM_;
    AscendC::GlobalTensor<dataType> scalesGM_;
    AscendC::GlobalTensor<int32_t> idxGM_;
    AscendC::GlobalTensor<int32_t> exportGM_;
    AscendC::GlobalTensor<dataType> biasGM_;
    AscendC::GlobalTensor<dataType> skip1GM_;
    AscendC::GlobalTensor<dataType> skip2GM_;
    AscendC::GlobalTensor<dataType> outputGM_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> rowValsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scaleBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> idxBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> expertBuf_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> rowValsFloatBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outFloatBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scaleTileFloatBuf_;

    AscendC::LocalTensor<dataType> rowValsLocal_;
    AscendC::LocalTensor<dataType> outLocal_;
    AscendC::LocalTensor<dataType> scaleLocal_;
    AscendC::LocalTensor<int32_t> idxLocal_;
    AscendC::LocalTensor<int32_t> expertLocal_;

    AscendC::LocalTensor<float> rowValsFloatLocal_;
    AscendC::LocalTensor<float> outFloatLocal_;
    AscendC::LocalTensor<float> scaleTileFloatLocal_;
};

#endif
