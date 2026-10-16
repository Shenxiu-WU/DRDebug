#ifndef IOU_KERNEL_H
#define IOU_KERNEL_H

#include <kernel_operator.h>
#include "iou_tiling.h"
#include "kernel_common.h"
#include "vector_tile.h"

template <typename dataType>
class IouKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR bboxes_x1, GM_ADDR bboxes_y1, GM_ADDR bboxes_x2, GM_ADDR bboxes_y2,
        GM_ADDR gtboxes_x1, GM_ADDR gtboxes_y1, GM_ADDR gtboxes_x2, GM_ADDR gtboxes_y2,
        GM_ADDR out, GM_ADDR tilingGM, AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);
        bboxes_x1_GM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(bboxes_x1), tiling_.N);
        bboxes_y1_GM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(bboxes_y1), tiling_.N);
        bboxes_x2_GM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(bboxes_x2), tiling_.N);
        bboxes_y2_GM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(bboxes_y2), tiling_.N);
        gtboxes_x1_GM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(gtboxes_x1), tiling_.M);
        gtboxes_y1_GM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(gtboxes_y1), tiling_.M);
        gtboxes_x2_GM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(gtboxes_x2), tiling_.M);
        gtboxes_y2_GM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(gtboxes_y2), tiling_.M);
        out_GM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(out), tiling_.M * tiling_.N);
        pipe_ = pipe;

        if ASCEND_IS_AIV {
            const uint32_t tileBytes = tiling_.tileN * sizeof(dataType);
            pipe_->InitBuffer(bboxes_x1_buf_, tileBytes);
            pipe_->InitBuffer(bboxes_y1_buf_, tileBytes);
            pipe_->InitBuffer(bboxes_x2_buf_, tileBytes);
            pipe_->InitBuffer(bboxes_y2_buf_, tileBytes);
            pipe_->InitBuffer(lt_x_buf_, tileBytes);
            pipe_->InitBuffer(lt_y_buf_, tileBytes);
            pipe_->InitBuffer(rb_x_buf_, tileBytes);
            pipe_->InitBuffer(rb_y_buf_, tileBytes);
            pipe_->InitBuffer(inter_buf_, tileBytes);
            pipe_->InitBuffer(tmp_buf_, tileBytes);
            pipe_->InitBuffer(result_buf_, tileBytes);
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
        const int coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        for (int localIdx = 0; localIdx < tiling_.tasksPerCore; ++localIdx) {
            int bx = coreIdx * tiling_.tasksPerCore + localIdx;
            if (bx >= tiling_.M) {
                continue;
            }
            float gt_x1_f = static_cast<float>(gtboxes_x1_GM_.GetValue(bx));
            float gt_y1_f = static_cast<float>(gtboxes_y1_GM_.GetValue(bx));
            float gt_x2_f = static_cast<float>(gtboxes_x2_GM_.GetValue(bx));
            float gt_y2_f = static_cast<float>(gtboxes_y2_GM_.GetValue(bx));
            float gt_area_f = (gt_x2_f - gt_x1_f) * (gt_y2_f - gt_y1_f);
            dataType gt_x1 = static_cast<dataType>(gt_x1_f);
            dataType gt_y1 = static_cast<dataType>(gt_y1_f);
            dataType gt_x2 = static_cast<dataType>(gt_x2_f);
            dataType gt_y2 = static_cast<dataType>(gt_y2_f);
            dataType gt_area = static_cast<dataType>(gt_area_f);

            int nNum = (tiling_.N + tiling_.tileN - 1) / tiling_.tileN;
            for (int ny = 0; ny < nNum; ++ny) {
                int nStart = ny * tiling_.tileN;
                int actualN = tiling_.tileN;
                if (nStart + actualN > tiling_.N) {
                    actualN = tiling_.N - nStart;
                }
                if (actualN <= 0) {
                    continue;
                }
                ComputeTile(bx, nStart, actualN, gt_x1, gt_y1, gt_x2, gt_y2, gt_area);
            }
        }
        }
    }

private:
    __aicore__ inline void ComputeTile(int bx, int nStart, int actualN,
                                       dataType gt_x1, dataType gt_y1,
                                       dataType gt_x2, dataType gt_y2,
                                       dataType gt_area)
    {
        auto bboxes_x1_loc = bboxes_x1_buf_.Get<dataType>();
        auto bboxes_y1_loc = bboxes_y1_buf_.Get<dataType>();
        auto bboxes_x2_loc = bboxes_x2_buf_.Get<dataType>();
        auto bboxes_y2_loc = bboxes_y2_buf_.Get<dataType>();
        auto lt_x_loc = lt_x_buf_.Get<dataType>();
        auto lt_y_loc = lt_y_buf_.Get<dataType>();
        auto rb_x_loc = rb_x_buf_.Get<dataType>();
        auto rb_y_loc = rb_y_buf_.Get<dataType>();
        auto inter_loc = inter_buf_.Get<dataType>();
        auto tmp_loc = tmp_buf_.Get<dataType>();
        auto result_loc = result_buf_.Get<dataType>();

        LoadGmToUb(bboxes_x1_loc, bboxes_x1_GM_[nStart], static_cast<uint32_t>(actualN));
        LoadGmToUb(bboxes_y1_loc, bboxes_y1_GM_[nStart], static_cast<uint32_t>(actualN));
        LoadGmToUb(bboxes_x2_loc, bboxes_x2_GM_[nStart], static_cast<uint32_t>(actualN));
        LoadGmToUb(bboxes_y2_loc, bboxes_y2_GM_[nStart], static_cast<uint32_t>(actualN));
        AscendC::PipeBarrier<PIPE_MTE2>();

        AscendC::Maxs(lt_x_loc, bboxes_x1_loc, gt_x1, static_cast<uint32_t>(actualN));
        AscendC::Maxs(lt_y_loc, bboxes_y1_loc, gt_y1, static_cast<uint32_t>(actualN));
        AscendC::Mins(rb_x_loc, bboxes_x2_loc, gt_x2, static_cast<uint32_t>(actualN));
        AscendC::Mins(rb_y_loc, bboxes_y2_loc, gt_y2, static_cast<uint32_t>(actualN));

        AscendC::Sub(lt_x_loc, rb_x_loc, lt_x_loc, static_cast<uint32_t>(actualN));
        AscendC::Sub(lt_y_loc, rb_y_loc, lt_y_loc, static_cast<uint32_t>(actualN));
        dataType zero = static_cast<dataType>(0.0f);
        AscendC::Maxs(lt_x_loc, lt_x_loc, zero, static_cast<uint32_t>(actualN));
        AscendC::Maxs(lt_y_loc, lt_y_loc, zero, static_cast<uint32_t>(actualN));

        AscendC::Mul(inter_loc, lt_x_loc, lt_y_loc, static_cast<uint32_t>(actualN));

        AscendC::Sub(tmp_loc, bboxes_x2_loc, bboxes_x1_loc, static_cast<uint32_t>(actualN));
        AscendC::Sub(result_loc, bboxes_y2_loc, bboxes_y1_loc, static_cast<uint32_t>(actualN));
        AscendC::Mul(result_loc, result_loc, tmp_loc, static_cast<uint32_t>(actualN));

        if (tiling_.mode == 0) {
            AscendC::Adds(result_loc, result_loc, gt_area, static_cast<uint32_t>(actualN));
            AscendC::Sub(result_loc, result_loc, inter_loc, static_cast<uint32_t>(actualN));
        } else {
            AscendC::Duplicate(result_loc, gt_area, static_cast<uint32_t>(actualN));
        }
        dataType eps = static_cast<dataType>(tiling_.eps);
        AscendC::Maxs(result_loc, result_loc, eps, static_cast<uint32_t>(actualN));

        AscendC::Div(result_loc, inter_loc, result_loc, static_cast<uint32_t>(actualN));

        AscendC::PipeBarrier<PIPE_V>();
        StoreUbToGm(out_GM_[bx * tiling_.N + nStart], result_loc, static_cast<uint32_t>(actualN));
        AscendC::PipeBarrier<PIPE_MTE3>();
    }

    IouKernelTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};

    AscendC::GlobalTensor<dataType> bboxes_x1_GM_;
    AscendC::GlobalTensor<dataType> bboxes_y1_GM_;
    AscendC::GlobalTensor<dataType> bboxes_x2_GM_;
    AscendC::GlobalTensor<dataType> bboxes_y2_GM_;
    AscendC::GlobalTensor<dataType> gtboxes_x1_GM_;
    AscendC::GlobalTensor<dataType> gtboxes_y1_GM_;
    AscendC::GlobalTensor<dataType> gtboxes_x2_GM_;
    AscendC::GlobalTensor<dataType> gtboxes_y2_GM_;
    AscendC::GlobalTensor<dataType> out_GM_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> bboxes_x1_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bboxes_y1_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bboxes_x2_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bboxes_y2_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> lt_x_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> lt_y_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rb_x_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rb_y_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inter_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> result_buf_;
};

#endif
