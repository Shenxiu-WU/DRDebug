#include "iou_kernel.h"

extern "C" __global__ __aicore__ void iou_kernel_fp32(
    GM_ADDR bboxes_x1, GM_ADDR bboxes_y1, GM_ADDR bboxes_x2, GM_ADDR bboxes_y2,
    GM_ADDR gtboxes_x1, GM_ADDR gtboxes_y1, GM_ADDR gtboxes_x2, GM_ADDR gtboxes_y2,
    GM_ADDR out, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    AscendC::TPipe pipe;
    IouKernel<float> kernel;
    kernel.Init(bboxes_x1, bboxes_y1, bboxes_x2, bboxes_y2,
                gtboxes_x1, gtboxes_y1, gtboxes_x2, gtboxes_y2,
                out, tiling, &pipe);
    kernel.Process();
}

extern "C" void iou_do_fp32(
    uint32_t blockDim,
    void *stream,
    uint8_t *bboxes_x1, uint8_t *bboxes_y1, uint8_t *bboxes_x2, uint8_t *bboxes_y2,
    uint8_t *gtboxes_x1, uint8_t *gtboxes_y1, uint8_t *gtboxes_x2, uint8_t *gtboxes_y2,
    uint8_t *out, uint8_t *tiling)
{
    iou_kernel_fp32<<<blockDim, nullptr, stream>>>(
        bboxes_x1, bboxes_y1, bboxes_x2, bboxes_y2,
        gtboxes_x1, gtboxes_y1, gtboxes_x2, gtboxes_y2,
        out, tiling);
}
