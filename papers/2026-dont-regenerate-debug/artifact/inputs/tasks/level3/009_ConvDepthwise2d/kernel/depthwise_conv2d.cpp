#include "depthwise_conv2d_kernel.h"

extern "C" __global__ __aicore__ void depthwise_conv2d_custom(
    GM_ADDR x,
    GM_ADDR w,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    DepthwiseConv2DKernel kernel;
    kernel.Init(x, w, bias, y, tiling, &pipe);
    kernel.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void depthwise_conv2d_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *w,
    uint8_t *bias,
    uint8_t *y,
    uint8_t *tiling)
{
    depthwise_conv2d_custom<<<blockDim, nullptr, stream>>>(x, w, bias, y, tiling);
}
#endif
