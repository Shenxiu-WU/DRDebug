#include "avg_pool3_d_multi_w_kernel.h"

extern "C" __global__ __aicore__ void avg_pool3_d_multi_w_custom(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    AvgPool3DMultiWKernel kernel;
    kernel.Init(x, y, tiling, &pipe);
    kernel.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void avg_pool3_d_multi_w_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *tiling)
{
    avg_pool3_d_multi_w_custom<<<blockDim, nullptr, stream>>>(x, y, tiling);
}
#endif
