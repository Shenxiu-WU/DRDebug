#include "conv1d_kernel.h"

extern "C" __global__ __aicore__ void conv1d_custom(GM_ADDR x, GM_ADDR weight,
                                                    GM_ADDR bias, GM_ADDR y,
                                                    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    Conv1DKernel kernel;
    kernel.Init(x, weight, bias, y, tiling, &pipe);
    kernel.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void conv1d_do(uint32_t blockDim, void *stream, uint8_t *x,
                          uint8_t *weight, uint8_t *bias, uint8_t *y,
                          uint8_t *tiling)
{
    conv1d_custom<<<blockDim, nullptr, stream>>>(x, weight, bias, y, tiling);
}
#endif
