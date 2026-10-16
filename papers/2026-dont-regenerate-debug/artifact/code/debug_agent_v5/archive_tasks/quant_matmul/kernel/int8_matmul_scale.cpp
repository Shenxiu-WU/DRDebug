#include "int8_matmul_scale.h"

extern "C" __global__ __aicore__ void int8_matmul_scale_custom(GM_ADDR a, GM_ADDR b, GM_ADDR scale,
                                                               GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    Int8MatmulScaleKernel kernel;
    kernel.Init(a, b, scale, c, workspace, tiling, &pipe);
    kernel.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void int8_matmul_scale_do(uint32_t blockDim, void *stream,
                                     uint8_t *a, uint8_t *b, uint8_t *scale,
                                     uint8_t *c, uint8_t *workspace, uint8_t *tiling)
{
    int8_matmul_scale_custom<<<blockDim, nullptr, stream>>>(a, b, scale, c, workspace, tiling);
}
#endif
