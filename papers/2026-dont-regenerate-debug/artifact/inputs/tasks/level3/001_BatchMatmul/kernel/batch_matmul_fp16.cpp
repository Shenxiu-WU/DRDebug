#include "batch_matmul.h"

extern "C" __global__ __aicore__ void batch_matmul_custom_fp16(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    AscendC::TPipe pipe;
    BatchMatmulKernel<half, half, float> kernel;
    kernel.Init(a, b, c, tiling, &pipe);
    kernel.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void batch_matmul_do_fp16(uint32_t blockDim, void *stream,
                                     uint8_t *a, uint8_t *b, uint8_t *c,
                                     uint8_t *tiling)
{
    batch_matmul_custom_fp16<<<blockDim, nullptr, stream>>>(a, b, c, tiling);
}
#endif // #ifndef ASCENDC_CPU_DEBUG
