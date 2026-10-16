#include "gather_elements_v2_last_dim_kernel.h"

extern "C" __global__ __aicore__ void gather_elements_v2_last_dim_fp16_custom(
    GM_ADDR x,
    GM_ADDR index,
    GM_ADDR rowMap,
    GM_ADDR y,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    GatherElementsV2LastDimKernel<half> kernel;
    kernel.Init(x, index, rowMap, y, tiling, &pipe);
    kernel.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void gather_elements_v2_last_dim_fp16_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *x,
    uint8_t *index,
    uint8_t *rowMap,
    uint8_t *y,
    uint8_t *tiling)
{
    gather_elements_v2_last_dim_fp16_custom<<<blockDim, nullptr, stream>>>(x, index, rowMap, y, tiling);
}
#endif
