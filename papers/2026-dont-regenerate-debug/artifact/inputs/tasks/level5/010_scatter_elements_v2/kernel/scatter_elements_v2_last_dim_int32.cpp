#include "scatter_elements_v2_last_dim_kernel.h"

extern "C" __global__ __aicore__ void scatter_elements_v2_last_dim_int32_custom(
    GM_ADDR selfGM,
    GM_ADDR indexGM,
    GM_ADDR updatesGM,
    GM_ADDR outGM,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    ScatterElementsV2LastDimKernel<int32_t> kernel;
    kernel.Init(selfGM, indexGM, updatesGM, outGM, tiling, &pipe);
    kernel.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void scatter_elements_v2_last_dim_int32_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *selfGM,
    uint8_t *indexGM,
    uint8_t *updatesGM,
    uint8_t *outGM,
    uint8_t *tiling)
{
    scatter_elements_v2_last_dim_int32_custom<<<blockDim, nullptr, stream>>>(selfGM, indexGM, updatesGM, outGM, tiling);
}
#endif
