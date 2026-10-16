#include "inplace_index_add_kernel.h"

extern "C" __global__ __aicore__ void inplace_index_add_fp32_custom(
    GM_ADDR var,
    GM_ADDR value,
    GM_ADDR sortedIndices,
    GM_ADDR alpha,
    GM_ADDR out,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::TPipe pipe;
    InplaceIndexAddKernel<float> kernel;
    kernel.Init(var, value, sortedIndices, alpha, out, tiling, &pipe);
    kernel.Process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void inplace_index_add_fp32_do(
    uint32_t blockDim,
    void *stream,
    uint8_t *var,
    uint8_t *value,
    uint8_t *sortedIndices,
    uint8_t *alpha,
    uint8_t *out,
    uint8_t *tiling)
{
    inplace_index_add_fp32_custom<<<blockDim, nullptr, stream>>>(var, value, sortedIndices, alpha, out, tiling);
}
#endif
